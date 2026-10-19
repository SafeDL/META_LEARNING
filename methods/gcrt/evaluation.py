"""Thirty-six isolated full trajectories and causal stopping/maintenance replay."""

import multiprocessing as mp
import hashlib
import platform
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from methods.baselines.hgt.io import read_json, write_json
from benchmark.config import SUT_IDS
from benchmark.risk_channel import disclose_risk
from benchmark.timing import QueryClock, TimedRiskOracle, now
from methods.gcrt.build import build_family, workspace_for
from methods.gcrt.config import BUDGET, INITIAL, ROOT, ROLES, SEED, SEGMENT
from methods.gcrt.controller import block_scores, threshold
from methods.gcrt.data import StateRecorder, forecast_task, save_task
from methods.gcrt.selection import select
from methods.gcrt.train import load_network, update

from benchmark.data import bank_path, measured_seconds
from methods.common.features import prepare_pool


def selector_worker(connection):
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            while True:
                command, task = connection.recv()
                if command == "stop":
                    return
                clock = QueryClock()
                recorder = StateRecorder(clock)
                result = select(
                    task["prior"],
                    TimedRiskOracle(connection, clock),
                    initial_indices=task["initial_indices"],
                    recorder=recorder,
                    clock=clock,
                    **task["parameters"],
                )
                result["timing"] = clock.report()
                result["state_recording_s"] = recorder.seconds
                result["records"] = recorder.records
                connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def evaluate_family(workspace):
    parameters = {
        **read_json(workspace.root / "prior.json")["selected"],
        "policy": read_json(workspace.root / "policy.json")["selected"],
    }
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    try:
        for pool in (0, 1, 2):
            cache = prepare_pool(workspace, pool)
            for sut in SUT_IDS:
                path = workspace.root / "trajectories" / f"pool_{pool}" / f"{sut}.json"
                if path.exists() and path.with_suffix(".pt").exists():
                    continue
                with np.load(bank_path(pool, sut)) as bank:
                    keep = bank["x"][:, 4] == workspace.family
                    np.testing.assert_array_equal(bank["x"][keep], cache["prior"].x)
                    truth = bank["risk"][keep].copy()
                    scenario_ids = bank["scenario_id"][keep].copy()
                seconds = measured_seconds(pool, sut, scenario_ids)
                result = disclose_risk(
                    parent,
                    {
                        "prior": cache["prior"],
                        "parameters": parameters,
                        "initial_indices": cache["common"]["initial_indices"],
                    },
                    truth,
                    process,
                )
                records = result.pop("records")
                selected = result["selected_indices"]
                if selected[:INITIAL] != cache["common"]["initial_indices"]:
                    raise ValueError("Initialization differs from the fixed comparison")
                found = np.cumsum(truth[selected] > 0.5).tolist()
                result.update(
                    family=workspace.root.name,
                    sut=sut,
                    pool=pool,
                    seed=SEED,
                    positives=int((truth > 0.5).sum()),
                    found=found,
                    H200=found[-1],
                    recall=(
                        found[-1] / int((truth > 0.5).sum())
                        if (truth > 0.5).any()
                        else None
                    ),
                    execution_s=seconds[selected].tolist(),
                    full_execution_s=float(seconds.sum()),
                    scenario_ids=scenario_ids[selected].tolist(),
                    physical_calls=0,
                )
                save_task(
                    path.with_suffix(".pt"),
                    records,
                    result,
                    sut=sut,
                    pool=pool,
                    budget=BUDGET,
                    initial=INITIAL,
                    positives=result["positives"],
                )
                write_json(path, result)
                write_json(
                    ROOT / "status.json",
                    {
                        "stage": "full_trajectories",
                        "status": "running",
                        "completed": len(list(ROOT.glob("*/trajectories/*/*.json"))),
                        "planned": 36,
                        "last_family": workspace.root.name,
                        "last_pool": pool,
                        "last_controller": sut,
                        "physical_calls": 0,
                    },
                )
                print(
                    f"TARGET {workspace.root.name} {pool} {sut} H200={found[-1]}",
                    flush=True,
                )
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()


def stopping_row(trace, prediction, gamma):
    hit = next(
        (r for r in prediction["forecast"] if r["deterministic"] or r["score"] > gamma),
        None,
    )
    used = hit["query"] if hit else BUDGET
    found = trace["found"][used - 1]
    full = trace["found"][-1]
    earliest = next(
        t for t in range(INITIAL, BUDGET + 1) if 20 * trace["found"][t - 1] >= 19 * full
    )
    check_seconds = sum(
        r["compute_s"] for r in prediction["forecast"] if r["query"] <= used
    )
    online = (
        trace["timing"]["task_initialization_s"]
        + sum(trace["timing"]["query_compute_s"][:used])
        + sum(trace["state_recording_s"][:used:SEGMENT])
        + sum(prediction["prequery_s"][:used])
        + sum(prediction["correction_s"][:used])
        + check_seconds
    )
    if hit:
        # Terminal state construction is needed although no query follows it.
        online += trace["state_recording_s"][used]
    positives = trace["positives"]
    return {
        "sut": trace["sut"],
        "pool": trace["pool"],
        "queries": used,
        "found": found,
        "full_found": full,
        "positives": positives,
        "recall": found / positives if positives else None,
        "retention": found / full if full else 1.0,
        "quality_failed": 20 * found < 19 * full,
        "missed": full - found,
        "excess_queries": max(used - earliest, 0),
        "online_s": online,
        "execution_s": sum(trace["execution_s"][:used]),
        "threshold": gamma,
        "terminal_decision_s": hit["compute_s"] if hit else 0.0,
        "H10": trace["found"][9],
        "H20": trace["found"][min(used, 20) - 1],
        "H50": trace["found"][min(used, 50) - 1],
        "H100": trace["found"][min(used, 100) - 1],
        "selected_indices": trace["selected_indices"][:used],
        "stop_reason": "quality_retained" if used < BUDGET else "budget_exhausted",
    }


def analyze_family(family):
    folder = ROOT / family
    settings = read_json(folder / "training.json")
    penalty = settings["penalty"]
    paths = sorted((folder / "trajectories").glob("*/*.json"))
    traces = [read_json(path) for path in paths]
    tasks = [
        torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
        for path in paths
    ]
    network = load_network(family)
    forecasts = [forecast_task(network, task, penalty, clock=now) for task in tasks]
    rng = np.random.default_rng(SEED)
    complete_controller = str(rng.choice(ROLES["training"]))
    incoming = []
    collection = []
    for task, trace, prediction in zip(tasks, traces, forecasts):
        if task["sut"] not in ROLES["training"]:
            continue
        row = stopping_row(trace, prediction, settings["collection_threshold"])
        visible = BUDGET if task["sut"] == complete_controller else row["queries"]
        incoming.append(
            {
                "sut": task["sut"],
                "pool": task["pool"],
                "budget": task["budget"],
                "initial": task["initial"],
                "records": task["records"][:visible],
                "labels": task["labels"][:visible],
                "natural_stop": row["queries"],
            }
        )
        collection.append(
            {
                "sut": task["sut"],
                "pool": task["pool"],
                "natural_stop": row["queries"],
                "actual_queries": visible,
                "random_complete": task["sut"] == complete_controller,
            }
        )
    write_json(
        folder / "collection.json",
        {
            "complete_probability": 0.5,
            "complete_controller": complete_controller,
            "threshold": settings["collection_threshold"],
            "tasks": collection,
            "threshold_source": "deterministic bootstrap; target calibration not consulted",
        },
    )
    if not ((folder / "updated.pt").exists() and (folder / "update.json").exists()):
        update(family, incoming)
    updated = load_network(family, updated=True)
    updated_forecasts = [
        forecast_task(updated, task, penalty, clock=now) for task in tasks
    ]
    without_correction = [
        forecast_task(network, task, penalty, correction=False, clock=now)
        for task in tasks
    ]
    direct = [
        forecast_task(network, task, penalty, direct_tilt=True, clock=now)
        for task in tasks
    ]
    rows = []
    calibration = {}
    for name, predictions in (
        ("initial", forecasts),
        ("updated", updated_forecasts),
        ("without_correction", without_correction),
        ("direct_tilt", direct),
    ):
        held = [p for p in predictions if p["sut"] in ROLES["calibration"]]
        scores = block_scores(held)
        calibration[name] = {
            "scores": scores,
            "empirical": threshold(scores.values()),
            "strict": threshold(scores.values(), "strict"),
            "contract": {
                "family": family,
                "network": name,
                "weights_sha256": hashlib.sha256(
                    (
                        folder / ("updated.pt" if name == "updated" else "model.pt")
                    ).read_bytes()
                ).hexdigest(),
                "budget": BUDGET,
                "initial": INITIAL,
                "pool_size": 1024,
                "segment": SEGMENT,
                "policy": read_json(folder / "policy.json")["selected"],
                "correction": (
                    "segment_rate_KL_projection"
                    if name != "direct_tilt"
                    else "direct_count_tilt"
                ),
            },
        }
        for trace, prediction in zip(traces, predictions):
            role = next(
                role
                for role, controllers in ROLES.items()
                if trace["sut"] in controllers
            )
            for mode in ("empirical", "strict"):
                row = stopping_row(trace, prediction, calibration[name][mode])
                row.update(model=name, mode=mode, role=role)
                rows.append(row)
    write_json(folder / "calibration.json", calibration)
    write_json(folder / "stopping.json", rows)
    write_json(
        folder / "forecasts.json",
        {"initial": forecasts, "updated": updated_forecasts, "direct_tilt": direct},
    )
    return traces, rows


def main():
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        ROOT / "protocol.json",
        {
            "method": {
                "id": "gcrt",
                "name_zh": "收益校正风险测试",
                "name_en": "Gain-Calibrated Risk Testing",
                "code": "methods/gcrt",
                "results": "results/gcrt",
                "tests": "tests/gcrt",
            },
            "seed": SEED,
            "roles": ROLES,
            "families": ["cutin", "brake"],
            "pools": [0, 1, 2],
            "budget": BUDGET,
            "initial": INITIAL,
            "retention": 0.95,
            "primary_calibration": "empirical",
            "physical_calls": 0,
            "new_full_trajectories": 36,
            "segment": SEGMENT,
            "historical_budget": BUDGET,
            "historical_initial": INITIAL,
            "historical_candidates": 1024,
            "hardware": {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "online_workers": 1,
                "torch_threads": 1,
            },
        },
    )
    with threadpool_limits(limits=1):
        workspaces = [workspace_for(f) for f in ("cutin", "brake")]
        for workspace in workspaces:
            build_family(workspace)
        protocol = read_json(ROOT / "protocol.json")
        protocol["family_parameters"] = {
            workspace.root.name: {
                "prior": read_json(workspace.root / "prior.json")["selected"],
                "policy": read_json(workspace.root / "policy.json")["selected"],
                "penalty": read_json(workspace.root / "training.json")["penalty"],
                "collection_threshold": read_json(workspace.root / "training.json")[
                    "collection_threshold"
                ],
            }
            for workspace in workspaces
        }
        write_json(ROOT / "protocol.json", protocol)
        project = Path(__file__).resolve().parents[2]
        snapshot_path = ROOT / "source_inputs.zip"
        if not snapshot_path.exists():
            with ZipFile(snapshot_path, "w", ZIP_DEFLATED) as archive:
                paths = sorted((project / "methods").rglob("*.py"))
                for workspace in workspaces:
                    paths.extend(
                        (
                            workspace.kernel_root / "models/seed_11.pt",
                            workspace.encoder_root / "models/decision_11.pt",
                            workspace.root / "model.pt",
                            workspace.root / "prior.json",
                            workspace.root / "training.json",
                        )
                    )
                for path in paths:
                    archive.write(path, path.relative_to(project).as_posix())
        for workspace in workspaces:
            evaluate_family(workspace)
        for workspace in workspaces:
            analyze_family(workspace.root.name)
        from .report import main as report

        report()
        results = read_json(ROOT / "summary.json")
        keys = (
            "first_two",
            "third",
            "updated_empirical_first_two",
            "updated_empirical_third",
        )
        accepted = all(
            all(r["acceptance"][key]["passed"] for key in keys)
            and r["lifecycle_quality"]["passed"]
            and r["cost_pass"]
            for r in results
        )
        write_json(
            ROOT / "status.json",
            {
                "stage": "acceptance",
                "status": "passed" if accepted else "not_passed",
                "experiments_complete": True,
                "completed": 36,
                "planned": 36,
                "physical_calls": 0,
            },
        )


if __name__ == "__main__":
    main()
