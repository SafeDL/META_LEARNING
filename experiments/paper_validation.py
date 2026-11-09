"""Reuse frozen HighwayEnv evidence and execute only the registered paper deficits."""

import csv
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import load_model

from .attribution import registered_runs
from .costs import PeakMemory
from .executions import ExecutionBank, collect_bank, execution_contract
from .history import build_source, finalize_task, load_training_archives, model_version
from .initialization import initial_indices
from .joint_validation import DEFAULT, run_one
from .pilot import PRIMARY_BASELINES, baseline_task, historical_classifiers
from .protocol import ROOT
from .scenario_sampling import sample_candidates
from .scenarios import DOMAINS, TEST_SCENARIOS, candidate_pool, validate
from .schema import Scenario
from .source_registration import specification_matches
from .storage import read_json, write_json

OLD = ROOT / "artifacts/runs/joint_response"
OUTPUT = OLD / "paper"
DATA = ROOT / "artifacts/data/joint_response"
EXPLORATION = ROOT / "artifacts/runs/joint_response/method_exploration/e2"
KERNELS = {"joint": JointResponseKernel, "shared_private": SharedPrivateKernel}
CONTROLS = ("collision", "no_history", "fixed_weights")
CONFIG = replace(DEFAULT, candidate_count=2000)
SEED = 11


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def freeze(path, value):
    """An existing registration must identify the same inputs on restart."""
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen paper inputs changed: {path}")
    else:
        write_json(path, value)


def load_models(kind, kernel):
    directory = (
        OLD / "models" / kind / "all_sources"
        if kernel == "joint"
        else EXPLORATION / "models" / kind
    )
    return tuple(
        load_model(directory / name, kernel_type=KERNELS[kernel])[0]
        for name in ("target_kernel.pt", "historical_encoder.pt")
    )


def register_parameters():
    """Separate unchanged engineering envelopes from the numbers actually in sources."""
    source = OLD / "reports/parameter_sources/parameters.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        references = {(r["scenario"], r["parameter"]): r for r in csv.DictReader(stream)}
    meaning = {
        "ego_speed_mps": "ego initial longitudinal speed",
        "lead_speed_mps": "lead initial longitudinal speed",
        "gap_m": "ego front to same-lane lead rear body clearance",
        "closing_speed_mps": "ego minus lead longitudinal speed; lead=ego-closing",
        "deceleration_mps2": "positive scripted lead braking magnitude",
        "event_time_s": "scripted event onset since simulation t=0; not disclosed to ADS",
        "brake_ramp_s": "linear build-up from zero to requested deceleration",
        "obstacle_distance_m": "initial obstacle centre x minus ego centre x",
        "lateral_duration_s": "complete scripted 4 m lateral shift with quintic position profile",
        "lead_gap_m": "ego front to original-lane lead rear body clearance",
        "rear_gap_m": "target-lane rear front to ego rear body clearance",
        "rear_speed_mps": "target-lane rear initial longitudinal speed",
        "adjacent_front_gap_m": "ego front to target-lane front vehicle rear clearance",
    }
    notes = {
        "ego_speed_mps": "Unchanged 64.8-122.4 km/h highway design envelope; not a fitted marginal.",
        "lead_speed_mps": "Independently designed actor speed; resulting closing speed is derived.",
        "gap_m": "Includes feasible short-headway states below standard test points; positive body clearance does not impose avoidability.",
        "closing_speed_mps": "Independent approach/recession design; derived lead speed is 10-38 m/s. This range is an engineering assumption, not a sourced empirical quantile.",
        "deceleration_mps2": "Contains 2 and 6 m/s^2 standard test levels and extends to a severe 9 m/s^2 scripted brake; not strict Euro NCAP reproduction.",
        "event_time_s": "Engineering onset window preserves a visible closed-loop pre-event phase within the 10 s execution.",
        "brake_ramp_s": "Engineering 0.1-1.5 s rise-time sweep; the cited standard's <=1 s criterion does not directly justify the entire interval.",
        "obstacle_distance_m": "Engineering centre-distance window, independent of trigger time; retain only predeclared lead-clearance geometry. It is not the standard fixed-TTC formula.",
        "lateral_duration_s": "Engineering mean lateral speed 1-2.22 m/s for a 4 m lane; includes faster manoeuvres than the cited naturalistic percentile envelope.",
        "lead_gap_m": "Independent engineering body-clearance envelope for the initial following phase.",
        "rear_gap_m": "Independent engineering body clearance, including close rear conflicts. Do not equate this interval to published time-gap IQRs.",
        "rear_speed_mps": "Engineering 79.2-136.8 km/h rear-speed envelope; closing speed is derived, not fitted jointly.",
        "adjacent_front_gap_m": "Independent engineering front clearance; target-lane front speed is fixed equal to initial ego speed. No cut-in parameter substitution.",
    }
    rows = []
    for kind in TEST_SCENARIOS:
        for name, (low, high) in DOMAINS[kind].bounds.items():
            reference = references.get((kind, name))
            if reference is None:
                raise ValueError(f"Missing source/engineering registration: {kind}/{name}")
            rows.append(
                {
                    "scenario": kind,
                    "parameter": name,
                    "unit": reference["unit"],
                    "reference_point": meaning[name],
                    "lower": low,
                    "upper": high,
                    "definition": "unchanged old HighwayEnv engineering envelope",
                    "source_url": reference["source"],
                    "source_location": reference["source_position"],
                    "source_values": reference["verified_source_value"],
                    "engineering_assumption": notes[name],
                }
            )
    path = OUTPUT / "reports/parameter_register.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def prepare():
    """Freeze nested coordinates, source records, model identities and reusable runs."""
    torch.set_num_threads(1)
    register_parameters()
    old = registered_runs()
    follow = candidate_pool("car_following", replicate=10)
    extra, sampling = sample_candidates(
        "car_following",
        DOMAINS["car_following"],
        count=1000,
        seed=62011,
        namespace="paper:supplement:10",
        feasible=lambda scene: validate(scene) is None,
        max_draws=10000,
        existing=follow,
    )
    if len(follow) != 1000 or len({s.scenario_id for s in follow + extra}) != 2000:
        raise ValueError("The nested follow library must preserve 1,000 old unique IDs")
    coordinates, sources, models, banks, reuse = {}, {}, {}, [], []
    for kind in TEST_SCENARIOS:
        candidates = (
            follow + extra if kind == "car_following" else candidate_pool(kind, replicate=10)
        )
        coordinates[kind] = [asdict(s) for s in candidates]
        freeze(OUTPUT / "candidates" / f"{kind}.json", coordinates[kind])
        history = load_training_archives(DATA / "manifest.json", kind)
        reference = read_json(old[CURRENT_INDEPENDENT_SUTS[0].sut_id, kind, "joint"]["path"])
        sources[kind] = reference["specification"]["source_snapshot"]["view_record_ids"]
        models[kind] = {}
        for kernel in KERNELS:
            pair = load_models(kind, kernel)
            for model in pair:
                construction = model.construction_specification
                if (
                    construction["scenario"] != kind
                    or construction["history_fingerprints"] != archive_fingerprints(history)
                    or construction["config"]["steps"] != 1500
                    or construction["config"]["seed"] != 1301
                    or construction["config"]["learning_rate"] != 0.001
                    or construction["config"]["historical_context"] != 64
                ):
                    raise ValueError("A reusable model differs from its historical construction")
            models[kind][kernel] = [model_version(m) for m in pair]
        for sut in CURRENT_INDEPENDENT_SUTS:
            bank = ExecutionBank(DATA / "targets/fixed" / sut.sut_id / kind)
            expected = candidates[:1000] if kind == "car_following" else candidates
            if (
                bank.candidates != expected
                or bank.manifest["environment_seed"] != CONFIG.environment_seed
                or not specification_matches(
                    bank.manifest["contract"], execution_contract(sut, kind=kind)
                )
            ):
                raise ValueError(
                    f"Old coordinates/ADS execution identity changed: {sut.sut_id}/{kind}"
                )
            missing = [i for i in range(len(expected)) if not bank.execution_path(i).is_file()]
            if missing:
                raise ValueError(
                    f"Old physical evidence is missing: {sut.sut_id}/{kind}/{missing[:5]}"
                )
            banks.append(
                {
                    "sut_id": sut.sut_id,
                    "scenario": kind,
                    "path": relative(bank.root),
                    "reused_physical_records": len(expected),
                }
            )
            for method in ("joint", *PRIMARY_BASELINES, *CONTROLS):
                entry = old[sut.sut_id, kind, method]
                path = Path(entry["path"])
                saved = read_json(path)
                spec, result = saved["specification"], saved["result"]
                initial = initial_indices(len(expected), 20, SEED)
                if (
                    spec["candidates"] != [asdict(s) for s in expected]
                    or result["selected_indices"][:20] != initial
                    or len(set(result["selected_indices"])) != 200
                    or spec["budget"] != 200
                    or spec["seed"] != SEED
                    or spec["environment_seed"] != CONFIG.environment_seed
                ):
                    raise ValueError(f"An old run cannot be reused: {path}")
                if method in ("joint", "collision", "fixed_weights"):
                    if (
                        [spec["target_model_version"], spec["encoder_version"]]
                        != models[kind]["joint"]
                        or spec["source_snapshot"]["view_record_ids"] != sources[kind]
                        or spec["feedback_mode"]
                        != ("collision" if method == "collision" else "joint")
                        or spec["update_source_weights"] != (method != "fixed_weights")
                    ):
                        raise ValueError(
                            f"Old control uses another model/source definition: {path}"
                        )
                elif method == "no_history" and (
                    spec["target_model_version"] is not None or spec["source_snapshot"] is not None
                ):
                    raise ValueError(f"Old no-history control contains a historical model: {path}")
                reuse.append(
                    {
                        "sut_id": sut.sut_id,
                        "scenario": kind,
                        "method": method,
                        "kernel": "joint" if method not in PRIMARY_BASELINES else None,
                        "path": relative(path),
                        "operation": "method_replay" if kind == "car_following" else "reuse",
                    }
                )
            e2 = EXPLORATION / "adaptive" / sut.sut_id / f"{kind}.json.gz"
            saved = read_json(e2)
            spec, result = saved["specification"], saved["result"]
            if (
                spec["target_model_version"] != models[kind]["shared_private"][0]
                or spec["source_snapshot"]["model_version"] != models[kind]["shared_private"][1]
                or spec["source_snapshot"]["view_record_ids"] != sources[kind]
                or spec["source_snapshot"]["candidates"] != [asdict(s) for s in expected]
                or result["selected_indices"][:20] != initial_indices(len(expected), 20, SEED)
                or len(set(result["selected_indices"])) != 200
            ):
                raise ValueError(f"E2 does not match the registered old comparison: {e2}")
            reuse.append(
                {
                    "sut_id": sut.sut_id,
                    "scenario": kind,
                    "method": "joint",
                    "kernel": "shared_private",
                    "path": relative(e2),
                    "operation": "method_replay" if kind == "car_following" else "reuse",
                }
            )
    protocol = {
        "format": "highway-paper-supplement",
        "platform": "HighwayEnv",
        "config": asdict(CONFIG),
        "systems": [asdict(s) for s in CURRENT_INDEPENDENT_SUTS],
        "candidate_counts": dict.fromkeys(TEST_SCENARIOS, 2000),
        "dimensions": {kind: len(DOMAINS[kind].bounds) for kind in TEST_SCENARIOS},
        "sampling": sampling,
        "follow_existing_prefix": 1000,
        "initial_pool_sizes": {
            kind: 1000 if kind == "car_following" else 2000 for kind in TEST_SCENARIOS
        },
        "initial_indices": {
            kind: initial_indices(
                2000, 20, SEED, pool_size=1000 if kind == "car_following" else 2000
            )
            for kind in TEST_SCENARIOS
        },
        "models": models,
        "source_view_record_ids": sources,
        "baselines": list(PRIMARY_BASELINES),
        "controls": list(CONTROLS),
        "model_selection": ["total_unique_collisions", "mean_discovery_auc", "joint_on_tie"],
        "selection_scope": "one global kernel type; target tasks used for model comparison",
        "old_banks": banks,
        "new_physical_executions": 8000,
        "new_history_records": 0,
        "new_training_steps": 0,
        "new_runs_if_joint": 88,
        "new_runs_if_shared_private": 160,
    }
    freeze(OUTPUT / "reports/protocol.json", json.loads(json.dumps(protocol)))
    freeze(OUTPUT / "reports/reuse.json", reuse)
    print("paper protocol frozen: 56,000 old truth records; 8,000 follow supplements", flush=True)
    return protocol


def collect():
    protocol = read_json(OUTPUT / "reports/protocol.json")
    candidates = [Scenario(**s) for s in read_json(OUTPUT / "candidates/car_following.json")]
    rows = []
    for sut in CURRENT_INDEPENDENT_SUTS:
        destination = DATA / "targets/paper_supplement" / sut.sut_id / "car_following"
        counts = Counter()
        for path in (destination / "attempts").glob("*.json"):
            attempt = read_json(path)
            if attempt["status"] != "complete":
                counts[attempt["scenario_id"]] += 1
        if any(n >= 2 for n in counts.values()):
            raise RuntimeError("A physical candidate already failed its one allowed retry")
        for attempt in range(2):
            try:
                new = collect_bank(
                    sut.sut_id,
                    candidates[1000:],
                    destination,
                    workers=12,
                    environment_seed=CONFIG.environment_seed,
                    current_registration=True,
                )
                break
            except RuntimeError:
                if attempt or counts:
                    raise
        old = ExecutionBank(DATA / "targets/fixed" / sut.sut_id / "car_following")
        if not specification_matches(old.manifest["contract"], new.manifest["contract"]):
            raise ValueError("The follow supplement uses another ADS execution identity")
        references = [
            relative(bank.execution_path(i))
            for bank in (old, new)
            for i in range(len(bank.candidates))
        ]
        merged = {
            "format": "collision-execution-bank",
            "status": "complete",
            "sut_id": sut.sut_id,
            "contract": old.manifest["contract"],
            "environment_seed": CONFIG.environment_seed,
            "scenarios": [asdict(s) for s in candidates],
            "candidate_count": 2000,
            "execution_refs": references,
            "physical_banks": [relative(old.root), relative(new.root)],
            "reused_physical_records": 1000,
            "new_physical_records": 1000,
        }
        freeze(DATA / "targets/paper" / sut.sut_id / "car_following/manifest.json", merged)
        rows.append(
            {
                "sut_id": sut.sut_id,
                "new_records": new.manifest["successful_calls"],
                "attempted_calls": new.manifest["attempted_calls"],
                "physical_execution_s": new.manifest["physical_execution_s"],
                "wall_s": new.manifest["total_successful_call_wall_s"],
            }
        )
        write_json(
            OUTPUT / "reports/collection.json",
            {
                "status": "complete" if len(rows) == len(protocol["systems"]) else "running",
                "new_physical_executions": sum(r["new_records"] for r in rows),
                "banks": rows,
            },
        )
    return rows


def task_bank(sut_id, kind):
    group = "paper" if kind == "car_following" else "fixed"
    return ExecutionBank(DATA / "targets" / group / sut_id / kind)


def prepare_sources():
    """One writer exports the expanded follow priors before concurrent task reads."""
    torch.set_num_threads(1)
    protocol = read_json(OUTPUT / "reports/protocol.json")
    candidates = [Scenario(**s) for s in read_json(OUTPUT / "candidates/car_following.json")]
    history = load_training_archives(DATA / "manifest.json", "car_following")
    rows = []
    with threadpool_limits(1):
        for kernel in KERNELS:
            _, encoder = load_models("car_following", kernel)
            source = build_source(
                encoder,
                history,
                candidates,
                view_record_ids=protocol["source_view_record_ids"]["car_following"],
                cache_directory=OUTPUT / "sources" / kernel / "car_following",
            )
            rows.append(
                {
                    "kernel": kernel,
                    "source_count": len(source["names"]),
                    "candidate_count": len(candidates),
                    "model_version": source["specification"]["model_version"],
                }
            )
    write_json(OUTPUT / "reports/source_preparation.json", {"status": "complete", "sources": rows})


def measure_follow_export():
    """Fill the missing construction timer with one checked repeat of the chosen export."""
    from .attribution import read_source

    torch.set_num_threads(1)
    selected = read_json(OUTPUT / "reports/model_selection.json")["selected_kernel"]
    protocol = read_json(OUTPUT / "reports/protocol.json")
    path = OUTPUT / "reports/source_export_timing.json"
    if path.exists():
        report = read_json(path)
        if report["model_version"] != protocol["models"]["car_following"][selected][1]:
            raise ValueError("Source timing belongs to a different frozen model")
        return report
    candidates = [Scenario(**s) for s in read_json(OUTPUT / "candidates/car_following.json")]
    _, encoder = load_models("car_following", selected)
    history = load_training_archives(DATA / "manifest.json", "car_following")
    cached = read_source(next((OUTPUT / "sources" / selected / "car_following").glob("*.npz")))
    with TemporaryDirectory(dir=OUTPUT / "sources", prefix="export-timing-") as temporary:
        with threadpool_limits(1), PeakMemory() as memory:
            repeated = build_source(
                encoder,
                history,
                candidates,
                view_record_ids=protocol["source_view_record_ids"]["car_following"],
                cache_path=Path(temporary) / "prior.npz",
            )
    errors = {
        name: float(np.max(np.abs(repeated[name] - cached[name])))
        for name in ("means", "blocks", "weights")
    }
    if max(errors.values()) > 1e-10 or repeated["specification"] != cached["specification"]:
        raise ValueError("Timed source export changed the frozen cache values")
    report = {
        "status": "passed",
        "kernel": selected,
        "model_version": model_version(encoder),
        "candidate_count": 2000,
        "source_count": len(history),
        "batch_size": 128,
        "scope": "One repeat of selected follow prior inference and NPZ export, excluding model/archive loading; initial export was not separately timed. This does not replace its first elapsed time.",
        "maximum_absolute_errors": errors,
        "resources": memory.result,
        "new_queries": 0,
        "new_training_steps": 0,
        "new_physical_executions": 0,
    }
    write_json(path, report)
    return report


def replay_task(task):
    torch.set_num_threads(1)
    kind, kernel, method, sut_id = (task[k] for k in ("scenario", "kernel", "method", "sut_id"))
    protocol = read_json(OUTPUT / "reports/protocol.json")
    sut = next(s for s in CURRENT_INDEPENDENT_SUTS if s.sut_id == sut_id)
    bank = task_bank(sut_id, kind)
    history = load_training_archives(DATA / "manifest.json", kind)
    destination = OUTPUT / "methods" / kernel / method / sut_id / kind
    initial_pool_size = protocol["initial_pool_sizes"][kind]
    with threadpool_limits(1):
        if method in PRIMARY_BASELINES:
            classifiers = historical_classifiers(
                history,
                OLD / "pilot/models" / kind,
                train_missing=False,
                names=("scenario_fuzz",) if method == "scenario_fuzz" else (),
            )
            saved, ledger = baseline_task(
                destination,
                bank,
                history,
                classifiers,
                CONFIG,
                SEED,
                task,
                initial_pool_size=initial_pool_size,
            )
        else:
            models = load_models(kind, kernel)
            if [model_version(m) for m in models] != protocol["models"][kind][kernel]:
                raise ValueError("A running task cannot switch its frozen model snapshot")
            saved, ledger = run_one(
                destination,
                bank,
                (None, None) if method == "no_history" else models,
                [] if method == "no_history" else history,
                CONFIG,
                SEED,
                feedback_mode="collision" if method == "collision" else "joint",
                update_source_weights=method != "fixed_weights",
                native_feedback=True,
                snapshot=task,
                initial_pool_size=initial_pool_size,
                view_record_ids=protocol["source_view_record_ids"][kind],
                source_cache_root=OUTPUT / "sources" / kernel / kind,
                source_cache_path=None
                if kind == "car_following"
                else (
                    ROOT / "artifacts/runs/joint_response/attribution/sources" / kind / "joint.npz"
                    if kernel == "joint"
                    else EXPLORATION / "sources" / f"{kind}.npz"
                ),
                raw_kernel_type=KERNELS[kernel],
            )
    finalize_task(
        destination / "completed.json.gz",
        ledger,
        execution_bank=bank,
        task_id=f"paper:{kernel}:{method}:{sut_id}:{kind}:{SEED}",
        sut_id=sut_id,
        sut_version=sut.version,
        family=sut.family,
        lineage=sut.lineage,
        original_role=sut.role,
        environment_seed=CONFIG.environment_seed,
        completed_order=1
        + TEST_SCENARIOS.index(kind) * 8
        + list(CURRENT_INDEPENDENT_SUTS).index(sut),
        branch_id=f"paper:{kernel}:{method}:{SEED}",
        execution_contract=bank.manifest["contract"],
        model_snapshot=saved["specification"],
    )
    return {
        **task,
        "kernel": None if method in PRIMARY_BASELINES else kernel,
        "path": relative(destination / "result.json.gz"),
        "operation": "new",
        "metrics": saved["metrics"],
    }


def execute_tasks(tasks, stage):
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(replay_task, task): task for task in tasks}
        for future in as_completed(pending):
            task = pending[future]
            try:
                rows.append(future.result())
            except Exception as error:
                write_json(
                    OUTPUT / f"reports/{stage}_failure.json",
                    {"task": task, "error": repr(error), "completed": rows},
                )
                raise
            write_json(
                OUTPUT / f"reports/{stage}.json",
                {
                    "status": "complete" if len(rows) == len(tasks) else "running",
                    "tasks": [
                        {key: value for key, value in row.items() if key != "metrics"}
                        for row in rows
                    ],
                    "wall_s": time.perf_counter() - started,
                },
            )
            print(stage, len(rows), "/", len(tasks), task, flush=True)
    return rows


def compare_kernels():
    registered = OUTPUT / "reports/model_selection.json"
    if registered.exists():
        saved = read_json(registered)
        identities = {(r["sut_id"], r["scenario"], r["kernel"]) for r in saved["runs"]}
        if (
            len(saved["runs"]) != 64
            or len(identities) != 64
            or any(read_json(ROOT / r["path"])["metrics"] != r["metrics"] for r in saved["runs"])
        ):
            raise ValueError("Completed kernel comparison lost its registered evidence")
        return saved
    tasks = [
        {
            "sut_id": sut.sut_id,
            "scenario": "car_following",
            "kernel": kernel,
            "method": "joint",
            "seed": SEED,
        }
        for sut in CURRENT_INDEPENDENT_SUTS
        for kernel in KERNELS
    ]
    new = execute_tasks(tasks, "kernel_replays")
    reused = [
        r
        for r in read_json(OUTPUT / "reports/reuse.json")
        if r["method"] == "joint" and r["operation"] == "reuse"
    ]
    rows = new + [{**r, "metrics": read_json(ROOT / r["path"])["metrics"]} for r in reused]
    totals = {}
    for kernel in KERNELS:
        metrics = [r["metrics"] for r in rows if r["kernel"] == kernel]
        if len(metrics) != 32:
            raise ValueError("Both kernels require all 32 tasks before model selection")
        totals[kernel] = {
            "failures": sum(m["unique_collisions"] for m in metrics),
            "mean_auc": sum(m["discovery_auc"] for m in metrics) / 32,
        }
    selected = max(
        KERNELS, key=lambda k: (totals[k]["failures"], totals[k]["mean_auc"], k == "joint")
    )
    selection = {
        "status": "complete",
        "selected_kernel": selected,
        "selection_rule": "total failures, mean AUC, original kernel on exact tie",
        "totals": totals,
        "runs": rows,
    }
    freeze(OUTPUT / "reports/model_selection.json", selection)
    return selection


def compare_methods(selection):
    registered = OUTPUT / "reports/summary.json"
    if registered.exists():
        saved = read_json(registered)
        if (
            saved["status"] != "complete"
            or saved["selected_kernel"] != selection["selected_kernel"]
            or len(saved["runs"]) != 320
            or any(read_json(ROOT / r["path"])["metrics"] != r["metrics"] for r in saved["runs"])
        ):
            raise ValueError("Completed paper matrix lost its registered evidence")
        return saved
    kernel = selection["selected_kernel"]
    reused = read_json(OUTPUT / "reports/reuse.json")
    rows = [r for r in selection["runs"] if r["kernel"] == kernel]
    tasks = []
    for sut in CURRENT_INDEPENDENT_SUTS:
        for kind in TEST_SCENARIOS:
            for method in (*PRIMARY_BASELINES, *CONTROLS):
                available = next(
                    (
                        r
                        for r in reused
                        if r["sut_id"] == sut.sut_id
                        and r["scenario"] == kind
                        and r["method"] == method
                        and r["operation"] == "reuse"
                        and (r["kernel"] is None or r["kernel"] == kernel)
                    ),
                    None,
                )
                if available is not None:
                    rows.append(
                        {**available, "metrics": read_json(ROOT / available["path"])["metrics"]}
                    )
                else:
                    tasks.append(
                        {
                            "sut_id": sut.sut_id,
                            "scenario": kind,
                            "method": method,
                            "kernel": kernel,
                            "seed": SEED,
                        }
                    )
    rows.extend(execute_tasks(tasks, "comparison_replays"))
    if len(rows) != 320:
        raise ValueError("The paper comparison requires 320 old-plus-new units")
    summary = {
        "status": "complete",
        "selected_kernel": kernel,
        "runs": rows,
        "logical_queries": 64000,
        "new_comparison_runs": len(tasks),
        "new_kernel_runs": 16,
        "new_physical_executions": 8000,
    }
    write_json(OUTPUT / "reports/summary.json", summary)
    return summary


def run():
    from .paper_report import archive_reused_tasks, audit_results, audit_supplement, report

    process = psutil.Process()
    progress = {
        "status": "running",
        "pid": process.pid,
        "process_created": process.create_time(),
        "started": time.time(),
    }
    stages = (
        ("reuse_registration", prepare),
        ("follow_collection", collect),
        ("data_audit", audit_supplement),
        ("source_caches", prepare_sources),
        ("kernel_comparison", compare_kernels),
    )
    try:
        for stage, action in stages:
            progress["stage"] = stage
            write_json(OUTPUT / "reports/progress.json", progress)
            result = action()
        measure_follow_export()
        progress["stage"] = "matched_comparisons"
        write_json(OUTPUT / "reports/progress.json", progress)
        compare_methods(result)
        progress["stage"] = "paid_archives"
        write_json(OUTPUT / "reports/progress.json", progress)
        archive_reused_tasks()
        progress["stage"] = "results_audit"
        write_json(OUTPUT / "reports/progress.json", progress)
        audit_results()
        progress["stage"] = "report_and_figures"
        write_json(OUTPUT / "reports/progress.json", progress)
        result = report()
    except Exception as error:
        progress.update(status="stopped", error=repr(error), stopped=time.time())
        write_json(OUTPUT / "reports/progress.json", progress)
        raise
    progress.update(status="complete", completed=time.time())
    write_json(OUTPUT / "reports/progress.json", progress)
    return result


if __name__ == "__main__":
    run()
