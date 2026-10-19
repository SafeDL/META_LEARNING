"""Build and evaluate independent functional-scenario tasks, with full costs."""

import multiprocessing as mp
import platform
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from methods.baselines.dart.encoder import fixed_context
from methods.baselines.dart.selection import select as dart_select
from methods.baselines.dart.train import (
    load_encoder,
    prepare_cases,
    train_seed as train_encoder,
)
from methods.baselines.hem.experts import HistoricalExperts
from methods.baselines.hgt.baseline import select as baseline_select
from methods.baselines.hgt.history import load_history, split_indices, subset
from methods.baselines.hgt.io import read_json, read_rows, write_json
from methods.baselines.hgt.search import TestingSession, select_session
from methods.baselines.hgt.train import load_kernel, train_seed as train_kernel
from benchmark.config import SUT_IDS
from benchmark.initialization import initial_indices
from benchmark.risk_channel import disclose_risk
from benchmark.timing import QueryClock, TimedRiskOracle, now

from methods.baselines.cg_brt import select as cg_select
from benchmark.data import bank_path, measured_seconds
from methods.common.features import prepare_pool
from methods.common.snapshots import current_path
from ..config import ROOT as LRT_ROOT
from ..data import generate
from methods.common.prior import build_prior
from ..selection import select
from ..train import load_value, train_seed as train_value
from methods.common.workspace import HistoricalWorkspace

ROOT = LRT_ROOT / "by_family"
PROJECT = Path(__file__).resolve().parents[4]
SEED = 11
BUDGET = 200
INITIAL = 10
FAMILIES = {"cutin": 0, "brake": 1}
METHODS = (
    "lrt",
    "cg_brt",
    "dart",
    "hgt",
    "gp_ei",
    "rf_bo",
    "uniform_random",
    "no_value_learning",
)
REUSE = (1, 10, 50, 100, 200)
TARGET_FRACTIONS = (0.8, 0.5)


def workspace_for(family, history=None):
    history = load_history() if history is None else history
    code = FAMILIES[family]
    filtered = {
        name: {
            key: value[source["x"][:, 4] == code].copy()
            for key, value in source.items()
        }
        for name, source in history.items()
    }
    return HistoricalWorkspace(ROOT / family, code, filtered)


def run_stage(workspace, name, action):
    path = workspace.root / "build_cost.json"
    ledger = read_json(path) if path.exists() else {"stages": {}, "physical_calls": 0}
    previous = ledger["stages"].get(name, {})
    if previous.get("status") == "complete":
        return
    started = now()
    ledger["stages"][name] = {
        "status": "running",
        "elapsed_s": previous.get("elapsed_s", 0),
    }
    write_json(path, ledger)
    try:
        action()
    except BaseException as error:
        ledger["stages"][name].update(status="failed", error=repr(error))
        raise
    else:
        ledger["stages"][name]["status"] = "complete"
    finally:
        ledger["stages"][name]["elapsed_s"] += now() - started
        write_json(path, ledger)
    print(
        f"BUILD {workspace.root.name} {name} {ledger['stages'][name]['elapsed_s']:.2f}s",
        flush=True,
    )


def build_family(workspace):
    run_stage(
        workspace,
        "kernel",
        lambda: train_kernel(
            workspace.history,
            SEED,
            root=workspace.kernel_root,
            modes=("global_feedback",),
        ),
    )
    run_stage(
        workspace,
        "encoder",
        lambda: train_encoder(
            SEED,
            "decision",
            prepare_cases(
                workspace.history,
                SEED,
                root=workspace.encoder_root,
                kernel_root=workspace.kernel_root,
                common_initial=True,
            ),
            root=workspace.encoder_root,
        ),
    )
    run_stage(
        workspace,
        "branch_labels",
        lambda: generate(SEED, ("full",), workspace=workspace),
    )
    run_stage(
        workspace, "value", lambda: train_value(SEED, "full", workspace=workspace)
    )
    counts = {"behavior": 0, "branches": 0}
    for path in (workspace.root / "historical").rglob("*.json"):
        data = read_json(path)
        if path.stem.endswith("_returns"):
            counts["branches"] += sum(
                row["paid_replay_queries"]
                for state in data.values()
                for row in state.values()
            )
        else:
            counts["behavior"] += len(data["queries"])
    write_json(
        workspace.root / "historical_queries.json", {**counts, "physical_calls": 0}
    )


def freeze_inputs(workspaces):
    protocol = {
        "seed": SEED,
        "candidates_per_family": 1024,
        "encoder_pool_size": 1024,
        "encoder_context_per_source": 128,
        "encoder_updates": 3200,
        "value_updates": 4000,
        "budget_per_family": BUDGET,
        "initial_queries": INITIAL,
        "families": FAMILIES,
        "pools": [0, 1, 2],
        "systems": SUT_IDS,
        "methods": METHODS,
        "metrics": ["H200", "Recall"],
        "risk_threshold": 0.5,
        "reuse_tasks": REUSE,
        "primary_reuse": 100,
        "target_fractions": TARGET_FRACTIONS,
        "new_physical_calls": 0,
        "previously_observed_pools": True,
        "training": {"candidates": 819, "budget": 160, "initial": 8},
        "validation": {"candidates": 205, "budget": 40, "initial": 2},
        "initialization": "label-independent seeded permutation, common to all methods",
        "cost": "serial-equivalent archived execution + measured synchronized compute",
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "online_workers": 1,
            "torch_threads": 1,
            "branch_workers": 4,
        },
    }
    weights = []
    for workspace in workspaces:
        for stage in read_json(workspace.root / "build_cost.json")["stages"].values():
            if stage["status"] != "complete":
                raise ValueError(
                    "Finish historical construction before target evaluation"
                )
        weights.extend(
            (
                workspace.kernel_root / "models" / f"seed_{SEED}.pt",
                workspace.encoder_root / "models" / f"decision_{SEED}.pt",
                workspace.root / "models" / f"full_{SEED}.pt",
            )
        )
    destination = ROOT / "frozen_inputs.zip"
    if destination.exists():
        with ZipFile(destination) as archive:
            entries = {current_path(name): name for name in archive.namelist()}
            for path in weights:
                if (
                    archive.read(entries[path.relative_to(PROJECT).as_posix()])
                    != path.read_bytes()
                ):
                    raise ValueError(f"Frozen family weight changed: {path}")
        return
    write_json(ROOT / "protocol.json", protocol)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in (
            sorted((PROJECT / "methods").rglob("*.py"))
            + weights
            + [ROOT / "protocol.json"]
        ):
            archive.write(path, path.relative_to(PROJECT).as_posix())


def selector_worker(connection):
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            while True:
                command, task = connection.recv()
                if command == "stop":
                    return
                clock = QueryClock()
                oracle = TimedRiskOracle(connection, clock)
                method, common = task["method"], task["common"]
                if method in ("lrt", "no_value_learning"):
                    network = (
                        load_value(
                            SEED,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            root=task["root"],
                        )
                        if method == "lrt"
                        else None
                    )
                    result = select(
                        **common,
                        transfer_prior=task["prior"],
                        value_model=network,
                        oracle=oracle,
                        variant="full" if method == "lrt" else method,
                        clock=clock,
                    )
                elif method == "dart":
                    result = dart_select(
                        **common,
                        proposal_covariance=task["prior"].enhanced_covariance,
                        oracle=oracle,
                        clock=clock,
                    )
                elif method == "cg_brt":
                    result = cg_select(**common, oracle=oracle, clock=clock)
                elif method == "hgt":
                    session = TestingSession(
                        common["x"],
                        SEED,
                        BUDGET,
                        covariance=common["covariance"],
                        prior=common["fixed_prior"],
                        initial_indices=common["initial_indices"],
                    )
                    result = select_session(
                        session, oracle, risk_only=True, clock=clock
                    )
                else:
                    result = baseline_select(
                        method,
                        common["x"],
                        np.zeros(len(common["x"]), int),
                        oracle,
                        SEED,
                        budget=BUDGET,
                        cell_count=1,
                        initial_indices=common["initial_indices"],
                        risk_only=True,
                        clock=clock,
                    )
                result["timing"] = clock.report()
                connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def evaluate_family(workspace):
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    try:
        for pool in (0, 1, 2):
            cache = prepare_pool(workspace, pool)
            for sut in SUT_IDS:
                with np.load(bank_path(pool, sut)) as bank:
                    keep = bank["x"][:, 4] == workspace.family
                    np.testing.assert_array_equal(bank["x"][keep], cache["common"]["x"])
                    truth = bank["risk"][keep].copy()
                    scenario_ids = bank["scenario_id"][keep].copy()
                durations = measured_seconds(pool, sut, scenario_ids)
                for method in METHODS:
                    destination = (
                        workspace.root
                        / "trajectories"
                        / f"pool_{pool}"
                        / sut
                        / f"{method}_{SEED}.json"
                    )
                    if destination.exists():
                        continue
                    task = {
                        "method": method,
                        "common": cache["common"],
                        "root": workspace.root,
                    }
                    if method in ("lrt", "dart", "no_value_learning"):
                        task["prior"] = cache["prior"]
                    started = time.perf_counter()
                    result = disclose_risk(parent, task, truth, process)
                    selected = result["selected_indices"]
                    if selected[:INITIAL] != cache["common"]["initial_indices"]:
                        raise ValueError(
                            "Methods must share exactly the same initialization"
                        )
                    found = np.cumsum(truth[selected] > 0.5).tolist()
                    result.update(
                        family=workspace.root.name,
                        pool=pool,
                        sut=sut,
                        seed=SEED,
                        method=method,
                        positives=int((truth > 0.5).sum()),
                        found=found,
                        H200=found[-1],
                        recall=(
                            found[-1] / int((truth > 0.5).sum())
                            if (truth > 0.5).any()
                            else None
                        ),
                        scenario_ids=scenario_ids[selected].tolist(),
                        execution_s=durations[selected].tolist(),
                        evaluator_elapsed_s=time.perf_counter() - started,
                        physical_calls=0,
                    )
                    write_json(destination, result)
                    print(
                        f"TARGET {workspace.root.name} {pool} {sut} {method} H200={found[-1]}",
                        flush=True,
                    )
                    write_json(
                        ROOT / "status.json",
                        {
                            "status": "running",
                            "stage": "evaluation",
                            "completed_trajectories": len(
                                list(ROOT.glob("*/trajectories/*/*/*.json"))
                            ),
                            "planned_trajectories": 288,
                            "physical_calls": 0,
                        },
                    )
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
        if process.exitcode != 0:
            raise RuntimeError("Family selector worker failed")


def main():
    torch.set_num_threads(1)
    workspaces = [workspace_for(family) for family in FAMILIES]
    with threadpool_limits(limits=1):
        for workspace in workspaces:
            build_family(workspace)
        freeze_inputs(workspaces)
        for workspace in workspaces:
            evaluate_family(workspace)
    from .family_report import main as report

    report()
    write_json(
        ROOT / "status.json",
        {
            "status": "complete",
            "stage": "report",
            "completed_trajectories": 288,
            "physical_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
