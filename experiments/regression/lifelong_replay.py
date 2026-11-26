"""Replay the lifelong acquisition against sealed same-lineage regression truth."""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch

from experiments.accumulation.lifelong import verify_stage_three
from experiments.history_transfer_experiment import freeze
from experiments.lifelong_seals import sha
from experiments.lifelong_transfer import OUTPUT, SETTINGS, _audit_forecast, selector_contract
from experiments.parallel import COLLECTION_WORKERS, initialize_worker, prepare_processes
from experiments.query import run_isolated
from experiments.regression.execution import ATTENTION
from experiments.regression.fixed_inputs import (
    VERSIONS, methods_for, model_root, query_directory, scene_root, version_directory,
)
from experiments.regression.fixed_replay import FixedReplayLedger
from experiments.regression.fixed_report import score
from experiments.regression.preflight import DQN_CONTINUAL
from experiments.resume import paid_feedback_hash
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.schema import Scenario
from experiments.storage import compact_result, read_json, write_json
from testing_methods.history_transfer.lifelong_selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import load_model

ROOT = OUTPUT / "policy_regression"
POLICIES = ("ppo", DQN_CONTINUAL, ATTENTION)
TARGET_VERSIONS = VERSIONS[1:4]


def _source(path, candidates, parent_specification):
    with np.load(path, allow_pickle=False) as saved:
        specification = json.loads(str(saved["specification"]))
        if (
            specification != parent_specification
            or specification["candidates"] != candidates
            or not np.isfinite(saved["means"]).all()
            or not np.isfinite(saved["blocks"]).all()
        ):
            raise ValueError("Fixed regression prior differs from the sealed original")
        return {
            "specification": specification, "names": saved["names"].tolist(),
            "weights": saved["weights"].copy(), "means": saved["means"].copy(),
            "blocks": saved["blocks"].copy(),
        }


def _checked_order(saved, truth, initial):
    rows, result = saved["query_ledger"], saved["result"]
    order = [row["index"] for row in rows]
    if (
        saved["paid_feedback_hash"] != paid_feedback_hash(rows)
        or saved["specification"]["truth_sha256"] != truth["execution_sha256"]
        or saved["specification"]["checkpoint_sha256"] != truth["checkpoint_sha256"]
        or order != result["selected_indices"] or order[:len(initial)] != initial
        or len(set(order)) != len(order)
        or any(
            type(index) is not int or not 0 <= index < len(truth["feedback"])
            or row["call"] != number or row["status"] != "complete"
            or row["feedback"] != truth["feedback"][index]
            for number, (row, index) in enumerate(zip(rows, order, strict=True), 1)
        )
    ):
        raise ValueError("Fixed regression replay differs from its paid sealed responses")
    return order


def replay_case(task, *, device="cpu", root=ROOT, workers=None):
    policy, kind, version = task
    base_root = scene_root(kind, policy)
    directory = root / policy / kind / version_directory(kind, "truth", version, policy).name
    registration = read_json(base_root / "registration.json")
    current_path = version_directory(kind, "truth", version, policy) / "sealed.json"
    previous_version = VERSIONS[VERSIONS.index(version) - 1]
    previous_path = version_directory(kind, "truth", previous_version, policy) / "sealed.json"
    truth, previous = read_json(current_path), read_json(previous_path)
    candidates = [Scenario(**row) for row in registration["candidates"]]
    ids = [scene.scenario_id for scene in candidates]
    if (
        len(ids) != 2000 or len(set(ids)) != len(ids)
        or any([row["scenario_id"] for row in bank["feedback"]] != ids for bank in (truth, previous))
    ):
        raise ValueError("Same-lineage paired truth does not share one fixed suite")
    initial = registration["initial_indices"]
    model_path = model_root(kind, policy) / "target_kernel.pt"
    source_path = base_root / "sources" / f"{version}.npz"
    control_paths = {
        method: query_directory(kind, version, method, policy) / "result.json.gz"
        for method in methods_for(policy)
    }
    controls = {method: read_json(path) for method, path in control_paths.items()}
    # The response-model prior must match the archive actually used by v1.
    sources = _source(source_path, registration["candidates"], controls["full"]["specification"]["source_snapshot"])
    spec = {
        "method": "lifelong_full", "version": version, "policy": policy, "scenario": kind,
        "checkpoint_sha256": truth["checkpoint_sha256"], "truth_sha256": truth["execution_sha256"],
        "sealed_truth_sha256": sha(current_path), "previous_truth_sha256": sha(previous_path),
        "source_sha256": sha(source_path), "base_model_sha256": sha(model_path),
        "source_snapshot": sources["specification"], "initial_indices": initial,
        "controls": {method: sha(path) for method, path in control_paths.items()},
        "selector_contract": selector_contract(), "online_settings": SETTINGS.online.to_dict(),
        "budget": len(ids), "termination": "later_of_Q90_and_500_paid_queries",
    }
    path = directory / "result.json.gz"
    if path.exists():
        saved = read_json(path)
        if saved["specification"] != spec:
            raise ValueError("Saved lifelong regression belongs to another registered replay")
    else:
        model, _ = load_model(model_path, kernel_type=SharedPrivateKernel)
        ledger = FixedReplayLedger(truth["feedback"], initial, directory, spec)
        result = run_isolated(select, {
            "candidates": candidates, "pretrained": model, "sources": sources,
            "initial_indices": initial, "online_settings": SETTINGS.online,
            "configuration": "lifelong_full", "budget": len(ids), "snapshot_counts": (),
            "implicit_evidence": True, "device": device,
        }, ledger, signal_evaluator_stop=True)
        ledger.finish()
        saved = {
            "specification": spec, "result": compact_result(result), "query_ledger": ledger.records,
            "paid_feedback_hash": paid_feedback_hash(ledger.records), "new_physical_calls": 0,
            "execution_schedule": {"device": device, "workers": workers, "threads_per_selector": 1},
        }
        write_json(path, saved)
        saved = read_json(path)
    _audit_forecast(saved["result"])
    reports = {}
    for method, row in {**controls, "lifelong_full": saved}.items():
        order = _checked_order(row, truth, initial)
        reports[method] = score(
            [item["collision"] for item in previous["feedback"]],
            [item["collision"] for item in truth["feedback"]],
            order,
        )
    write_json(directory / "paired_metrics.json", reports)
    return task


def run(workers):
    verify_stage_three()
    tasks = [(policy, kind, version) for policy in POLICIES for kind in FORMAL_SCENARIOS for version in TARGET_VERSIONS]
    protocol_path = ROOT / "protocol.json"
    registration_workers = read_json(protocol_path)["workers"] if protocol_path.exists() else workers
    freeze(protocol_path, {
        "tasks": [list(task) for task in tasks], "physical_calls": 0,
        "main_metrics": ["q90", "new_in_first_100"], "selector_contract": selector_contract(),
        "device": "cpu", "workers": registration_workers, "threads_per_selector": 1,
    })
    pending = []
    for policy, kind, version in tasks:
        folder = ROOT / policy / kind / version_directory(kind, "truth", version, policy).name
        if not all((folder / filename).is_file() for filename in ("result.json.gz", "paired_metrics.json")):
            pending.append((policy, kind, version))
    print(f"{len(tasks) - len(pending)}/{len(tasks)} replays already complete; {len(pending)} pending", flush=True)
    if not pending:
        return
    tasks = pending
    task_times = {}
    for policy, kind, version in tasks:
        origin = query_directory(kind, version, "full", policy) / "result.json.gz"
        task_times[policy, kind, version] = read_json(origin)["result"]["resources"]["wall_s"]
    # Start the longest measured sealed workloads first to keep the worker pool occupied.
    tasks.sort(key=task_times.__getitem__, reverse=True)
    if 1 < workers < len(tasks):
        # One short registered case checks the complete output path early.
        tasks.insert(workers - 1, tasks.pop())
    prepare_processes()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker,
    ) as pool:
        for policy, kind, version in pool.map(partial(replay_case, workers=workers), tasks):
            print("complete", policy, kind, version, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(4, COLLECTION_WORKERS))
    args = parser.parse_args()
    if not 0 < args.workers <= COLLECTION_WORKERS:
        parser.error("--workers must fit the available CPU task concurrency")
    torch.set_num_threads(1)
    run(args.workers)


if __name__ == "__main__":
    main()
