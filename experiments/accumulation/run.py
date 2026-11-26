"""Execute the registered four-scene knowledge accumulation study."""

import argparse
import hashlib
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import psutil

from controllers.highway.catalog import FORMAL_TARGET_SUTS
from controllers.highway.learning import CHECKPOINT_ROOT
from testing_methods.history_transfer.training import DEFAULT as TRAINING_CONFIG

from experiments.budget import protocol as budget_protocol
from experiments.accumulation import sequence as transfer_sequence
from experiments.collect_transfer_records import ApprovedCase
from experiments.regression.dqn_evaluation import PREDICTION, REVISED, STUDY, TARGET
from experiments.executions import ExecutionBank
from experiments.history_transfer_experiment import OUTPUT as INPUT, RUN_SCENARIOS, freeze, model_directory, registered_history_views, scenario_history
from experiments.method_contracts import method_contract
from experiments.parallel import METHOD_WORKERS, initialize_worker, prepare_processes
from experiments.protocol import ROOT
from experiments.weight_ablation.run import load_context as load_admitted_models
from experiments.storage import read_json, write_json
from experiments.transfer_execution import approval_path, canonical
from experiments.transfer_settings import native_settings

OUTPUT = ROOT / "results/knowledge_accumulation"
SEED, BUDGET, INITIAL = budget_protocol.SEED, budget_protocol.BUDGET, budget_protocol.INITIAL
LIMITS = budget_protocol.LIMITS
PARTITION_SEED, BANK_SIZE, BLOCK_SIZE = 71011, 2000, 500
GROUPS = {
    "group_a": ("target_idm_mobil", "target_linear_mpc", "target_uct", "target_ppo"),
    "group_b": ("target_nonlinear_trajectory", "target_fvdm_lane", TARGET, "target_beam"),
}
ORDERS = ("forward", "reverse")
SETTINGS = native_settings()
SUTS = {s.sut_id: s for s in FORMAL_TARGET_SUTS}
sha = budget_protocol.sha


class CandidateBlock:
    """Evaluator-only view of a fixed subset in its registered local ordering."""

    def __init__(self, parent, parent_indices):
        self.parent = parent
        self.parent_indices = parent_indices
        self.candidates = [parent.candidates[i] for i in parent_indices]
        self.manifest = parent.manifest
        self.domain = parent.domain

    def __call__(self, scenario):
        return self.parent(scenario)

    def execution_path(self, index):
        return self.parent.execution_path(self.parent_indices[index])

    @cached_property
    def snapshot_fingerprint(self):
        digest = hashlib.sha256(self.parent.snapshot_fingerprint.encode())
        digest.update(np.asarray(self.parent_indices, dtype="<i8").tobytes())
        return digest.hexdigest()


def sequences():
    return [(kind, group, order) for kind in RUN_SCENARIOS for group in GROUPS for order in ORDERS]


def sequence(kind, group, order, regime):
    targets = GROUPS[group] if order == "forward" else GROUPS[group][::-1]
    return {
        "scenario": kind, "seed": SEED, "regime": regime,
        "namespace": [group, order],
        "branch_id": f"{kind}:{group}:{order}:{regime}",
        "tasks": [
            {
                "ordinal": i, "sut_id": sut, "initialization_seed": SEED,
                "initial_queries": INITIAL, "total_budget": BUDGET,
                "update_after_task": regime == "update_shared" and i == 2,
            }
            for i, sut in enumerate(targets, 1)
        ],
    }


def location(kind, group, order, regime):
    return transfer_sequence.directory(OUTPUT, sequence(kind, group, order, regime))


def bank_path(kind, target):
    root = PREDICTION if target == TARGET else INPUT
    return root / "physical/target" / kind / target


def load_blocks(kind):
    parents = {target: ExecutionBank(bank_path(kind, target)) for target in SUTS}
    coordinates = {s.scenario_id: s for s in parents[GROUPS["group_a"][0]].candidates}
    if len(coordinates) != BANK_SIZE or len({s.coordinate() for s in coordinates.values()}) != BANK_SIZE:
        raise ValueError("Each scene must have 2,000 unique public coordinates")
    for target, bank in parents.items():
        if (
            {s.scenario_id: s for s in bank.candidates} != coordinates
            or len(bank.candidates) != BANK_SIZE
            or bank.manifest["sut_id"] != target
            or bank.manifest["successful_calls"] != BANK_SIZE
            or bank.manifest["candidate_count"] != BANK_SIZE
            or bank.manifest["environment_seed"] != 81001
            or any(s.kind != kind for s in bank.candidates)
        ):
            raise ValueError(f"Invalid E05 execution bank: {kind}/{target}")
    ordered_ids = np.asarray(sorted(coordinates))
    ids = ordered_ids[np.random.default_rng(PARTITION_SEED).permutation(BANK_SIZE)]
    return {
        target: CandidateBlock(parents[target], [
            parents[target].indices[sid] for sid in ids[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
        ])
        for targets in GROUPS.values() for i, target in enumerate(targets)
    }


def declaration():
    return {
        "experiment": "knowledge_accumulation",
        "scenes": list(RUN_SCENARIOS), "groups": GROUPS, "orders": ORDERS,
        "regimes": transfer_sequence.REGIMES, "seed": SEED,
        "initial_count": INITIAL, "budget": BUDGET, "partition_seed": PARTITION_SEED,
        "bank_size": BANK_SIZE, "block_size": BLOCK_SIZE,
        "settings": SETTINGS.to_dict(), "limits": LIMITS, "training": asdict(TRAINING_CONFIG),
        "training_update_after_task": 2, "metric": "mean cumulative collisions at queries 1..200",
        "result_positions": 192, "unique_paths": 112, "logical_queries": 22400,
        "model_updates": 16, "new_physical_executions": 0,
        "method_contract": method_contract("full"),
        "driver_sha256": sha(Path(__file__)),
        "sequence_sha256": sha(Path(transfer_sequence.__file__)),
    }


def admit():
    """Seal coordinates and verify the source of each complete physical bank."""
    release = STUDY / "publication.json"
    published = read_json(release)
    checkpoint = CHECKPOINT_ROOT / REVISED.training_run / "best.zip"
    if (published["status"] != "published" or published["training_run"] != REVISED.training_run
            or published["checkpoint_sha256"] != sha(checkpoint)):
        raise ValueError("Published revised DQN checkpoint is unavailable")
    acceptance = INPUT / "reports/method_acceptance.json"
    proof = read_json(acceptance)
    if proof["status"] != "complete":
        raise ValueError("E05 requires the admitted original historical inputs")
    freeze(OUTPUT / "protocol.json", declaration())
    for kind in RUN_SCENARIOS:
        blocks = load_blocks(kind)
        recorded = {}
        for target, block in blocks.items():
            bank = block.parent
            identity = bank.snapshot_fingerprint
            if target == TARGET:
                receipt = bank.root / "quality.json"
                if (bank.manifest["contract"]["runtime_contract"]["checkpoint_sha256"]
                        != published["checkpoint_sha256"]
                        or read_json(receipt)["bank_sha256"] != identity):
                    raise ValueError("E05 DQN block differs from the published target bank")
            else:
                case = ApprovedCase("budget", target, kind, SEED, "full", None)
                receipt = approval_path(budget_protocol.OUTPUT, case)
                sealed = read_json(receipt)
                spec = sealed["specification"]
                if (sealed["scope"] != "formal" or sealed["key"] != list(case.key)
                        or spec["task_snapshot"]["bank_snapshot"] != identity
                        or spec["bank_contract"] != bank.manifest["contract"]
                        or spec["candidates"] != canonical([asdict(s) for s in bank.candidates])):
                    raise ValueError("E05 block differs from the original formal target bank")
            expected_sut = REVISED if target == TARGET else SUTS[target]
            if bank.manifest["contract"]["object"] != asdict(expected_sut):
                raise ValueError("E05 target system identity differs from the admitted bank")
            recorded[target] = {
                "bank": bank.root.relative_to(ROOT).as_posix(), "bank_sha256": identity,
                "receipt": receipt.relative_to(ROOT).as_posix(), "receipt_sha256": sha(receipt),
                "parent_indices": block.parent_indices,
                "candidate_ids": [s.scenario_id for s in block.candidates],
            }
        models = [model_directory(INPUT, kind) / name
                  for name in ("target_kernel.pt", "historical_encoder.pt")]
        history = [
            *models, *(INPUT / "data/history" / kind).glob("*.npz"),
            INPUT / "data/history" / kind / "manifest.json",
            INPUT / "reports/history_views" / f"{kind}.json",
        ]
        if len(history) != 12:
            raise ValueError("E05 requires eight admitted historical archives")
        history_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in history}
        if any(proof["proof_sha256"].get(path) != digest for path, digest in history_hashes.items()):
            raise ValueError("E05 historical inputs differ from their original admission")
        freeze(OUTPUT / "admission" / f"{kind}.json", {
            "targets": recorded, "historical_inputs": history_hashes, "scene": kind,
            "original_admission_sha256": sha(acceptance), "dqn_publication_sha256": sha(release),
        })
        print({"admitted_scene": kind, "targets": len(blocks)}, flush=True)


@lru_cache(maxsize=4)
def load_context(kind):
    admission = read_json(OUTPUT / "admission" / f"{kind}.json")
    blocks = load_blocks(kind)
    for target, block in blocks.items():
        record = admission["targets"][target]
        if record["parent_indices"] != block.parent_indices:
            raise ValueError("E05 block order changed after admission")
        # The driver has independently rehashed every admitted parent before
        # launching workers. Retain that identity in the local view.
        block.parent.snapshot_fingerprint = record["bank_sha256"]
    # The already exercised compatibility loader verifies the sealed model
    # bytes while accepting a newer implementation of the same model class.
    case = ApprovedCase("budget", "target_ppo", kind, SEED, "full", None)
    approved = replace(case, specification=read_json(
        approval_path(budget_protocol.OUTPUT, case))["specification"])
    proof = read_json(INPUT / "reports/method_acceptance.json")["proof_sha256"]
    models = load_admitted_models(kind, proof, approved)["models"]
    history = scenario_history(INPUT, kind)
    _, views = registered_history_views(INPUT, kind, history)
    dqn_version = blocks[TARGET].manifest["contract"]["runtime_contract"]["checkpoint_sha256"]
    return {
        "models": models, "history": history, "view_record_ids": views["selected_record_ids"],
        "suts": {**SUTS, TARGET: replace(REVISED, version=f"{REVISED.version}@{dqn_version}")},
        "banks": blocks,
    }


def run_branch(kind, group, order, regime, *, prefix=False):
    job = (kind, group, order, regime)
    output = location(*job)
    seq = sequence(*job)
    status = output / "progress.json"
    if status.exists():
        saved = read_json(status)
        if saved["status"] == "complete" or (prefix and len(saved["completed"]) >= 2):
            return job
    if regime == "update_shared":
        origin = location(kind, group, order, "append_sources")
        reuse = {1: location(kind, group, "forward", "fixed_history"), 2: origin}
        training = output / "models/after_02_tasks"
        for name in ("target_kernel", "historical_encoder"):
            diagnostic = training / f"{name}.json"
            if diagnostic.exists() and read_json(diagnostic)["status"] == "running":
                interrupted = training / "interrupted" / str(time.time_ns())
                interrupted.mkdir(parents=True)
                diagnostic.replace(interrupted / diagnostic.name)
                checkpoint = training / f"{name}.pt"
                if checkpoint.exists():
                    checkpoint.replace(interrupted / checkpoint.name)
    elif regime == "append_sources":
        reuse = {1: location(kind, group, "forward", "fixed_history")}
    else:
        reuse = {}
    print({"running": job, "prefix_only": prefix, "pid": os.getpid()}, flush=True)
    try:
        transfer_sequence.execute_sequence(
            OUTPUT, seq, load_context(kind), settings=SETTINGS, limits=LIMITS,
            scope="formal", fixed_reference="execute", reuse_from=reuse,
            stop_after_tasks=2 if prefix else None,
        )
    except transfer_sequence.SequenceInterrupted:
        if not prefix:
            raise
    print({"finished": job, "prefix_only": prefix}, flush=True)
    return job


def execute(*, workers=METHOD_WORKERS, update_workers=16):
    prepare_processes()
    initialize_worker()
    admit()
    spawn = mp.get_context("spawn")
    branches = sequences()
    with ProcessPoolExecutor(max_workers=workers, mp_context=spawn,
                             initializer=initialize_worker) as queries:
        fixed = [queries.submit(run_branch, kind, group, "forward", "fixed_history")
                 for kind, group, order in branches if order == "forward"]
        for result in as_completed(fixed):
            result.result()
        prefixes = [queries.submit(run_branch, *branch, "append_sources", prefix=True)
                    for branch in branches]
        with ProcessPoolExecutor(max_workers=update_workers, mp_context=spawn,
                                 initializer=initialize_worker) as updates:
            training, remaining = {}, []
            for result in as_completed(prefixes):
                branch = result.result()[:3]
                training[updates.submit(run_branch, *branch, "update_shared", prefix=True)] = branch
                remaining.append(queries.submit(run_branch, *branch, "append_sources"))
            for result in as_completed(training):
                result.result()
                remaining.append(queries.submit(run_branch, *training[result], "update_shared"))
            for result in as_completed(remaining):
                result.result()


def guarded_run(workers, update_workers):
    """Serialize official execution on this workspace using an OS-held lock."""
    import msvcrt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("An E05 driver already holds the workspace lock") from error
        birth = psutil.Process().create_time()
        status = OUTPUT / "runner.json"
        state = {"pid": os.getpid(), "process_created_at": birth, "started_at": time.time(),
                 "query_workers": workers, "update_workers": update_workers}
        write_json(status, {**state, "status": "running"})
        try:
            execute(workers=workers, update_workers=update_workers)
            from experiments.accumulation.report import report

            result = report()
            write_json(status, {**state, "status": result["status"], "ended_at": time.time()})
        except BaseException as error:
            write_json(status, {**state, "status": "failed", "ended_at": time.time(),
                                "error": repr(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=METHOD_WORKERS)
    parser.add_argument("--update-workers", type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.workers <= METHOD_WORKERS or not 0 < args.update_workers <= 16:
        parser.error("Worker counts must be within the registered CPU concurrency limits")
    guarded_run(args.workers, args.update_workers)


if __name__ == "__main__":
    main()
