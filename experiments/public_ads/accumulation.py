"""Run and audit E05 on fixed, disjoint blocks of the three sealed public ADS banks."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from controllers.highway.catalog import SUT
from experiments.accumulation import sequence
from experiments.accumulation.report import audit_scene, describe, differences
from experiments.accumulation.run import CandidateBlock, ORDERS
from experiments.budget.protocol import BUDGET, INITIAL, LIMITS, SEED, sha
from experiments.budget.report import write_csv
from experiments.executions import ExecutionBank
from experiments.history_transfer_experiment import freeze, registered_history_views, scenario_history
from experiments.parallel import initialize_worker, prepare_processes
from experiments.public_ads import extend
from experiments.public_ads.evaluate_e01 import ORIGINAL, OUTPUT as PUBLIC, TARGETS, prepare_history
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.storage import read_json, write_json
from experiments.transfer_settings import native_settings

OUTPUT = PUBLIC / "knowledge_accumulation"
BANK_SIZE, BLOCK_SIZE, PARTITION_SEED = 2000, 600, 71023
SETTINGS = native_settings()
GROUP = "public"
FAMILIES = ("online_mc_ppo", "attention_ddqn", "decision_transformer")


def sequences():
    return [(kind, GROUP, order) for kind in FORMAL_SCENARIOS for order in ORDERS]


def plan(kind, group, order, regime):
    targets = TARGETS if order == "forward" else TARGETS[::-1]
    return {
        "scenario": kind, "seed": SEED, "regime": regime, "namespace": [group, order],
        "branch_id": f"{kind}:{group}:{order}:{regime}",
        "tasks": [
            {"ordinal": i, "sut_id": target, "initialization_seed": SEED,
             "initial_queries": INITIAL, "total_budget": BUDGET,
             "update_after_task": i == 2 and regime == "update_shared"}
            for i, target in enumerate(targets, 1)
        ],
    }


def location(kind, group, order, regime):
    return sequence.directory(OUTPUT, plan(kind, group, order, regime))


def blocks(kind, parents):
    reference = {s.scenario_id: s for s in parents[TARGETS[0]].candidates}
    if len(reference) != BANK_SIZE or len({s.coordinate() for s in reference.values()}) != BANK_SIZE:
        raise ValueError("Public E05 requires 2,000 unique coordinates")
    if any(
        len(bank.candidates) != BANK_SIZE or
        {s.scenario_id: s for s in bank.candidates} != reference or
        bank.manifest["sut_id"] != target or bank.manifest["environment_seed"] != 81001 or
        any(s.kind != kind for s in bank.candidates)
        for target, bank in parents.items()
    ):
        raise ValueError("Public E05 banks disagree on the registered coordinates")
    ids = np.asarray(sorted(reference))[np.random.default_rng(PARTITION_SEED).permutation(BANK_SIZE)]
    return {
        target: CandidateBlock(parents[target], [parents[target].indices[sid] for sid in ids[
            i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE
        ]])
        for i, target in enumerate(TARGETS)
    }


def receipt(kind, target, block):
    acceptance = PUBLIC / "qualification" / target / "acceptance.json"
    origin = PUBLIC / "k_shot" / target / kind / "11/full.json.gz"
    return {
        "parent_snapshot": block.parent.snapshot_fingerprint, "block_snapshot": block.snapshot_fingerprint,
        "indices": block.parent_indices, "candidate_ids": [s.scenario_id for s in block.candidates],
        "acceptance_sha256": sha(acceptance), "public_e01_sha256": sha(origin),
    }


def admit(kind):
    frozen = read_json(PUBLIC / "k_shot" / TARGETS[0] / kind / "11/full.json.gz")
    extend.initialize({"sources": {"specification": frozen["specification"]["source_snapshot"]}})
    views = blocks(kind, {t: extend.bank_for(kind, t) for t in TARGETS})
    return freeze(OUTPUT / "admission" / f"{kind}.json",
                  {target: receipt(kind, target, block) for target, block in views.items()})


def admit_case(root, case, bank):
    if (case.experiment != "budget" or case.sut_id not in TARGETS or
            case.scenario not in FORMAL_SCENARIOS or not Path(root).is_relative_to(OUTPUT)
            or not isinstance(bank, CandidateBlock)):
        raise ValueError("Formal execution requires a registered target ADS")
    registered = read_json(OUTPUT / "admission" / f"{case.scenario}.json")[case.sut_id]
    if receipt(case.scenario, case.sut_id, bank) != registered:
        raise ValueError("Public E05 case differs from its sealed candidate block")


@lru_cache(maxsize=4)
def context(kind):
    admitted = read_json(OUTPUT / "admission" / f"{kind}.json")
    banks = blocks(kind, {
        target: ExecutionBank(PUBLIC / "physical/target" / kind / target) for target in TARGETS
    })
    for target, block in banks.items():
        block.parent.snapshot_fingerprint = admitted[target]["parent_snapshot"]
        if receipt(kind, target, block) != admitted[target]:
            raise ValueError("Public E05 block differs from admission")
    historical = scenario_history(ORIGINAL, kind)
    _, views = registered_history_views(ORIGINAL, kind, historical)
    if len(historical) != 8 or sum(len(a.scenarios) for a in historical) != 4000:
        raise ValueError("Public E05 requires its eight frozen historical sources")
    models = prepare_history(kind)["models"]
    suts = {
        target: SUT(target, family, target,
                    banks[target].manifest["contract"]["public_ads"]["checkpoint_sha256"], "target", {})
        for target, family in zip(TARGETS, FAMILIES, strict=True)
    }
    return dict(banks=banks, models=models, history=historical,
                view_record_ids=views["selected_record_ids"], suts=suts)


def run_branch(kind, group, order, regime):
    branch = location(kind, group, order, regime)
    status = branch / "progress.json"
    if status.exists() and read_json(status)["status"] == "complete":
        return
    fixed = location(kind, group, "forward", "fixed_history")
    append = location(kind, group, order, "append_sources")
    reuse = ({1: fixed, 2: append} if regime == "update_shared"
             else {1: fixed} if regime == "append_sources"
             else {1: fixed, 2: fixed, 3: fixed} if order == "reverse" else {})
    print("Running", kind, order, regime, os.getpid(), flush=True)
    sequence.execute_sequence(
        OUTPUT, plan(kind, group, order, regime), context(kind), settings=SETTINGS, limits=LIMITS,
        scope="formal", fixed_reference="execute", reuse_from=reuse,
    )
    print("Finished", kind, order, regime, flush=True)


def execute(kinds, workers):
    prepare_processes()
    freeze(OUTPUT / "protocol.json", {
        "scenes": FORMAL_SCENARIOS, "targets": TARGETS, "families": FAMILIES,
        "orders": ORDERS, "regimes": sequence.REGIMES, "seed": SEED, "budget": BUDGET,
        "initial": INITIAL, "partition_seed": PARTITION_SEED, "bank_size": BANK_SIZE,
        "block_size": BLOCK_SIZE, "settings": SETTINGS.to_dict(), "limits": LIMITS,
        "training": asdict(sequence.TRAINING_CONFIG), "driver_sha256": sha(Path(__file__)),
        "result_positions": 72, "distinct_paths": 36, "logical_queries": 7200, "model_updates": 8,
        "new_physical_executions": 0,
    })
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        for completed in as_completed([pool.submit(admit, kind) for kind in kinds]):
            completed.result()
        for regime in sequence.REGIMES:
            jobs = [(s, g, o) for s, g, o in sequences() if s in kinds]
            if regime == "fixed_history":
                for order in ORDERS:
                    for result in as_completed([
                        pool.submit(run_branch, *job, regime) for job in jobs if job[2] == order
                    ]):
                        result.result()
            else:
                for result in as_completed([pool.submit(run_branch, *job, regime) for job in jobs]):
                    result.result()


def paid_content_hash(path):
    archive = read_json(path)
    content = {
        "sut_id": archive["sut_id"], "sut_version": archive["sut_version"], "order": archive["completed_order"],
        "executions": [{k: v for k, v in row.items() if k != "record_id"} for row in archive["executions"]],
        "attempts": archive["attempts"],
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def audit(kind):
    admit(kind)
    views = context(kind)["banks"]
    groups = [{s.coordinate() for s in bank.candidates} for bank in views.values()]
    if any(len(group) != BLOCK_SIZE for group in groups) or len(set.union(*groups)) != 1800:
        raise ValueError("Public E05 tasks reuse physical coordinates")
    prefixes = []
    for order in ORDERS:
        combined = []
        for regime in ("append_sources", "update_shared"):
            root = location(kind, GROUP, order, regime)
            progress = read_json(root / "progress.json")
            combined.append([paid_content_hash(root / row["archive"]) for row in progress["completed"][:2]])
        if combined[0] != combined[1]:
            raise ValueError("Append and update branches received different paid archive contents")
        prefixes.append({"scene": kind, "order": order, "common_paid_archive_content_sha256": combined[0]})
    return (*audit_scene(
        kind, output=OUTPUT, scene_context=context, sequence_jobs=sequences,
        sequence_builder=plan, sequence_location=location,
    ), prefixes)


def report():
    prepare_processes()
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        data = list(pool.map(audit, FORMAL_SCENARIOS))
    rows = [r for scene, _, _, _ in data for r in scene]
    updates = [r for _, events, _, _ in data for r in events]
    prefixes = [r for _, _, _, checks in data for r in checks]
    pairs = differences(rows)
    final = [r for r in pairs if r["ordinal"] == 3]
    logical_queries = sum(n for _, _, n, _ in data)
    distinct = len({r["result"] for r in rows})
    if (len(rows), distinct, len(updates), len(pairs), len(final)) != (72, 36, 8, 16, 8):
        raise ValueError("Public E05 output differs from its registered matrix")
    directory = OUTPUT / "reports"
    tables = {"task_results.csv": rows, "model_updates.csv": updates, "paired_differences.csv": pairs}
    for name, table in tables.items():
        write_csv(directory / name, table)
    write_json(directory / "completion.json", {
        "status": "complete" if all(r["status"] == "complete" for r in rows) else "completed_with_failures",
        "positions": len(rows), "distinct_paths": distinct, "logical_queries": logical_queries,
        "model_updates_accepted": sum(r["accepted"] for r in updates),
        "new_physical_executions": 0, "prefix_audit": prefixes,
        "summary": {
            f"{comparison}_{metric}": describe([r[f"{comparison}_{metric}"] for r in entries])
            for comparison, entries in (("append_fixed", pairs), ("update_append", final), ("update_fixed", final))
            for metric in ("auc", "c200")
        },
        "files": {name: sha(directory / name) for name in tables},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("run", "report"), default="run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kind", action="append", choices=FORMAL_SCENARIOS)
    args = parser.parse_args()
    if args.stage == "report":
        report()
    else:
        execute(args.kind or FORMAL_SCENARIOS, args.workers)


if __name__ == "__main__":
    main()
