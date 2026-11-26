"""Cold-start, cross-architecture relay and original eight-ADS replay branches."""

import argparse
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.accumulation import run as original
from experiments.collect_transfer_records import ApprovedCase
from experiments.history import build_source, model_version
from experiments.history_transfer_experiment import freeze, scenario_history
from experiments.lifelong_seals import sha
from experiments.lifelong_transfer import (
    CROSS_ARCHITECTURE, IN_FAMILY, OUTPUT, PREDICTION, SETTINGS, TARGETS,
    bank_for, evaluate_task, _verify_receipt, verify_stage_one,
)
from experiments.parallel import COLLECTION_WORKERS, METHOD_WORKERS, initialize_worker, prepare_processes
from experiments.public_ads import evaluate_e01
from experiments.public_ads.evaluate_continuous_sac import _prediction_destination
from experiments.rss import measurement_contract
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.schema import QueryFeedback
from experiments.storage import read_json, write_json
from experiments.transfer_execution import _paths
from testing_methods.common.response import ResponseArchive, archive_fingerprints
from testing_methods.history_transfer import lifelong_archive, lifelong_training, training
from testing_methods.history_transfer.shared_private import SharedPrivateKernel

ROOT = OUTPUT / "knowledge_accumulation"
RULE_HISTORY = ("history_idm_mobil", "history_fvdm_lane")
A_GROUPS = {
    "forward_policy": (IN_FAMILY[0], IN_FAMILY[2], IN_FAMILY[4], CROSS_ARCHITECTURE[0]),
    "reverse_policy": (CROSS_ARCHITECTURE[1], IN_FAMILY[6], IN_FAMILY[3], IN_FAMILY[1]),
    "forward_learning": (IN_FAMILY[5], IN_FAMILY[7], CROSS_ARCHITECTURE[2], CROSS_ARCHITECTURE[3]),
}
REGIMES = ("fixed", "naive_append", "debiased_append", "full_lifelong")
PARTITION_SEEDS = {"cold_start": 76011, "cross_architecture": 76023, "in_family": 71011}


@dataclass(frozen=True)
class Sequence:
    study: str
    name: str
    target_ids: tuple[str, ...]
    block_ids: tuple[int, ...]


def sequences(study):
    if study == "cold_start":
        if sorted(t for group in A_GROUPS.values() for t in group) != sorted(TARGETS):
            raise ValueError("Cold-start sequences must cover every registered target exactly once")
        return [Sequence(study, name, targets, tuple(range(4))) for name, targets in A_GROUPS.items()]
    if study == "in_family":
        return [
            Sequence(study, f"{name}_{order}", targets[::step], tuple(range(4))[::step])
            for name, targets in original.GROUPS.items()
            for order, step in (("forward", 1), ("reverse", -1))
        ]
    if study == "cross_architecture":
        return [
            Sequence(study, order, (*CROSS_ARCHITECTURE[::step], "target_ppo"), (*range(4)[::step], 4))
            for order, step in (("forward", 1), ("reverse", -1))
        ]
    raise ValueError(f"Unknown continual transfer study: {study}")


def _blocks(banks, study):
    source = banks[0]
    key = {scene.scenario_id: scene for scene in source.candidates}
    if len(key) != 2000 or len({scene.coordinate() for scene in key.values()}) != 2000:
        raise ValueError("E05 requires a complete 2,000-point target population")
    for bank in banks:
        if {scene.scenario_id: scene for scene in bank.candidates} != key:
            raise ValueError("E05 target banks must share exactly the same coordinates")
    ids = np.asarray(sorted(key))
    generator = np.random.default_rng(PARTITION_SEEDS[study])
    blocks = list(np.split(ids[generator.permutation(len(ids))], 4))
    if study == "cross_architecture":
        blocks = [*[b[:400] for b in blocks], np.concatenate([b[400:] for b in blocks])]
    return blocks


def _archive(bank, saved, source_id, ordinal):
    queries, ledger = saved["result"]["queries"], saved["query_ledger"]
    if (
        saved["result"]["actual_queries"] != 200
        or len(queries) != len(ledger) or len(ledger) != 200
        or [row["index"] for row in queries] != [row["index"] for row in ledger]
        or any(
            row["status"] != "complete" or row["purpose"] != ("initial" if i < 20 else "target")
            for i, row in enumerate(ledger)
        )
    ):
        raise ValueError("Only a complete 200-query task can enter the continual archive")
    feedback = [QueryFeedback.from_dict(row["feedback"]) for row in ledger]
    scenes = [bank.candidates[row["index"]] for row in ledger]
    if any(f.scenario_id != s.scenario_id or f.rss_contract != measurement_contract()
           or f.safety.rss_margin is None for f, s in zip(feedback, scenes, strict=True)):
        raise ValueError("Archive feedback must come from the completed task's paid coordinates")
    archive = ResponseArchive(
        source_id, "target", bank.manifest["sut_id"], "target", scenes,
        np.asarray([f.collision for f in feedback]),
        np.asarray([[f.safety.min_clearance_m, f.safety.min_ttc_s] for f in feedback]),
        provenance={"completed_task": True, "completed_order": ordinal,
                    "contract": bank.manifest["contract"], "record_ids": [s.scenario_id for s in scenes]},
        rss_margin=np.asarray([f.safety.rss_margin for f in feedback]),
        rss_contract=measurement_contract(), domain=bank.domain,
    )
    probability = [row["prequery_probability"] for row in queries]
    return archive, lifelong_archive.archive_weights(archive.collision, probability)


def _sealed_in_family(kind, sequence, regime, target):
    group, order = sequence.name.rsplit("_", 1)
    branch = "fixed_history" if regime == "fixed" else "update_shared"
    location = original.location(kind, group, order, branch)
    ordinal = sequence.target_ids.index(target)
    entry = read_json(location / "progress.json")["completed"][ordinal]
    if entry["sut_id"] != target or entry["ordinal"] != ordinal + 1 or entry["status"] != "complete":
        raise ValueError("Original in-family position differs from its sealed branch")
    saved_path = entry["result_path"].replace("\\", "/").split("/knowledge_accumulation/", 1)[1]
    source = original.OUTPUT / saved_path
    case = ApprovedCase("budget", target, kind, 11, "full", None)
    directory = source.parents[5]
    path, paid = _paths(directory, case)
    if path != source:
        raise ValueError("Original in-family receipt uses an unexpected task location")
    return _verify_receipt(directory, case), (path, paid)


def branch_result(kind, sequence, regime, target):
    branch = sequence.name
    if regime == "fixed" and sequence.study == "in_family":
        # Forward and reverse assign every target to the same coordinate block.
        branch = f"{branch.rsplit('_', 1)[0]}_forward"
    directory = ROOT / sequence.study / kind / branch / regime / "tasks" / target
    if sequence.study == "in_family" and regime == "naive_append":
        saved, paths = _sealed_in_family(kind, sequence, regime, target)
        if read_json(directory / "sealed_origin.json") != _origin_receipt(paths):
            raise ValueError("Original in-family source changed after reuse")
        return saved
    case = ApprovedCase("budget", target, kind, 11, "lifelong_full", None)
    return _verify_receipt(directory, case)


def _origin_receipt(paths):
    return {
        path.relative_to(original.OUTPUT).as_posix(): sha(path)
        for path in paths
    }


def _update(directory, models, archives, weighting):
    report_path, model_path = directory / "gate.json", directory / "shared_models.pt"
    key = {
        "parents": [model_version(model) for model in models],
        "archives": archive_fingerprints(archives),
        "weighting": {
            name: {"weights": weights.tolist(), "collision_anchor": anchor}
            for name, (weights, anchor) in weighting.items()
        },
        "update_source": sha(Path(lifelong_training.__file__)),
    }
    if report_path.exists():
        saved = read_json(report_path)
        if saved["inputs"] != key:
            raise ValueError("Continual update origin changed on resume")
        if saved["decision"] != "gate_accepted":
            return models
        if sha(model_path) != saved["checkpoint_sha256"]:
            raise ValueError("Published continual representation changed")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        fresh = tuple(copy.deepcopy(model) for model in models)
        for model, weights in zip(fresh, state, strict=True):
            model.load_state_dict(weights)
        return fresh
    fresh, gate = lifelong_training.update_shared(models, archives, weighting, SETTINGS.online)
    if gate["decision"] == "gate_accepted":
        directory.mkdir(parents=True, exist_ok=True)
        pending = model_path.with_suffix(".tmp")
        torch.save([model.state_dict() for model in fresh], pending)
        pending.replace(model_path)
        gate["checkpoint_sha256"] = sha(model_path)
    write_json(report_path, {**gate, "inputs": key})
    return fresh


def run_branch(kind, sequence, regime, history, models):
    if sequence.study == "in_family" and regime == "fixed" and sequence.name.endswith("_reverse"):
        return
    directory = ROOT / sequence.study / kind / sequence.name / regime
    parents = [bank_for(kind, target) for target in sequence.target_ids]
    blocks = _blocks(parents, sequence.study)
    tasks = [
        original.CandidateBlock(bank, [bank.indices[sid] for sid in blocks[block]])
        for bank, block in zip(parents, sequence.block_ids, strict=True)
    ]
    coordinates = [scene.coordinate() for bank in tasks for scene in bank.candidates]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Continual targets overlap in physical coordinates")
    history = tuple(history)
    weighting = {}
    legacy = regime == "naive_append"
    for ordinal, bank in enumerate(tasks, 1):
        target = sequence.target_ids[ordinal - 1]
        task_directory = directory / "tasks" / target
        if sequence.study == "in_family" and regime == "naive_append":
            saved, paths = _sealed_in_family(kind, sequence, regime, target)
            if (
                saved["specification"]["candidates"] != [asdict(scene) for scene in bank.candidates]
                or saved["specification"]["bank_contract"] != bank.manifest["contract"]
            ):
                raise ValueError("Original E05-C matched outcome has different physical coordinates")
            freeze(task_directory / "sealed_origin.json", _origin_receipt(paths))
            continue
        if legacy:
            sources = build_source(models[1], history, bank.candidates, domain=bank.domain)
        else:
            sources = lifelong_archive.source_snapshot(models[1], history, bank.candidates, weighting)
        saved = evaluate_task(
            bank, {"models": models, "sources": sources}, "lifelong_full", destination=task_directory,
        )
        if regime == "fixed" or ordinal == len(tasks):
            continue
        archive, weights = _archive(bank, saved, f"paid_{target}", ordinal)
        history = (*history, archive)
        if not legacy:
            weighting[archive.source_id] = weights
        freeze(task_directory / "archive.json", {
            "completed_order": ordinal, "source_id": archive.source_id,
            "archive_fingerprint": archive_fingerprints([archive])[archive.source_id],
            "paid_ids": [scene.scenario_id for scene in archive.scenarios],
            "weights": np.ones(len(weights[0])).tolist() if legacy else weights[0].tolist(),
            "collision_anchor": None if legacy else weights[1],
        })
        if regime == "full_lifelong":
            models = _update(task_directory, models, history, weighting)
        print("complete", kind, sequence.name, regime, ordinal, target, flush=True)


def _models(kind, study):
    history = scenario_history(evaluate_e01.ORIGINAL, kind)
    if study == "cold_start":
        history = [archive for archive in history if archive.source_id in RULE_HISTORY]
        if {archive.source_id for archive in history} != set(RULE_HISTORY):
            raise ValueError("Cold start requires the two registered rule sources")
        directory = ROOT / study / kind / "shared_models"
        training.train(history, directory, kernel_type=SharedPrivateKernel)
        models = tuple(
            training.load_model(directory / f"{name}.pt", kernel_type=SharedPrivateKernel)[0]
            for name in ("target_kernel", "historical_encoder")
        )
    else:
        with _prediction_destination(PREDICTION):
            models = evaluate_e01.prepare_history(kind)["models"]
    return history, models


def verify_stage_two():
    from experiments.lifelong_transfer import BUDGET, CONFIGURATIONS

    verify_stage_one()
    for kind in FORMAL_SCENARIOS:
        for target in TARGETS:
            for configuration in CONFIGURATIONS:
                _verify_receipt(BUDGET, ApprovedCase("budget", target, kind, 11, configuration, None))
            _verify_receipt(PREDICTION, ApprovedCase("prediction", target, kind, 11, "lifelong_full", None))


def verify_stage_three():
    verify_stage_two()
    for study in PARTITION_SEEDS:
        verify_study(study)


def verify_study(study):
    registration = read_json(ROOT / study / "protocol.json")
    tasks = sequences(study)
    completed = 0
    for kind in FORMAL_SCENARIOS:
        for sequence in tasks:
            for regime in registration["regimes"]:
                folder = ROOT / study / kind / sequence.name / regime / "tasks"
                for ordinal, target in enumerate(sequence.target_ids, 1):
                    result = branch_result(kind, sequence, regime, target)
                    if result["result"]["selected_indices"][:20] != result["specification"]["initial_indices"]:
                        raise ValueError("Continual branch changed its common random initialization")
                    if regime == "full_lifelong" and ordinal < len(sequence.target_ids):
                        receipt = read_json(folder / target / "gate.json")
                        if receipt["decision"] not in ("gate_accepted", "gate_rejected_retained_parent"):
                            raise ValueError("Continual model update lacks an auditable gate decision")
                    completed += 1
            if study == "in_family":
                first_target = sequence.target_ids[0]
                fixed = branch_result(kind, sequence, "fixed", first_target)
                full = branch_result(kind, sequence, "full_lifelong", first_target)
                if (
                    fixed["result"]["selected_indices"] != full["result"]["selected_indices"]
                    or fixed["result"]["prediction_snapshots"] != full["result"]["prediction_snapshots"]
                ):
                    raise ValueError("E05-C first position must reproduce the fixed predictions and order")
    return completed


def _run_independent_branch(job):
    kind, sequence, regime, history, models = job
    run_branch(kind, sequence, regime, history, models)
    return kind, sequence.name, regime


def run(study, *, workers=None):
    registered = sequences(study)
    verify_stage_two()
    freeze(ROOT / study / "protocol.json", {
        "scenes": list(FORMAL_SCENARIOS), "partition_seed": PARTITION_SEEDS[study],
        "branches": {s.name: {"targets": list(s.target_ids), "block_ids": list(s.block_ids)}
                     for s in registered},
        "regimes": list(REGIMES),
        "naive_append": (
            "reuse the sealed v1 update_shared branch" if study == "in_family"
            else "v1 archive posterior and equal source weights with shared acquisition"
        ),
        "archive_order": "paid sequence within each branch",
        "candidates_per_block": 400 if study == "cross_architecture" else 500, "physical_calls": 0,
        "initial_queries": 20, "budget": 200,
    })
    prepare_processes()
    initialize_worker()
    job_context = {kind: _models(kind, study) for kind in FORMAL_SCENARIOS}
    jobs = [
        (kind, sequence, regime, *job_context[kind])
        for regime in ("full_lifelong", "debiased_append", "naive_append", "fixed")
        for kind in FORMAL_SCENARIOS for sequence in registered
    ]
    stage_limit = min(12, COLLECTION_WORKERS) if study == "cold_start" else METHOD_WORKERS
    concurrency = stage_limit if workers is None else min(stage_limit, workers)
    with ProcessPoolExecutor(
        max_workers=concurrency, mp_context=mp.get_context("spawn"), initializer=initialize_worker,
    ) as pool:
        for finished in pool.map(_run_independent_branch, jobs):
            print("complete", study, *finished, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=tuple(PARTITION_SEEDS))
    parser.add_argument("--workers", type=int, help="Upper limit on concurrent branches")
    args = parser.parse_args()
    if args.workers is not None and not 0 < args.workers <= COLLECTION_WORKERS:
        parser.error("--workers must fit the available CPU task concurrency")
    study = ROOT / args.study
    study.mkdir(parents=True, exist_ok=True)
    completion = study / "completion.json"
    if not completion.exists():
        run(args.study, workers=args.workers)
    completed = verify_study(args.study)
    freeze(completion, {"study": args.study, "positions": completed, "new_physical_executions": 0})
    print("verified", args.study, completed, "positions", flush=True)


if __name__ == "__main__":
    main()
