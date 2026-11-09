"""Fixed-history and chronological paid-feedback experiments for the joint method."""

import argparse
import copy
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.selection import aggregate_history, select
from testing_methods.history_transfer.training import (
    DEFAULT as TRAINING_DEFAULT,
)
from testing_methods.history_transfer.training import (
    load_model,
    model_directory,
    training_contract,
)

from .costs import PeakMemory
from .discovery_config import read_json, write_json
from .domain_risk import disclose_driving_quality, disclose_impact
from .executions import collect_bank
from .failure_metrics import METRIC_CONTRACT, summarize
from .history import (
    build_source,
    completed_archives,
    finalize_task,
    load_training_archives,
    model_version,
    update_shared_model,
)
from .initialization import initial_indices
from .prepare_joint import DATASET_ROOT
from .protocol import ROOT, RUNS_ROOT
from .query import QueryLedger, run_isolated
from .rss import measurement_contract
from .scenarios import TEST_SCENARIOS, candidate_pool
from .scenarios import candidate_count as scenario_candidate_count

OUTPUT = RUNS_ROOT / "joint_response"
REGIMES = ("fixed_history", "append_sources", "update_shared")


@dataclass(frozen=True)
class Config:
    candidate_count: int | None = None
    budget: int = 200
    initial_count: int = 20
    algorithm_seeds: tuple = (11,)
    scenarios: tuple = TEST_SCENARIOS
    update_every_systems: int = 2
    batch_size: int = 128
    environment_seed: int = 81001
    device: str = "cpu"

    def __post_init__(self):
        if not 0 < self.initial_count <= self.budget:
            raise ValueError("Invalid experiment budget")
        if self.candidate_count is not None and self.candidate_count <= 0:
            raise ValueError("Candidate count must be positive")
        if not self.scenarios or any(kind not in TEST_SCENARIOS for kind in self.scenarios):
            raise ValueError("Experiment requires current test scenarios")
        if self.update_every_systems < 1:
            raise ValueError("Update period must be positive")
        if torch.device(self.device).type not in ("cpu", "cuda"):
            raise ValueError("Joint inference supports CPU or CUDA")


DEFAULT = Config()


def freeze_protocol(root, suite, config, systems, dataset_manifest, *, extra=None):
    paths = (
        "experiments/joint_validation.py",
        "experiments/initialization.py",
        "experiments/candidate_sensitivity.py",
        "experiments/parallel_comparisons.py",
        "experiments/history.py",
        "experiments/query.py",
        "experiments/schema.py",
        "experiments/failure_metrics.py",
        "testing_methods/history_transfer/selection.py",
    )
    specification = json.loads(
        json.dumps(
            {
                "format": "joint-response-discovery",
                "cpu_torch_threads": torch.get_num_threads(),
                "suite": suite,
                "config": asdict(config),
                "candidate_counts": {
                    kind: config.candidate_count or scenario_candidate_count(kind)
                    for kind in config.scenarios
                },
                "systems": [asdict(s) for s in systems],
                "rss_contract": measurement_contract(),
                "training_contract": training_contract(),
                "runner_contract": {
                    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
                },
                "dataset_sha256": hashlib.sha256(Path(dataset_manifest).read_bytes()).hexdigest(),
                "source_availability": "completed paid tasks in the same regime and seed, strictly before this ADS",
                **(extra or {}),
            }
        )
    )
    path = Path(root) / "reports/protocol.json"
    if path.exists() and read_json(path) != specification:
        raise ValueError(
            "Existing experiment protocol differs; use a distinct experiment destination"
        )
    write_json(path, specification)
    return specification


def initial_models(kind, held_family=None):
    root = model_directory(kind, held_family)
    models = tuple(
        load_model(root / name)[0] for name in ("target_kernel.pt", "historical_encoder.pt")
    )
    if any(m.construction_specification["held_family"] != held_family for m in models):
        raise ValueError("Model construction scope differs from the requested family holdout")
    return models


def target_bank(root, sut, kind, config, *, replicate=10):
    # New coordinate namespaces are separate from recorded target pools and from each other.
    candidates = candidate_pool(kind, config.candidate_count, replicate=replicate)
    return collect_bank(
        sut.sut_id,
        candidates,
        Path(root) / sut.sut_id / kind,
        environment_seed=config.environment_seed,
        current_registration=True,
    )


def run_one(
    destination,
    bank,
    models,
    history,
    config,
    seed,
    *,
    feedback_mode="joint",
    snapshot=None,
    source_cache_root=None,
    native_feedback=False,
    prior_mode="sources",
    historical_rss=True,
    update_source_weights=True,
    initial_pool_size=None,
):
    """Keep the execution bank in the parent and send only public priors to the selector."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    budget = min(config.budget, len(bank.candidates))
    initial_count = min(config.initial_count, budget)
    initial = initial_indices(
        len(bank.candidates), initial_count, seed, pool_size=initial_pool_size
    )
    base, encoder = models
    if prior_mode not in ("sources", "embedding", "aggregate"):
        raise ValueError("Unknown historical prior control")
    if history and (base is None or encoder is None):
        raise ValueError("Historical transfer requires its trained model pair")
    available = archive_fingerprints(history)
    for model in (base, encoder) if history else (base,):
        if hasattr(model, "construction_specification"):
            trained = model.construction_specification["history_fingerprints"]
            if any(available.get(name) != fingerprint for name, fingerprint in trained.items()):
                raise ValueError(
                    "A shared model contains responses outside the admitted task snapshot"
                )
            if model.construction_specification["scenario"] != bank.candidates[0].kind:
                raise ValueError("Shared model belongs to another functional scenario")
    with PeakMemory(device=config.device) as construction:
        sources = (
            build_source(
                copy.deepcopy(encoder).to(config.device),
                history,
                bank.candidates,
                cache_path=destination / "source_priors.npz" if source_cache_root is None else None,
                cache_directory=source_cache_root,
                use_rss=historical_rss,
            )
            if history and prior_mode != "embedding"
            else None
        )
        if sources is not None and prior_mode == "aggregate":
            sources = aggregate_history(sources)
    spec = {
        "target_model_version": None if base is None else model_version(base),
        "encoder_version": None if encoder is None else model_version(encoder),
        "source_snapshot": None if sources is None else sources["specification"],
        "bank_contract": bank.manifest["contract"],
        "environment_seed": bank.manifest["environment_seed"],
        "candidates": [asdict(scene) for scene in bank.candidates],
        "feedback_mode": feedback_mode,
        "prior_mode": prior_mode,
        "historical_rss": historical_rss,
        "update_source_weights": update_source_weights,
        "feedback_contract": "paid-collision-rss-actual-trajectory",
        "budget": budget,
        "initial_count": initial_count,
        "seed": seed,
        "batch_size": config.batch_size,
        "task_snapshot": snapshot,
        "metric_contract": METRIC_CONTRACT,
        "device": config.device,
        "prediction_snapshot_counts": [0, 20, 50, 200],
        "native_feedback": native_feedback,
    }
    if initial_pool_size is not None:
        spec.update(initial_pool_size=initial_pool_size, initial_indices=initial)
    public = dict(
        candidates=bank.candidates,
        pretrained=base,
        sources=sources,
        initial_indices=initial,
        budget=budget,
        batch_size=config.batch_size,
        feedback_mode=feedback_mode,
        mode="H0F1" if prior_mode == "aggregate" else "H1F1",
        update_source_weights=update_source_weights,
        device=config.device,
    )
    return execute_selector(
        destination,
        bank,
        select,
        public,
        spec,
        construction_resources=construction.result,
        native_feedback=native_feedback,
    )


def execute_selector(
    destination,
    bank,
    selector,
    public,
    specification,
    *,
    construction_resources=None,
    native_feedback=False,
):
    """Run any method with the same evaluator-owned truth, budget and disclosures."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    specification = json.loads(json.dumps(specification))
    budget, initial_count = specification["budget"], specification["initial_count"]

    def execute(scene):
        execution = bank(scene)
        if native_feedback:
            execution = disclose_driving_quality(disclose_impact(execution))
        return execution

    path = destination / "result.json"
    if path.exists():
        saved = read_json(path)
        if saved["specification"] != specification:
            raise ValueError("Saved result differs from the fixed model/source/task snapshot")
        # Recover only already-paid traces when resuming, never disclose extra outcomes.
        ledger = QueryLedger(bank.candidates, budget, execute, joint_feedback=True)
        ledger.records = saved["query_ledger"]
        ledger.trajectories = [
            bank(bank.candidates[r["index"]]).trajectory
            for r in ledger.records
            if r["status"] == "complete"
        ]
        return saved, ledger
    ledger = QueryLedger(bank.candidates, budget, execute, joint_feedback=True)
    try:
        result = run_isolated(selector, public, ledger)
    except Exception:
        write_json(
            destination / f"failed_attempts_{time.time_ns()}.json",
            {"specification": specification, "query_ledger": ledger.records},
        )
        raise
    labels = [bank(s).feedback.collision for s in bank.candidates]
    metrics = summarize(
        bank.candidates,
        labels,
        result["queries"],
        ledger.trajectories,
        initial_count=initial_count,
        query_ledger=ledger.records,
    )
    saved = {
        "specification": specification,
        "result": result,
        "metrics": metrics,
        "query_ledger": ledger.records,
        "source_construction_resources": construction_resources,
        "evaluation": {"candidate_collisions": labels},
    }
    write_json(path, saved)
    return saved, ledger


def update_scenario(
    destination,
    history,
    models,
    *,
    completed,
    new_source_ids,
    config,
    training_config,
):
    """Update one scenario and replay its paid tasks without changing another scope."""
    torch.set_num_threads(1)
    active, publication = update_shared_model(
        destination, history, models, new_source_ids=new_source_ids, config=training_config
    )
    if publication["accepted"]:
        from .retention import compare

        compare(Path(destination) / "retention.json", models, active, history, completed, config)
    return active, publication


def sequential(
    root=OUTPUT / "sequential",
    *,
    config=DEFAULT,
    systems=CURRENT_INDEPENDENT_SUTS,
    dataset_manifest=DATASET_ROOT / "manifest.json",
    model_loader=initial_models,
    training_config=None,
):
    torch.set_num_threads(1)
    root = Path(root)
    training_config = training_config or TRAINING_DEFAULT
    update_workers = (
        min(4, len(config.scenarios))
        if torch.device(training_config.device).type == torch.device(config.device).type == "cpu"
        else 1
    )
    freeze_protocol(
        root,
        "sequential",
        config,
        systems,
        dataset_manifest,
        extra={
            "shared_update_training": asdict(training_config),
            "shared_update_workers": update_workers,
        },
    )
    rows, updates = [], []
    for seed in config.algorithm_seeds:
        order = list(np.random.default_rng(seed).permutation(len(systems)))
        for regime in REGIMES:
            branch_id = f"{regime}:{seed}"
            branch = root / "methods" / regime / str(seed)
            active = {kind: model_loader(kind) for kind in config.scenarios}
            learned_ids = {kind: set() for kind in config.scenarios}
            completed = []
            for ordinal, index in enumerate(order, 1):
                sut = systems[index]
                for kind in config.scenarios:
                    history = load_training_archives(dataset_manifest, kind)
                    past = completed_archives(
                        completed,
                        before_order=ordinal,
                        branch_id=branch_id,
                        current_system=(sut.sut_id, sut.version),
                    )
                    past = [a for a in past if a.scenarios[0].kind == kind]
                    allowed = history if regime == "fixed_history" else [*history, *past]
                    bank = target_bank(
                        Path(dataset_manifest).parent / "targets/sequential",
                        sut,
                        kind,
                        config,
                        replicate=20,
                    )
                    task = {
                        "sut_id": sut.sut_id,
                        "family": sut.family,
                        "lineage": sut.lineage,
                        "scenario": kind,
                        "ordinal": ordinal,
                        "regime": regime,
                        "seed": seed,
                        "branch_id": branch_id,
                    }
                    destination = branch / "queries" / f"{ordinal:02d}_{sut.sut_id}" / kind
                    saved, ledger = run_one(
                        destination,
                        bank,
                        active[kind],
                        allowed,
                        config,
                        seed,
                        snapshot=task,
                        source_cache_root=root / "models/source_cache",
                    )
                    task_id = f"{seed}:{regime}:{ordinal}:{sut.sut_id}:{kind}"
                    path = finalize_task(
                        branch / "completed" / f"{ordinal:02d}_{sut.sut_id}_{kind}.json.gz",
                        ledger,
                        task_id=task_id,
                        sut_id=sut.sut_id,
                        sut_version=sut.version,
                        family=sut.family,
                        lineage=sut.lineage,
                        environment_seed=config.environment_seed,
                        completed_order=ordinal,
                        branch_id=branch_id,
                        original_role=sut.role,
                        execution_contract=bank.manifest["contract"],
                        model_snapshot={
                            "target_model_version": saved["specification"]["target_model_version"],
                            "encoder_version": saved["specification"]["encoder_version"],
                        },
                    )
                    completed.append(path)
                    rows.append(
                        {
                            **task,
                            "metrics": saved["metrics"],
                            "resources": saved["result"]["resources"],
                            "source_ids": [a.source_id for a in allowed],
                            "result_path": (destination / "result.json")
                            .relative_to(root)
                            .as_posix(),
                        }
                    )
                if (
                    regime == "update_shared"
                    and ordinal % config.update_every_systems == 0
                    and ordinal < len(order)
                ):
                    jobs = []
                    for kind in config.scenarios:
                        history = load_training_archives(dataset_manifest, kind)
                        past = [
                            a
                            for a in completed_archives(
                                completed, before_order=ordinal + 1, branch_id=branch_id
                            )
                            if a.scenarios[0].kind == kind
                        ]
                        allowed = [*history, *past]
                        new_ids = [
                            a.source_id for a in past if a.source_id not in learned_ids[kind]
                        ]
                        update_root = branch / "models" / kind / f"after_{ordinal:02d}_systems"
                        jobs.append(
                            {
                                "destination": update_root,
                                "history": allowed,
                                "models": active[kind],
                                "completed": past,
                                "new_source_ids": new_ids,
                                "config": config,
                                "training_config": training_config,
                            }
                        )
                    if update_workers == 1:
                        results = [update_scenario(**job) for job in jobs]
                    else:
                        with ProcessPoolExecutor(max_workers=update_workers) as pool:
                            futures = [pool.submit(update_scenario, **job) for job in jobs]
                            results = [future.result() for future in futures]
                    # All scopes finish before the next ADS takes its model snapshot.
                    for kind, job, (models, publication) in zip(
                        config.scenarios, jobs, results, strict=True
                    ):
                        active[kind] = models
                        if publication["accepted"]:
                            learned_ids[kind].update(job["new_source_ids"])
                        updates.append(
                            {
                                "branch_id": branch_id,
                                "after_systems": ordinal,
                                "scenario": kind,
                                "publication": publication,
                                "retention_path": (job["destination"] / "retention.json")
                                .relative_to(root)
                                .as_posix()
                                if publication["accepted"]
                                else None,
                            }
                        )
                write_json(
                    root / "reports/summary.json",
                    {"status": "running", "runs": rows, "updates": updates},
                )
    write_json(
        root / "reports/summary.json", {"status": "complete", "runs": rows, "updates": updates}
    )
    return rows, updates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("fixed", "sequential"), default="fixed")
    args = parser.parse_args()
    if args.suite == "fixed":
        from .pilot import fixed_comparison

        fixed_comparison(
            OUTPUT / "fixed",
            systems=CURRENT_INDEPENDENT_SUTS,
            suite="fixed",
            reuse_root=OUTPUT / "pilot",
        )
    else:
        sequential()


if __name__ == "__main__":
    main()
