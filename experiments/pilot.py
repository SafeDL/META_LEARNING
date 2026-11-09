"""Fixed-history comparisons with shared queries, complete baselines and reusable pilot runs."""

import gzip
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from controllers.highway.catalog import CURRENT_BY_ID
from testing_methods.baselines.av_bo_rf_ei import method as rf_ei
from testing_methods.baselines.behavexplor import pool as behavexplor
from testing_methods.baselines.indago import method as indago_model
from testing_methods.baselines.indago import pool as indago
from testing_methods.baselines.paired_system_active_evaluation import method as paired
from testing_methods.baselines.response_baselines import collision as random
from testing_methods.baselines.samota import pool as samota
from testing_methods.baselines.scenario_fuzz import model as fuzz_model
from testing_methods.baselines.scenario_fuzz import pool as scenario_fuzz
from testing_methods.common.response import archive_fingerprints, shared_domain

from .costs import PeakMemory, measure_resources
from .failure_metrics import METRIC_CONTRACT
from .history import finalize_task, load_training_archives, model_version
from .initialization import initial_indices
from .joint_validation import (
    DEFAULT,
    OUTPUT,
    execute_selector,
    freeze_protocol,
    initial_models,
    run_one,
    target_bank,
)
from .prepare_joint import DATASET_ROOT
from .protocol import ROOT
from .storage import read_json, write_json

SYSTEMS = (CURRENT_BY_ID["independent_idm_mobil"], CURRENT_BY_ID["independent_ppo"])
BASELINES = {
    "random": random,
    "av_bo_rf_ei": rf_ei,
    "samota": samota,
    "scenario_fuzz": scenario_fuzz,
    "paired_system_active_evaluation": paired,
    "indago": indago,
    "behavexplor": behavexplor,
}
MAIN_METHODS = ("joint", *BASELINES)
PRIMARY_BASELINES = (
    "random",
    "av_bo_rf_ei",
    "samota",
    "scenario_fuzz",
    "paired_system_active_evaluation",
    "behavexplor",
)
CONTROLS = ("collision", "no_history")
MECHANISMS = ("historical_collision", "embedding_only", "aggregate", "fixed_weights")


def method_configurations(methods=tuple(BASELINES)):
    return {
        **{
            name: asdict(module.DEFAULT)
            for name, module in BASELINES.items()
            if name in methods and name != "random"
        },
        **({"indago_training": asdict(indago_model.DEFAULT)} if "indago" in methods else {}),
        **(
            {"scenario_fuzz_training": asdict(fuzz_model.DEFAULT)}
            if "scenario_fuzz" in methods
            else {}
        ),
    }


def implementation_contract():
    paths = [
        *sorted((ROOT / "testing_methods/baselines").rglob("*.py")),
        *sorted((ROOT / "testing_methods/common").glob("*.py")),
        ROOT / "experiments/pilot.py",
        ROOT / "experiments/domain_risk.py",
        ROOT / "experiments/initialization.py",
    ]
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def historical_classifiers(
    history,
    destination,
    *,
    train_missing=True,
    names=("indago", "scenario_fuzz"),
    graph_fn=None,
    graph_definition=None,
):
    """Train/reuse only same-scenario histories, including their full supplemental batches."""
    if len({a.scenarios[0].kind for a in history}) != 1:
        raise ValueError("Baseline construction requires one functional scenario")
    domain = shared_domain(history)
    if domain is None and (graph_fn is not None or graph_definition is not None):
        raise ValueError("A custom graph builder requires an explicit parameter definition")
    classifiers, reports = {}, {}
    modules = {"indago": indago_model, "scenario_fuzz": fuzz_model}
    for name in names:
        options = {}
        if domain is not None:
            if name != "scenario_fuzz":
                raise ValueError("Only the six main baselines are admitted to new-domain tasks")
            if graph_fn is None or not graph_definition:
                raise ValueError("New-domain graph training requires its registered public builder")
            options = {"graph_fn": graph_fn, "graph_definition": graph_definition}
        module = modules[name]
        path = Path(destination) / name / "classifier.pt"
        if train_missing and not path.exists():
            module.train(history, path, config=module.DEFAULT, **options)
        model, checkpoint = module.load(path)
        if (
            checkpoint["history_fingerprints"] != archive_fingerprints(history)
            or checkpoint["config"] != asdict(module.DEFAULT)
            or checkpoint["held_family"] is not None
        ):
            raise ValueError("Baseline classifier differs from its historical construction")
        if checkpoint.get("parameter_definition") != (None if domain is None else domain.to_dict()):
            raise ValueError("Baseline classifier uses another parameter definition")
        if domain is not None and checkpoint.get("graph_definition") != graph_definition:
            raise ValueError("Baseline classifier uses another public graph definition")
        classifiers[name] = model
        reports[name] = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    if train_missing:
        write_json(Path(destination) / "construction.json", reports)
    return classifiers


@measure_resources
def baseline_selector(
    candidates, oracle, *, method, initial_indices, budget, seed, domain=None, **state
):
    if domain is not None:
        if method in ("av_bo_rf_ei", "paired_system_active_evaluation"):
            state["feature_fn"] = domain.features
        elif method in ("samota", "scenario_fuzz", "behavexplor"):
            state["domain"] = domain
        elif method != "random":
            raise ValueError("Only the six main baselines are admitted to new-domain tasks")
    return BASELINES[method].run(
        candidates, oracle, initial_indices=initial_indices, budget=budget, seed=seed, **state
    )


def baseline_task(
    destination,
    bank,
    history,
    classifiers,
    config,
    seed,
    task,
    *,
    initial_pool_size=None,
    graph_fn=None,
    graph_definition=None,
):
    method = task["method"]
    domain = getattr(bank, "domain", None)
    if (
        domain is None
        and method == "scenario_fuzz"
        and (graph_fn is not None or graph_definition is not None)
    ):
        raise ValueError("A custom graph builder requires an explicit parameter definition")
    if domain is not None and config.budget > len(bank.candidates):
        raise ValueError("Registered task budget exceeds its fixed candidate library")
    budget = min(config.budget, len(bank.candidates))
    initial_count = min(config.initial_count, budget)
    initial = initial_indices(
        len(bank.candidates), initial_count, seed, pool_size=initial_pool_size
    )
    public = dict(
        candidates=bank.candidates,
        method=method,
        initial_indices=initial,
        budget=budget,
        seed=seed,
    )
    if domain is not None:
        public["domain"] = domain
        if method == "scenario_fuzz":
            if graph_fn is None or not graph_definition:
                raise ValueError("New-domain ScenarioFuzz requires its registered public builder")
            public.update(graph_fn=graph_fn, graph_definition=graph_definition)
    construction = {}
    if method != "random":
        public["config"] = BASELINES[method].DEFAULT
    with PeakMemory() as memory:
        if method in classifiers:
            classifier = classifiers[method]
            public["classifier"] = classifier
            construction = {
                "model_version": model_version(classifier),
                "history_fingerprints": archive_fingerprints(history),
            }
            if domain is not None:
                construction["graph_definition"] = graph_definition
        elif method == "paired_system_active_evaluation":
            options = {} if domain is None else {"feature_fn": domain.features}
            public["proxy"], construction = paired.build_proxy(bank.candidates, history, **options)
    spec = {
        "task_snapshot": task,
        "method": method,
        "configuration": {} if method == "random" else asdict(BASELINES[method].DEFAULT),
        "construction": construction,
        "bank_contract": bank.manifest["contract"],
        "environment_seed": bank.manifest["environment_seed"],
        "candidates": [asdict(s) for s in bank.candidates],
        "budget": budget,
        "initial_count": initial_count,
        "seed": seed,
        "feedback_contract": "paid-collision-rss-native-risk-actual-trajectory",
        "metric_contract": METRIC_CONTRACT,
    }
    # Runtime is reported separately and does not identify a reproducible source snapshot.
    spec["construction"] = {k: v for k, v in construction.items() if k != "construction_s"}
    if initial_pool_size is not None:
        spec.update(initial_pool_size=initial_pool_size, initial_indices=initial)
    if domain is not None:
        spec["parameter_definition"] = domain.to_dict()
    return execute_selector(
        destination,
        bank,
        baseline_selector,
        public,
        spec,
        construction_resources=memory.result,
        native_feedback=True,
    )


def fixed_comparison(
    root=OUTPUT / "pilot",
    *,
    config=DEFAULT,
    systems=SYSTEMS,
    dataset_manifest=DATASET_ROOT / "manifest.json",
    model_loader=initial_models,
    classifier_builder=historical_classifiers,
    suite="pilot",
    methods=(*MAIN_METHODS, *CONTROLS),
    reuse_root=None,
    historical_collision_loader=None,
    candidate_replicate=10,
    bank_loader=None,
    initial_pool_size=None,
):
    torch.set_num_threads(1)
    root = Path(root)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(name not in (*MAIN_METHODS, *CONTROLS, *MECHANISMS) for name in methods)
    ):
        raise ValueError("Comparison requires distinct registered methods")
    if "historical_collision" in methods and historical_collision_loader is None:
        raise ValueError("Historical RSS control requires collision-only trained models")
    if suite == "sensitivity" and initial_pool_size is None:
        raise ValueError("Candidate sensitivity requires an explicit shared initialization pool")
    expected_tasks = len(systems) * len(config.scenarios) * len(config.algorithm_seeds)
    main_methods = [m for m in methods if m in MAIN_METHODS]
    controls = [m for m in methods if m not in MAIN_METHODS]
    protocol = freeze_protocol(
        root,
        suite,
        config,
        systems,
        dataset_manifest,
        extra={
            "main_methods": main_methods,
            "controls": controls,
            "main_runs": expected_tasks * len(main_methods),
            "control_runs": expected_tasks * len(controls),
            "method_configurations": method_configurations(methods),
            "baseline_contract": implementation_contract(),
            "source_availability": "fixed original historical ADSs; completed tasks remain sealed",
            "candidate_replicate": candidate_replicate,
            "reuse_root": None if reuse_root is None else os.path.relpath(reuse_root, root),
            **({"initial_pool_size": initial_pool_size} if initial_pool_size is not None else {}),
        },
    )
    reused = {}
    if reuse_root is not None:
        reuse_root = Path(reuse_root)
        original = read_json(reuse_root / "reports/protocol.json")
        for key in (
            "config",
            "candidate_counts",
            "rss_contract",
            "training_contract",
            "runner_contract",
            "dataset_sha256",
            "method_configurations",
            "baseline_contract",
            "candidate_replicate",
            "cpu_torch_threads",
        ):
            if original[key] != protocol[key]:
                raise ValueError(f"Reused comparison differs in {key}")
        summary = read_json(reuse_root / "reports/summary.json")
        if summary["status"] != "complete":
            raise ValueError("Only completed comparisons may supply reusable runs")
        for sut in systems:
            original_suts = [s for s in original["systems"] if s["sut_id"] == sut.sut_id]
            if original_suts and original_suts != [asdict(sut)]:
                raise ValueError("Reused ADS registration differs from the current task")
        reused = {(r["sut_id"], r["scenario"], r["seed"], r["method"]): r for r in summary["runs"]}
    prepared = {}
    for kind in config.scenarios:
        history = load_training_archives(dataset_manifest, kind)
        models = model_loader(kind)
        classifier_root = root if reuse_root is None else reuse_root
        classifiers = (
            classifier_builder(history, classifier_root / "models" / kind)
            if any(name in methods for name in ("indago", "scenario_fuzz"))
            else {}
        )
        collision_models = (
            historical_collision_loader(kind) if "historical_collision" in methods else None
        )
        prepared[kind] = history, models, classifiers, collision_models
    rows = []
    for sut in systems:
        for kind in config.scenarios:
            history, models, classifiers, collision_models = prepared[kind]
            bank = (target_bank if bank_loader is None else bank_loader)(
                Path(dataset_manifest).parent / "targets/fixed",
                sut,
                kind,
                config,
                replicate=candidate_replicate,
            )
            for seed in config.algorithm_seeds:
                for method in methods:
                    task = {
                        "sut_id": sut.sut_id,
                        "family": sut.family,
                        "lineage": sut.lineage,
                        "scenario": kind,
                        "held_family": None,
                        "seed": seed,
                        "method": method,
                    }
                    destination = root / "methods" / method / sut.sut_id / kind / str(seed)
                    previous = reused.get((sut.sut_id, kind, seed, method))
                    if previous is not None:
                        destination = (reuse_root / previous["result_path"]).parent
                    if method in BASELINES:
                        saved, ledger = baseline_task(
                            destination,
                            bank,
                            history,
                            classifiers,
                            config,
                            seed,
                            task,
                            initial_pool_size=initial_pool_size,
                        )
                    else:
                        saved, ledger = run_one(
                            destination,
                            bank,
                            (None, None)
                            if method == "no_history"
                            else (collision_models if method == "historical_collision" else models),
                            [] if method == "no_history" else history,
                            config,
                            seed,
                            feedback_mode="collision" if method == "collision" else "joint",
                            snapshot=task,
                            source_cache_root=root / "models/source_cache",
                            native_feedback=True,
                            prior_mode={
                                "embedding_only": "embedding",
                                "aggregate": "aggregate",
                            }.get(method, "sources"),
                            historical_rss=method != "historical_collision",
                            update_source_weights=method != "fixed_weights",
                            initial_pool_size=initial_pool_size,
                        )
                    original_task = (
                        None
                        if previous is None
                        else json.loads(
                            gzip.decompress((destination / "completed.json.gz").read_bytes())
                        )
                    )
                    task_id = (
                        f"{suite}:{method}:{sut.sut_id}:{kind}:{seed}"
                        if original_task is None
                        else original_task["task_id"]
                    )
                    finalize_task(
                        destination / "completed.json.gz",
                        ledger,
                        execution_bank=bank,
                        task_id=task_id,
                        sut_id=sut.sut_id,
                        sut_version=sut.version,
                        family=sut.family,
                        lineage=sut.lineage,
                        environment_seed=config.environment_seed,
                        completed_order=len(rows) + 1
                        if original_task is None
                        else original_task["completed_order"],
                        branch_id=f"{suite}:{method}:{seed}"
                        if original_task is None
                        else original_task["branch_id"],
                        original_role=sut.role,
                        execution_contract=bank.manifest["contract"],
                        model_snapshot=saved["specification"],
                    )
                    rows.append(
                        {
                            **task,
                            "metrics": saved["metrics"],
                            "resources": saved["result"]["resources"],
                            "result_path": os.path.relpath(
                                destination / "result.json.gz", root
                            ).replace("\\", "/"),
                            "comparison": "main" if method in MAIN_METHODS else "control",
                            "reused_from": None
                            if previous is None
                            else os.path.relpath(reuse_root, root).replace("\\", "/"),
                        }
                    )
                    write_json(root / "reports/summary.json", {"status": "running", "runs": rows})
                    print(
                        suite,
                        len(rows),
                        "/",
                        expected_tasks * len(methods),
                        sut.sut_id,
                        kind,
                        method,
                        flush=True,
                    )
    write_json(root / "reports/summary.json", {"status": "complete", "runs": rows})
    return rows
