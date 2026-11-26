"""Formal task declaration and batches using validated fixed GP defaults.

Study-level readiness is checked before task declarations or target queries.
Root study commands and the direct CLI use this same admitted runner.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import torch

from controllers.highway.catalog import FORMAL_TARGET_SUTS
from testing_methods.baselines.scenario_fuzz.model import DEFAULT as CLASSIFIER_CONFIG
from testing_methods.common.response import archive_fingerprints

from . import transfer_execution as execution
from .collect_transfer_records import _check_formal_specification, _formal_context
from .costs import PeakMemory
from .executions import ExecutionBank, execution_contract
from .history import build_source, model_version
from .history_transfer_experiment import (
    BUDGET_METHODS,
    CHECKPOINTS,
    OUTPUT,
    PREDICTION_METHODS,
    SEEDS,
    TARGET_COUNT,
    freeze,
    models_and_history,
    pooled_history,
    registered_history_views,
    registered_populations,
    scenarios_for,
)
from .k_shot import split_candidates
from .parallel import (
    METHOD_WORKERS,
    TRAINING_DEVICE,
    initialize_worker,
    prediction_device,
    prediction_workers,
    prepare_processes,
)
from .baselines import historical_classifiers
from .protocol import ROOT
from .storage import read_json, write_json
from .transfer_settings import REGISTRY, load_settings

WORKERS = METHOD_WORKERS
ACCEPTANCE = "reports/method_acceptance.json"
_CONTEXT = None
_BANKS = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limits_from_acceptance(root):
    from .transfer_admission import load_acceptance

    return load_acceptance(root)["execution_contract"]["limits"]


def classifier_input(root, kind, history):
    directory = Path(root) / "models/baselines" / kind
    completed = read_json(directory / "training_completion.json")
    model = historical_classifiers(
        history, directory, names=("scenario_fuzz",), train_missing=False
    )["scenario_fuzz"]
    if (
        completed["status"] != "complete"
        or completed["specification"]["device"] != TRAINING_DEVICE
        or completed["training_device"] != TRAINING_DEVICE
        or completed["specification"]["history_fingerprints"] != archive_fingerprints(history)
        or completed["model_version"] != model_version(model)
        or completed["weights_sha256"] != _sha(directory / "scenario_fuzz/classifier.pt")
        or completed["epochs"] != CLASSIFIER_CONFIG.epochs
        or completed["specification"]["configuration"] != asdict(CLASSIFIER_CONFIG)
        or any(
            _sha(ROOT / name) != digest
            for name, digest in completed["specification"]["source_sha256"].items()
        )
    ):
        raise ValueError("Formal ScenarioFuzz input differs from its completed historical training")
    return {
        "model": model,
        "construction": {
            "model_version": model_version(model),
            "history_fingerprints": archive_fingerprints(history),
        },
    }


def prepare_context(root, kind, mode):
    root = Path(root)
    models, history = models_and_history(root, kind)
    views, view = registered_history_views(root, kind, history)
    population = registered_populations(root)["target", kind]
    banks = {}
    for sut in FORMAL_TARGET_SUTS:
        bank = ExecutionBank(root / "physical/target" / kind / sut.sut_id)
        if (
            len(bank.candidates) != TARGET_COUNT
            or bank.candidates != population
            or bank.manifest["sut_id"] != sut.sut_id
            or bank.manifest["contract"] != execution_contract(sut, kind=kind)
            or bank.manifest["environment_seed"] != 81001
        ):
            raise ValueError("Formal bank differs from its registered target and public population")
        banks[sut.sut_id] = bank
    cache = root / "sources" / kind
    existing = set(cache.glob("*.npz"))
    with PeakMemory() as resources:
        sources = build_source(
            models[1],
            history,
            population,
            cache_directory=cache,
            view_record_ids=view["selected_record_ids"],
        )
    key = hashlib.sha256(json.dumps(sources["specification"], sort_keys=True).encode()).hexdigest()
    receipt = root / "reports/source_construction" / kind / f"{key}.json"
    if not receipt.exists():
        write_json(
            receipt,
            {
                "specification": sources["specification"],
                "resources": resources.result,
                "scope": (
                    "cache_load" if cache / f"{key}.npz" in existing else "source_construction"
                ),
                "shared_across_target_ADSes_and_seeds": True,
            },
        )
    else:
        if read_json(receipt)["specification"] != sources["specification"]:
            raise ValueError("Shared source-construction receipt changed")
    return {
        "models": models,
        "sources": sources,
        "pooled": pooled_history(views, view),
        "classifier": classifier_input(root, kind, history) if mode == "budget" else None,
        "source_resource_receipt": str(receipt.relative_to(root)),
    }, banks


def make_cases(mode, banks, context, settings, limits):
    if mode not in ("prediction", "budget"):
        raise ValueError("Unknown registered evaluation mode")
    if set(banks) != {s.sut_id for s in FORMAL_TARGET_SUTS}:
        raise ValueError("Declare the complete eight-ADS task cohort")
    cases = []
    for sut in FORMAL_TARGET_SUTS:
        bank = banks[sut.sut_id]
        for seed in SEEDS:
            methods = PREDICTION_METHODS if mode == "prediction" else BUDGET_METHODS
            for method in methods:
                models = context["models"] if method in ("full", "pooled_history") else (None, None)
                sources = context["sources"] if method == "full" else None
                if mode == "prediction":
                    case = execution.prediction_case(
                        bank,
                        split_candidates(bank.candidates, seed),
                        seed,
                        method,
                        models,
                        context["pooled"] if method == "pooled_history" else None,
                        sources,
                        settings=settings,
                        checkpoints=CHECKPOINTS,
                        limits=limits,
                    )
                else:
                    case = execution.budget_case(
                        bank,
                        seed,
                        method,
                        models,
                        sources,
                        settings=settings,
                        initial_count=20,
                        budget=200,
                        limits=limits,
                        classifier=context["classifier"] if method == "scenario_fuzz" else None,
                    )
                cases.append(case)
    return cases


def declare_batch(root, mode, kind, cases, settings, limits):
    root = Path(root)
    methods = PREDICTION_METHODS if mode == "prediction" else BUDGET_METHODS
    expected = {
        (mode, s.sut_id, kind, seed, m)
        for s in FORMAL_TARGET_SUTS
        for seed in SEEDS
        for m in methods
    }
    if {c.key for c in cases} != expected or len(cases) != len(expected):
        raise ValueError("Formal task declaration omits or duplicates a registered case")
    context = _formal_context(root, cases)
    if settings.to_dict() != context["settings"].to_dict():
        raise ValueError("Formal batch settings differ from the registered fixed defaults")
    for case in cases:
        if case.specification["evaluate_complete_bank"] is not True:
            raise ValueError("Formal tasks require complete fixed evaluation after paid querying")
        if case.specification["execution_contract"] != execution.execution_contract(limits):
            raise ValueError("Formal methods must share the admitted execution limits")
        _check_formal_specification(case, context)
    directory = root / "reports/execution_tasks" / mode / kind
    protocol = freeze(
        directory / "protocol.json",
        {
            "mode": mode,
            "scenario": kind,
            "entry_source_sha256": _sha(__file__),
            "execution_contract": execution.execution_contract(limits),
            "settings": settings.to_dict(),
            "settings_sha256": _sha(root / REGISTRY),
            "candidate_count": TARGET_COUNT,
            "workers": WORKERS,
            "pooled_workers": prediction_workers("pooled_history") if mode == "prediction" else 0,
            "prediction_devices": (
                {m: prediction_device(m) for m in methods} if mode == "prediction" else {}
            ),
            "threads_per_selector": 1,
            "cases": [
                {
                    "key": list(c.key),
                    "specification_sha256": hashlib.sha256(
                        json.dumps(c.specification, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest(),
                }
                for c in cases
            ],
        },
    )
    for case in cases:
        execution.declare(root, case, scope="formal")
    return directory, protocol


def _initialize(root, context, settings):
    global _CONTEXT, _BANKS
    initialize_worker()
    _CONTEXT = {"root": Path(root), "inputs": context, "settings": settings}
    _BANKS = {}


def _run_case(case):
    context = _CONTEXT
    root = context["root"]
    inputs = context["inputs"]
    settings = context["settings"]
    key = case.sut_id, case.scenario
    if key not in _BANKS:
        _BANKS[key] = ExecutionBank(root / "physical/target" / case.scenario / case.sut_id)
    bank = _BANKS[key]
    models = inputs["models"] if case.method in ("full", "pooled_history") else (None, None)
    sources = inputs["sources"] if case.method == "full" else None
    try:
        if case.experiment == "prediction":
            execution.run_prediction(
                root,
                case,
                bank,
                models,
                inputs["pooled"] if case.method == "pooled_history" else None,
                sources,
                settings=settings,
                paid_execution=lambda i: bank(bank.candidates[i]),
            )
        else:
            execution.run_budget(
                root,
                case,
                bank,
                models,
                sources,
                settings=settings,
                paid_execution=lambda i: bank(bank.candidates[i]),
                classifier=inputs["classifier"] if case.method == "scenario_fuzz" else None,
                construction_resources=(
                    {
                        "shared_receipt": inputs["source_resource_receipt"],
                        "scope": "shared_scenario_constructor",
                    }
                    if case.method == "full"
                    else None
                ),
            )
    except Exception as error:
        return {"key": list(case.key), "status": "failed", "error": repr(error)}
    return {"key": list(case.key), "status": "complete"}


def run(mode, root=OUTPUT):
    from .history_transfer_experiment import require_method_evaluation
    from .transfer_admission import require_experiment

    require_experiment(mode)
    root = Path(root)
    require_method_evaluation(root)
    settings = load_settings(root)
    limits = limits_from_acceptance(root)
    pooled_workers = prediction_workers("pooled_history") if mode == "prediction" else 0
    prepare_processes()
    torch.set_num_threads(1)
    for kind in scenarios_for(root):
        context, banks = prepare_context(root, kind, mode)
        cases = make_cases(mode, banks, context, settings, limits)
        directory, protocol = declare_batch(root, mode, kind, cases, settings, limits)
        completed = []
        with ExitStack() as stack:
            futures = {}
            groups = (
                ([c for c in cases if c.method != "pooled_history"], WORKERS),
                (
                    [c for c in cases if c.method == "pooled_history"],
                    pooled_workers,
                ),
            )
            for tasks, workers in groups:
                if not tasks:
                    continue
                pool = stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=mp.get_context("spawn"),
                        initializer=_initialize,
                        initargs=(root, context, settings),
                    )
                )
                futures.update({pool.submit(_run_case, c): c for c in tasks})
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as error:
                    row = {
                        "key": list(futures[future].key),
                        "status": "failed",
                        "error": repr(error),
                    }
                completed.append(row)
                write_json(
                    directory / "progress.json",
                    {"protocol": protocol, "completed": completed, "expected": len(cases)},
                )
                print(row, flush=True)


def approved_cases(root, mode):
    """Load independent declarations; terminal status comes from the batch ledger."""
    from .collect_transfer_records import ApprovedCase

    if mode not in ("prediction", "budget"):
        raise ValueError("Unknown registered evaluation mode")
    root = Path(root)
    cases = []
    for kind in scenarios_for(root):
        directory = root / "reports/execution_tasks" / mode / kind
        protocol = read_json(directory / "protocol.json")
        if protocol["entry_source_sha256"] != _sha(__file__) or protocol["settings_sha256"] != _sha(
            root / REGISTRY
        ):
            raise ValueError("Formal task declarations changed after admission")
        if protocol["execution_contract"] != execution.execution_contract(
            protocol["execution_contract"]["limits"]
        ):
            raise ValueError("Formal executor changed after task admission")
        progress = (
            read_json(directory / "progress.json")
            if (directory / "progress.json").exists()
            else {"completed": []}
        )
        if (directory / "progress.json").exists() and progress["protocol"] != protocol:
            raise ValueError("Formal progress belongs to another declaration")
        statuses = {tuple(r["key"]): r for r in progress["completed"]}
        for declaration in protocol["cases"]:
            key = tuple(declaration["key"])
            case = ApprovedCase(*key, specification=None)
            saved = read_json(execution.approval_path(root, case))
            if (
                saved["key"] != list(key)
                or saved["scope"] != "formal"
                or hashlib.sha256(
                    json.dumps(saved["specification"], sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                != declaration["specification_sha256"]
            ):
                raise ValueError("Formal task approval differs from its pre-query declaration")
            cases.append(
                replace(
                    case,
                    specification=saved["specification"],
                    final_failure=statuses.get(key, {}).get("status") == "failed",
                )
            )
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prediction", "budget", "status"))
    args = parser.parse_args()
    if args.mode == "status":
        from .history_transfer_experiment import readiness

        print(readiness(), flush=True)
    else:
        run(args.mode)
