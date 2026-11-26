"""Run the preregistered 12-ADS v2 discovery and prediction matrix."""

import argparse
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from experiments import transfer_execution as execution
from experiments.budget import protocol
from experiments.collect_transfer_records import ApprovedCase
from experiments.executions import ExecutionBank
from experiments.history_transfer_experiment import freeze, registered_populations
from experiments.k_shot import CHECKPOINTS, split_candidates
from experiments.parallel import METHOD_WORKERS, initialize_worker, prepare_processes
from experiments.public_ads import evaluate_e01
from experiments.public_ads.collect_continuous_sac import PHYSICAL
from experiments.public_ads.continuous_sac import selected_target
from experiments.public_ads.evaluate_continuous_sac import (
    BUDGET, FROZEN, OUTPUT, PREDICTION, PREDICTORS, _prediction_destination,
)
from experiments.regression.dqn_evaluation import PREDICTION as REVISED_PREDICTION, TARGET
from experiments.resume import ResumableLedger, paid_feedback_hash
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.storage import compact_feedback, read_json, write_json
from experiments.transfer_admission import LIMITS
from experiments.transfer_settings import native_settings
from testing_methods.history_transfer.lifelong_selection import CONFIGURATIONS, select

IN_FAMILY = (
    "target_idm_mobil", "target_fvdm_lane", "target_linear_mpc",
    "target_nonlinear_trajectory", "target_uct", "target_beam", "target_ppo", TARGET,
)
NEW_TARGET = selected_target()
CROSS_ARCHITECTURE = (*evaluate_e01.TARGETS, NEW_TARGET)
TARGETS = (*IN_FAMILY, *CROSS_ARCHITECTURE)
SCOPE = "lifelong_transfer"
SETTINGS = native_settings()


def bank_for(kind, target):
    if target == NEW_TARGET:
        path = PHYSICAL / kind / target
    else:
        root = (
            evaluate_e01.OUTPUT if target in CROSS_ARCHITECTURE else
            REVISED_PREDICTION if target == TARGET else evaluate_e01.ORIGINAL
        )
        path = root / "physical/target" / kind / target
    return ExecutionBank(path)


def selector_contract():
    root = protocol.ROOT
    files = [
        "experiments/lifelong_transfer.py",
        "testing_methods/history_transfer/lifelong_reliability.py",
        "testing_methods/history_transfer/lifelong_selection.py",
    ]
    return {
        **protocol.contract("full"),
        "extensions": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
    }


def _verify_receipt(directory, case, *, derived=False):
    result_path, paid_path = execution._paths(directory, case)
    saved = read_json(result_path)
    result = saved["result"]
    if not derived:
        paid = read_json(paid_path)
        if (
            paid["status"] != "complete" or saved["query_ledger"] != paid["query_ledger"]
            or paid["paid_feedback_hash"] != paid_feedback_hash(saved["query_ledger"])
            or saved["specification"] != paid["specification"]["task"]
        ):
            raise ValueError(f"Paid task result does not match its receipt: {result_path}")
    if len(result["queries"]) != 200 or len(set(result["selected_indices"])) != 200:
        raise ValueError(f"Missing or duplicated admitted queries: {result_path}")
    return saved


def verify_stage_one():
    from experiments.public_ads.continuous_sac import verify_release

    release = verify_release()
    for kind in FORMAL_SCENARIOS:
        bank = bank_for(kind, NEW_TARGET)
        if (
            bank.manifest["contract"]["public_ads"] != release["runtime_contract"]
            or bank.manifest["audit"]["executions"] != 2000
            or bank.manifest["audit"]["maximum_rss_error"] > 1e-12
        ):
            raise ValueError("Stage one admission lacks the complete audited SAC bank")
        for directory, experiment, methods in (
            (PREDICTION, "prediction", PREDICTORS),
            (BUDGET, "budget", protocol.METHODS),
            (FROZEN, "prediction", ("full",)),
            (FROZEN, "budget", ("full",)),
        ):
            for method in methods:
                case = ApprovedCase(experiment, NEW_TARGET, kind, protocol.SEED, method, None)
                saved = _verify_receipt(directory, case, derived=method == "random")
                specification = saved["specification"]
                fingerprint = specification.get(
                    "bank_snapshot", specification.get("task_snapshot", {}).get("bank_snapshot"),
                )
                if fingerprint != bank.snapshot_fingerprint:
                    raise ValueError("Stage one result uses a different physical execution bank")
                if experiment == "prediction" and len(saved["metrics"]) != len(CHECKPOINTS):
                    raise ValueError("Stage one prediction checkpoints are incomplete")


def _audit_forecast(result):
    for query in result["queries"]:
        w = np.array(query["prequery_component_weights"])
        p = np.array(query["prequery_component_probabilities"])
        variance = np.array(query["prequery_component_epistemic_variances"])
        mixture = float(w @ p)
        sigma = float(np.sqrt(w @ (variance + (p - mixture) ** 2)))
        values = (
            (float(w.sum()), 1), (mixture, query["prequery_probability"]),
            (sigma, query["prequery_epistemic_standard_deviation"]),
            (mixture + query["risk_beta"] * sigma, query["prequery_acquisition"]),
        )
        if (
            not np.isfinite([w, p, variance]).all()
            or np.any((w < 0) | (p < 0) | (p > 1) | (variance < 0))
            or any(not np.isclose(actual, expected, atol=1e-12, rtol=0) for actual, expected in values)
            or (result["local_reliability"] and w[-2:].sum() < 0.05 - 1e-12)
        ):
            raise ValueError("A paid forecast fails weighted probability or uncertainty recomputation")


def run_case(directory, case, bank, options):
    execution.declare(directory, case, scope=SCOPE)
    path, checkpoint = execution._paths(directory, case)
    spec = case.specification
    native = case.experiment == "budget"

    def reveal(scene):
        paid = bank(scene)
        return execution._native(paid) if native else paid

    ledger = ResumableLedger(
        bank.candidates, options["budget"], reveal, checkpoint_path=checkpoint,
        specification=spec, paid_execution=lambda i: reveal(bank.candidates[i]),
        joint_feedback=True, domain=bank.domain, allow_registration=False,
    )
    if path.exists():
        result = _verify_receipt(directory, case)
        if result["specification"] != spec:
            raise ValueError("Completed lifelong task differs from its registered inputs")
        execution._validate_output(case, result["result"], ledger.records)
        _audit_forecast(result["result"])
        return result
    limits = spec["execution_contract"]["limits"]
    result = execution.run_isolated(
        select, {"candidates": bank.candidates, **options},
        ledger, wall_limit_s=limits["wall_s"], memory_limit_bytes=limits["child_memory_bytes"],
    )
    execution._validate_output(case, result, ledger.records)
    _audit_forecast(result)
    ledger.finish()
    saved = execution._finalize(case, bank, result, ledger, scope="formal", construction_resources=None)
    saved["new_physical_calls"] = 0
    write_json(path, saved)
    return read_json(path)


def evaluate_task(bank, context, configuration, *, prediction=False, destination=None):
    models, sources = context["models"], context["sources"]
    seed = protocol.SEED
    if prediction:
        split = split_candidates(bank.candidates, seed)
        case = execution.prediction_case(
            bank, split, seed, "full", models, None, sources, settings=SETTINGS,
            checkpoints=CHECKPOINTS, limits=LIMITS,
        )
        initial = split["query_order"][:protocol.INITIAL]
        checkpoints = CHECKPOINTS
        options = {"query_order": split["query_order"]}
        destination = PREDICTION if destination is None else destination
    else:
        case = execution.budget_case(
            bank, seed, "full", models, sources, settings=SETTINGS,
            initial_count=protocol.INITIAL, budget=protocol.BUDGET, limits=protocol.LIMITS,
        )
        initial = case.specification["initial_indices"]
        checkpoints = tuple(case.specification["prediction_snapshot_counts"])
        options = {}
        destination = BUDGET if destination is None else destination
    spec = {
        **case.specification, "lifelong_configuration": configuration,
        "lifelong_selector_contract": selector_contract(),
    }
    case = replace(case, method=configuration, specification=spec)
    options.update(
        pretrained=models[0], sources=sources, online_settings=SETTINGS.online,
        initial_indices=initial, budget=protocol.BUDGET, snapshot_counts=checkpoints, device="cpu",
    )
    options["configuration"] = configuration
    return run_case(destination, case, bank, options)


def _job(job):
    kind, target, configuration, prediction, context, expected = job
    bank = bank_for(kind, target)
    if bank.candidates != expected or bank.manifest["environment_seed"] != 81001:
        raise ValueError("Lifelong matrix requires the same admitted target coordinates")
    evaluate_task(bank, context, configuration, prediction=prediction)
    return kind, target, configuration, prediction


def evaluate(workers):
    verify_stage_one()
    freeze(OUTPUT / "main_protocol.json", {
        "in_family": list(IN_FAMILY), "cross_architecture": list(CROSS_ARCHITECTURE),
        "scenes": list(FORMAL_SCENARIOS), "budget": protocol.BUDGET,
        "initial_queries": protocol.INITIAL, "seed": protocol.SEED,
        "configurations": {name: list(flags) for name, flags in CONFIGURATIONS.items()},
        "prediction_configuration": "lifelong_full", "prediction_checkpoints": list(CHECKPOINTS),
        "budget_results": 144, "prediction_results": 48, "new_physical_calls": 0,
        "bootstrap": {"unit": "ADS", "replicates": 10000, "seed": 1301},
        "selector_contract": selector_contract(),
    })
    expected = registered_populations(evaluate_e01.ORIGINAL)
    prepare_processes()
    initialize_worker()
    for kind in FORMAL_SCENARIOS:
        with _prediction_destination(PREDICTION):
            context = evaluate_e01.prepare_history(kind)
        jobs = [
            (kind, target, configuration, False, context, expected["target", kind])
            for target in TARGETS for configuration in CONFIGURATIONS
        ]
        jobs.extend(
            (kind, target, "lifelong_full", True, context, expected["target", kind])
            for target in TARGETS
        )
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker,
        ) as pool:
            for result in pool.map(_job, jobs):
                print("complete", *result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=METHOD_WORKERS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    evaluate(args.workers)


if __name__ == "__main__":
    main()
