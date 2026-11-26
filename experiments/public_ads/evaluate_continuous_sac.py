"""Complete the seventh-method baseline matrix for the admitted continuous ADS."""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

from experiments import transfer_execution as execution
from experiments.budget import protocol
from experiments.budget.audit import derive_random
from experiments.collect_transfer_records import ApprovedCase
from experiments.executions import ExecutionBank
from experiments.history_transfer_experiment import freeze, registered_populations
from experiments.k_shot import CHECKPOINTS, split_candidates
from experiments.parallel import TRAINING_WORKERS, initialize_worker, prepare_processes
from experiments.protocol import ROOT
from experiments.public_ads import evaluate_e01, extend
from experiments.public_ads.collect_continuous_sac import PHYSICAL
from experiments.public_ads.continuous_sac import verify_release
from experiments.record_audit import BankTruth
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.transfer_admission import LIMITS
from experiments.transfer_settings import native_settings

OUTPUT = ROOT / "results/lifelong_transfer"
PREDICTION = OUTPUT / "k_shot_adaptation"
BUDGET = OUTPUT / "budget_discovery"
FROZEN = OUTPUT / "weight_adaptation"
PREDICTORS = ("full", "target_only", "pooled_history")


@contextmanager
def _prediction_destination(root):
    previous = evaluate_e01.OUTPUT
    evaluate_e01.OUTPUT = root
    try:
        yield
    finally:
        evaluate_e01.OUTPUT = previous


def _prediction(bank, method, context, *, frozen=False):
    settings = native_settings()
    models = context["models"] if method != "target_only" else (None, None)
    pooled = context["pooled"] if method == "pooled_history" else None
    sources = context["sources"] if method == "full" else None
    case = execution.prediction_case(
        bank, split_candidates(bank.candidates, protocol.SEED), protocol.SEED,
        method, models, pooled, sources, settings=settings,
        checkpoints=CHECKPOINTS, limits=dict(LIMITS),
    )
    if frozen:
        case = ApprovedCase(
            case.experiment, case.sut_id, case.scenario, case.seed, case.method,
            {**case.specification, "update_source_weights": False},
        )
    destination = FROZEN if frozen else PREDICTION
    execution.declare(destination, case, scope="public_extension")
    with _prediction_destination(destination):
        return evaluate_e01._run_prediction_case(case, bank, models, pooled, sources, settings)


@contextmanager
def _budget_context(context):
    previous = extend.OUTPUT, extend.BUDGET, extend._CONTEXT
    extend.OUTPUT, extend.BUDGET, extend._CONTEXT = PREDICTION, BUDGET, context
    try:
        yield
    finally:
        extend.OUTPUT, extend.BUDGET, extend._CONTEXT = previous


def _evaluate_scene(sut, kind, context, *, pooled):
    bank = ExecutionBank(PHYSICAL / kind / sut)
    methods = ("pooled_history",) if pooled else ("full", "target_only")
    for method in methods:
        _prediction(bank, method, context)
        print("complete", "prediction", sut, kind, method, flush=True)
    if not pooled:
        _prediction(bank, "full", context, frozen=True)
    budget_methods = (
        ("pooled_history",) if pooled else tuple(m for m in protocol.METHODS if m != "pooled_history")
    )
    with _budget_context(context):
        for method in budget_methods:
            if method == "random":
                source = PREDICTION / "k_shot" / sut / kind / str(protocol.SEED)
                destination = BUDGET / "budget" / sut / kind / str(protocol.SEED) / "random"
                derive_random(
                    bank, BankTruth(bank), destination=destination / "result.json.gz",
                    source=(source / "target_only.json.gz", source / "target_only_paid_feedback.json.gz"),
                )
            else:
                case, public = extend.budget_inputs(bank, method)
                extend.execute_budget(bank, case, public, BUDGET)
            print("complete", "budget", sut, kind, method, flush=True)
        if not pooled:
            case, public = extend.budget_inputs(bank, "full", frozen=True)
            extend.execute_budget(bank, case, public, FROZEN)
            print("complete", "frozen controls", sut, kind, flush=True)


def evaluate(kinds):
    acceptance = verify_release()
    sut = acceptance["sut_id"]
    populations = registered_populations(evaluate_e01.ORIGINAL)
    settings = native_settings()
    freeze(OUTPUT / "continuous_sac_protocol.json", {
        "target": sut, "scenarios": list(FORMAL_SCENARIOS),
        "prediction_methods": list(PREDICTORS), "budget_methods": list(protocol.METHODS),
        "seed": protocol.SEED, "budget": protocol.BUDGET, "initial_count": protocol.INITIAL,
        "prediction_checkpoints": list(CHECKPOINTS), "settings": settings.to_dict(),
        "expected_prediction_trajectories": 12, "expected_budget_trajectories": 28,
        "frozen_prediction_pairs": 4, "frozen_budget_pairs": 4, "new_physical_executions": 0,
        "runtime_contract": acceptance["runtime_contract"],
    })
    prepare_processes()
    initialize_worker()
    contexts = {}
    for kind in kinds:
        bank = ExecutionBank(PHYSICAL / kind / sut)
        if (
            bank.manifest["contract"]["public_ads"] != acceptance["runtime_contract"]
            or bank.manifest["environment_seed"] != 81001
            or bank.candidates != populations["target", kind]
            or bank.manifest["audit"]["executions"] != 2000
        ):
            raise ValueError("Continuous SAC bank differs from its frozen inputs")
        with _prediction_destination(PREDICTION):
            contexts[kind] = evaluate_e01.prepare_history(kind)
    worker = dict(mp_context=mp.get_context("spawn"), initializer=initialize_worker)
    with ProcessPoolExecutor(max_workers=TRAINING_WORKERS, **worker) as cpu, (
        ProcessPoolExecutor(max_workers=1, **worker)
    ) as gpu:
        jobs = [
            pool.submit(_evaluate_scene, sut, kind, context, pooled=pooled)
            for pooled, pool in ((False, cpu), (True, gpu)) for kind, context in contexts.items()
        ]
        for job in as_completed(jobs):
            job.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", choices=FORMAL_SCENARIOS, default=FORMAL_SCENARIOS)
    args = parser.parse_args()
    evaluate(args.kinds)


if __name__ == "__main__":
    main()
