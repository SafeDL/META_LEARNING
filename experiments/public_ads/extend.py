"""Execute the preregistered public ADS E02/E03 extension without new simulation."""

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace

from experiments import transfer_execution as execution
from experiments.budget import protocol
from experiments.budget.execution import make_case
from experiments.budget.selectors import run as budget_selector
from experiments.collect_transfer_records import ApprovedCase
from experiments.executions import ExecutionBank
from experiments.history_transfer_experiment import freeze
from experiments.parallel import initialize_worker, prepare_processes
from experiments.public_ads.evaluate_e01 import OUTPUT, TARGETS, _run_prediction_case, prepare_history as fresh_history
from experiments.resume import ResumableLedger
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.storage import compact_feedback, read_json, write_json
from experiments.transfer_settings import native_settings

BUDGET = OUTPUT / "budget_discovery"
FROZEN = OUTPUT / "weight_adaptation"
BASELINES = protocol.OUTPUT
_CONTEXT = None


def prepare_history(kind):
    import json
    import numpy as np
    from experiments.weight_ablation.run import digest

    context = fresh_history(kind)
    expected = read_json(OUTPUT / "k_shot" / TARGETS[0] / kind / "11/full.json.gz")["specification"]["source_snapshot"]
    fresh = context["sources"]["specification"]
    if ({k: v for k, v in expected.items() if k != "source_contract"}
            != {k: v for k, v in fresh.items() if k != "source_contract"}
            or expected["source_contract"]["packages"] != fresh["source_contract"]["packages"]):
        raise ValueError("Historical weights, data, candidates or numerical packages changed")
    path = OUTPUT / "sources" / kind / f"{digest(expected)}.npz"
    with np.load(path, allow_pickle=False) as saved:
        errors = {k: float(np.max(np.abs(saved[k] - context["sources"][k])))
                  for k in ("means", "blocks", "weights")}
        if (json.loads(str(saved["specification"])) != expected
                or saved["names"].tolist() != context["sources"]["names"]
                or any(not np.allclose(saved[k], context["sources"][k], atol=1e-10, rtol=0)
                       for k in ("means", "blocks", "weights"))):
            raise ValueError("Rebuilt source posterior differs numerically from sealed public E01")
        context["sources"].update({k: saved[k].copy() for k in ("means", "blocks", "weights")})
    context["sources"]["specification"] = expected
    write_json(BUDGET / "input_admission" / f"{kind}.json", {
        "sealed_source": str(path.relative_to(protocol.ROOT)), "sealed_sha256": protocol.sha(path),
        "rebuilt_maximum_absolute_errors": errors, "tolerance": 1e-10,
        "source_contract_recorded": expected["source_contract"], "source_contract_current": fresh["source_contract"],
        "used_for_selection": "sealed numerical source posterior", "new_physical_executions": 0,
    })
    return context


@contextmanager
def public_paths():
    """Bind legacy path-based readers within this process; never alter sealed inputs."""
    old = protocol.INPUT, protocol.OUTPUT
    protocol.INPUT, protocol.OUTPUT = OUTPUT, BUDGET
    try:
        yield
    finally:
        protocol.INPUT, protocol.OUTPUT = old


def bank_for(kind, target):
    from controllers.highway.public_agents import CACHE, CHECKPOINTS, WEIGHTS
    from experiments.qualification import verify_course_evidence

    bank = ExecutionBank(OUTPUT / "physical/target" / kind / target)
    acceptance_path = OUTPUT / "qualification" / target / "acceptance.json"
    acceptance = read_json(acceptance_path)
    contract = acceptance["runtime_contract"]
    # Replay verifies the executing identity recorded at collection, without
    # importing or executing an ADS that the statistical selector never uses.
    recorded = bank.manifest["contract"]["public_ads"]
    if (target not in TARGETS or len(bank.candidates) != 2000
            or {k: v for k, v in recorded.items() if k != "source_files"}
            != {k: v for k, v in contract.items() if k != "source_files"}
            or not acceptance["qualified"]
            or not acceptance["independent_policy_rerun_identical"]):
        raise ValueError("Public bank no longer matches its admitted frozen ADS")
    evidence = verify_course_evidence(OUTPUT / "qualification" / target / "qualification")
    if evidence["courses"] != acceptance["courses"]:
        raise ValueError("Frozen public driving qualification changed")
    for name, checksum in contract["source_files"].items():
        if protocol.sha(protocol.ROOT / name) != checksum:
            raise ValueError(f"Frozen ADS source changed: {name}")
    checkpoint = CACHE / WEIGHTS[target][0] if target in WEIGHTS else CHECKPOINTS / target / "policy.pt"
    if protocol.sha(checkpoint) != contract["checkpoint_sha256"]:
        raise ValueError("Frozen ADS weights changed")
    for name, checksum in contract["upstream_assets"].items():
        if name == "training.json":
            asset = CHECKPOINTS / target / name
        elif name.startswith("modules."):
            asset = CACHE / "highway_dt/modules" / (name.split(".")[-1] + ".py")
        elif name.startswith("rl_agents."):
            asset = next((CACHE / "upstream").glob("rl-agents-*/rl_agents/agents/common/models.py"))
        else:
            asset = CACHE / "online_mc" / name
        if protocol.sha(asset) != checksum:
            raise ValueError(f"Frozen upstream asset changed: {name}")
    saved = read_json(OUTPUT / "k_shot" / target / kind / "11/full.json.gz")
    if (saved["specification"]["bank_snapshot"] != bank.snapshot_fingerprint
            or saved["specification"]["bank_contract"] != bank.manifest["contract"]
            or saved["specification"]["candidates"] != execution.canonical([asdict(s) for s in bank.candidates])
            or saved["specification"]["source_snapshot"] != _CONTEXT["sources"]["specification"]):
        raise ValueError("Public bank or historical posteriors differ from sealed E01")
    return bank


def budget_inputs(bank, method, *, frozen=False):
    # Baseline construction evidence belongs to the original historical study.
    if method in ("scenario_fuzz", "paired_system_active_evaluation"):
        case, public = baseline_inputs(bank, method)
    else:
        with public_paths():
            case, public = make_case(bank, method, _CONTEXT)
    if frozen:
        if method != "full":
            raise ValueError("Only the complete mixture has a frozen-weight control")
        case = replace(case, specification={**case.specification, "update_source_weights": False})
        public["update_source_weights"] = False
    return case, public


def baseline_inputs(bank, method):
    import numpy as np
    from experiments.baselines import BASELINES as definitions
    from experiments.budget.prepare import construction_specification
    from experiments.history_transfer_experiment import scenario_history
    from testing_methods.baselines.scenario_fuzz import model as fuzz

    kind = bank.candidates[0].kind
    directory = BASELINES / "models" / kind
    completion = read_json(directory / "training_completion.json")
    recorded = completion["specification"]
    current = construction_specification(scenario_history(protocol.INPUT, kind))
    if (completion["status"] != "complete"
            or {k: v for k, v in recorded.items() if k != "source_sha256"}
            != {k: v for k, v in current.items() if k != "source_sha256"}
            or {k: v for k, v in recorded["source_sha256"].items() if k.startswith("testing_methods/")}
            != {k: v for k, v in current["source_sha256"].items() if k.startswith("testing_methods/")}):
        raise ValueError("Historical baseline data, settings or numerical implementation changed")
    with public_paths():
        template, _ = make_case(bank, "target_only", _CONTEXT)
    spec = {k: v for k, v in template.specification.items() if k not in (
        "online_settings", "update_source_weights", "models", "source_snapshot", "raw_kernel", "device")}
    spec["task_snapshot"] = {**spec["task_snapshot"], "method": method, "method_contract": protocol.contract(method)}
    spec.update(configuration=asdict(definitions[method].DEFAULT), construction=completion)
    public = dict(method=method, candidates=bank.candidates, initial_indices=spec["initial_indices"],
                  budget=200, device="cpu", seed=11, config=definitions[method].DEFAULT)
    if method == "scenario_fuzz":
        path = directory / "scenario_fuzz/classifier.pt"
        if protocol.sha(path) != completion["weights_sha256"]:
            raise ValueError("Historical graph classifier weights changed")
        public["classifier"] = fuzz.load(path)[0]
    else:
        path = directory / "paired_proxy.json"
        proxy = read_json(path)
        if (protocol.sha(path) != completion["paired_proxy_sha256"]
                or proxy["candidate_ids"] != [s.scenario_id for s in bank.candidates]):
            raise ValueError("Historical paired proxy changed")
        public["proxy"] = np.asarray(proxy["values"])
    return replace(template, method=method, specification=spec), public


def execute_budget(bank, case, public, destination):
    # Public ADS admission is separate from the original eight-target catalog.
    execution.declare(destination, case, scope="formal")
    path, checkpoint = execution._paths(destination, case)
    spec = case.specification
    ledger = ResumableLedger(
        bank.candidates, 200, lambda s: execution._native(bank(s)),
        checkpoint_path=checkpoint, specification=spec,
        paid_execution=lambda i: execution._native(bank(bank.candidates[i])),
        joint_feedback=True, domain=bank.domain, allow_registration=False,
    )
    if path.exists():
        saved = read_json(path)
        if (saved["specification"] != spec or read_json(checkpoint)["status"] != "complete"
                or saved["query_ledger"] != execution.canonical(compact_feedback(ledger.records))):
            raise ValueError("Completed public budget task changed")
        execution._validate_output(case, saved["result"], ledger.records)
        return saved
    started = time.perf_counter()
    try:
        result = execution.run_isolated(budget_selector, public, ledger,
            wall_limit_s=protocol.LIMITS["wall_s"],
            memory_limit_bytes=protocol.LIMITS["child_memory_bytes"])
        execution._validate_output(case, result, ledger.records)
        ledger.finish()
        saved = execution._finalize(case, bank, result, ledger, scope="formal", construction_resources=None)
        saved["wall_s"] = time.perf_counter() - started
        write_json(path, saved)
        return saved
    except Exception as error:
        write_json(path.parent / f"failed_attempt_{time.time_ns()}.json", {
            "specification": spec, "error": repr(error), "query_ledger": compact_feedback(ledger.records)})
        raise


def initialize(context):
    global _CONTEXT
    initialize_worker()
    _CONTEXT = context


def run_case(kind, target, stage, method):
    bank = bank_for(kind, target)
    print("running", stage, target, kind, method, flush=True)
    if stage == "budget":
        case, public = budget_inputs(bank, method)
        execute_budget(bank, case, public, BUDGET)
    elif stage == "frozen-budget":
        case, public = budget_inputs(bank, "full", frozen=True)
        execute_budget(bank, case, public, FROZEN)
    else:
        original = read_json(OUTPUT / "k_shot" / target / kind / "11/full.json.gz")
        spec = {**original["specification"], "update_source_weights": False,
                "execution_contract": execution.execution_contract(protocol.LIMITS)}
        case = ApprovedCase("prediction", target, kind, 11, "full", spec)
        execution.declare(FROZEN, case, scope="public_extension")
        # This executor accepts an explicit destination through a local binding.
        from experiments.public_ads import evaluate_e01
        old = evaluate_e01.OUTPUT
        evaluate_e01.OUTPUT = FROZEN
        try:
            _run_prediction_case(case, bank, _CONTEXT["models"], None,
                                 _CONTEXT["sources"], native_settings())
        finally:
            evaluate_e01.OUTPUT = old
    print("complete", stage, target, kind, method, flush=True)
    return target, kind, stage, method


def register():
    freeze(BUDGET / "protocol.json", {
        "scope": "public_e02_extension", "targets": list(TARGETS),
        "scenarios": list(FORMAL_SCENARIOS), "methods": list(protocol.METHODS),
        "budget": 200, "initial_count": 20, "seed": 11,
        "settings": native_settings().to_dict(), "limits": protocol.LIMITS,
        "new_physical_executions": 0, "expected_results": 84,
        "historical_models": str(BASELINES.relative_to(protocol.ROOT)),
        "statistics": "separate from the original eight ADS; equal scenes per ADS",
        "method_contracts": {m: protocol.contract(m) for m in protocol.METHODS},
        "driver_sha256": protocol.sha(__file_path()),
    })
    freeze(FROZEN / "protocol.json", {
        "scope": "public_e03_extension", "targets": list(TARGETS),
        "scenarios": list(FORMAL_SCENARIOS), "seed": 11,
        "settings": native_settings().to_dict(), "limits": protocol.LIMITS,
        "checkpoints": [0, 5, 10, 20, 50, 100, 200], "few_shot": [5, 10, 20, 50],
        "update_source_weights": False, "target_feedback_updates": True,
        "new_physical_executions": 0, "prediction_pairs": 12, "budget_pairs": 12,
        "metrics": ["heldout_nll", "heldout_brier", "sequential_nll", "sequential_brier"],
        "driver_sha256": protocol.sha(__file_path()),
    })


def __file_path():
    from pathlib import Path
    return Path(__file__)


def run(stage, targets=TARGETS, kinds=FORMAL_SCENARIOS, methods=None, workers=5):
    register()
    prepare_processes()
    initialize_worker()
    for kind in kinds:
        context = prepare_history(kind)
        groups = [([m for m in (methods or protocol.METHODS) if m not in ("random", "pooled_history")], workers),
                  ([m for m in (methods or protocol.METHODS) if m == "pooled_history"], 1)]
        if stage != "budget":
            groups = [(["full"], workers)]
        for selected, count in groups:
            if not selected:
                continue
            with ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context("spawn"),
                                     initializer=initialize, initargs=(context,)) as pool:
                futures = [pool.submit(run_case, kind, target, stage, method)
                           for target in targets for method in selected]
                errors = []
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(repr(error))
                        print("failed", repr(error), flush=True)
                if errors:
                    raise RuntimeError(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("budget", "frozen-prediction", "frozen-budget", "register"))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument("--kinds", nargs="+", choices=FORMAL_SCENARIOS, default=FORMAL_SCENARIOS)
    parser.add_argument("--methods", nargs="+", choices=protocol.METHODS)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "register":
        register()
    else:
        run(args.stage, args.targets, args.kinds, args.methods, args.workers)


if __name__ == "__main__":
    main()
