"""Run the registered E03 weight ablation against the sealed E01/E02 controls."""

import argparse
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from controllers.highway.catalog import FORMAL_TARGET_SUTS
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import validate_model

from experiments.budget import protocol
from experiments import transfer_execution as execution
from experiments.budget.selectors import run as select_budget
from experiments.collect_transfer_records import ApprovedCase
from experiments.executions import ExecutionBank
from experiments.history import model_version
from experiments.history_transfer_experiment import freeze, model_directory
from experiments.initialization import initial_indices
from experiments.k_shot import split_candidates
from experiments.method_contracts import method_contract
from experiments.parallel import METHOD_WORKERS, initialize_worker, prepare_processes
from experiments.resume import ResumableLedger
from experiments.storage import read_json, write_json
from experiments.transfer_batches import CHECKPOINTS
from experiments.transfer_settings import native_settings

ROOT = protocol.ROOT
INPUT = protocol.INPUT
OUTPUT = ROOT / "results/weight_adaptation"
SCENES = protocol.RUN_SCENARIOS
TARGETS = tuple(s.sut_id for s in FORMAL_TARGET_SUTS)
MODES = ("prediction", "budget")
SEED = protocol.SEED
LIMITS = protocol.LIMITS
TOLERANCE = 1e-10
PREDICTION_REPORT = INPUT / "reports/main_results/prediction"
BUDGET_REPORT = protocol.OUTPUT / "reports"
ADMISSION = INPUT / "reports/method_acceptance.json"
SETTINGS = native_settings()
sha = protocol.sha
_CONTEXT = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def declaration():
    return {
        "experiment": "weight_adaptation",
        "scenes": list(SCENES), "targets": list(TARGETS), "modes": list(MODES),
        "seed": SEED, "budget": protocol.BUDGET, "initial_count": protocol.INITIAL,
        "evaluation_count": 1800, "checkpoints": list(CHECKPOINTS),
        "few_shot_summary": [5, 10, 20, 50],
        "metrics": ["nll", "recall_100", "unique_collisions", "discovery_auc"],
        "settings": SETTINGS.to_dict(), "limits": LIMITS,
        "update_source_weights": False, "device": "cpu", "threads_per_selector": 1,
        "workers": METHOD_WORKERS, "implicit_evidence": False,
        "bootstrap_unit": "ADS", "bootstrap_replicates": 10000, "bootstrap_seed": 1301,
        "compatibility_tolerance": TOLERANCE,
        "new_trajectories": 64, "new_physical_executions": 0,
        "method_contract": method_contract("full"),
        "budget_selector_sha256": sha(ROOT / "experiments/budget/selectors.py"),
        "driver_sha256": sha(Path(__file__)),
    }


def input_receipts():
    prediction = read_json(PREDICTION_REPORT / "completion.json")
    budget = read_json(BUDGET_REPORT / "completion.json")
    if (
        (prediction["status"], prediction["cases"], prediction["metric_rows"]) != ("complete", 96, 672)
        or (budget["status"], budget["cases"], budget["curve_rows"]) != ("complete", 224, 44800)
        or prediction["protocol"]["admission_sha256"] != sha(ADMISSION)
        or read_json(protocol.OUTPUT / "input_admission.json")["acceptance_sha256"] != sha(ADMISSION)
        or read_json(protocol.OUTPUT / "protocol.json") != protocol.declaration()
    ):
        raise ValueError("The complete sealed E01/E02 inputs are required")
    for directory, receipt in ((PREDICTION_REPORT, prediction), (BUDGET_REPORT, budget)):
        for name, checksum in receipt["files"].items():
            if sha(directory / name) != checksum:
                raise ValueError(f"Sealed input table changed: {directory / name}")
    registered = read_json(INPUT / "reports/method_settings.json")
    old = registered["methods"]["full"]
    current = method_contract("full")
    if (
        registered["settings"] != SETTINGS.to_dict() or old["packages"] != current["packages"]
        or old["source_files"].keys() != current["source_files"].keys()
    ):
        raise ValueError("Fixed settings, package versions or method dependencies changed")
    changes = {
        path: {"sealed": checksum, "current": current["source_files"][path]}
        for path, checksum in old["source_files"].items()
        if checksum != current["source_files"][path]
    }
    freeze(OUTPUT / "input_admission.json", {
        "files": {relative(path): sha(path) for path in (
            ADMISSION, INPUT / "reports/method_settings.json",
            PREDICTION_REPORT / "completion.json", BUDGET_REPORT / "completion.json",
            protocol.OUTPUT / "protocol.json", protocol.OUTPUT / "input_admission.json",
        )},
        "method_source_differences": changes,
        "compatibility_gate": "Per-task replay of sealed paid prefixes, checkpoints and all decisions",
    })
    return read_json(ADMISSION)["proof_sha256"], {
        tuple(row["key"]): row["specification_sha256"]
        for row in prediction["protocol"]["approved_cases"]
    }


def sealed_specification(mode, bank, approved):
    kind, sut = bank.candidates[0].kind, bank.manifest["sut_id"]
    case = ApprovedCase(mode, sut, kind, SEED, "full", None)
    source = INPUT if mode == "prediction" else protocol.OUTPUT
    path = execution.approval_path(source, case)
    approval = read_json(path)
    spec = approval["specification"]
    bank_hash = spec["bank_snapshot"] if mode == "prediction" else spec["task_snapshot"]["bank_snapshot"]
    if (
        approval["key"] != list(case.key) or approval["scope"] != "formal"
        or bank_hash != bank.snapshot_fingerprint
        or spec["bank_contract"] != bank.manifest["contract"]
        or spec["candidates"] != execution.canonical([asdict(s) for s in bank.candidates])
        or spec["online_settings"] != SETTINGS.online.to_dict()
        or spec["update_source_weights"] is not True
        or spec["execution_contract"]["limits"] != LIMITS or spec["device"] != "cpu"
        or spec["evaluate_complete_bank"] is not True
    ):
        raise ValueError(f"Sealed approval differs from its bank or common settings: {path}")
    if mode == "prediction":
        if digest(spec) != approved[case.key] or spec["split"] != split_candidates(bank.candidates, SEED) or spec["checkpoints"] != list(CHECKPOINTS):
            raise ValueError("E01 approval or shared evaluation split changed")
    else:
        current = protocol.contract("full")
        recorded = spec["task_snapshot"]["method_contract"]
        if (
            recorded["method"] != "full" or recorded["packages"] != current["packages"]
            or spec["seed"] != SEED or spec["budget"] != protocol.BUDGET
            or spec["initial_count"] != protocol.INITIAL
            or spec["initial_indices"] != initial_indices(len(bank.candidates), protocol.INITIAL, SEED)
            or spec["new_physical_executions"] != 0
            or any(recorded["source_files"][name] != current["source_files"][name]
                   for name in ("experiments/budget/protocol.py", "experiments/budget/execution.py", "experiments/budget/selectors.py"))
        ):
            raise ValueError("Sealed E02 budget or selector dispatch changed")
    return ApprovedCase(*case.key, spec)


def load_context(kind, proof, first):
    """Read the exact admitted historical weights and the frozen source posterior."""
    models = []
    model_hashes = {}
    for name, part in (("target_kernel.pt", "target"), ("historical_encoder.pt", "historical_encoder")):
        path = model_directory(INPUT, kind) / name
        model_hashes[relative(path)] = sha(path)
        if model_hashes[relative(path)] != proof[relative(path)]:
            raise ValueError(f"Historical checkpoint changed: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        spec = saved["specification"]
        if (
            saved["format"] != "sequential-joint-collision-rss" or saved["part"] != part
            or saved["step"] != 1500 or spec["config"]["steps"] != 1500
            or spec["scenario"] != kind or spec["training_device"] != "cpu"
        ):
            raise ValueError(f"Historical checkpoint identity differs: {path}")
        model = SharedPrivateKernel(saved["dimension"]).double().eval()
        model.load_state_dict(saved["state_dict"])
        model.construction_specification = spec
        validate_model(model)
        models.append(model)
    expected = first.specification["source_snapshot"]
    key = digest(expected)
    cache = INPUT / "sources" / kind / f"{key}.npz"
    with np.load(cache, allow_pickle=False) as saved:
        sources = {name: saved[name].copy() for name in ("means", "blocks", "weights")}
        sources.update(names=saved["names"].tolist(), specification=json.loads(str(saved["specification"])))
    if (
        sources["specification"] != expected
        or expected["model_version"] != model_version(models[1])
        or expected["model_construction"] != models[1].construction_specification
        or expected["candidates"] != first.specification["candidates"]
        or len(sources["names"]) != 8 or set(sources["names"]) != set(expected["archives"])
        or sources["means"].shape != (8, 2000, 2)
        or sources["blocks"].shape != (8, 2000, 2, 2)
        or sources["weights"].shape != (8,) or not np.all(sources["weights"] == 1 / 8)
        or not all(np.isfinite(sources[name]).all() for name in ("means", "blocks"))
    ):
        raise ValueError("Source posterior differs from the sealed models, history or candidate library")
    return {"models": models, "sources": sources, "input_hashes": {**model_hashes, relative(cache): sha(cache)}}


def selector_inputs(case, candidates, context, *, dynamic):
    spec = case.specification
    public = {
        "candidates": candidates, "pretrained": context["models"][0], "sources": context["sources"],
        "online_settings": SETTINGS.online, "device": "cpu", "feedback_mode": "joint",
        "update_source_weights": dynamic, "batch_size": 128, "implicit_evidence": False,
    }
    if case.experiment == "prediction":
        order = spec["split"]["query_order"]
        return select, dict(public, query_order=order, initial_indices=order[:1],
                            budget=len(order), snapshot_counts=spec["checkpoints"])
    return select_budget, dict(public, method="full", initial_indices=spec["initial_indices"],
                               budget=spec["budget"], snapshot_counts=(0, 20, 50, 100, 200))


def frozen_case(original, result_path, paid_path, context):
    spec = original.specification
    if spec["source_snapshot"] != context["sources"]["specification"]:
        raise ValueError("The two controls use different source posteriors")
    versions = [model_version(model) for model in context["models"]]
    if (
        (original.experiment == "prediction" and (
            spec["model"] != versions[0] or spec["model_construction"] != context["models"][0].construction_specification
        ))
        or (original.experiment == "budget" and spec["models"] != versions)
    ):
        raise ValueError("Sealed online model differs from the admitted weights")
    now = declaration()
    contract = now["method_contract"]
    if original.experiment == "budget":
        contract = {**contract, "source_files": {**contract["source_files"],
                    "experiments/budget/selectors.py": now["budget_selector_sha256"]}}
    frozen = {
        **spec, "update_source_weights": False, "execution_contract": execution.execution_contract(LIMITS),
        "weight_ablation": "all mixture weights fixed at the joint prior",
        "input_sha256": context["input_hashes"],
        "control_sha256": {relative(p): sha(p) for p in (result_path, paid_path)},
        "protocol_sha256": sha(OUTPUT / "protocol.json"),
    }
    if original.experiment == "prediction":
        frozen["method_contract"] = contract
    else:
        frozen["task_snapshot"] = {**spec["task_snapshot"], "method_contract": contract}
    return ApprovedCase(*original.key, frozen)


def compare_replay(actual, expected):
    errors = {}

    def numeric(name, new, saved):
        new, saved = np.asarray(new, dtype=float), np.asarray(saved, dtype=float)
        if new.shape != saved.shape or not np.isfinite(new).all() or not np.isfinite(saved).all():
            raise ValueError(f"Replay {name} is incomplete or nonfinite")
        error = float(np.max(np.abs(new - saved)))
        if error > TOLERANCE:
            raise ValueError(f"Replay {name} differs by {error:.6g}")
        errors[name] = max(errors.get(name, 0), error)

    for name in ("selected_indices", "component_names", "termination", "online_settings",
                 "target_feedback_updates", "source_weight_updates", "online_feature_precision"):
        if actual[name] != expected[name]:
            raise ValueError(f"Replay changed {name}")
    for new, old in zip(actual["queries"], expected["queries"], strict=True):
        for name in ("index", "conditioning_queries", "reason"):
            if new[name] != old[name]:
                raise ValueError(f"Replay changed query {name}")
        for name in ("prequery_probability", "prequery_component_probabilities",
                     "prequery_component_weights", "component_weights_after_query", "prequery_log_likelihood"):
            numeric(name, new[name], old[name])
    for new, old in zip(actual["prediction_snapshots"], expected["prediction_snapshots"], strict=True):
        for name in ("paid_queries", "observed_indices"):
            if new[name] != old[name]:
                raise ValueError(f"Replay changed checkpoint {name}")
        for name in ("probabilities", "component_weights"):
            numeric(name, new[name], old[name])
        for name in ("target_parameters", "physical_parameters"):
            if new[name].keys() != old[name].keys():
                raise ValueError(f"Replay changed fitted {name}")
            numeric(name, list(new[name].values()), [old[name][p] for p in new[name]])
    return errors


def run_target(kind, sut, approved, modes):
    bank = ExecutionBank(INPUT / "physical/target" / kind / sut)
    status = []
    for mode in modes:
        stage = "admission"
        try:
            original = sealed_specification(mode, bank, approved)
            source = INPUT if mode == "prediction" else protocol.OUTPUT
            result_path, paid_path = execution._paths(source, original)
            case = frozen_case(original, result_path, paid_path, _CONTEXT)
            execution.declare(OUTPUT, case, scope="formal")
            saved = read_json(result_path)
            if saved["specification"] != original.specification:
                raise ValueError("Sealed result disagrees with its approved specification")

            def paid(index):
                return bank(bank.candidates[index])

            stage = "compatibility"
            bridge_path = OUTPUT / "compatibility" / kind / sut / f"{mode}.json"
            identity = {"control_sha256": case.specification["control_sha256"],
                        "case_sha256": digest(case.specification)}
            if bridge_path.exists():
                if read_json(bridge_path)["identity"] != identity:
                    raise ValueError("Compatibility receipt was produced by different inputs")
            else:
                replay = ResumableLedger(
                    bank.candidates, protocol.BUDGET, None, checkpoint_path=paid_path,
                    specification=original.specification, paid_execution=paid,
                    domain=bank.domain, allow_registration=False, joint_feedback=True,
                )
                checkpoint = read_json(paid_path)
                if (
                    checkpoint["status"] != "complete" or replay.saved_count != protocol.BUDGET
                    or checkpoint["query_ledger"] != saved["query_ledger"]
                ):
                    raise ValueError("Compatibility replay requires a complete sealed paid ledger")
                selector, public = selector_inputs(original, bank.candidates, _CONTEXT, dynamic=True)
                actual = execution.run_isolated(selector, public, replay,
                    wall_limit_s=LIMITS["wall_s"], memory_limit_bytes=LIMITS["child_memory_bytes"])
                maximum_errors = compare_replay(actual, saved["result"])
                freeze(bridge_path, {
                    "identity": identity, "paid_queries_replayed": replay.replayed_count,
                    "new_paid_queries": 0, "checkpoints": len(actual["prediction_snapshots"]),
                    "selected_order_equal": True, "maximum_errors": maximum_errors,
                })
            stage = "frozen_execution"
            selector, public = selector_inputs(case, bank.candidates, _CONTEXT, dynamic=False)
            execution._execute(OUTPUT, case, bank, selector, public, paid_execution=paid, scope="formal")
            row = {"scene": kind, "target": sut, "mode": mode, "status": "complete"}
        except Exception as error:
            row = {"scene": kind, "target": sut, "mode": mode, "status": "failed",
                   "stage": stage, "error": repr(error)}
        write_json(OUTPUT / "status" / kind / sut / f"{mode}.json", row)
        print(row, flush=True)
        status.append(row)
    return status


def _initialize(context):
    global _CONTEXT
    initialize_worker()
    _CONTEXT = context


def run(*, scenes=SCENES, targets=TARGETS, modes=MODES):
    initialize_worker()
    freeze(OUTPUT / "protocol.json", declaration())
    proof, approved = input_receipts()
    prepare_processes()
    rows = []
    for kind in scenes:
        print({"scene": kind, "stage": "sealed_context"}, flush=True)
        bank = ExecutionBank(INPUT / "physical/target" / kind / targets[0])
        context = load_context(kind, proof, sealed_specification("prediction", bank, approved))
        with ProcessPoolExecutor(max_workers=min(METHOD_WORKERS, len(targets)),
                                 mp_context=mp.get_context("spawn"),
                                 initializer=_initialize, initargs=(context,)) as pool:
            tasks = [pool.submit(run_target, kind, sut, approved, modes) for sut in targets]
            for future in as_completed(tasks):
                rows.extend(future.result())
    if any(row["status"] != "complete" for row in rows):
        raise RuntimeError("E03 task failures are recorded under its status directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", choices=SCENES)
    parser.add_argument("--target", action="append", choices=TARGETS)
    parser.add_argument("--mode", action="append", choices=MODES)
    args = parser.parse_args()
    run(scenes=args.scene or SCENES, targets=args.target or TARGETS, modes=args.mode or MODES)
