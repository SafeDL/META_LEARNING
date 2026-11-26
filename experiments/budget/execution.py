"""Admit and execute only E02 tasks against unchanged, evaluator-owned banks."""

from dataclasses import asdict

import numpy as np

from testing_methods.baselines.scenario_fuzz import model as fuzz

from experiments.budget import protocol
from experiments import transfer_execution as execution
from experiments.baselines import BASELINES
from experiments.budget.selectors import run, static_order
from experiments.collect_transfer_records import ApprovedCase
from experiments.history import model_version
from experiments.history_transfer_experiment import freeze
from experiments.initialization import initial_indices
from experiments.storage import read_json
from experiments.transfer_settings import native_settings


def e01_paths(sut_id, kind, method):
    directory = protocol.INPUT / "k_shot" / sut_id / kind / str(protocol.SEED)
    return directory / f"{method}.json.gz", directory / f"{method}_paid_feedback.json.gz"


def initial_prediction(bank):
    path, ledger = e01_paths(bank.manifest["sut_id"], bank.candidates[0].kind, "full")
    saved = read_json(path)
    spec = saved["specification"]
    if (spec["split"]["candidate_ids"] != [s.scenario_id for s in bank.candidates]
            or spec["bank_contract"] != bank.manifest["contract"]):
        raise ValueError("E01 initial predictions belong to another target bank")
    initial = [s for s in saved["result"]["prediction_snapshots"] if s["paid_queries"] == 0]
    if (len(initial) != 1 or initial[0]["observed_indices"]
            or len(initial[0]["probabilities"]) != len(bank.candidates)
            or not np.isfinite(initial[0]["probabilities"]).all()):
        raise ValueError("A complete, feedback-free E01 prediction is required")
    return initial[0]["probabilities"], {
        "result": str(path.relative_to(protocol.ROOT)), "sha256": protocol.sha(path),
        "snapshot": 0, "source_snapshot": spec["source_snapshot"],
        "model": spec["model"], "component_weights": initial[0]["component_weights"],
    }


def make_case(bank, method, context):
    if method not in protocol.METHODS or method == "random":
        raise ValueError("Random is derived directly from E01; it opens no query interface")
    settings = native_settings()
    kind, sut_id = bank.candidates[0].kind, bank.manifest["sut_id"]
    initial = initial_indices(len(bank.candidates), protocol.INITIAL, protocol.SEED)
    spec = {
        "task_snapshot": {"sut_id": sut_id, "scenario": kind, "method": method,
                          "method_contract": protocol.contract(method),
                          "bank_snapshot": bank.snapshot_fingerprint},
        "bank_contract": bank.manifest["contract"], "environment_seed": bank.manifest["environment_seed"],
        "candidates": [asdict(s) for s in bank.candidates],
        "budget": protocol.BUDGET, "initial_count": protocol.INITIAL, "initial_indices": initial,
        "seed": protocol.SEED, "metric_contract": execution.METRIC_CONTRACT,
        "execution_contract": execution.execution_contract(protocol.LIMITS),
        "evaluate_complete_bank": True, "new_physical_executions": 0,
        "feedback_contract": "only this task's paid collision, RSS and native trajectory",
    }
    public = {"method": method, "candidates": bank.candidates, "initial_indices": initial,
              "budget": protocol.BUDGET, "device": "cpu"}
    if method in ("full", "target_only"):
        public.update(pretrained=context["models"][0] if method == "full" else None,
                      sources=context["sources"] if method == "full" else None,
                      online_settings=settings.online, snapshot_counts=(0, 20, 50, 100, 200))
        spec.update(online_settings=settings.online.to_dict(), update_source_weights=True,
                    models=None if method == "target_only" else [model_version(m) for m in context["models"]],
                    source_snapshot=None if method == "target_only" else context["sources"]["specification"],
                    raw_kernel="SharedPrivateKernel", device="cpu")
    elif method == "pooled_history":
        public.update(pretrained=context["models"][0], history=context["pooled"],
                      fit_settings=settings.pooled, snapshot_counts=(0, 20, 50, 100), device="cuda")
        spec.update(model=model_version(context["models"][0]),
                    history_view=context["pooled"]["view_specification"],
                    fit_settings=settings.pooled.to_dict(), device="cuda", inference_backend="grouped")
    elif method == "static_prior":
        probability, origin = initial_prediction(bank)
        # Reuse only the matching E01 historical mixture, never a K>0 state.
        if (origin["model"] != model_version(context["models"][0])
                or origin["source_snapshot"] != context["sources"]["specification"]):
            raise ValueError("Static K=0 history differs from current Ours inputs")
        public["probabilities"] = probability
        spec.update(origin=origin, fixed_order=static_order(bank.candidates, probability, initial, protocol.BUDGET))
    else:
        directory = protocol.OUTPUT / "models" / kind
        completion = read_json(directory / "training_completion.json")
        if completion["status"] != "complete":
            raise ValueError("E02 historical baseline construction is incomplete")
        from experiments.budget.prepare import construction_specification
        from experiments.history_transfer_experiment import scenario_history

        if completion["specification"] != construction_specification(scenario_history(protocol.INPUT, kind)):
            raise ValueError("Baseline historical inputs or graph implementation changed")
        public.update(seed=protocol.SEED, config=BASELINES[method].DEFAULT)
        spec.update(configuration=asdict(BASELINES[method].DEFAULT), construction=completion)
        if method == "scenario_fuzz":
            path = directory / "scenario_fuzz/classifier.pt"
            if protocol.sha(path) != completion["weights_sha256"]:
                raise ValueError("ScenarioFuzz weights changed after training")
            public["classifier"] = fuzz.load(path)[0]
        else:
            path = directory / "paired_proxy.json"
            proxy = read_json(path)
            if (protocol.sha(path) != completion["paired_proxy_sha256"]
                    or proxy["candidate_ids"] != [s.scenario_id for s in bank.candidates]):
                raise ValueError("Historical proxy differs from the public candidate bank")
            public["proxy"] = np.asarray(proxy["values"])
    case = ApprovedCase("budget", sut_id, kind, protocol.SEED, method, execution.canonical(spec))
    return case, public


def execute(bank, method, context):
    if (bank.manifest["environment_seed"] != 81001 or len(bank.candidates) != 2000):
        raise ValueError("E02 requires the unchanged E01 target library")
    admission = read_json(protocol.OUTPUT / "input_admission.json")
    if admission["acceptance_sha256"] != protocol.sha(protocol.INPUT / "reports/method_acceptance.json"):
        raise ValueError("Reused E01 admission changed")
    origin, _ = e01_paths(bank.manifest["sut_id"], bank.candidates[0].kind, "target_only")
    approved = read_json(origin)["specification"]
    if (approved["bank_snapshot"] != bank.snapshot_fingerprint
            or approved["candidates"] != execution.canonical([asdict(s) for s in bank.candidates])):
        raise ValueError("E02 bank differs from the approved E01 bank")
    if method in ("full", "pooled_history", "static_prior"):
        path, _ = e01_paths(bank.manifest["sut_id"], bank.candidates[0].kind,
                            "pooled_history" if method == "pooled_history" else "full")
        inherited = read_json(path)["specification"]
        if inherited["model"] != model_version(context["models"][0]):
            raise ValueError("E02 historical model differs from E01")
        if method != "pooled_history" and inherited["source_snapshot"] != context["sources"]["specification"]:
            raise ValueError("E02 sources differ from E01")
        if method == "pooled_history" and inherited["history_view"] != context["pooled"]["view_specification"]:
            raise ValueError("E02 Pooled historical records differ from E01")
    case, public = make_case(bank, method, context)
    execution.declare(protocol.OUTPUT, case, scope="formal")
    return execution._execute(protocol.OUTPUT, case, bank, run, public,
                              paid_execution=lambda i: bank(bank.candidates[i]), scope="formal")
