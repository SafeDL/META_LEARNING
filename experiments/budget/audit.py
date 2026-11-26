"""Independent E02 receipt, forecast, discovery and Random-reuse verification."""

from dataclasses import asdict

import numpy as np
import torch

from testing_methods.baselines.paired_system_active_evaluation.method import build_proxy
from testing_methods.baselines.scenario_fuzz.graphs import graph
from testing_methods.baselines.scenario_fuzz.model import batch_graphs

from experiments.budget import protocol
from experiments.budget.execution import e01_paths, make_case
from experiments.budget.selectors import greedy_index
from experiments.baselines import BASELINES
from experiments.costs import PeakMemory
from experiments.executions import ExecutionBank
from experiments.failure_metrics import summarize
from experiments.initialization import initial_indices
from experiments.history_transfer_experiment import scenario_history
from experiments.record_audit import BankTruth, _canonical_feedback, verify_metrics
from experiments.resume import paid_feedback_hash, trajectory_hash
from experiments.storage import read_json, write_json
from experiments.transfer_prediction_audit import verify_forecasts, verify_weight_updates
from experiments.transfer_settings import native_settings


def result_directory(sut_id, kind, method):
    return protocol.OUTPUT / "budget" / sut_id / kind / str(protocol.SEED) / method


def verify_configuration(method, result):
    device = "cuda" if method == "pooled_history" else "cpu"
    if result["resources"]["device"] != device:
        raise ValueError("E02 selector ran on a different registered device")
    settings = native_settings()
    if method in ("full", "target_only"):
        expected = {
            "method": "H1F1", "response_kernel": "SharedPrivateKernel",
            "feedback_used": "joint", "target_feedback_updates": True,
            "explicit_source_components": method == "full",
            "source_weight_updates": method == "full", "local_reliability": False,
            "representation": "historical_embedding" if method == "full" else "normalized_parameters",
            "online_feature_precision": "float64", "online_settings": settings.online.to_dict(),
            "conditional_inference_device": device, "kernel_and_adaptation_device": device,
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError("GP result changed the registered feedback, branch, weight or fitting rules")
        for snapshot in result["prediction_snapshots"]:
            count = snapshot["paid_queries"]
            if not count:
                continue
            fits = [(snapshot["target_fit"], settings.online.learned if method == "full" else settings.online.raw)]
            if method == "full":
                fits.append((snapshot["physical_fit"], settings.online.raw))
            for fit, expected_fit in fits:
                if (fit["paid_observations"] != count or fit["settings"] != expected_fit.to_dict()
                        or not np.isfinite([fit["objective"], fit["gradient_max"]]).all()):
                    raise ValueError("GP adaptation receipt differs from its paid prefix or ten-step settings")
    elif method == "pooled_history":
        if (result["fit_settings"] != settings.pooled.to_dict()
                or result["inference_backend"] != "grouped"
                or result["conditional_inference_device"] != device
                or result["feedback_used"] != "joint" or result["historical_count"] != 4000):
            raise ValueError("Pooled GP changed its admitted historical inputs or default inference")
        for snapshot in result["prediction_snapshots"]:
            if snapshot["conditional_mode"]["original_records"] != 4000 + snapshot["paid_queries"]:
                raise ValueError("Pooled GP conditioning count differs from history and its own paid prefix")
    elif method == "static_prior":
        if result["target_feedback_updates"] is not False:
            raise ValueError("Static prior unexpectedly adapted to target feedback")
    elif method in ("scenario_fuzz", "paired_system_active_evaluation"):
        if result["config"] != asdict(BASELINES[method].DEFAULT):
            raise ValueError("Recent baseline did not execute its registered default configuration")
        if method == "scenario_fuzz" and result["online_classifier_updates"] != 0:
            raise ValueError("ScenarioFuzz changed its fixed classifier within the 200-query budget")


def verify_historical_baselines(bank, context):
    """Rebuild fixed historical predictions without any target response."""
    kind = bank.candidates[0].kind
    case, public = make_case(bank, "scenario_fuzz", context)
    classifier = public["classifier"].eval()
    with torch.no_grad():
        probabilities = np.concatenate([
            classifier(batch_graphs([graph(s) for s in bank.candidates[start:start + 73]]))
            .sigmoid().cpu().numpy()
            for start in range(0, len(bank.candidates), 73)
        ])
    if (probabilities.shape != (2000,) or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Reconstructed graph classifier predictions are invalid")
    _, paired_public = make_case(bank, "paired_system_active_evaluation", context)
    history = scenario_history(protocol.INPUT, kind)
    actual, _ = build_proxy(bank.candidates, history)
    error = float(np.max(np.abs(actual - paired_public["proxy"])))
    if not np.isfinite(error) or error > 1e-12:
        raise ValueError("Paired proxy cannot be reconstructed from the registered history")
    return {
        "scenario": kind, "status": "complete", "candidate_count": 2000,
        "history_records": sum(len(a.scenarios) for a in history),
        "classifier_model_version": case.specification["construction"]["model_version"],
        "graph_batch_size": 73, "graph_precision": "float32",
        "proxy_maximum_error": error, "target_feedback_used": 0,
    }, probabilities


def derive_random(bank, truth, *, destination=None, source=None):
    """No query client is opened; every prefix refers to the original E01 receipt."""
    sut_id, kind = bank.manifest["sut_id"], bank.candidates[0].kind
    if destination is None:
        destination = result_directory(sut_id, kind, "random") / "result.json.gz"
    path, receipt = source or e01_paths(sut_id, kind, "target_only")
    # Rebuild the public order before opening any saved responses.
    order = np.random.default_rng(protocol.SEED).permutation(len(bank.candidates))[:protocol.BUDGET].tolist()
    initial = initial_indices(len(bank.candidates), protocol.INITIAL, protocol.SEED)
    if order[:protocol.INITIAL] != initial:
        raise ValueError("Random order changed its registered initialization")
    with PeakMemory() as extraction:
        origin, paid = read_json(path), read_json(receipt)
        spec, records = origin["specification"], origin["query_ledger"]
        if (spec["bank_snapshot"] != bank.snapshot_fingerprint
                or spec["bank_contract"] != bank.manifest["contract"]
                or spec["split"]["candidate_ids"] != [s.scenario_id for s in bank.candidates]
                or spec["feedback_seed"] != protocol.SEED
                or spec["split"]["query_order"] != order
                or origin["result"]["selected_indices"] != order
                or [r["index"] for r in records] != order
                or paid["query_ledger"] != records
                or paid["paid_feedback_hash"] != paid_feedback_hash(records)
                or paid["status"] != "complete"):
            raise ValueError("E01 cannot supply the registered Random trajectory")
        queries, traces = [], []
        for position, (index, record) in enumerate(zip(order, records), 1):
            if _canonical_feedback(record["feedback"]) != truth.paid_feedback(index, native=False):
                raise ValueError("Reused Random feedback differs from the original native execution")
            trace = bank(bank.candidates[index]).trajectory
            if trajectory_hash(trace) != paid["trajectory_hashes"][position - 1]:
                raise ValueError("Reused Random trajectory changed")
            traces.append(trace)
            queries.append({"query": position, "index": index, "feedback": record["feedback"],
                            "reason": "common_initialization" if position <= protocol.INITIAL else "random",
                            "source_paid_position": position})
        metrics = summarize(bank.candidates, truth.labels.tolist(), queries, traces,
                            initial_count=protocol.INITIAL, query_ledger=records, domain=bank.domain)
        verify_metrics(bank.candidates, truth.labels.tolist(), order, queries, records, metrics,
                       protocol.INITIAL, domain=bank.domain)
    # These fields are not measurements of a newly run Random selector.
    metrics.update(selection_s=None, replay_wall_s=None, rss_compute_s=None,
                   equivalent_testing_curve_s=None)
    for name in ("selection", "replay", "rss"):
        metrics["cumulative_costs_s"][name] = [None] * protocol.BUDGET
    specification = {
        "method": "random", "seed": protocol.SEED, "budget": protocol.BUDGET,
        "initial_indices": initial, "bank_snapshot": bank.snapshot_fingerprint,
        "bank_contract": bank.manifest["contract"],
        "origin_result": str(path.relative_to(protocol.ROOT)), "origin_result_sha256": protocol.sha(path),
        "origin_receipt": str(receipt.relative_to(protocol.ROOT)), "origin_receipt_sha256": protocol.sha(receipt),
        "source_positions": list(range(1, protocol.BUDGET + 1)),
        "budget_to_source_index": order, "new_paid_reveals": 0, "reused_paid_reveals": protocol.BUDGET,
        "selector_cost_measured": False,
    }
    if destination.exists():
        saved = read_json(destination)
        if (saved["specification"] != specification or saved["query_ledger"] != records
                or saved["result"]["queries"] != queries
                or saved["metrics"] != metrics):
            raise ValueError("Existing Random derivation differs from the source ledger")
        return saved
    saved = {
        "specification": specification, "query_ledger": records, "metrics": metrics,
        "result": {"method": "random", "queries": queries, "selected_indices": order,
                   "actual_queries": protocol.BUDGET, "maximum_queries": protocol.BUDGET,
                   "termination": "budget_exhausted"},
        "extraction_resources": extraction.result, "new_physical_calls": 0,
    }
    write_json(destination, saved)
    return saved


def check_case(bank, truth, method, context, *, root=None, inputs=None):
    root = protocol.OUTPUT if root is None else root
    case, public = make_case(bank, method, context) if inputs is None else inputs
    directory = root / "budget" / case.sut_id / case.scenario / str(protocol.SEED) / method
    path = directory / "result.json.gz"
    saved = read_json(path)
    approval = read_json(root / "approvals/budget" / case.sut_id / case.scenario
                         / str(protocol.SEED) / f"{method}.json")
    if (saved["specification"] != case.specification
            or approval != {"key": list(case.key), "specification": case.specification, "scope": "formal"}):
        raise ValueError("E02 result differs from independently reconstructed inputs or its admission")
    records, result = saved["query_ledger"], saved["result"]
    verify_configuration(method, result)
    paid = read_json(directory / "paid_feedback.json.gz")
    indices = [r["index"] for r in records]
    if (len(indices) != protocol.BUDGET or len(set(indices)) != protocol.BUDGET
            or len(result["queries"]) != protocol.BUDGET
            or indices[:protocol.INITIAL] != case.specification["initial_indices"]
            or result["selected_indices"] != indices
            or result["termination"] != "budget_exhausted"
            or result["actual_queries"] != protocol.BUDGET
            or result["maximum_queries"] != protocol.BUDGET
            or paid["query_ledger"] != records or paid["status"] != "complete"
            or paid["successful_paid_responses"] != protocol.BUDGET
            or paid["specification"]["task"] != case.specification
            or paid["paid_feedback_hash"] != paid_feedback_hash(records)
            or len(paid["trajectory_hashes"]) != protocol.BUDGET):
        raise ValueError("E02 budget, initialization or paid-recovery receipt disagrees")
    for position, (record, query) in enumerate(zip(records, result["queries"]), 1):
        index = record["index"]
        expected = truth.paid_feedback(index, native=True)
        if (record["call"] != position or record["status"] != "complete"
                or _canonical_feedback(record["feedback"]) != expected
                or query["index"] != index or _canonical_feedback(query["feedback"]) != expected):
            raise ValueError("E02 paid response differs from the original execution")
        if trajectory_hash(bank(bank.candidates[index]).trajectory) != paid["trajectory_hashes"][position - 1]:
            raise ValueError("E02 saved paid trajectory differs from native history")
    if saved["evaluation"]["candidate_collisions"] != truth.labels.tolist():
        raise ValueError("E02 evaluator labels differ from native events")
    verify_metrics(bank.candidates, truth.labels.tolist(), indices, result["queries"], records,
                   saved["metrics"], protocol.INITIAL, domain=bank.domain)
    audit = {"method": method, "queries": len(records), "configuration_verified": True,
             "receipt_hash": protocol.sha(directory / "paid_feedback.json.gz")}
    if method in ("full", "target_only", "pooled_history"):
        audit["forecasts"] = verify_forecasts(
            bank.candidates, result["prediction_snapshots"], records, truth.margins[indices],
            method=method,
            models=context["models"] if method != "target_only" else (None, None),
            sources=context["sources"] if method == "full" else None,
            pooled=context["pooled"] if method == "pooled_history" else None,
            settings=native_settings())
        for snapshot in result["prediction_snapshots"]:
            count = snapshot["paid_queries"]
            if protocol.INITIAL <= count < protocol.BUDGET:
                remaining = set(range(len(bank.candidates))) - set(indices[:count])
                expected = greedy_index(bank.candidates, snapshot["probabilities"], remaining)
                if indices[count] != expected:
                    raise ValueError("Saved acquisition checkpoint did not choose the highest collision probability")
        if method == "full":
            audit["weights"] = verify_weight_updates(result, records, context["sources"])
        for query in result["queries"]:
            if query["conditioning_queries"] != query["query"] - 1:
                raise ValueError("GP acquisition used feedback outside its paid prefix")
    elif method == "static_prior":
        if indices != case.specification["fixed_order"] or result["fixed_order"] != indices:
            raise ValueError("Static prior changed its predeclared order")
        if any(q["conditioning_queries"] != 0 for q in result["queries"]):
            raise ValueError("Static prior adapted to target feedback")
        probabilities = public["probabilities"]
        if any(q["prequery_probability"] != probabilities[q["index"]] for q in result["queries"]):
            raise ValueError("Static prior differs from the frozen K=0 probabilities")
    elif method == "paired_system_active_evaluation":
        if result["target_fit_sample_counts"] != list(range(20, 200, 5)):
            raise ValueError("Paired-system adaptation changed its prescribed batches")
        if [q["batch"] for q in result["queries"]] != [0] * 20 + [i for i in range(1, 37) for _ in range(5)]:
            raise ValueError("Paired-system query batches changed")
    elif method == "scenario_fuzz":
        expected = context["scenario_fuzz_probabilities"][indices]
        actual = np.asarray([q["prequery_probability"] for q in result["queries"]])
        error = float(np.max(np.abs(actual - expected)))
        if not np.isfinite(error) or error > 1e-6:
            raise ValueError("ScenarioFuzz risk cannot be reconstructed from its historical classifier")
        audit["maximum_graph_probability_error"] = error
    return saved, audit


def audit_target(kind, sut_id, context):
    bank = ExecutionBank(protocol.INPUT / "physical/target" / kind / sut_id)
    truth = BankTruth(bank)
    saved, audits = {}, []
    for method in protocol.METHODS:
        if method == "random":
            saved[method] = derive_random(bank, truth)
            audits.append({"method": method, "queries": 200, "source_receipt_verified": True,
                           "new_paid_reveals": 0})
        else:
            saved[method], audit = check_case(bank, truth, method, context)
            audits.append(audit)
    return check_common_feedback(bank, truth, saved, audits, protocol.OUTPUT)


def check_common_feedback(bank, truth, saved, audits, root):
    kind, sut_id = bank.candidates[0].kind, bank.manifest["sut_id"]
    reference = saved["full"]["query_ledger"][:protocol.INITIAL]
    # Native baseline-only scalar disclosures do not change the common collision/RSS response.
    for method in protocol.METHODS:
        current = saved[method]["query_ledger"][:protocol.INITIAL]
        for left, right in zip(reference, current):
            if (left["index"] != right["index"]
                    or left["feedback"]["collision"] != right["feedback"]["collision"]
                    or left["feedback"]["safety"]["rss_margin"] != right["feedback"]["safety"]["rss_margin"]):
                raise ValueError("Methods do not share the same initial paid responses")
    for k in (0, 20):
        ours = next(s for s in saved["full"]["result"]["prediction_snapshots"] if s["paid_queries"] == k)
        raw = next(s for s in saved["target_only"]["result"]["prediction_snapshots"] if s["paid_queries"] == k)
        if ours["physical_parameters"] != raw["target_parameters"]:
            raise ValueError("Raw target branch differs from Target-only under common feedback")
    outcome = {"scene": kind, "target": sut_id, "status": "complete", "audits": audits,
               "labels": truth.labels.tolist(), "bank_snapshot": bank.snapshot_fingerprint}
    write_json(root / "audits" / kind / f"{sut_id}.json", outcome)
    print({"audit": "complete", "scene": kind, "target": sut_id}, flush=True)
    return outcome
