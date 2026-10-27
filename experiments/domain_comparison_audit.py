"""Check frozen domain runs against evaluator-owned executions and causal predictions."""

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from testing_methods.common.fsbo.method import load as load_target
from testing_methods.history_transfer.domain_build import OUTPUT as MODELS
from testing_methods.history_transfer.domain_build import load_sources
from testing_methods.history_transfer.domain_discovery import aggregate_history

from .collision_metrics import collision_recall, summarize
from .construction import load_construction
from .discovery_config import development_tasks, read_json, write_json
from .domain_comparison import METHODS, OUTPUT, WORKERS, run_path, validate_protocol
from .domain_main_validation import reconstruct
from .domain_risk import disclose_driving_quality, disclose_impact
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import ROOT, TEST_RECORDS_ROOT
from .schema import QueryFeedback


def check_record(
    saved, task, method, candidates, executions, pool_labels, protocol_digest, *, acquisition=True
):
    setting, result = saved["specification"], saved["result"]
    queries, ledger = result["queries"], saved["query_ledger"]
    selected = [q["index"] for q in queries]
    initial = initial_indices(512, 8, task["algorithm_seed"])
    if (
        setting["task"] != task
        or setting["method"] != method
        or setting["protocol_sha256"] != protocol_digest
        or setting["candidate_ids"] != [s.scenario_id for s in candidates]
        or len(queries) != 128
        or len(ledger) != 128
        or len(set(selected)) != 128
        or selected[:8] != initial
        or saved["initial_indices"] != initial
        or result["selected_indices"] != selected
        or result["actual_queries"] != 128
        or result["maximum_queries"] != 128
        or result["termination"] != "budget_exhausted"
    ):
        raise ValueError(
            "A domain run differs from its frozen task, complete budget or initialization"
        )
    for number, (query, record) in enumerate(zip(queries, ledger), 1):
        index = query["index"]
        if (
            query["query"] != number
            or record["call"] != number
            or record["index"] != index
            or record["status"] != "complete"
            or record["account"] != "target"
            or record["scenario_id"] != candidates[index].scenario_id
            or record["purpose"] != ("initial" if number <= 8 else "target")
            or record["feedback"] != query["feedback"]
            or QueryFeedback.from_dict(query["feedback"]).to_dict()
            != executions[index].feedback.to_dict()
        ):
            raise ValueError(
                "A paid scalar or query ledger differs from its original executed trajectory"
            )
    metrics = {
        **summarize(candidates, queries, [executions[i].trajectory for i in selected]),
        **collision_recall(candidates, pool_labels, queries),
    }
    metrics = json.loads(json.dumps(metrics, allow_nan=False))
    if metrics != saved["metrics"]:
        raise ValueError(
            "Domain metrics differ from the complete collision denominator or paid trajectories"
        )
    if any(b < a for a, b in zip(metrics["discovery_curve"], metrics["discovery_curve"][1:])):
        raise ValueError("A collision discovery curve decreased")
    if any(v is not None and not 0 <= v <= 1 for v in metrics["recall_curve"]):
        raise ValueError("Recall escaped its complete-pool bounds")
    checked_choices = 0
    if method in ("gcrt", "H0F0", "H1F0", "H0F1"):
        mode = "H1F1" if method == "gcrt" else method
        if result["method"] != mode or result["online_feature_precision"] != "float64":
            raise ValueError("The main cell used a different mechanism or feature precision")
        sources = load_sources(task["scenario"])
        if method.startswith("H0"):
            sources = aggregate_history(sources)
        expected_names = [*sources["names"], "independent_target", "physical_target"]
        if (
            result["component_names"] != expected_names
            or result["physical_mass"] != 0.1
            or result["beta"] != 1
        ):
            raise ValueError("The H-by-F components or shared mixture settings differ")
        log_weights = np.log(np.r_[0.75 * sources["weights"], 0.25])
        for step, query in enumerate(queries):
            expected = np.r_[0.9 * np.exp(log_weights - logsumexp(log_weights)), 0.1]
            np.testing.assert_allclose(query["prequery_component_weights"], expected, atol=1e-12)
            if query["conditioning_queries"] != (step if mode.endswith("F1") else 0):
                raise ValueError("F0 used labels or F1 conditioned on unavailable feedback")
            if mode.endswith("F1"):
                probability = np.asarray(query["prequery_component_probabilities"][:-1])
                likelihood = probability if query["feedback"]["collision"] else 1 - probability
                log_likelihood = np.log(np.maximum(likelihood, 1e-15))
                np.testing.assert_allclose(
                    query["prequery_log_likelihood"], log_likelihood, atol=1e-12
                )
                log_weights += log_likelihood
                for diagnostic in (*query["conditional_modes"], query["physical_mode"]):
                    if diagnostic["gradient_max"] > 1e-10:
                        raise ValueError("A conditional Bernoulli mode did not converge")
            elif query["prequery_log_likelihood"] is not None:
                raise ValueError("F0 recorded a target-label likelihood update")
            after = np.r_[0.9 * np.exp(log_weights - logsumexp(log_weights)), 0.1]
            np.testing.assert_allclose(query["component_weights_after_query"], after, atol=1e-12)
        if acquisition:
            base, _ = load_target(MODELS / task["scenario"] / "target_kernel.pt")
            for number in (9, 64, 128):
                row = queries[number - 1]
                index, probability, weights, components = reconstruct(
                    candidates, base, sources, queries[: number - 1], mode, initial
                )
                if index != row["index"]:
                    raise ValueError(
                        "Independent full-pool main acquisition chose a different candidate"
                    )
                np.testing.assert_allclose(
                    probability[index], row["prequery_probability"], atol=1e-7
                )
                np.testing.assert_allclose(
                    components[:, index], row["prequery_component_probabilities"], atol=1e-7
                )
                np.testing.assert_allclose(weights, row["prequery_component_weights"], atol=1e-12)
                checked_choices += 1
    elif method == "random":
        order = initial_indices(512, 512, task["algorithm_seed"])
        expected = initial + [i for i in order if i not in initial][:120]
        if selected != expected:
            raise ValueError("Random selection differs from its label-independent seed permutation")
    if method == "av_bo_rf_ei" and [r["paid_observations"] for r in result["tuning"]] != [50, 100]:
        raise ValueError("RF-EI tuned on a different paid sample count")
    if method == "scenario_fuzz" and result["online_classifier_updates"] != 0:
        raise ValueError("ScenarioFuzz retrained before the declared data increment")
    return {
        "queries": len(queries),
        "independent_main_choices": checked_choices,
        "metrics": metrics,
    }


def audit_group(target_id, kind, rows, digest):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        _, development = load_construction(kind)
        archive = next(a for a in development if a.source_id == target_id)
        bank = ExecutionBank(TEST_RECORDS_ROOT / "physical/development" / kind / target_id)
        executions = [
            disclose_driving_quality(disclose_impact(bank(scene))) for scene in bank.candidates
        ]
        labels = [e.feedback.collision for e in executions]
        if labels != archive.collision.tolist():
            raise ValueError("The exhaustive denominator differs from its construction archive")
        checks = []
        for row in rows:
            task = {key: row[key] for key in development_tasks()[0]}
            saved = read_json(run_path(task, row["method"]))
            if saved["specification"]["bank_contract"] != bank.manifest["contract"]:
                raise ValueError("A run changed its execution contract")
            check = check_record(
                saved, task, row["method"], bank.candidates, executions, labels, digest
            )
            if any(row[key] != value for key, value in check["metrics"].items()):
                raise ValueError("A run index differs from its saved metrics")
            checks.append(
                {
                    "path": row["path"],
                    "queries": check["queries"],
                    "independent_main_choices": check["independent_main_choices"],
                }
            )
        return checks


def compare_first_with_stage_a(rows):
    matches = []
    for row in rows:
        method = row["method"]
        task = {key: row[key] for key in development_tasks()[0]}
        if method == "random":
            continue
        mode = "H1F1" if method == "gcrt" else method
        stage = (
            ROOT
            / "experiment_outputs/gcrt/collision/discovery/domain_comparison/stage_a"
            / f"{mode}.json"
            if method in ("gcrt", "H0F0", "H1F0", "H0F1")
            else ROOT
            / f"experiment_outputs/{method}/collision/discovery/domain_comparison/stage_a/first_run.json"
        )
        previous = read_json(stage)["result"]
        actual = read_json(run_path(task, method))["result"]
        if previous["selected_indices"] != actual["selected_indices"]:
            raise ValueError(
                f"The frozen {method} first run differs from its mechanism-checked sequence"
            )
        for before, after in zip(previous["queries"], actual["queries"]):
            for key in (
                "prequery_probability",
                "prequery_component_weights",
                "prequery_component_probabilities",
                "prequery_risk_mean",
                "prequery_risk_variance",
                "expected_improvement",
            ):
                if key in before and before[key] is not None:
                    np.testing.assert_allclose(before[key], after[key], atol=1e-7)
        matches.append(method)
    return matches


def audit(*, first_only=False):
    started = time.perf_counter()
    specification = read_json(OUTPUT / "protocol.json")
    validate_protocol(specification)
    index = read_json(OUTPUT / ("first.json" if first_only else "run_index.json"))
    rows = index["runs"]
    tasks = development_tasks()[:1] if first_only else development_tasks()
    expected = {
        (t["target_id"], t["scenario"], t["algorithm_seed"], m) for t in tasks for m in METHODS
    }
    actual = [(r["target_id"], r["scenario"], r["algorithm_seed"], r["method"]) for r in rows]
    if index["status"] != "complete" or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Audit requires the complete exact frozen matrix")
    digest = hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()
    groups = sorted({(r["target_id"], r["scenario"]) for r in rows})
    checks = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(
                audit_group,
                target,
                kind,
                [r for r in rows if (r["target_id"], r["scenario"]) == (target, kind)],
                digest,
            )
            for target, kind in groups
        ]
        for future in as_completed(futures):
            checks.extend(future.result())
            print("domain_audit", len(checks), "/", len(rows), flush=True)
    report = {
        "status": "passed",
        "runs": len(rows),
        "queries": sum(r["queries"] for r in checks),
        "independent_main_choices": sum(r["independent_main_choices"] for r in checks),
        "stage_a_sequence_matches": compare_first_with_stage_a(rows) if first_only else None,
        "mechanism_overlap": "H1F1 and gcrt share the same 40 files and all paid states",
        "checks": sorted(checks, key=lambda r: r["path"]),
        "new_physical_calls": 0,
        "wall_s": time.perf_counter() - started,
    }
    write_json(OUTPUT / ("first_audit.json" if first_only else "audit.json"), report)
    return report
