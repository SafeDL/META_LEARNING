"""Registered method exploration using paid responses from the old platform."""

from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from testing_methods.history_transfer.selection import select

from .attribution import (
    CHECKPOINTS,
    DATA,
    compact_result,
    csv_report,
    models,
    prediction_metrics,
    read_source,
    registered_runs,
    source_path,
)
from .attribution import (
    OUTPUT as ATTRIBUTION,
)
from .executions import ExecutionBank
from .failure_metrics import summarize
from .history import model_version
from .protocol import ROOT
from .query import QueryLedger, run_isolated
from .scenarios import DOMAINS
from .schema import Scenario
from .storage import read_json, write_json

OUTPUT = ROOT / "artifacts/runs/public_ads/method_exploration"

E1_CONFIGURATION = {
    "implementation": "testing_methods/history_transfer/reliability.py",
    "feature": "frozen target-kernel historical embedding",
    "distance": "Euclidean in that embedding; Gaussian weights exp(-d^2/(2*h^2))",
    "bandwidth": "h=exp(frozen target-kernel log_lengthscale), lower bound 0.01",
    "error": "prequery collision log likelihood, clipped at 1e-15",
    "local_mean": "l_s(x)=sum_i K_i(x)*log_likelihood_i,s / sum_i K_i(x)",
    "global_mean": "g_s=mean_i log_likelihood_i,s",
    "shrinkage": "alpha(x)=sum_i K_i(x)/(sum_i K_i(x)+20)",
    "weight": "0.9*softmax_s(global_log_score_s+alpha(x)*t*(l_s(x)-g_s))",
    "fallback": "no paid records or zero kernel mass gives global weights",
    "raw_component_weight": 0.1,
    "weighted_components": "all historical sources and independent target; raw excluded",
    "additional_trainable_parameters": 0,
    "additional_optimization_steps": 0,
    "device": "cpu",
    "precision": "float64",
    "torch_threads": 1,
    "batch_size": 128,
    "max_task_wall_s": 1800,
    "max_task_cpu_gib": 8,
    "max_task_gpu_gib": 0,
    "diagnostic": {
        "split": "fixed midpoint of each original continuous parameter range",
        "midpoint_assignment": "lower side includes midpoint",
        "minimum_paid_records_per_side": 5,
        "criterion": "same source-pair Brier ordering reversal at consecutive checkpoints",
        "tie_tolerance": 1e-12,
        "historical_sources_only": True,
    },
    "check_command": "python -B -m pytest tests/history_transfer/test_reliability.py "
    "tests/experiments/test_method_exploration.py",
    "expected_checks": "normalized positive weights, global fallback, local reversal, "
    "chunk invariance and future-feedback isolation",
}


def complementarity(candidates, queries, source_names, *, checkpoints=CHECKPOINTS):
    """Inspect only the supplied paid prefix, never the complete candidate responses."""
    domain = DOMAINS[candidates[0].kind]
    count = len(source_names)
    rows, reversals, previous = [], [], {}
    for prefix in checkpoints:
        paid = queries[:prefix]
        if len(paid) != prefix or len({q["index"] for q in paid}) != prefix:
            raise ValueError("The diagnostic needs a complete, unique paid prefix")
        p = np.asarray([q["prequery_component_probabilities"][:count] for q in paid])
        if p.shape != (prefix, count) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Source probabilities must be finite prequery probabilities")
        y = np.asarray([q["feedback"]["collision"] for q in paid], dtype=float)
        errors = (p - y[:, None]) ** 2
        current = {}
        for parameter, (low, high) in domain.bounds.items():
            midpoint = (low + high) / 2
            lower = np.asarray(
                [candidates[q["index"]].parameters[parameter] <= midpoint for q in paid]
            )
            sizes = int(lower.sum()), int((~lower).sum())
            eligible = min(sizes) >= 5
            means = [errors[side].mean(0) if side.any() else None for side in (lower, ~lower)]
            for source, name in enumerate(source_names):
                rows.append(
                    {
                        "paid_queries": prefix,
                        "parameter": parameter,
                        "midpoint": midpoint,
                        "lower_count": sizes[0],
                        "upper_count": sizes[1],
                        "eligible": eligible,
                        "source": name,
                        "lower_brier": None if means[0] is None else float(means[0][source]),
                        "upper_brier": None if means[1] is None else float(means[1][source]),
                    }
                )
            if not eligible:
                continue
            for a, b in combinations(range(count), 2):
                delta_lower = float(means[0][a] - means[0][b])
                delta_upper = float(means[1][a] - means[1][b])
                if abs(delta_lower) <= 1e-12 or abs(delta_upper) <= 1e-12:
                    continue
                if np.sign(delta_lower) == np.sign(delta_upper):
                    continue
                key = parameter, a, b, int(np.sign(delta_lower))
                current[key] = prefix
                if key in previous:
                    reversals.append(
                        {
                            "parameter": parameter,
                            "source_a": source_names[a],
                            "source_b": source_names[b],
                            "previous_prefix": previous[key],
                            "paid_queries": prefix,
                            "lower_count": sizes[0],
                            "upper_count": sizes[1],
                            "lower_brier_a_minus_b": delta_lower,
                            "upper_brier_a_minus_b": delta_upper,
                        }
                    )
        previous = current
    return rows, reversals


def task_inputs():
    runs = registered_runs()
    return [
        {
            "sut_id": sut,
            "scenario": kind,
            "joint": str(Path(row["path"]).relative_to(ROOT)),
            "common_feedback": str(
                (ATTRIBUTION / "common_feedback" / sut / kind / "joint.json.gz").relative_to(ROOT)
            ),
        }
        for (sut, kind, method), row in sorted(runs.items())
        if method == "joint"
    ]


def register_protocol(tasks):
    protocol = {
        "purpose": "old-platform method development, not independent validation",
        "platform": "HighwayEnv",
        "new_physical_executions": 0,
        "algorithm_seed": 11,
        "initial_count": 20,
        "budget": 200,
        "checkpoints": list(CHECKPOINTS),
        "tasks": tasks,
        "controls": "reuse original global-weight full method and common Random replays",
        "records": "fixed source record IDs and model versions in each original specification",
        "query_order": "saved Random sequence for prefixes; adaptive thereafter in separate runs",
        "metrics": "failures, post-initial failures, AUC, candidate-truth coverage, "
        "Gower pair/nearest-neighbor distances at equal budget and equal discovery count, "
        "NLL/Brier and costs",
        "e1": E1_CONFIGURATION,
        "e2": {"configuration": "unfrozen", "status": "not_started"},
        "e3": {"configuration": "unfrozen", "status": "not_started"},
    }
    path = OUTPUT / "protocol.json"
    if path.exists():
        previous = read_json(path)
        expected = {**protocol, "e2": previous["e2"], "e3": previous["e3"]}
        if previous != expected:
            raise ValueError("Exploration configuration differs from its registered protocol")
        return previous
    else:
        write_json(path, protocol)
    return protocol


def diagnose_e1():
    tasks = task_inputs()
    register_protocol(tasks)
    rows, reversals, task_results = [], [], []
    for task in tasks:
        original = read_json(ROOT / task["joint"])
        common = read_json(ROOT / task["common_feedback"])
        candidates = [Scenario(**s) for s in original["specification"]["candidates"]]
        queries = common["result"]["queries"]
        names = common["result"]["component_names"][:-2]
        if len(queries) != 200 or common["specification"]["query_order"] != [
            q["index"] for q in queries
        ]:
            raise ValueError("Diagnostic input differs from the saved Random query order")
        if (
            common["specification"]["target_model_version"]
            != original["specification"]["target_model_version"]
        ):
            raise ValueError("Diagnostic model differs from its original registered model")
        for position, query in enumerate(queries):
            if (
                query["conditioning_queries"] != position
                or query["feedback"]["scenario_id"] != candidates[query["index"]].scenario_id
            ):
                raise ValueError("Source error must use the corresponding prequery prediction")
        regional, stable = complementarity(candidates, queries, names)
        identity = {"sut_id": task["sut_id"], "scenario": task["scenario"]}
        rows.extend({**identity, **r} for r in regional)
        reversals.extend({**identity, **r} for r in stable)
        task_results.append(
            {**identity, "stable_reversals": len(stable), "gate_passed": bool(stable)}
        )
    csv_report(OUTPUT / "e1/region_errors.csv", rows)
    if reversals:
        csv_report(OUTPUT / "e1/stable_reversals.csv", reversals)
    summary = {
        "status": "diagnostic_passed" if reversals else "ended_no_local_complementarity",
        "tasks": task_results,
        "gate_passed_tasks": sum(t["gate_passed"] for t in task_results),
        "stable_reversals": len(reversals),
        "existing_paid_records_reused": len(tasks) * 200,
        "new_logical_queries": 0,
        "new_physical_executions": 0,
        "adaptive_runs": 0,
        "interpretation": "reversal is an exploration gate, not evidence of discovery benefit",
    }
    write_json(OUTPUT / "e1/diagnostic.json", summary)
    return summary


def replay_e1(task, phase):
    torch.set_num_threads(1)
    original = read_json(ROOT / task["joint"])
    common = read_json(ROOT / task["common_feedback"])
    base, _ = models(task["scenario"])
    if model_version(base) != original["specification"]["target_model_version"]:
        raise ValueError("E1 must use the exact original full-method target model")
    source = read_source(source_path(task["scenario"], True))
    if source["specification"] != original["specification"]["source_snapshot"]:
        raise ValueError("E1 must use the exact original full-method source snapshot")
    specification = {
        "platform": "HighwayEnv",
        "stage": "method exploration E1",
        "sut_id": task["sut_id"],
        "scenario": task["scenario"],
        "phase": phase,
        "reference_joint": task["joint"],
        "reference_common": task["common_feedback"],
        "source_ref": str(source_path(task["scenario"], True).relative_to(ROOT)),
        "target_model_version": original["specification"]["target_model_version"],
        "configuration": E1_CONFIGURATION,
        "new_physical_executions": 0,
    }
    path = OUTPUT / "e1" / phase / task["sut_id"] / f"{task['scenario']}.json.gz"
    if path.exists():
        saved = read_json(path)
        if saved["specification"] != specification:
            raise ValueError("Saved E1 replay has a different frozen configuration")
        return str(path.relative_to(OUTPUT))
    bank = ExecutionBank(DATA / "targets/fixed" / task["sut_id"] / task["scenario"])
    if list(bank.candidates) != [Scenario(**s) for s in original["specification"]["candidates"]]:
        raise ValueError("E1 candidate coordinates differ from the original comparison")
    ledger = QueryLedger(bank.candidates, 200, bank, joint_feedback=True)
    public = {
        "candidates": bank.candidates,
        "pretrained": base,
        "sources": source,
        "initial_indices": original["result"]["selected_indices"][:20],
        "budget": 200,
        "snapshot_counts": CHECKPOINTS,
        "query_order": common["result"]["selected_indices"] if phase == "common" else None,
        "local_reliability": True,
        "device": "cpu",
        "batch_size": 128,
    }
    try:
        result = run_isolated(
            select,
            public,
            ledger,
            wall_limit_s=E1_CONFIGURATION["max_task_wall_s"],
            memory_limit_bytes=E1_CONFIGURATION["max_task_cpu_gib"] * 2**30,
        )
        labels = original["evaluation"]["candidate_collisions"]
        metrics = summarize(
            bank.candidates,
            labels,
            result["queries"],
            ledger.trajectories,
            initial_count=20,
            query_ledger=ledger.records,
        )
        if phase == "common":
            for actual, expected in zip(result["queries"], common["result"]["queries"]):
                np.testing.assert_allclose(
                    actual["prequery_component_probabilities"],
                    expected["prequery_component_probabilities"],
                    atol=1e-6,
                    rtol=0,
                )
        for snapshot in result["prediction_snapshots"]:
            snapshot["evaluation"] = prediction_metrics(snapshot, labels)
        write_json(
            path,
            {
                "specification": specification,
                "result": compact_result(result, bank.root),
                "metrics": metrics,
            },
        )
    except Exception as error:
        write_json(
            path.with_name(f"{task['scenario']}_failure.json"),
            {
                "specification": specification,
                "error": repr(error),
                "completed_queries": len(ledger.records),
                "last_index": ledger.records[-1]["index"] if ledger.records else None,
                "physical_executions": 0,
            },
        )
        raise
    return str(path.relative_to(OUTPUT))


def run_e1():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    diagnostic = read_json(OUTPUT / "e1/diagnostic.json")
    if diagnostic["status"] != "diagnostic_passed":
        raise ValueError("The local-complementarity gate must pass before E1 replay")
    tasks = task_inputs()
    register_protocol(tasks)
    completed = []
    for phase in ("common", "adaptive"):
        with ProcessPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(replay_e1, task, phase): task for task in tasks}
            for future in as_completed(pending):
                path = future.result()
                completed.append(path)
                write_json(
                    OUTPUT / "e1/progress.json",
                    {"status": "running", "phase": phase, "completed": completed},
                )
                print(phase, len(completed), path, flush=True)
    write_json(OUTPUT / "e1/progress.json", {"status": "complete", "completed": completed})


if __name__ == "__main__":
    if not (OUTPUT / "e1/diagnostic.json").exists():
        result = diagnose_e1()
        print(result["status"], result["gate_passed_tasks"], "of", len(result["tasks"]), flush=True)
    run_e1()
