"""Apply the predeclared K=4 versus K=0 Vanilla physical-pilot decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from pearl_learning.src.io import content_hash, read_config, write_json


def _success_count(summary: Mapping[str, Any]) -> int:
    episodes = int(summary["episodes"])
    count = float(summary["valid_critical_strict_rate"]) * episodes
    rounded = int(round(count))
    if abs(count - rounded) > 1e-7:
        raise ValueError("VCSR is inconsistent with the recorded episode count")
    return rounded


def vanilla_adaptation_report(
    suite: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    if suite.get("split") != "meta_validation":
        raise ValueError("Vanilla pilot decision requires meta_validation results")
    regime = suite.get("evaluation_regimes", {}).get("validation_known_logical_type")
    if not isinstance(regime, Mapping):
        raise ValueError("validation evaluation regime is missing")
    primary_mode = str(decision["primary_query_execution_mode"])
    mode = regime.get("query_modes", {}).get(primary_mode)
    if not isinstance(mode, Mapping):
        raise ValueError("primary deterministic-mean query mode is missing")
    expected_step = int(decision["fixed_checkpoint_step"])
    actual_step = int(mode.get("provenance", {}).get("checkpoint_step", -1))
    if actual_step != expected_step:
        raise ValueError(
            f"decision requires fixed step {expected_step}, received checkpoint step {actual_step}"
        )
    tasks = dict(mode.get("tasks", {}))
    expected_tasks = int(decision["expected_validation_tasks"])
    query_cases = int(decision["query_cases_per_task"])
    if len(tasks) != expected_tasks:
        raise ValueError("Vanilla pilot result has the wrong validation-task count")
    baseline_shot = str(int(decision["baseline_shot"]))
    decision_shot = str(int(decision["decision_shot"]))
    rows = []
    baseline_total = decision_total = 0
    for task_id, shots in tasks.items():
        if baseline_shot not in shots or decision_shot not in shots:
            raise ValueError(f"task {task_id} lacks a predeclared decision shot")
        baseline = shots[baseline_shot]["summary"]
        adapted = shots[decision_shot]["summary"]
        if int(baseline["episodes"]) != query_cases or int(adapted["episodes"]) != query_cases:
            raise ValueError("each validation task must use exactly four paired query cases")
        baseline_count = _success_count(baseline)
        adapted_count = _success_count(adapted)
        baseline_total += baseline_count
        decision_total += adapted_count
        rows.append({
            "task_id": task_id,
            "k0_valid_critical_count": baseline_count,
            "k4_valid_critical_count": adapted_count,
            "valid_critical_count_gain": adapted_count - baseline_count,
            "k0_summary": dict(baseline),
            "k4_summary": dict(adapted),
        })
    total_queries = expected_tasks * query_cases
    gain = decision_total - baseline_total
    minimum_gain = int(decision["minimum_aggregate_success_count_gain"])
    passed = gain >= minimum_gain
    return {
        "schema": "vanilla_physical_pearl_adaptation_gate_v1",
        "status": "pass" if passed else "fail",
        "hard_gate_metric": "aggregate paired valid-critical success count",
        "primary_query_execution_mode": primary_mode,
        "fixed_checkpoint_step": expected_step,
        "baseline_shot": int(baseline_shot),
        "decision_shot": int(decision_shot),
        "total_query_cases": total_queries,
        "k0_valid_critical_count": baseline_total,
        "k4_valid_critical_count": decision_total,
        "aggregate_success_count_gain": gain,
        "aggregate_vcsr_gain": float(gain / total_queries),
        "minimum_aggregate_success_count_gain": minimum_gain,
        "diagnostic_metrics": [
            "mean_episode_return", "invalid_rate", "target_collision_rate",
            "episodes_to_first_valid_critical", "environment_steps_to_first_valid_critical",
        ],
        "tasks": rows,
        "next_allowed_stage": "structure_aware_pearl" if passed else None,
        "failure_action": (
            None if passed else "audit_support_to_posterior_and_posterior_to_actor; do_not_enable_structure_aware"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    suite = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    report = vanilla_adaptation_report(suite, config["vanilla_pilot_decision"])
    report["inputs"] = {
        "evaluation_path": str(Path(args.evaluation).resolve()),
        "evaluation_hash": content_hash(suite),
    }
    write_json(args.output, report)
    print(f"Vanilla physical adaptation: {report['status']}")
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
