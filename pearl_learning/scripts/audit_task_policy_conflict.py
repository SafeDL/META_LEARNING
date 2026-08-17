"""Run Gate 1: matched scripted longitudinal policy conflict audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pearl_learning.src.benchmark_calibration import apply_calibration_manifest
from pearl_learning.src.casebook import MECHANISM_CASEBOOK_SCHEMA, load_casebook
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.mechanism_audit import (
    SCRIPTED_POLICIES,
    policy_conflict_report,
    rollout_scripted_policy,
)
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _select_tasks(taskbook, requested: list[str]):
    wanted = set(map(str, requested))
    tasks = [task for split in taskbook.values() for task in split if task.task_id in wanted or task.geometry_id in wanted]
    if len(tasks) != 2 or len(tasks) != len(wanted):
        raise ValueError("Gate 1 requires exactly two unique frozen task ids/geometry ids")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--split", default="train_pool")
    parser.add_argument(
        "--metric-profile-policy",
        choices=("taskwise_validation_calibrated", "common_validation_componentwise_strictest"),
        default="taskwise_validation_calibrated",
        help="taskwise profiles establish within-task feasibility; common strictest is a sensitivity diagnostic",
    )
    args = parser.parse_args()
    manifest = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
    config = apply_calibration_manifest(read_config(args.config), manifest)
    if args.metric_profile_policy == "common_validation_componentwise_strictest":
        # Sensitivity mode: remove logical-type profiles and evaluate both
        # tasks under the same validation-derived strictest threshold triple.
        config["critical_metric"].pop("threshold_profiles", None)
    if not bool(config.get("control", {}).get("mechanism_longitudinal_only", False)):
        raise ValueError("Gate 1 requires control.mechanism_longitudinal_only=true")
    if int(config["environment"]["action_dim"]) != 1:
        raise ValueError("Gate 1 requires environment.action_dim=1")
    taskbook = load_taskbook(args.taskbook)
    tasks = _select_tasks(taskbook, args.task_id)
    rows = []
    casebook_hashes = {}
    for task in tasks:
        book = load_casebook(task, args.casebook_root, required_schema=MECHANISM_CASEBOOK_SCHEMA)
        cases = list(book[args.split])
        if not cases:
            raise ValueError(f"mechanism casebook {task.task_id} has no cases in {args.split}")
        casebook_hashes[task.task_id] = content_hash(book)
        for case in cases:
            for policy in SCRIPTED_POLICIES:
                rows.append(rollout_scripted_policy(task, case, config, policy))
    task_ids = [task.task_id for task in tasks]
    matrix, rankings, gate = policy_conflict_report(rows, task_ids)
    provenance = {
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": casebook_hashes,
        "config_hash": content_hash(config),
        "critical_threshold_hash": manifest["calibration_hash"],
        "action_contract": "longitudinal_only_1d_with_deterministic_route_tracker",
        "metric_profile_policy": args.metric_profile_policy,
    }
    matrix["provenance"] = provenance
    rankings["provenance"] = provenance
    gate["inputs"] = provenance
    root = Path(args.output)
    write_json(root / "scripted_policy_matrix.json", matrix)
    write_json(root / "scripted_policy_case_rankings.json", rankings)
    # These compact per-rollout records preserve the raw evidence needed to
    # recompute Gate 1 if a future criterion changes, without silently
    # rerunning a different simulation.
    write_json(root / "scripted_policy_rollouts.json", {"schema": "scripted_policy_rollouts_v1", "provenance": provenance, "rows": rows})
    write_json(root / "policy_conflict_gate.json", gate)
    print(f"Gate 1 task-policy conflict: {gate['status']} ({gate['criterion_count']}/4 criteria)")


if __name__ == "__main__":
    main()
