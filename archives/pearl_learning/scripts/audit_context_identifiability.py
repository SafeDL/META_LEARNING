"""Run Gate 2 with a fixed probing policy and transition-only features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from archives.pearl_learning.src.benchmark_calibration import apply_calibration_manifest
from archives.pearl_learning.src.casebook import MECHANISM_CASEBOOK_SCHEMA, load_casebook
from archives.pearl_learning.src.context_identifiability import (
    collect_probe_trajectory,
    context_identifiability_report,
)
from archives.pearl_learning.src.io import content_hash, read_config, write_json
from archives.pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _select_tasks(taskbook, requested: list[str]):
    wanted = set(map(str, requested))
    tasks = [task for split in taskbook.values() for task in split if task.task_id in wanted or task.geometry_id in wanted]
    if len(tasks) != 2 or len(tasks) != len(wanted):
        raise ValueError("Gate 2 requires exactly two unique frozen task ids/geometry ids")
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
    parser.add_argument("--probing-policy", default="P6_arrival_gap_heuristic")
    args = parser.parse_args()
    manifest = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
    config = apply_calibration_manifest(read_config(args.config), manifest)
    if not bool(config.get("control", {}).get("mechanism_longitudinal_only", False)):
        raise ValueError("Gate 2 requires the same 1-D route-tracker action contract as Gate 1")
    taskbook = load_taskbook(args.taskbook)
    tasks = _select_tasks(taskbook, args.task_id)
    trajectories = {}
    casebook_hashes = {}
    for task in tasks:
        book = load_casebook(task, args.casebook_root, required_schema=MECHANISM_CASEBOOK_SCHEMA)
        cases = list(book[args.split])
        casebook_hashes[task.task_id] = content_hash(book)
        trajectories[task.task_id] = [
            collect_probe_trajectory(task, case, config, args.probing_policy)
            for case in cases
        ]
    task_ids = [task.task_id for task in tasks]
    metrics, distance, gate = context_identifiability_report(task_ids, trajectories)
    provenance = {
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": casebook_hashes,
        "config_hash": content_hash(config),
        "critical_threshold_hash": manifest["calibration_hash"],
        "probing_policy": args.probing_policy,
        "fixed_policy_across_tasks": True,
    }
    metrics["provenance"] = provenance
    distance["provenance"] = provenance
    gate["inputs"] = provenance
    root = Path(args.output)
    write_json(root / "context_probe_metrics.json", metrics)
    write_json(root / "context_feature_distance.json", distance)
    write_json(root / "context_identifiability_gate.json", gate)
    print(f"Gate 2 context identifiability: {gate['status']} (accuracy={metrics['held_out_accuracy']:.3f})")


if __name__ == "__main__":
    main()
