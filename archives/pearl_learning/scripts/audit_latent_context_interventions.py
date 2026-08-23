from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from archives.pearl_learning.src.benchmark_calibration import apply_calibration_manifest
from archives.pearl_learning.src.casebook import CASEBOOK_SCHEMA, load_casebook
from archives.pearl_learning.src.causal_audit import audit_task_context_interventions
from archives.pearl_learning.src.checkpoint import load_checkpoint
from archives.pearl_learning.src.io import read_config, write_json
from archives.pearl_learning.src.pearl_agent import PEARLAgent
from archives.pearl_learning.src.taskbook import load_taskbook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--split", choices=["meta_validation", "meta_test_template", "meta_test_logical"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
    cfg = apply_calibration_manifest(read_config(args.config), manifest)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["experiment"].get("device") != "cpu" else "cpu")
    agent = PEARLAgent(int(cfg["environment"]["observation_dim"]), int(cfg["environment"]["action_dim"]), cfg, device)
    load_checkpoint(args.checkpoint, agent, device)
    taskbook = load_taskbook(args.taskbook)
    all_tasks = [task for tasks in taskbook.values() for task in tasks]
    by_geometry = {task.geometry_id: task for task in all_tasks}
    target_ids = set(cfg["method_flow_pilot"]["task_ids"].get(args.split, []))
    if args.split == "meta_test_logical":
        target_ids = {"y_merge_32"}
    targets = [task for task in taskbook[args.split] if task.geometry_id in target_ids]
    pair = {
        "lane_drop_40": "bottleneck_40", "bottleneck_40": "lane_drop_40",
        "lane_drop_48": "bottleneck_48", "bottleneck_48": "lane_drop_48",
        "y_merge_32": "bottleneck_40",
    }
    needed = {task.task_id: task for task in targets}
    for task in targets:
        wrong = by_geometry[pair[task.geometry_id]]
        needed[wrong.task_id] = wrong
    books = {
        task_id: load_casebook(task, args.casebook_root, required_schema=CASEBOOK_SCHEMA)
        for task_id, task in needed.items()
    }
    results = {}
    for task in targets:
        wrong = by_geometry[pair[task.geometry_id]]
        wrong_split = "meta_validation" if wrong.split == "meta_validation" else args.split
        results[task.task_id] = audit_task_context_interventions(
            agent, cfg, task, wrong, books[task.task_id], books[wrong.task_id],
            split=args.split,
            wrong_support_key=("validation_support" if wrong_split == "meta_validation" else "test_support"),
        )
        results[task.task_id]["wrong_evidence_source_split"] = wrong_split
    write_json(args.output, {
        "schema": "latent_context_causal_audit_suite_v1",
        "critical_metric_schema": manifest["critical_metric_schema"],
        "calibration_hash": manifest["calibration_hash"],
        "tasks": results,
    })


if __name__ == "__main__":
    main()
