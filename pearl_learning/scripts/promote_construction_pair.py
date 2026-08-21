"""Freeze a passed construction pair into a fresh Gate-only taskbook."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pearl_learning.src.io import content_hash, file_sha256, prepare_run_manifest, read_config, write_json
from pearl_learning.src.taskbook import TASKBOOK_SCHEMA, load_taskbook, save_taskbook, taskbook_payload, validate_taskbook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    selection = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
    if selection.get("schema") != "physical_task_construction_selection_v1" or selection.get("status") != "pass":
        raise ValueError("promotion requires a passed immutable construction selection manifest")
    selected = [str(value) for value in selection["selected_pair"]["geometry_ids"]]
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError("selection manifest must contain exactly two geometries")
    parent = load_taskbook(args.taskbook)
    all_tasks = [task for tasks in parent.values() for task in tasks]
    if {task.geometry_id for task in all_tasks if task.geometry_id in selected} != set(selected):
        raise ValueError("selected pair is absent from the frozen construction taskbook")
    promoted = {split: [] for split in parent}
    promoted_task_ids: list[str] = []
    for split, tasks in parent.items():
        for task in tasks:
            if task.geometry_id in selected:
                task = replace(task, split="meta_train", task_id=f"meta_train_{task.geometry_id}")
                promoted_task_ids.append(task.task_id)
                promoted["meta_train"].append(task)
            else:
                promoted[split].append(task)
    validate_taskbook(promoted)
    root = Path(args.output)
    manifest = {
        "schema": "physical_task_construction_promotion_v1", "run_name": "physical_gate_v2_taskbook",
        "source_config_sha256": file_sha256(args.config), "resolved_config_sha256": content_hash(cfg),
        "taskbook_hash": content_hash(taskbook_payload(parent)), "casebook_hashes": {},
        "critical_threshold_hash": "construction_only_calibration_pending",
        "selection_hash": selection["selection_hash"], "selected_geometry_ids": selected,
        "promoted_task_ids": promoted_task_ids,
    }
    prepare_run_manifest(root, manifest, resume=False)
    digest = save_taskbook(promoted, root / "taskbooks")
    write_json(root / "taskbooks" / "taskbook_provenance.json", {
        "schema": TASKBOOK_SCHEMA, "task_schema": "logical_merge_task", "taskbook_hash": digest,
        "parent_taskbook_hash": manifest["taskbook_hash"], "selection_hash": selection["selection_hash"],
        "selected_geometry_ids": selected, "promoted_task_ids": promoted_task_ids,
    })
    write_json(root / "promotion_manifest.json", {**manifest, "promoted_taskbook_hash": digest})
    print(f"promoted construction pair: {selected}")


if __name__ == "__main__":
    main()
