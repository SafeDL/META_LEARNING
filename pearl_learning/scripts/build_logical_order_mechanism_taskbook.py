"""Derive a two-rule, physically matched task pair for the mechanism gate."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pearl_learning.src.io import content_hash, write_json
from pearl_learning.src.taskbook import (
    TASKBOOK_SCHEMA,
    load_taskbook,
    save_taskbook,
    taskbook_payload,
    validate_taskbook,
)


def derive_logical_order_taskbook(parent: dict, geometry_id: str) -> tuple[dict, list]:
    """Replace one frozen physical task with two rule-only variants."""
    matches = [task for task in parent["meta_train"] if task.geometry_id == geometry_id]
    if len(matches) != 1:
        raise ValueError("--geometry-id must identify one non-variant frozen meta-train geometry")
    base = matches[0]
    variants = []
    for order in ("adversary_first", "sut_first"):
        variant_geometry_id = f"{base.geometry_id}__logical_order_{order}"
        priority = {
            **base.priority_spec,
            "target_contact_entry_order": order,
            "target_contact_entry_order_semantics": "continuous_route_entry_interpolation",
            "target_contact_speed_relation": "any",
            "target_contact_speed_margin_mps": 0.0,
        }
        variants.append(replace(
            base,
            task_id=f"meta_train_{variant_geometry_id}",
            geometry_id=variant_geometry_id,
            priority_spec=priority,
            # A distinct seed affects only case sampling; routes/map hashes
            # intentionally remain identical to make the pair physical match.
            case_seed=int(content_hash({"parent": base.task_id, "logical_order": order})[:16], 16) % (2**31 - 2) + 1,
        ))
    derived = {split: list(tasks) for split, tasks in parent.items()}
    derived["meta_train"] = [task for task in derived["meta_train"] if task.task_id != base.task_id] + variants
    validate_taskbook(derived)
    return derived, variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-taskbook", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-id", default="lane_drop_24")
    args = parser.parse_args()
    parent = load_taskbook(args.parent_taskbook)
    derived, variants = derive_logical_order_taskbook(parent, args.geometry_id)
    base = next(task for task in parent["meta_train"] if task.geometry_id == args.geometry_id)
    parent_hash = content_hash(taskbook_payload(parent))
    digest = save_taskbook(derived, args.output)
    write_json(Path(args.output) / "taskbook_provenance.json", {
        "schema": TASKBOOK_SCHEMA,
        "task_schema": "logical_merge_task",
        "taskbook_hash": digest,
        "parent_taskbook_hash": parent_hash,
        "purpose": "logical_order_mechanism_identifiability",
        "logical_rule_hidden_from_network_inputs": True,
        "derived_geometry_id": base.geometry_id,
        "variant_task_ids": [task.task_id for task in variants],
    })
    print(f"wrote logical-order mechanism taskbook sha256={digest}")


if __name__ == "__main__":
    main()
