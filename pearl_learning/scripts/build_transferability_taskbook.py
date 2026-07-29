"""Build a candidate frozen taskbook with enough validation geometries for calibration.

The generated taskbook is deliberately separate from the selected experiment.
It expands only ``meta_validation`` with new physical merge lengths and never
changes the canonical taskbook or any selected checkpoint.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from pearl_learning.scripts.build_taskbook import _resolve_task
from pearl_learning.src.casebook import save_casebook, validate_casebook_disjoint
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.taskbook import TASKBOOK_SCHEMA, build_taskbook, save_taskbook, validate_taskbook


def extend_validation_catalog(config: Mapping[str, Any], lengths: list[float]) -> dict[str, Any]:
    """Clone audited 40 m recipes into new, physically distinct validation maps."""
    if not lengths or any(length <= 0.0 for length in lengths) or len(set(lengths)) != len(lengths):
        raise ValueError("validation merge lengths must be unique positive values")
    result = deepcopy(dict(config))
    catalog = result["geometry_catalog"]
    geometries = list(catalog["geometries"])
    by_id = {str(item["geometry_id"]): item for item in geometries}
    rules = dict(catalog["target_contact_rules"])
    orders = dict(catalog["target_contact_entry_orders"])
    for family in ("lane_drop", "bottleneck"):
        source_id = f"{family}_40"
        if source_id not in by_id:
            raise ValueError(f"missing audited template geometry {source_id}")
        for index, length in enumerate(lengths):
            rendered = int(length) if float(length).is_integer() else str(length).replace(".", "p")
            geometry_id = f"calibration_{family}_{rendered}"
            if geometry_id in by_id:
                raise ValueError(f"duplicate calibration geometry {geometry_id}")
            item = deepcopy(by_id[source_id])
            item["geometry_id"] = geometry_id
            item["split"] = "meta_validation"
            item["map_config"]["merge_length_m"] = float(length)
            geometries.append(item)
            # Alternating frozen rules ensure that validation includes both
            # support-identifiable target-contact relations.
            adversary_first = (index + (0 if family == "lane_drop" else 1)) % 2 == 0
            orders[geometry_id] = "adversary_first" if adversary_first else "sut_first"
            rules[geometry_id] = {
                "target_contact_speed_relation": "adversary_faster" if adversary_first else "sut_faster",
                "target_contact_speed_margin_mps": 3.0,
            }
    catalog["geometries"] = geometries
    catalog["target_contact_rules"] = rules
    catalog["target_contact_entry_orders"] = orders
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", nargs="+", type=float, default=[36.0, 44.0, 52.0])
    args = parser.parse_args()
    config = extend_validation_catalog(read_config(args.config), list(args.lengths))
    candidate = build_taskbook(config)
    resolved: dict[str, list[Any]] = {split: [] for split in candidate}
    casebooks: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, tasks in candidate.items():
        for task in tasks:
            frozen, book = _resolve_task(task, config)
            resolved[split].append(frozen)
            casebooks[frozen.task_id] = book
    validate_taskbook(resolved)
    validate_casebook_disjoint(casebooks)
    root = Path(args.output)
    digest = save_taskbook(resolved, root)
    case_root = root.parent
    case_hashes = {
        task_id: save_casebook(next(task for tasks in resolved.values() for task in tasks if task.task_id == task_id), book, str(case_root))
        for task_id, book in casebooks.items()
    }
    write_json(root / "taskbook_provenance.json", {
        "schema": TASKBOOK_SCHEMA,
        "task_schema": "logical_merge_task",
        "taskbook_hash": digest,
        "casebook_hashes": case_hashes,
        "geometry_catalog_hash": content_hash(config["geometry_catalog"]),
        "purpose": "transferability_validation_expansion",
        "validation_merge_lengths_m": [float(length) for length in args.lengths],
    })
    print(f"wrote transferability candidate taskbook sha256={digest} with {len(resolved['meta_validation'])} validation tasks")


if __name__ == "__main__":
    main()
