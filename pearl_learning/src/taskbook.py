"""Frozen, hashed logical-scenario taskbooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import content_hash, write_json
from .task_spec import LogicalScenarioTaskSpec


SPLIT_LAYOUT = {
    "meta_train": {"on_ramp_merge": 4, "lane_drop_merge": 4, "bottleneck_merge": 4},
    "meta_validation": {"on_ramp_merge": 1, "lane_drop_merge": 1, "bottleneck_merge": 1},
    "meta_test_template": {"on_ramp_merge": 1, "lane_drop_merge": 1, "bottleneck_merge": 1},
    "meta_test_logical": {"y_merge": 4},
}


def _map_config(logical_type: str, template: int) -> dict[str, Any]:
    # These are physical map recipes, never learned labels.  The topology
    # audit resolves every lane graph before a taskbook may be used in training.
    recipes = {
        "on_ramp_merge": {"kind": "in_ramp", "map": "SrS", "lane_num": 2},
        "lane_drop_merge": {"kind": "lane_drop", "bottle_lane_num": 3, "neck_lane_num": 2},
        "bottleneck_merge": {"kind": "bottleneck", "bottle_lane_num": 3, "neck_lane_num": 1},
        "y_merge": {"kind": "y_merge", "map": "r", "lane_num": 1},
    }
    recipe = dict(recipes[logical_type])
    recipe["template_index"] = template
    recipe["merge_length_m"] = 32.0 + 4.0 * (template % 4)
    return recipe


def build_taskbook(config: Mapping[str, Any]) -> dict[str, list[LogicalScenarioTaskSpec]]:
    """Generate the prescribed, non-overlapping Stage 2 split structure."""
    result: dict[str, list[LogicalScenarioTaskSpec]] = {}
    seed = int(config.get("experiment", {}).get("taskbook_seed", 7301))
    for split, by_type in SPLIT_LAYOUT.items():
        tasks: list[LogicalScenarioTaskSpec] = []
        offset = 0
        for logical_type, count in by_type.items():
            for local_index in range(count):
                template = offset + local_index
                task = LogicalScenarioTaskSpec(
                    task_id=f"{split}_{logical_type}_{local_index:02d}", split=split, logical_type=logical_type,
                    map_config=_map_config(logical_type, template),
                    conflict_spec={"conflict_radius_m": 6.0, "merge_length_m": 32.0 + 4.0 * (template % 4)},
                    case_seed=seed + 1009 * len(tasks) + 17 * template,
                )
                task.validate()
                tasks.append(task)
            offset += count
        result[split] = tasks
    validate_taskbook(result)
    return result


def taskbook_payload(taskbook: Mapping[str, Iterable[LogicalScenarioTaskSpec]]) -> dict[str, list[dict[str, Any]]]:
    return {split: [task.to_dict() for task in tasks] for split, tasks in taskbook.items()}


def validate_taskbook(taskbook: Mapping[str, Iterable[LogicalScenarioTaskSpec]]) -> None:
    expected = {key: sum(value.values()) for key, value in SPLIT_LAYOUT.items()}
    seen_ids: set[str] = set()
    for split, expected_count in expected.items():
        tasks = list(taskbook.get(split, []))
        if len(tasks) != expected_count:
            raise ValueError(f"{split} expected {expected_count} tasks, got {len(tasks)}")
        for task in tasks:
            task.validate()
            if task.task_id in seen_ids:
                raise ValueError(f"duplicate task id: {task.task_id}")
            if task.split != split:
                raise ValueError(f"task {task.task_id} is in wrong split")
            seen_ids.add(task.task_id)
    held_out = {task.logical_type for task in taskbook["meta_test_logical"]}
    train = {task.logical_type for task in taskbook["meta_train"]}
    if held_out & train:
        raise ValueError("held-out logical type leaked into meta-train")


def save_taskbook(taskbook: Mapping[str, Iterable[LogicalScenarioTaskSpec]], output_dir: str | Path) -> str:
    payload = taskbook_payload(taskbook)
    digest = content_hash(payload)
    root = Path(output_dir)
    for split, tasks in payload.items():
        write_json(root / f"{split}_tasks.json", tasks)
    write_json(root / "taskbook_hash.json", {"sha256": digest})
    return digest

