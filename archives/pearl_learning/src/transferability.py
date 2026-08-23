"""Leakage-safe task descriptors and meta-training coverage diagnostics.

The score in this module is deliberately a *descriptor-space coverage* score.
It is not a learned prediction of PEARL's query performance.  A calibrated
transferability model needs task-level, budget-matched adaptation outcomes from
the validation split; this module creates the frozen, pre-adaptation inputs for
that later calibration without reading query cases or query metrics.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import math

import numpy as np

from .io import content_hash


TASK_DESCRIPTOR_SCHEMA = "logical_merge_task_descriptor_v1"
TRANSFERABILITY_REPORT_SCHEMA = "logical_merge_transferability_report_v1"
_LOGICAL_TYPES = ("on_ramp_merge", "lane_drop_merge", "bottleneck_merge", "y_merge")
_MAP_KINDS = ("on_ramp_srs", "bottleneck", "y_merge")
_SUPPORT_SPLIT = {
    "meta_train": "train_pool",
    "meta_validation": "validation_support",
    "meta_test_template": "test_support",
    "meta_test_logical": "test_support",
}


def support_case_group(split: str) -> str:
    """Return the only case group usable before query evaluation for ``split``."""
    try:
        return _SUPPORT_SPLIT[str(split)]
    except KeyError as exc:
        raise ValueError(f"unsupported task split for transferability: {split!r}") from exc


def _one_hot(value: str, options: tuple[str, ...]) -> list[float]:
    return [1.0 if value == option else 0.0 for option in options]


def _lane_index(route: Mapping[str, Any]) -> float:
    lane = route["initial_lane"]
    return float(lane[2])


def _route_lane_changes(route: Mapping[str, Any]) -> float:
    lanes = list(route["lane_sequence"])
    return float(sum(lanes[index][2] != lanes[index - 1][2] for index in range(1, len(lanes))))


def _case_values(cases: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    rows = [dict(case) for case in cases]
    if not rows:
        raise ValueError("a task descriptor requires at least one allowed support case")
    required = ("adversary_speed_mps", "adversary_spawn_m", "sut_spawn_m")
    if any(any(key not in row for key in required) for row in rows):
        raise ValueError("support cases lack required pre-adaptation fields")
    speed = np.asarray([float(row["adversary_speed_mps"]) for row in rows], dtype=float)
    adversary_spawn = np.asarray([float(row["adversary_spawn_m"]) for row in rows], dtype=float)
    sut_spawn = np.asarray([float(row["sut_spawn_m"]) for row in rows], dtype=float)
    if not all(np.all(np.isfinite(value)) for value in (speed, adversary_spawn, sut_spawn)):
        raise ValueError("support case descriptor values must be finite")
    summary: dict[str, float] = {}
    for name, values, scale in (
        ("adversary_speed", speed, 30.0),
        ("adversary_spawn", adversary_spawn, 100.0),
        ("sut_spawn", sut_spawn, 100.0),
        ("spawn_difference", adversary_spawn - sut_spawn, 100.0),
    ):
        summary[f"{name}_mean"] = float(values.mean() / scale)
        summary[f"{name}_std"] = float(values.std(ddof=0) / scale)
    return summary


def task_descriptor(task: Any, support_cases: Iterable[Mapping[str, Any]], *, include_hidden_rules: bool = False) -> dict[str, Any]:
    """Create a semantic descriptor without task IDs, hashes, or query data.

    ``target_contact_speed_relation`` and ``target_contact_entry_order`` are
    intentionally hidden from the policy observation contract.  They therefore
    stay out of the default descriptor and can only be included for explicitly
    marked oracle/offline explanations.
    """
    map_config = dict(task.map_config)
    priority = dict(task.priority_spec)
    conflict = dict(task.conflict_spec)
    geometry_categorical = {
        "logical_type": _one_hot(str(task.logical_type), _LOGICAL_TYPES),
        "map_kind": _one_hot(str(map_config.get("kind", "")), _MAP_KINDS),
    }
    geometry_continuous = {
        "bottle_lane_count": float(map_config.get("bottle_lane_num", 0.0)) / 5.0,
        "neck_lane_count": float(map_config.get("neck_lane_num", 0.0)) / 5.0,
        "merge_length": float(map_config.get("merge_length_m", 0.0)) / 100.0,
        "adversary_route_segments": float(len(task.adversary_route["lane_sequence"])) / 8.0,
        "sut_route_segments": float(len(task.sut_route["lane_sequence"])) / 8.0,
        "adversary_initial_lane": _lane_index(task.adversary_route) / 5.0,
        "sut_initial_lane": _lane_index(task.sut_route) / 5.0,
        "adversary_lane_changes": _route_lane_changes(task.adversary_route) / 5.0,
        "sut_lane_changes": _route_lane_changes(task.sut_route) / 5.0,
        "conflict_radius": float(conflict["conflict_radius_m"]) / 10.0,
        "max_route_distance": float(conflict.get("max_route_distance_m", 0.0)) / 10.0,
        "adversary_spawn_midpoint": float(sum(task.spawn_regions["adversary"]) / 2.0) / 100.0,
        "adversary_spawn_width": float(task.spawn_regions["adversary"][1] - task.spawn_regions["adversary"][0]) / 100.0,
        "sut_spawn_midpoint": float(sum(task.spawn_regions["sut"]) / 2.0) / 100.0,
        "sut_spawn_width": float(task.spawn_regions["sut"][1] - task.spawn_regions["sut"][0]) / 100.0,
    }
    rules: dict[str, Any] = {"visible_priority": {"sut_has_priority": [float(bool(priority["sut_has_priority"]))]}}
    if include_hidden_rules:
        rules["oracle_hidden"] = {
            "speed_relation": _one_hot(str(priority.get("target_contact_speed_relation", "any")), ("any", "adversary_faster", "sut_faster")),
            "entry_order": _one_hot(str(priority.get("target_contact_entry_order", "any")), ("any", "adversary_first", "sut_first")),
            "speed_margin": [float(priority.get("target_contact_speed_margin_mps", 0.0)) / 10.0],
        }
    return {
        "schema": TASK_DESCRIPTOR_SCHEMA,
        "task_id": str(task.task_id),
        "task_split": str(task.split),
        "case_group": support_case_group(str(task.split)),
        "uses_query_cases": False,
        "uses_hidden_rules": bool(include_hidden_rules),
        "groups": {
            "geometry": {"categorical": geometry_categorical, "continuous": geometry_continuous},
            "interaction_prior": {"continuous": _case_values(support_cases)},
            "rules": rules,
        },
    }


def _flatten(values: Mapping[str, Any]) -> np.ndarray:
    flat: list[float] = []
    for key in sorted(values):
        value = values[key]
        if isinstance(value, Mapping):
            flat.extend(_flatten(value).tolist())
        elif isinstance(value, (list, tuple)):
            flat.extend(float(item) for item in value)
        else:
            flat.append(float(value))
    result = np.asarray(flat, dtype=float)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("descriptor group must contain finite values")
    return result


def _group_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    first, second = _flatten(left), _flatten(right)
    if first.shape != second.shape:
        raise ValueError("descriptor group shapes differ")
    return float(np.sqrt(np.mean(np.square(first - second))))


def descriptor_distance(left: Mapping[str, Any], right: Mapping[str, Any], *, weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Return per-group and weighted descriptor distances.

    Values are scaled by physical units when descriptors are built, so the
    distance is comparable across task groups without fitting on query tasks.
    """
    if left.get("schema") != TASK_DESCRIPTOR_SCHEMA or right.get("schema") != TASK_DESCRIPTOR_SCHEMA:
        raise ValueError("unsupported task descriptor schema")
    if bool(left.get("uses_hidden_rules")) != bool(right.get("uses_hidden_rules")):
        raise ValueError("cannot compare descriptors with different rule visibility")
    chosen = {"geometry": 0.5, "interaction_prior": 0.4, "rules": 0.1} | dict(weights or {})
    if any(value < 0.0 for value in chosen.values()) or not any(chosen.values()):
        raise ValueError("descriptor weights must be non-negative with a positive total")
    group_distances = {name: _group_distance(left["groups"][name], right["groups"][name]) for name in chosen}
    total = math.sqrt(sum(chosen[name] * value * value for name, value in group_distances.items()) / sum(chosen.values()))
    return group_distances | {"total": float(total)}


def transferability_report(taskbook: Mapping[str, list[Any]], casebooks: Mapping[str, Mapping[str, list[Mapping[str, Any]]]], *, candidate_split: str, include_hidden_rules: bool = False, similarity_temperature: float = 0.5) -> dict[str, Any]:
    """Report meta-train coverage for each candidate task without query access."""
    if candidate_split not in taskbook or "meta_train" not in taskbook:
        raise ValueError("taskbook lacks required meta_train or candidate split")
    if similarity_temperature <= 0.0:
        raise ValueError("similarity_temperature must be positive")
    reference = list(taskbook["meta_train"])
    candidates = list(taskbook[candidate_split])
    if not reference or not candidates:
        raise ValueError("reference and candidate task sets must be non-empty")

    def describe(task: Any) -> dict[str, Any]:
        if task.task_id not in casebooks:
            raise ValueError(f"casebook is missing for {task.task_id}")
        group = support_case_group(task.split)
        return task_descriptor(task, casebooks[task.task_id][group], include_hidden_rules=include_hidden_rules)

    reference_descriptors = {task.task_id: describe(task) for task in reference}
    train_logical_types = {str(task.logical_type) for task in reference}
    train_map_kinds = {str(task.map_config.get("kind", "")) for task in reference}
    candidates_payload = []
    for task in candidates:
        descriptor = describe(task)
        ranked = []
        for train_task in reference:
            distance = descriptor_distance(descriptor, reference_descriptors[train_task.task_id])
            ranked.append({"task_id": train_task.task_id, "distance": distance, "similarity": float(math.exp(-distance["total"] ** 2 / (2.0 * similarity_temperature ** 2)))})
        ranked.sort(key=lambda row: (row["distance"]["total"], row["task_id"]))
        candidates_payload.append({
            "task_id": task.task_id,
            "descriptor": descriptor,
            "nearest_meta_train": ranked[0],
            "top_meta_train_neighbors": ranked[:3],
            "coverage_flags": {
                "unseen_logical_type": str(task.logical_type) not in train_logical_types,
                "unseen_map_kind": str(task.map_config.get("kind", "")) not in train_map_kinds,
                "uses_only_pre_adaptation_inputs": True,
            },
        })
    return {
        "schema": TRANSFERABILITY_REPORT_SCHEMA,
        "status": "diagnostic_only_not_calibrated",
        "candidate_split": candidate_split,
        "taskbook_hash": content_hash({split: [task.to_dict() for task in tasks] for split, tasks in taskbook.items()}),
        "reference_split": "meta_train",
        "reference_task_count": len(reference),
        "descriptor_schema": TASK_DESCRIPTOR_SCHEMA,
        "uses_hidden_rules": bool(include_hidden_rules),
        "similarity_temperature": float(similarity_temperature),
        "limitations": [
            "Similarity measures coverage in the declared descriptor space, not a calibrated prediction of adaptation gain.",
            "No query case, query metric, task ID, or task hash is used as a descriptor feature.",
            "Calibration requires task-level, equal-new-task-budget outcomes on a validation split.",
        ],
        "candidates": candidates_payload,
    }
