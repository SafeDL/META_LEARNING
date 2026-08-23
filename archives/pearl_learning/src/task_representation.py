"""Supervision targets for an optional disentangled PEARL task posterior."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .observation import OBS_FIELDS


INTERACTION_OBSERVATION_FIELDS = (
    "arrival_time_difference",
    "relative_route_speed",
    "ttc",
)
INTERACTION_OBSERVATION_INDEXES = tuple(OBS_FIELDS.index(name) for name in INTERACTION_OBSERVATION_FIELDS)


def representation_target(task: Any) -> dict[str, np.ndarray]:
    """Return training-only semantic targets; task IDs and hashes are excluded."""
    map_config, conflict, priority = dict(task.map_config), dict(task.conflict_spec), dict(task.priority_spec)
    geometry = np.asarray([
        float(map_config.get("bottle_lane_num", 0.0)) / 5.0,
        float(map_config.get("neck_lane_num", 0.0)) / 5.0,
        float(map_config.get("merge_length_m", 0.0)) / 100.0,
        float(conflict["conflict_radius_m"]) / 10.0,
        float(conflict.get("max_route_distance_m", 0.0)) / 10.0,
    ], dtype=np.float32)
    order = str(priority.get("target_contact_entry_order", "any"))
    if order not in {"adversary_first", "sut_first"}:
        raise ValueError("disentangled rule supervision requires a frozen binary entry order")
    return {"geometry": geometry, "entry_order": np.asarray(float(order == "adversary_first"), dtype=np.float32)}


def configure_disentangled_representation(config: Mapping[str, Any], *, enabled: bool, latent_dims: list[int], geometry_weight: float, interaction_weight: float, rule_weight: float) -> dict[str, Any]:
    """Return an explicit experimental configuration without changing defaults."""
    if not enabled:
        return dict(config)
    if len(latent_dims) != 3 or any(int(value) < 1 for value in latent_dims):
        raise ValueError("representation latent dims must contain three positive integers")
    if min(geometry_weight, interaction_weight, rule_weight) < 0.0:
        raise ValueError("disentangled auxiliary weights must be non-negative")
    return dict(config) | {"task_representation": {
        "enabled": True, "latent_dims": [int(value) for value in latent_dims],
        "geometry_weight": float(geometry_weight), "interaction_weight": float(interaction_weight), "rule_weight": float(rule_weight),
    }}
