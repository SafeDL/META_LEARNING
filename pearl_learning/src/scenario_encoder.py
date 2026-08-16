"""Static logical-scenario descriptors and a task-conditioned PEARL prior.

The descriptor intentionally reads only the frozen task-level geometry.  It
never receives case values, replay transitions, rewards, or hidden-rule labels.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import torch
from torch import nn


DESCRIPTOR_SCHEMA = "logical_merge_static_descriptor_v1"
LOGICAL_TYPES = ("on_ramp_merge", "lane_drop_merge", "bottleneck_merge", "y_merge")
DESCRIPTOR_FIELDS = (
    "logical_type:on_ramp_merge", "logical_type:lane_drop_merge",
    "logical_type:bottleneck_merge", "logical_type:y_merge", "bottle_lane_num",
    "neck_lane_num", "merge_length_m", "adversary_route_lanes",
    "sut_route_lanes", "conflict_radius_m", "route_lane_difference",
)


def build_task_descriptor(task: Any, runtime_geometry: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return the unique normalized static descriptor for a task.

    ``runtime_geometry`` is optional and may only refine topology quantities;
    it is rejected if it contains any case/reward/posterior-like key.
    """
    forbidden = {"case_id", "case_seed", "reward", "return", "query", "support", "hidden_rule"}
    if runtime_geometry and any(any(token in str(key).lower() for token in forbidden) for key in runtime_geometry):
        raise ValueError("task descriptor must not consume case, outcome, or hidden-rule data")
    if task.logical_type not in LOGICAL_TYPES:
        raise ValueError(f"unsupported logical type for descriptor: {task.logical_type!r}")
    map_config = dict(task.map_config)
    runtime = dict(runtime_geometry or {})
    logical_onehot = [float(task.logical_type == name) for name in LOGICAL_TYPES]
    bottle = float(runtime.get("bottle_lane_num", map_config.get("bottle_lane_num", 1.0))) / 5.0
    neck = float(runtime.get("neck_lane_num", map_config.get("neck_lane_num", 1.0))) / 5.0
    merge = float(runtime.get("merge_length_m", map_config.get("merge_length_m", 0.0))) / 100.0
    adv_lanes = float(runtime.get("adversary_route_lanes", len(task.adversary_route["lane_sequence"]))) / 8.0
    sut_lanes = float(runtime.get("sut_route_lanes", len(task.sut_route["lane_sequence"]))) / 8.0
    radius = float(runtime.get("conflict_radius_m", task.conflict_spec["conflict_radius_m"])) / 20.0
    lane_difference = (float(task.adversary_route["initial_lane"][2]) - float(task.sut_route["initial_lane"][2])) / 5.0
    value = np.asarray(logical_onehot + [bottle, neck, merge, adv_lanes, sut_lanes, radius, lane_difference], dtype=np.float32)
    if value.shape != (len(DESCRIPTOR_FIELDS),) or not np.isfinite(value).all():
        raise ValueError("task descriptor is malformed or non-finite")
    return value


def _mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden_sizes:
        layers.extend((nn.Linear(last, int(width)), nn.ReLU()))
        last = int(width)
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class ScenarioEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.model = _mlp(len(DESCRIPTOR_FIELDS), hidden_sizes, embedding_dim)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim != 2 or descriptor.shape[-1] != len(DESCRIPTOR_FIELDS):
            raise ValueError("descriptor must have shape [tasks, descriptor_dim]")
        return self.model(descriptor)


class ScenarioConditionedPrior(nn.Module):
    def __init__(self, embedding_dim: int, latent_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.model = _mlp(embedding_dim, hidden_sizes, 2 * latent_dim)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.model(embedding).chunk(2, dim=-1)
        return mu, log_var.clamp(-10.0, 5.0)
