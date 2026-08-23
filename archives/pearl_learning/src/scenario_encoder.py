"""Minimal physical-merge descriptors and a task-conditioned PEARL prior.

The descriptor intentionally reads only the frozen task-level geometry.  It
never receives case values, replay transitions, rewards, or hidden-rule labels.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import nn


DESCRIPTOR_SCHEMA = "physical_merge_minimal_descriptor_v1"
LOGICAL_TYPES = ("lane_drop_merge", "bottleneck_merge")
DESCRIPTOR_FIELDS = (
    "merge_subtype:lane_drop", "merge_subtype:bottleneck", "merge_length_m/100",
)


def build_task_descriptor(task: Any) -> np.ndarray:
    """Return [lane-drop one-hot, bottleneck one-hot, merge length / 100]."""
    if task.logical_type not in LOGICAL_TYPES:
        raise ValueError(f"unsupported logical type for descriptor: {task.logical_type!r}")
    map_config = dict(task.map_config)
    logical_onehot = [float(task.logical_type == name) for name in LOGICAL_TYPES]
    merge = float(map_config.get("merge_length_m", 0.0)) / 100.0
    value = np.asarray(logical_onehot + [merge], dtype=np.float32)
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
