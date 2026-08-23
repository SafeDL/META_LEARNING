"""Complete test rollout encoder; one call produces one evidence token."""
from __future__ import annotations

import torch
from torch import nn


TRAJECTORY_FIELDS = (
    "relative_x", "relative_y", "relative_speed", "sut_acceleration", "sut_speed", "sut_lateral_offset",
    "adversary_progress", "sut_progress", "ttc", "pet", "pair_distance", "conflict_timing",
)


class TrajectoryEncoder(nn.Module):
    def __init__(self, feature_dim: int = len(TRAJECTORY_FIELDS), hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.project = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.feature_dim or mask.shape != sequence.shape[:2]:
            raise ValueError("trajectory and boolean mask shapes do not match the context contract")
        lengths = mask.long().sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("each trajectory must include at least one valid step")
        encoded = self.project(sequence) * mask.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden[-1]
