from __future__ import annotations

import torch
from torch import nn

from .trajectory_encoder import TrajectoryEncoder


class EpisodeTokenBuilder(nn.Module):
    """Combines explicit test conditions, map state, trajectory, and outcome."""
    def __init__(self, map_dim: int, config_dim: int, option_count: int, trajectory_encoder: TrajectoryEncoder, token_dim: int = 128, outcome_dim: int = 5) -> None:
        super().__init__()
        self.trajectory_encoder = trajectory_encoder
        self.option_embedding = nn.Embedding(option_count, 16)
        self.output = nn.Sequential(nn.Linear(map_dim + config_dim + 16 + trajectory_encoder.hidden_dim + outcome_dim, token_dim), nn.ReLU(), nn.Linear(token_dim, token_dim))

    def forward(self, map_embedding: torch.Tensor, config: torch.Tensor, option_index: torch.Tensor, sequence: torch.Tensor, mask: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
        trajectory = self.trajectory_encoder(sequence, mask)
        if map_embedding.shape[0] != trajectory.shape[0] or config.shape[0] != trajectory.shape[0] or outcome.shape[0] != trajectory.shape[0]:
            raise ValueError("episode-token batch dimensions must agree")
        return self.output(torch.cat((map_embedding, config, self.option_embedding(option_index), trajectory, outcome), dim=-1))
