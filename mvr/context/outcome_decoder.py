"""Outcome target batch and decoder for probabilistic context training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PosteriorTrainingBatch:
    support_tokens: torch.Tensor
    support_mask: torch.Tensor
    target_scene: torch.Tensor
    target_concrete: torch.Tensor
    target_outcome: torch.Tensor
    support_episode_ids: tuple[tuple[str, ...], ...]
    target_episode_id: tuple[str, ...]

    def validate(self) -> None:
        batch = self.support_tokens.shape[0]
        if self.support_tokens.ndim != 3 or self.support_mask.shape != self.support_tokens.shape[:2]:
            raise ValueError("posterior support tensors have incompatible shapes")
        if self.target_scene.shape[0] != batch or self.target_concrete.shape[0] != batch:
            raise ValueError("posterior target tensors do not match the batch dimension")
        if self.target_outcome.shape != (batch, 5):
            raise ValueError("posterior target tensors do not match the batch dimension")
        if len(self.support_episode_ids) != batch or len(self.target_episode_id) != batch:
            raise ValueError("posterior episode ids do not match the batch dimension")
        for support, target in zip(self.support_episode_ids, self.target_episode_id):
            if target in support:
                raise ValueError("target_episode_id must not appear in support_episode_ids")


class VulnerabilityOutcomeDecoder(nn.Module):
    def __init__(self, latent_dim: int, scene_dim: int, concrete_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + scene_dim + concrete_dim, 128), nn.ReLU(), nn.Linear(128, 5)
        )

    def forward(self, latent: torch.Tensor, scene: torch.Tensor, concrete: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((latent, scene, concrete), dim=-1))
