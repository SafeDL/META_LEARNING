"""Shared feature composition with an explicit no-SUT-identity guard."""
from __future__ import annotations

from typing import Mapping
import torch
from torch import nn


FORBIDDEN_MODEL_KEYS = frozenset({"sut_ref", "sut_id", "profile_id", "algorithm_id", "controller_id"})


class SharedFeatureEncoder(nn.Module):
    def __init__(self, state_dim: int, map_dim: int, latent_dim: int, concrete_dim: int, output_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim + map_dim + latent_dim + concrete_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim), nn.ReLU())

    @staticmethod
    def validate_metadata(metadata: Mapping[str, object]) -> None:
        forbidden = FORBIDDEN_MODEL_KEYS.intersection(metadata)
        if forbidden:
            raise ValueError(f"SUT identity must not enter model inputs: {sorted(forbidden)}")

    def forward(self, state: torch.Tensor, map_embedding: torch.Tensor, latent: torch.Tensor, concrete: torch.Tensor, *, metadata: Mapping[str, object] | None = None) -> torch.Tensor:
        if metadata is not None:
            self.validate_metadata(metadata)
        return self.network(torch.cat((state, map_embedding, latent, concrete), dim=-1))
