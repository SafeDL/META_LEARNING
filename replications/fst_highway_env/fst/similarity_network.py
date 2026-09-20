"""Learned query-normalized inverse-distance similarity from the FST paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FeatureTransform:
    """Frozen input-only transform: standardized coordinates plus mode one-hot."""

    mean: np.ndarray
    scale: np.ndarray
    mode_names: tuple[str, ...]

    @classmethod
    def fit(cls, anchors: np.ndarray, modes: np.ndarray) -> "FeatureTransform":
        x = np.asarray(anchors, dtype=float)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(mean, scale, tuple(sorted(set(str(v) for v in modes))))

    def transform(self, anchors: np.ndarray, modes: np.ndarray) -> np.ndarray:
        x = (np.asarray(anchors, dtype=float) - self.mean) / self.scale
        labels = np.asarray(modes).astype(str)
        one_hot = np.column_stack([(labels == name).astype(float) for name in self.mode_names])
        return np.column_stack([x, one_hot]).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "mode_names": list(self.mode_names),
            "semantics": "standardized [configured_gap, relative_speed] plus unordered mode one-hot",
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FeatureTransform":
        return cls(
            np.asarray(value["mean"], dtype=float),
            np.asarray(value["scale"], dtype=float),
            tuple(str(v) for v in value["mode_names"]),
        )


class SimilarityNetwork(nn.Module):
    """MLP encoder with inverse-L2 logits and softmax across selected queries.

    Tensor contract:
      X [L,d_input], selected_indices [B,n], embedding_all [L,d_latent],
      attention_S [B,n,L], weights_w [B,n].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        linear_layers: int = 8,
        latent_dim: int | None = None,
        distance_epsilon: float = 1e-6,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if linear_layers < 2:
            raise ValueError("linear_layers counts linear layers and must be at least 2")
        if distance_epsilon <= 0 or temperature <= 0:
            raise ValueError("distance_epsilon and temperature must be positive")
        latent = hidden_dim if latent_dim is None else latent_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(linear_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, latent))
        self.encoder = nn.Sequential(*layers)
        self.distance_epsilon = float(distance_epsilon)
        self.temperature = float(temperature)
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "linear_layers": linear_layers,
            "latent_dim": latent,
            "distance_epsilon": self.distance_epsilon,
            "temperature": self.temperature,
            "similarity": "inverse_l2_query_softmax",
        }

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [L,d_input]")
        return self.encoder(features)

    def attention_from_embeddings(
        self, embedding_all: torch.Tensor, selected_indices: torch.Tensor
    ) -> torch.Tensor:
        if selected_indices.ndim == 1:
            selected_indices = selected_indices.unsqueeze(0)
        query = embedding_all[selected_indices]  # [B,n,d_latent]
        squared = ((query[:, :, None, :] - embedding_all[None, None, :, :]) ** 2).sum(-1)
        # At a query's self-key, sqrt(squared) has an undefined derivative even
        # though the paper's denominator epsilon keeps its value finite.  The
        # smooth norm below preserves the same self-logit 1/epsilon while making
        # the backward pass finite: 1 / sqrt(||q-k||^2 + epsilon^2).
        logits = torch.rsqrt(squared + self.distance_epsilon ** 2)
        logits = logits / self.temperature
        # Paper Eq. (15): normalize along the n-query dimension for each key.
        return torch.softmax(logits, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        selected_indices: torch.Tensor,
        probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encode(features)
        attention = self.attention_from_embeddings(embedding, selected_indices)
        weights = torch.einsum("bnl,l->bn", attention, probability)
        return embedding, attention, weights


def source_estimates(
    weights: torch.Tensor, source_response: torch.Tensor, selected_indices: torch.Tensor
) -> torch.Tensor:
    """Return [B,M] estimates from weights [B,n] and F_source [M,L]."""
    if selected_indices.ndim == 1:
        selected_indices = selected_indices.unsqueeze(0)
    selected = source_response[:, selected_indices].permute(1, 0, 2)
    return torch.einsum("bn,bmn->bm", weights, selected)


def minimax_loss_per_set(
    weights: torch.Tensor,
    source_response: torch.Tensor,
    selected_indices: torch.Tensor,
    probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    estimates = source_estimates(weights, source_response, selected_indices)
    truth = torch.einsum("ml,l->m", source_response, probability)
    loss = torch.max(torch.abs(estimates - truth[None, :]), dim=1).values
    return loss, estimates
