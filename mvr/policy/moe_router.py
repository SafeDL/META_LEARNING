"""Learned task-aware router for shared initial-condition experts."""
from __future__ import annotations

import torch
from torch import nn


class TaskAwareMoERouter(nn.Module):
    def __init__(self, scene_dim: int, latent_dim: int, num_experts: int) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("a MoE router requires at least two experts")
        self.network = nn.Sequential(
            nn.Linear(scene_dim + latent_dim, 256), nn.ReLU(), nn.Linear(256, num_experts)
        )
        self.num_experts = num_experts

    def forward(self, scene_embedding: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if scene_embedding.shape[:-1] != latent.shape[:-1]:
            raise ValueError("router scene and latent batch dimensions must match")
        return self.network(torch.cat((scene_embedding, latent), dim=-1))

    @staticmethod
    def load_balance_loss(logits: torch.Tensor) -> torch.Tensor:
        probabilities = logits.softmax(dim=-1).mean(dim=0)
        return logits.new_tensor(float(logits.shape[-1])) * probabilities.square().sum()
