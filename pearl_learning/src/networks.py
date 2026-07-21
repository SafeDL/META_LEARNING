"""Tanh-Gaussian actor and twin critics; task labels have no input path."""
from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers = []; last = input_dim
    for width in hidden_sizes: layers += [nn.Linear(last, width), nn.ReLU()]; last = width
    layers.append(nn.Linear(last, output_dim)); return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, observation_dim: int, latent_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__(); self.backbone = mlp(observation_dim + latent_dim, hidden_sizes, action_dim * 2); self.action_dim = action_dim
    def forward(self, observation: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.backbone(torch.cat([observation, latent], dim=-1)).chunk(2, dim=-1)
        return mean, log_std.clamp(-20, 2)
    def sample(self, observation: torch.Tensor, latent: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation, latent); normal = Normal(mean, log_std.exp()); raw = mean if deterministic else normal.rsample(); action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(1 - action.square() + 1e-6); return action, log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int, hidden_sizes: list[int]):
        super().__init__(); self.model = mlp(observation_dim + action_dim + latent_dim, hidden_sizes, 1)
    def forward(self, observation: torch.Tensor, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([observation, action, latent], dim=-1))
