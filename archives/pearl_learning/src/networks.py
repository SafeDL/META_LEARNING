"""Tanh-Gaussian actor and twin critics; task labels have no input path."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for width in hidden_sizes:
        layers.extend((nn.Linear(last_dim, width), nn.ReLU()))
        last_dim = width
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, observation_dim: int, latent_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.backbone = mlp(observation_dim + latent_dim, hidden_sizes, action_dim * 2)
        self.action_dim = action_dim

    def forward(self, observation: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.backbone(torch.cat([observation, latent], dim=-1)).chunk(2, dim=-1)
        return mean, log_std.clamp(-20, 2)

    def sample(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation, latent)
        normal = Normal(mean, log_std.exp())
        raw_action = mean if deterministic else normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.model = mlp(observation_dim + action_dim + latent_dim, hidden_sizes, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([observation, action, latent], dim=-1))


class LatentFiLMCritic(nn.Module):
    """Critic where the latent explicitly modulates state-action features.

    The concatenated Critic can treat the latent as one more input feature and
    still learn a Q that is nearly constant along the task-discriminating
    latent direction.  Here the latent produces a FiLM modulation of the
    state-action trunk, so z is an explicit value-function conditioning
    variable with a dedicated gradient path.  Capacity stays in the same
    order as the dense twin critic; only the conditioning structure changes.
    """

    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int, hidden_sizes: list[int]):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("FiLM critic needs a non-empty hidden size list")
        self.feature_dim = int(hidden_sizes[-1])
        self.trunk = mlp(observation_dim + action_dim, list(hidden_sizes), self.feature_dim)
        self.modulator = mlp(latent_dim, [max(16, latent_dim * 4), 2 * self.feature_dim], 2 * self.feature_dim)
        self.head = torch.nn.Linear(self.feature_dim, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        features = self.trunk(torch.cat([observation, action], dim=-1))
        gamma, beta = self.modulator(latent).chunk(2, dim=-1)
        modulated = (1.0 + gamma) * features + beta
        return self.head(modulated)


class LatentGammaOnlyFiLMCritic(nn.Module):
    """FiLM Critic whose latent can only rescale state-action features.

    This intentionally keeps the historical FiLM Critic untouched.  The
    modulator has the same hidden depth and widths as ``LatentFiLMCritic``;
    only its final output drops the latent-only beta path.  A zero-initialized
    final layer starts the critic at the ordinary, latent-independent Q(s, a)
    solution and lets the multiplicative path emerge during training.
    """

    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int, hidden_sizes: list[int]):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("Gamma-only FiLM critic needs a non-empty hidden size list")
        self.feature_dim = int(hidden_sizes[-1])
        self.trunk = mlp(observation_dim + action_dim, list(hidden_sizes), self.feature_dim)
        self.modulator = mlp(
            latent_dim,
            [max(16, latent_dim * 4), 2 * self.feature_dim],
            self.feature_dim,
        )
        last_layer = self.modulator[-1]
        if not isinstance(last_layer, nn.Linear):  # Defensive: mlp always terminates in Linear.
            raise RuntimeError("Gamma-only FiLM modulator must end in a linear layer")
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        self.head = nn.Linear(self.feature_dim, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        features = self.trunk(torch.cat([observation, action], dim=-1))
        gamma = torch.tanh(self.modulator(latent))
        return self.head((1.0 + gamma) * features)
