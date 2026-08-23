"""Product-of-Gaussians episode-context inference without legacy imports."""
from __future__ import annotations

import torch
from torch import nn


class PearlContextEncoder(nn.Module):
    """Permutation-invariant q(z|C) built from independent evidence factors."""

    def __init__(self, token_dim: int = 128, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.factor = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(), nn.Linear(token_dim, 2 * latent_dim)
        )

    def prior(self, batch_size: int = 1, *, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(batch_size, self.latent_dim, device=device), torch.zeros(batch_size, self.latent_dim, device=device)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("context expects [batch, episodes, token] and matching mask")
        factor_mean, factor_logvar = self.factor(tokens).chunk(2, dim=-1)
        factor_logvar = factor_logvar.clamp(-10.0, 10.0)
        precision = (-factor_logvar).exp() * mask.unsqueeze(-1)
        variance = 1.0 / (1.0 + precision.sum(dim=1))
        mean = variance * (precision * factor_mean).sum(dim=1)
        logvar = variance.log()
        empty = ~mask.any(dim=1)
        mean = torch.where(empty[:, None], torch.zeros_like(mean), mean)
        logvar = torch.where(empty[:, None], torch.zeros_like(logvar), logvar)
        return mean, logvar

    @staticmethod
    def sample(mean: torch.Tensor, logvar: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * (0.5 * logvar).exp()

    @staticmethod
    def kl_to_prior(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1)
