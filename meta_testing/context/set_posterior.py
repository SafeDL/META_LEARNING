from __future__ import annotations

import torch
from torch import nn


class SetPosterior(nn.Module):
    """Permutation-invariant diagonal Gaussian q(z | complete test episodes)."""
    def __init__(self, token_dim: int = 128, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Sequential(nn.Linear(token_dim, token_dim), nn.ReLU(), nn.Linear(token_dim, token_dim))
        self.head = nn.Linear(token_dim, 2 * latent_dim)

    def prior(self, batch_size: int = 1, *, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(batch_size, self.latent_dim, device=device), torch.zeros(batch_size, self.latent_dim, device=device)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("posterior expects [batch, episodes, token] and matching mask")
        encoded = self.embed(tokens) * mask.unsqueeze(-1)
        counts = mask.sum(dim=1, keepdim=True)
        pooled = encoded.sum(dim=1) / counts.clamp(min=1)
        mean, logvar = self.head(pooled).chunk(2, dim=-1)
        prior_mean, prior_logvar = self.prior(tokens.shape[0], device=tokens.device)
        empty = counts.squeeze(-1) == 0
        return torch.where(empty[:, None], prior_mean, mean), torch.where(empty[:, None], prior_logvar, logvar.clamp(-12.0, 8.0))

    @staticmethod
    def sample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_to_unit(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1)


class VulnerabilityOutcomeDecoder(nn.Module):
    """Held-out-outcome decoder used to train z without a SUT-ID feature."""
    def __init__(self, latent_dim: int, map_dim: int, config_dim: int, option_count: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.options = nn.Embedding(option_count, 16)
        self.network = nn.Sequential(nn.Linear(latent_dim + map_dim + config_dim + 16, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 5))

    def forward(self, latent: torch.Tensor, map_embedding: torch.Tensor, config: torch.Tensor, option_index: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((latent, map_embedding, config, self.options(option_index)), dim=-1))
