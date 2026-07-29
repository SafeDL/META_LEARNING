"""Product-of-Gaussians context posterior with an exact empty-context prior."""
from __future__ import annotations

import torch
from torch import nn


def product_of_gaussians(mu: torch.Tensor, log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate independent diagonal Normal factors along context dimension 1."""
    variance = torch.exp(log_var).clamp_min(1e-7)
    precision = variance.reciprocal()
    posterior_variance = precision.sum(dim=1).reciprocal()
    posterior_mu = (mu * precision).sum(dim=1) * posterior_variance
    return posterior_mu, torch.log(posterior_variance)


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(last_dim, width), nn.ReLU()))
            last_dim = width
        layers.append(nn.Linear(last_dim, 2 * latent_dim))
        self.model = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 4:
            raise ValueError("context must have shape [tasks, episodes, transitions, features]")
        # Adjacent transitions are correlated.  Pool each episode before the
        # product, so an episode contributes one evidence factor, not hundreds.
        output = self.model(context).mean(dim=2)
        mu, log_var = output.chunk(2, dim=-1)
        return product_of_gaussians(mu, log_var.clamp(-10, 5))

    def prior(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.latent_dim)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @staticmethod
    def kl_to_unit_normal(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(torch.exp(log_var) + mu.square() - 1.0 - log_var, dim=-1).mean()
