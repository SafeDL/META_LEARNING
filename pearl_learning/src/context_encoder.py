"""PEARL Product-of-Gaussians posterior with an exact empty-context prior."""
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


def product_of_gaussians_with_prior(
    evidence_mu: torch.Tensor,
    evidence_log_var: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_log_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine one static prior factor with independent transition evidence."""
    if evidence_mu.ndim != 3 or evidence_log_var.shape != evidence_mu.shape:
        raise ValueError("evidence factors must have shape [tasks, factors, latent]")
    if prior_mu.shape != (evidence_mu.shape[0], evidence_mu.shape[2]) or prior_log_var.shape != prior_mu.shape:
        raise ValueError("prior must have shape [tasks, latent]")
    variance = torch.exp(evidence_log_var).clamp_min(1e-7)
    precision = variance.reciprocal()
    prior_precision = torch.exp(prior_log_var).clamp_min(1e-7).reciprocal()
    total_precision = prior_precision + precision.sum(dim=1)
    posterior_variance = total_precision.reciprocal()
    posterior_mu = posterior_variance * (prior_precision * prior_mu + (precision * evidence_mu).sum(dim=1))
    return posterior_mu, torch.log(posterior_variance)


def kl_diag_normal(q_mu: torch.Tensor, q_log_var: torch.Tensor, p_mu: torch.Tensor, p_log_var: torch.Tensor) -> torch.Tensor:
    """Mean KL[q || p] for diagonal Normal distributions."""
    if q_mu.shape != q_log_var.shape or p_mu.shape != q_mu.shape or p_log_var.shape != q_mu.shape:
        raise ValueError("all Normal parameters must have the same shape")
    ratio = torch.exp(q_log_var - p_log_var)
    squared = (q_mu - p_mu).square() * torch.exp(-p_log_var)
    return 0.5 * torch.sum(p_log_var - q_log_var + ratio + squared - 1.0, dim=-1).mean()


CONTEXT_AGGREGATIONS = {"transition_product", "episode_product"}


class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_sizes: list[int],
        aggregation: str = "transition_product",
    ):
        super().__init__()
        if aggregation not in CONTEXT_AGGREGATIONS:
            raise ValueError(f"unsupported context aggregation: {aggregation!r}")
        layers: list[nn.Module] = []
        last_dim = input_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(last_dim, width), nn.ReLU()))
            last_dim = width
        layers.append(nn.Linear(last_dim, 2 * latent_dim))
        self.model = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.aggregation = aggregation

    def forward(self, context: torch.Tensor, prior: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 4:
            raise ValueError("context must have shape [tasks, episodes, transitions, features]")
        output = self.model(context)
        if self.aggregation == "transition_product":
            # PEARL Eq. (2): every transition is an independent Gaussian
            # evidence factor. Flattening only removes episode boundaries, so
            # regrouping the same transition set cannot change the posterior.
            evidence = output.flatten(start_dim=1, end_dim=2)
        else:
            # Explicit ablation for correlated-episode studies. This is not the
            # paper-faithful PEARL posterior and must be named in provenance.
            evidence = output.mean(dim=2)
        mu, log_var = evidence.chunk(2, dim=-1)
        log_var = log_var.clamp(-10, 5)
        if prior is None:
            return product_of_gaussians(mu, log_var)
        return product_of_gaussians_with_prior(mu, log_var, prior[0], prior[1])

    def prior(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.latent_dim)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @staticmethod
    def kl_to_unit_normal(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return kl_diag_normal(mu, log_var, torch.zeros_like(mu), torch.zeros_like(log_var))
