"""Shared zero-inflated beta-binomial count distribution."""

import torch
from torch.nn import functional as F


def count_log_prob(parameters, lengths, count):
    """Direct cumulative-count model; no independence assumption on true labels."""
    zero_logit, alpha, beta = parameters.unbind(-1)
    n = torch.as_tensor(lengths, dtype=alpha.dtype, device=alpha.device)
    k = torch.as_tensor(count, dtype=alpha.dtype, device=alpha.device)
    bb = (
        torch.lgamma(n + 1)
        - torch.lgamma(k + 1)
        - torch.lgamma(n - k + 1)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        - torch.lgamma(n + alpha + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )
    ordinary = F.logsigmoid(-zero_logit) + bb
    return torch.where(
        k == 0, torch.logaddexp(F.logsigmoid(zero_logit), ordinary), ordinary
    )


def expected_count(parameters, lengths):
    z, a, b = parameters.unbind(-1)
    n = torch.as_tensor(lengths, dtype=a.dtype, device=a.device)
    return torch.sigmoid(-z) * n * a / (a + b)


def count_cdf(parameters, length, limit):
    if limit < 0:
        return torch.zeros(parameters.shape[:-1], device=parameters.device)
    if limit >= length:
        return torch.ones(parameters.shape[:-1], device=parameters.device)
    k = torch.arange(limit + 1, dtype=torch.float64, device=parameters.device)
    logp = count_log_prob(parameters.double().unsqueeze(-2), length, k)
    return logp.logsumexp(-1).exp().clamp(0, 1)
