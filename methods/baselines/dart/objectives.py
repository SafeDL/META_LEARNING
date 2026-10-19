"""Differentiable Student-mixture likelihood and decision-related supervision."""

from functools import lru_cache
import math

import numpy as np
from scipy.stats import gamma
import torch
from torch.nn import functional as F

from .config import BUDGET, JITTER, NOISE, PRIOR_DF, SCALE_NODES, THRESHOLD


@lru_cache(maxsize=None)
def scale_quadrature(alpha):
    nodes, weights = np.polynomial.legendre.leggauss(SCALE_NODES)
    return gamma.ppf(0.5 * (nodes + 1), a=alpha), 0.5 * weights


def event_probability(mean, variance, alpha, beta, log_weight):
    precision, weights = scale_quadrature(float(alpha))
    precision = torch.as_tensor(precision, device=mean.device, dtype=mean.dtype)
    weights = torch.as_tensor(weights, device=mean.device, dtype=mean.dtype)
    z = (mean - THRESHOLD)[..., None] * torch.sqrt(
        precision / (variance[..., None, None] * beta[..., None])
    )
    component = (torch.special.ndtr(z) * weights).sum(-1)
    return (component * log_weight.exp()).sum(-1)


def joint_density(mean, covariance, alpha, beta, truth):
    count = len(truth)
    factor = torch.linalg.cholesky(covariance)
    residual = truth[:, None] - mean
    quadratic = (residual * torch.cholesky_solve(residual, factor)).sum(0)
    a = beta.new_tensor(alpha)
    return (
        torch.lgamma(a + count / 2)
        - torch.lgamma(a)
        - count / 2 * math.log(2 * math.pi)
        - factor.diag().log().sum()
        + a * beta.log()
        - (a + count / 2) * (beta + 0.5 * quadratic).log()
    )


def conditioned(covariance, priors, probability, truth, count):
    alpha = PRIOR_DF / 2
    beta = priors.new_full((priors.shape[1],), (PRIOR_DF - 2) / 2)
    log_weight = probability.log()
    mean, block = priors[count:], covariance[count:, count:]
    if count:
        support = covariance[:count, :count] + (NOISE + JITTER) * torch.eye(
            count,
            device=priors.device,
            dtype=priors.dtype,
        )
        log_weight = log_weight + joint_density(
            priors[:count], support, alpha, beta, truth[:count]
        )
        factor = torch.linalg.cholesky(support)
        residual = truth[:count, None] - priors[:count]
        solved_residual = torch.cholesky_solve(residual, factor)
        beta = beta + 0.5 * (residual * solved_residual).sum(0)
        cross = covariance[count:, :count]
        mean = mean + cross @ solved_residual
        solved = torch.linalg.solve_triangular(factor, cross.T, upper=False)
        block = block - solved.T @ solved
        alpha += count / 2
    block = 0.5 * (block + block.T) + (NOISE + JITTER) * torch.eye(
        len(mean),
        device=priors.device,
        dtype=priors.dtype,
    )
    log_weight = log_weight - torch.logsumexp(log_weight, 0)
    return mean, block, alpha, beta, log_weight


def probe_probability(mean, covariance, alpha, beta, log_weight, probe, risk):
    denominator = covariance[probe, probe]
    residual = risk - mean[probe]
    updated_mean = mean + covariance[:, probe, None] * residual / denominator
    variance = covariance.diag() - covariance[:, probe].square() / denominator
    updated_beta = beta + 0.5 * residual.square() / denominator
    scale = beta / alpha * denominator
    a = beta.new_tensor(alpha)
    density = (
        torch.lgamma(a + 0.5)
        - torch.lgamma(a)
        - 0.5 * (2 * a * math.pi * scale).log()
        - (a + 0.5) * torch.log1p(residual.square() / (2 * a * scale))
    )
    updated_weight = log_weight + density
    updated_weight = updated_weight - torch.logsumexp(updated_weight, 0)
    # The probe itself is excluded; its zero conditional variance is never used.
    keep = torch.arange(len(mean), device=mean.device) != probe
    return (
        event_probability(
            updated_mean[keep],
            variance[keep],
            alpha + 0.5,
            updated_beta,
            updated_weight,
        ),
        keep,
    )


def training_loss(
    covariance, priors, probability, truth, count, mode="decision", *, pool_size=2048
):
    mean, block, alpha, beta, log_weight = conditioned(
        covariance,
        priors,
        probability,
        truth,
        count,
    )
    query_truth = truth[count:]
    joint = -torch.logsumexp(
        log_weight
        + joint_density(
            mean,
            block,
            alpha,
            beta,
            query_truth,
        ),
        0,
    ) / len(mean)
    probability_query = event_probability(
        mean, block.diag(), alpha, beta, log_weight
    ).clamp(1e-8, 1 - 1e-8)
    high = (query_truth > THRESHOLD).to(mean.dtype)
    if mode == "conventional":
        mse = ((mean * log_weight.exp()).sum(1) - query_truth).square().mean()
        return joint + 5 * mse, {"joint": joint.detach(), "mse": mse.detach()}
    event = F.binary_cross_entropy(probability_query, high)
    fraction = (BUDGET - count) / (pool_size - count)
    boundary_position = max(1, min(len(mean), round(len(mean) * fraction))) - 1
    sorted_probability = probability_query.detach().sort(descending=True).values
    boundary = sorted_probability[boundary_position]
    near = (probability_query.detach() - boundary).abs().argsort(stable=True)[:8]
    logits = probability_query.log() - torch.log1p(-probability_query)
    positive = torch.nonzero(high > 0).flatten()
    negative = torch.nonzero(high == 0).flatten()
    rank = logits.sum() * 0
    if len(positive) and len(negative):
        pairs = torch.isin(positive[:, None], near) | torch.isin(negative[None], near)
        if pairs.any():
            rank = F.softplus(-(logits[positive, None] - logits[negative][None]))[
                pairs
            ].mean()
    updates = []
    for probe in near[:4]:
        predicted, keep = probe_probability(
            mean, block, alpha, beta, log_weight, int(probe), query_truth[probe]
        )
        updates.append(
            F.binary_cross_entropy(predicted.clamp(1e-8, 1 - 1e-8), high[keep])
        )
    update = torch.stack(updates).mean()
    loss = joint + event + 0.2 * rank + update
    return loss, {
        "joint": joint.detach(),
        "event": event.detach(),
        "rank": rank.detach(),
        "update": update.detach(),
    }
