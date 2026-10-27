"""Historical posterior transfer with an explicitly scaled target residual kernel."""

import math

import torch


def cached_covariance(encoded, log_lengthscale, log_scale):
    distance = torch.cdist(encoded, encoded) / log_lengthscale.exp().clamp_min(0.01)
    radius = math.sqrt(5) * distance
    return log_scale.exp().square() * (1 + radius + radius.square() / 3) * (-radius).exp()


def residual_prior(
    target_mean, target_covariance, source_means, source_covariances, reference_mean, beta
):
    if beta < 0:
        raise ValueError("Residual amplitude must be nonnegative")
    count = target_covariance.shape[-1]
    independent_mean = target_mean.expand(count)
    means = source_means + (target_mean - reference_mean)
    covariances = source_covariances + beta**2 * target_covariance[None]
    return (
        torch.cat((means, independent_mean[None])),
        torch.cat((covariances, target_covariance[None])),
    )


def update_reliability(log_weights, probabilities, index, label, public_x, bandwidth):
    likelihood = (
        (probabilities[:, index] if label else 1 - probabilities[:, index]).clamp_min(1e-15).log()
    )
    influence = (
        torch.ones(len(public_x), dtype=log_weights.dtype, device=log_weights.device)
        if bandwidth is None
        else (-torch.square(public_x - public_x[index]).mean(1) / (2 * bandwidth**2)).exp()
    )
    updated = log_weights + likelihood[:, None] * influence
    return updated - torch.logsumexp(updated, dim=0)
