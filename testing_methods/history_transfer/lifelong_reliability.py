"""Spatial reliability and epistemic acquisition for the lifelong protocol."""

import torch

from testing_methods.common.classification import NODES, WEIGHTS

DISCOUNT = 0.985
TEMPERATURE = 1.0
TARGET_FLOOR = 0.05
BETA = 0.35
MISMATCH_MULTIPLIER = 1.5


def update_scores(scores, probabilities, collision, *, tempered):
    likelihood = (probabilities if collision else 1 - probabilities).clamp_min(1e-15).log()
    discount = DISCOUNT if tempered else 1.0
    return torch.log_softmax(discount * scores + likelihood, dim=0), likelihood


def target_floor(logits):
    history, target = logits[:-2], logits[-2:]
    log_ratio = target.logsumexp(dim=0) - history.logsumexp(dim=0)
    target_mass = log_ratio.sigmoid().clamp_min(TARGET_FLOOR)
    history_mass = (-log_ratio).sigmoid().clamp_max(1 - TARGET_FLOOR)
    return torch.cat((
        history.softmax(dim=0) * history_mass,
        target.softmax(dim=0) * target_mass,
    ), dim=0)


def effective_weights(scores, locations, paid_locations, likelihoods, bandwidth, *, spatial, batch_size):
    if not spatial:
        return scores.softmax(0)[:, None].expand(-1, len(locations))
    count = len(paid_locations)
    if len(likelihoods) != count:
        raise ValueError("Every reliability observation must match a paid location")
    if not count:
        return target_floor(scores / TEMPERATURE)[:, None].expand(-1, len(locations))
    likelihood = torch.stack(likelihoods)
    global_mean = likelihood.mean(0)
    outputs = []
    for batch in locations.split(batch_size):
        kernel = (-0.5 * (torch.cdist(batch, paid_locations) / bandwidth).square()).exp()
        mass = kernel.sum(1)
        # Algebraically equal to m/(m+20) times the local-minus-global residual.
        residual = kernel @ (likelihood - global_mean)
        adjustment = count * residual / (mass + 20)[:, None]
        outputs.append(target_floor((scores[:, None] + adjustment.T) / TEMPERATURE))
    return torch.cat(outputs, dim=1)


def logistic_moments(mean, variance):
    nodes = torch.as_tensor(NODES, dtype=mean.dtype, device=mean.device)
    weights = torch.as_tensor(WEIGHTS, dtype=mean.dtype, device=mean.device)
    latent = mean[..., None] + (2 * variance.clamp_min(1e-12)).sqrt()[..., None] * nodes
    probability = latent.sigmoid()
    average = probability @ weights
    variance = (probability.square() @ weights - average.square()).clamp_min(0)
    return average, variance


def mixture_moments(weights, means, variances):
    mean = (weights * means).sum(0)
    variance = (weights * (variances + (means - mean).square())).sum(0)
    return mean, variance.clamp_min(0).sqrt()


def risk_beta(query, budget, initial_count, historical_mass):
    if not 0 <= initial_count < query <= budget or not 0 <= historical_mass <= 1:
        raise ValueError("Risk-UCB requires an active query and a valid historical weight")
    remaining_fraction = (budget - query) / (budget - initial_count)
    return BETA * remaining_fraction * (1 + MISMATCH_MULTIPLIER * (1 - historical_mass))
