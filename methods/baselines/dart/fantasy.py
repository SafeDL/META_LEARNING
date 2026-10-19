"""Pure one-observation Student-mixture conditioning; virtual values are real."""

import numpy as np
from scipy.special import logsumexp, stdtr

from methods.baselines.saem.posterior import student_log_density

from .config import JITTER, NOISE, THRESHOLD


def conditional_parameters(model, candidate, values, indices, spatial=True):
    indices = np.asarray(indices, dtype=int)
    values = np.asarray(values, dtype=float).reshape(-1)
    denominator = model.variance[candidate] + NOISE + JITTER
    cross = (
        model.covariance[indices, candidate]
        - model.solved_cross[indices] @ model.solved_cross[candidate]
    )
    if not spatial:
        cross = np.zeros_like(cross)
    residual = values[:, None] - model.source_mean[candidate]
    mean = (
        model.source_mean[indices][None]
        + cross[None, :, None] * residual[:, None] / denominator
    )
    variance = model.variance[indices] + NOISE + JITTER - cross**2 / denominator
    beta = model.beta[None] + 0.5 * residual**2 / denominator
    log_weight = model.log_weight + student_log_density(
        residual,
        model.beta / model.alpha * denominator,
        2 * model.alpha,
    )
    log_weight -= logsumexp(log_weight, axis=1, keepdims=True)
    return mean, variance, model.alpha + 0.5, beta, log_weight


def fantasy_probabilities(model, candidate, values, indices, spatial=True):
    mean, variance, alpha, beta, log_weight = conditional_parameters(
        model,
        candidate,
        values,
        indices,
        spatial,
    )
    z = (mean - THRESHOLD) / np.sqrt(variance[None, :, None] * beta[:, None] / alpha)
    return np.sum(stdtr(2 * alpha, z) * np.exp(log_weight[:, None]), axis=-1)
