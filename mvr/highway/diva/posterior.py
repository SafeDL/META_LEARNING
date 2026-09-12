"""Analytic Gaussian posterior for K-shot target adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvr.highway.diva.low_rank_prior import LowRankPrior


@dataclass(frozen=True)
class LatentPosterior:
    """Posterior distribution and resulting complete-anchor prediction."""

    mean: np.ndarray
    covariance: np.ndarray
    prediction: np.ndarray


def adapt_posterior(
    prior: LowRankPrior,
    support_indices: np.ndarray,
    support_responses: np.ndarray,
    observation_noise: float = 0.03,
) -> LatentPosterior:
    """Update the target latent profile from only revealed support outcomes."""
    indices = np.asarray(support_indices, dtype=int)
    outcomes = np.asarray(support_responses, dtype=float)
    if len(indices) != len(outcomes):
        raise ValueError("support_indices and support_responses must have equal length")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("support indices must be unique")
    if len(indices) == 0:
        return LatentPosterior(
            np.zeros(prior.rank), prior.latent_covariance.copy(), prior.mean.copy()
        )
    design = prior.basis[indices]
    residual = outcomes - prior.mean[indices]
    prior_precision = np.linalg.inv(prior.latent_covariance)
    precision = prior_precision + design.T @ design / observation_noise**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ design.T @ residual / observation_noise**2
    return LatentPosterior(mean, covariance, prior.predict(mean))
