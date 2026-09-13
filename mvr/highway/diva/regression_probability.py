"""Posterior critical-event probabilities for PR-BRVT."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np

from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.posterior import LatentPosterior


def critical_probability(
    prior: LowRankPrior,
    posterior: LatentPosterior,
    threshold: float = 0.75,
    observation_noise: float = 0.03,
) -> np.ndarray:
    """Return ``P(f_target(x) >= threshold | revealed target outcomes)``.

    The observation noise is retained inside the predictive variance, rather
    than becoming a separate exploration bonus.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if observation_noise <= 0.0:
        raise ValueError("observation_noise must be positive")
    if posterior.mean.shape != (prior.rank,):
        raise ValueError("posterior mean has incompatible latent dimension")
    if posterior.covariance.shape != (prior.rank, prior.rank):
        raise ValueError("posterior covariance has incompatible latent dimension")
    prediction = prior.predict(posterior.mean)
    variance = np.einsum(
        "ij,jk,ik->i", prior.basis, posterior.covariance, prior.basis
    ) + observation_noise**2
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    z_score = (threshold - prediction) / standard_deviation
    return 0.5 * np.fromiter(
        (erfc(float(value) / sqrt(2.0)) for value in z_score),
        dtype=float,
        count=len(z_score),
    )
