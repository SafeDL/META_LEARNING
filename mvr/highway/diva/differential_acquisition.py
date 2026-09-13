"""Acquisition functions for population-referenced differential mining."""

from __future__ import annotations

import numpy as np

from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.population_reference import PopulationReference
from mvr.highway.diva.posterior import LatentPosterior


def posterior_std(prior: LowRankPrior, posterior: LatentPosterior) -> np.ndarray:
    """Return per-anchor latent uncertainty under the current posterior."""
    variance = np.einsum(
        "ij,jk,ik->i", prior.basis, posterior.covariance, prior.basis
    )
    return np.sqrt(np.maximum(variance, 0.0))


def differential_acquisition(
    prior: LowRankPrior,
    posterior: LatentPosterior,
    population: PopulationReference,
    differential_weight: float = 1.0,
    shared_weight: float = 0.10,
    uncertainty_weight: float = 0.25,
) -> np.ndarray:
    """Score unobserved anchors without reading the target response bank row."""
    prediction = np.clip(posterior.prediction, 0.0, 1.0)
    novelty = 1.0 - population.smoothed_failure_prevalence
    differential_utility = prediction * novelty
    return (
        differential_weight * differential_utility
        + shared_weight * prediction
        + uncertainty_weight * posterior_std(prior, posterior)
    )


def select_next_differential_index(
    prior: LowRankPrior,
    posterior: LatentPosterior,
    population: PopulationReference,
    excluded: np.ndarray | list[int],
    differential_weight: float = 1.0,
    shared_weight: float = 0.10,
    uncertainty_weight: float = 0.25,
) -> int:
    """Select the highest-scoring available anchor, resolving ties by index."""
    scores = differential_acquisition(
        prior,
        posterior,
        population,
        differential_weight,
        shared_weight,
        uncertainty_weight,
    )
    if scores.shape != population.mean_vulnerability.shape:
        raise ValueError("prior and population must contain the same anchors")
    blocked = set(np.asarray(excluded, dtype=int).tolist())
    candidates = np.asarray(
        [index for index in range(len(scores)) if index not in blocked], dtype=int
    )
    if len(candidates) == 0:
        raise ValueError("no unqueried anchors remain")
    order = np.lexsort((candidates, -scores[candidates]))
    return int(candidates[order[0]])
