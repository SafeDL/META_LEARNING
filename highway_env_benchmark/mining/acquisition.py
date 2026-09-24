"""Support acquisition rules for target-SUT diagnosis and failure mining."""

from __future__ import annotations

import numpy as np

from highway_env_benchmark.mining.low_rank_prior import LowRankPrior
from highway_env_benchmark.mining.posterior import LatentPosterior, adapt_posterior


def highest_risk_indices(
    scores: np.ndarray, count: int, excluded: np.ndarray | list[int] = ()
) -> np.ndarray:
    """Choose the highest-scoring unobserved anchors with deterministic ties."""
    blocked = set(np.asarray(excluded, dtype=int).tolist())
    candidates = np.array([i for i in range(len(scores)) if i not in blocked], dtype=int)
    if count > len(candidates):
        raise ValueError("Cannot choose more anchors than remain")
    order = np.lexsort((candidates, -np.asarray(scores)[candidates]))
    return candidates[order[:count]]


def boundary_weight(prediction: np.ndarray, boundary: float = 0.75) -> np.ndarray:
    """Give maximal weight to predictions on the collision/near-miss boundary."""
    values = np.asarray(prediction, dtype=float)
    return np.clip(1.0 - np.abs(values - boundary), 0.0, 1.0)


def boundary_aware_support_index(
    prior: LowRankPrior,
    posterior: LatentPosterior,
    excluded: np.ndarray | list[int] = (),
) -> int:
    """Select one unobserved probe using information gain times boundary proximity."""
    blocked = set(np.asarray(excluded, dtype=int).tolist())
    candidates = np.array(
        [index for index in range(len(prior.mean)) if index not in blocked], dtype=int
    )
    if not len(candidates):
        raise ValueError("No anchors remain for diagnostic support")
    basis = prior.basis[candidates]
    information_gain = np.einsum(
        "ij,jk,ik->i", basis, posterior.covariance, basis
    )
    scores = information_gain * boundary_weight(posterior.prediction[candidates])
    order = np.lexsort((candidates, -scores))
    return int(candidates[order[0]])


def diagnostic_support_indices(
    prior: LowRankPrior,
    target_responses: np.ndarray,
    count: int,
    observation_noise: float = 0.03,
) -> np.ndarray:
    """Reveal K supports sequentially, updating the target posterior after each probe."""
    truth = np.asarray(target_responses, dtype=float)
    if truth.shape != prior.mean.shape:
        raise ValueError("target_responses must have one value per anchor")
    if count > len(prior.mean):
        raise ValueError("support count exceeds candidate pool")
    selected: list[int] = []
    posterior = adapt_posterior(
        prior,
        np.asarray(selected, dtype=int),
        np.asarray([], dtype=float),
        observation_noise,
    )
    for _ in range(count):
        chosen = boundary_aware_support_index(prior, posterior, selected)
        selected.append(chosen)
        support = np.asarray(selected, dtype=int)
        posterior = adapt_posterior(
            prior, support, truth[support], observation_noise
        )
    return np.asarray(selected, dtype=int)
