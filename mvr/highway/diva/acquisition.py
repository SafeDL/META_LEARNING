"""Diagnostic support and posterior-risk acquisition rules."""

from __future__ import annotations

import numpy as np

from mvr.highway.diva.low_rank_prior import LowRankPrior


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


def diagnostic_support_indices(
    prior: LowRankPrior, count: int, observation_noise: float = 0.03
) -> np.ndarray:
    """Greedily maximize latent predictive variance for K diagnostic tests."""
    if count > len(prior.mean):
        raise ValueError("support count exceeds candidate pool")
    covariance = prior.latent_covariance.copy()
    available = set(range(len(prior.mean)))
    selected: list[int] = []
    for _ in range(count):
        candidates = np.asarray(sorted(available), dtype=int)
        basis = prior.basis[candidates]
        scores = np.einsum("ij,jk,ik->i", basis, covariance, basis)
        chosen = int(candidates[np.argmax(scores)])
        selected.append(chosen)
        available.remove(chosen)
        vector = prior.basis[chosen : chosen + 1]
        covariance = covariance - (
            covariance @ vector.T @ vector @ covariance
            / (observation_noise**2 + float((vector @ covariance @ vector.T).item()))
        )
    return np.asarray(selected, dtype=int)
