"""Leakage-free discrete low-rank vulnerability prior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LowRankPrior:
    """SVD prior: response = scenario_mean + basis @ latent_profile."""

    mean: np.ndarray
    basis: np.ndarray
    latent_covariance: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    def predict(self, latent_mean: np.ndarray | None = None) -> np.ndarray:
        latent = np.zeros(self.rank) if latent_mean is None else latent_mean
        return self.mean + self.basis @ latent

    @classmethod
    def fit(cls, source_responses: np.ndarray, rank: int) -> "LowRankPrior":
        """Fit only on source rows; callers are responsible for LOSO splitting."""
        responses = np.asarray(source_responses, dtype=float)
        if responses.ndim != 2 or responses.shape[0] < 2:
            raise ValueError("source_responses must contain at least two SUT rows")
        max_rank = min(responses.shape)
        if not 1 <= rank <= max_rank:
            raise ValueError(f"rank must be in [1, {max_rank}]")
        mean = responses.mean(axis=0)
        centered = responses - mean
        left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
        basis = right[:rank].T
        latent_profiles = left[:, :rank] * singular_values[:rank]
        covariance = np.atleast_2d(np.cov(latent_profiles, rowvar=False, ddof=1))
        covariance += np.eye(rank) * 1e-8
        total_variance = float(np.sum(singular_values**2))
        explained = (
            singular_values**2 / total_variance
            if total_variance > 0
            else np.zeros_like(singular_values)
        )
        return cls(mean, basis, covariance, explained)
