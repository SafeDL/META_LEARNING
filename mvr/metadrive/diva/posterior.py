"""Analytic working posterior for the DIVA low-rank response model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LatentVulnerabilityPosterior:
    """Gaussian posterior for a frozen linear response surrogate, not simulator Bayes."""

    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def standard_normal(cls, rank: int) -> "LatentVulnerabilityPosterior":
        if rank < 0:
            raise ValueError("rank must be non-negative")
        return cls(np.zeros(rank, dtype=np.float64), np.eye(rank, dtype=np.float64))

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.covariance.shape != (len(self.mean), len(self.mean)):
            raise ValueError("posterior covariance shape does not match mean")
        np.linalg.cholesky(self.covariance)

    def information_gain(self, basis: np.ndarray, observation_noise_var: float) -> float:
        vector = np.asarray(basis, dtype=np.float64).reshape(-1)
        if vector.shape != self.mean.shape or observation_noise_var <= 0.0:
            raise ValueError("invalid basis or observation noise variance")
        return float(0.5 * np.log1p(vector @ self.covariance @ vector / observation_noise_var))

    def update(
        self,
        basis: np.ndarray,
        centered_score: float,
        observation_noise_var: float,
    ) -> None:
        vector = np.asarray(basis, dtype=np.float64).reshape(-1)
        if vector.shape != self.mean.shape or observation_noise_var <= 0.0:
            raise ValueError("invalid basis or observation noise variance")
        if not len(vector):
            return
        projected = self.covariance @ vector
        denominator = float(observation_noise_var + vector @ projected)
        gain = projected / denominator
        self.mean = self.mean + gain * (float(centered_score) - vector @ self.mean)
        self.covariance = self.covariance - np.outer(gain, projected)
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        np.linalg.cholesky(self.covariance)

    def predict(
        self,
        mean_component: float,
        basis: np.ndarray,
        observation_noise_var: float,
    ) -> tuple[float, float]:
        vector = np.asarray(basis, dtype=np.float64).reshape(-1)
        if vector.shape != self.mean.shape or observation_noise_var < 0.0:
            raise ValueError("invalid predictive parameters")
        location = float(mean_component + vector @ self.mean)
        variance = float(vector @ self.covariance @ vector + observation_noise_var)
        return location, variance
