"""Sequential design for uncertainty about IDM-to-target transfer."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


KERNEL_LENGTH_SCALE = 0.3
OBSERVATION_NOISE = 0.25


def physical_kernel(coordinates: np.ndarray) -> np.ndarray:
    distance = cdist(coordinates, coordinates, "sqeuclidean")
    return np.exp(-distance / (2 * KERNEL_LENGTH_SCALE**2))


class TransferUncertainty:
    """Gaussian residual surrogate with weighted variance-reduction queries.

    Target feedback is treated as a noisy observation of the target-minus-history
    residual; RAS uses binary collision outcomes. Its variance is an exploration
    score, not a calibrated coverage or target failure probability guarantee.
    """

    def __init__(self, prior: np.ndarray, kernel: np.ndarray):
        self.prior = np.asarray(prior, dtype=np.float64)
        self.covariance = np.asarray(kernel, dtype=np.float64).copy()
        if self.covariance.shape != (len(self.prior), len(self.prior)):
            raise ValueError("kernel and prior shapes differ")
        self.mean_residual = np.zeros_like(self.prior)
        self.selected = np.zeros(len(self.prior), dtype=bool)
        self.weights = 1 - self.prior
        self.initial_information = float(np.max(self._information()))

    def _information(self) -> np.ndarray:
        weighted = (self.covariance**2) @ self.weights
        return weighted / (self.weights.sum() *
                           (np.diag(self.covariance) + OBSERVATION_NOISE))

    def risk(self) -> np.ndarray:
        return np.clip(self.prior + self.mean_residual, 0, 1)

    def information_gain(self) -> np.ndarray:
        return self._information() / self.initial_information

    def observe(self, index: int, risk: float | None) -> None:
        self.selected[index] = True
        if risk is None:
            return
        column = self.covariance[:, index].copy()
        denominator = column[index] + OBSERVATION_NOISE
        innovation = (risk - self.prior[index]) - self.mean_residual[index]
        self.mean_residual += column * (innovation / denominator)
        self.covariance -= np.outer(column, column) / denominator
        self.covariance = (self.covariance + self.covariance.T) / 2
