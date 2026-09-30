"""Sequential design for uncertainty about IDM-to-target transfer."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


KERNEL_LENGTH_SCALE = 0.3
OBSERVATION_NOISE = 0.25
RISK_WEIGHT = 0.25
MISSED_FAILURE_WEIGHT = 0.25
INFORMATION_WEIGHT = 0.5


def physical_kernel(coordinates: np.ndarray) -> np.ndarray:
    distance = cdist(coordinates, coordinates, "sqeuclidean")
    return np.exp(-distance / (2 * KERNEL_LENGTH_SCALE**2))


class TransferUncertainty:
    """Gaussian residual surrogate with weighted variance-reduction queries.

    The binary FVDM feedback is treated as a noisy observation of the
    target-minus-history residual. Its variance is a model-based exploration
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
        if self.initial_information <= 0:
            raise ValueError("candidate bank has no transfer information")

    def _information(self) -> np.ndarray:
        weighted = (self.covariance**2) @ self.weights
        return weighted / (self.weights.sum() *
                           (np.diag(self.covariance) + OBSERVATION_NOISE))

    def risk(self) -> np.ndarray:
        return np.clip(self.prior + self.mean_residual, 0, 1)

    def acquisition(self) -> np.ndarray:
        risk = self.risk()
        information = self._information() / self.initial_information
        score = (RISK_WEIGHT * risk +
                 MISSED_FAILURE_WEIGHT * self.weights * risk +
                 INFORMATION_WEIGHT * information)
        score[self.selected] = -np.inf
        return score

    def choose(self) -> int:
        return int(np.argmax(self.acquisition()))

    def observe(self, index: int, label: int | None) -> None:
        if self.selected[index]:
            raise ValueError("repeated target query")
        self.selected[index] = True
        if label is None:
            return
        if label not in (0, 1):
            raise ValueError("target feedback is not binary")
        column = self.covariance[:, index].copy()
        denominator = column[index] + OBSERVATION_NOISE
        innovation = (label - self.prior[index]) - self.mean_residual[index]
        self.mean_residual += column * (innovation / denominator)
        self.covariance -= np.outer(column, column) / denominator
        self.covariance = (self.covariance + self.covariance.T) / 2


def select_transfer_sequence(prior: np.ndarray, kernel: np.ndarray,
                             oracle, budget: int) -> tuple[list[int], list[int | None],
                                                           np.ndarray]:
    model = TransferUncertainty(prior, kernel)
    selected, observed = [], []
    for _ in range(budget):
        index = model.choose()
        label = oracle.query(index)
        selected.append(index)
        observed.append(label)
        model.observe(index, label)
    return selected, observed, model.risk()
