"""Risk, residual propagation, and coverage selection on one candidate bank."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def similarities(x: np.ndarray, responses: np.ndarray,
                 sigma_x: float, sigma_r: float | None) -> np.ndarray:
    if sigma_x <= 0 or sigma_r is not None and sigma_r <= 0:
        raise ValueError("similarity bandwidths must be positive")
    squared = cdist(x, x, "sqeuclidean") / (2 * sigma_x**2)
    if sigma_r is not None:
        squared += cdist(responses, responses, "sqeuclidean") / (
            2 * responses.shape[1] * sigma_r**2)
    return np.exp(-squared).astype(np.float32)


def corrected_risk(prior: np.ndarray, similarity: np.ndarray,
                   selected: list[int], observed: list[float | None],
                   regularizer: float) -> np.ndarray:
    if not selected:
        return prior.copy()
    valid = [(index, label) for index, label in zip(selected, observed)
             if label is not None]
    if not valid:
        return prior.copy()
    indices = np.asarray([item[0] for item in valid], dtype=int)
    residual = np.asarray([item[1] - prior[item[0]] for item in valid])
    weights = similarity[:, indices]
    delta = weights @ residual / (regularizer + weights.sum(axis=1))
    return np.clip(prior + delta, 0, 1)
