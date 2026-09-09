"""Source-only evaluability smoother used as an acquisition weight."""
from __future__ import annotations

import numpy as np


class RBFEvaluability:
    def __init__(self, features: np.ndarray, eligible: np.ndarray, bandwidth: float = 0.35) -> None:
        self.features = np.asarray(features, dtype=np.float64)
        self.eligible = np.asarray(eligible, dtype=np.float64).reshape(-1)
        self.bandwidth = float(bandwidth)
        if self.features.ndim != 2 or len(self.features) != len(self.eligible):
            raise ValueError("evaluator features and values must align")
        if self.bandwidth <= 0.0 or not np.isfinite(self.features).all():
            raise ValueError("invalid RBF evaluability model")

    def predict(self, query: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(query, dtype=np.float64))
        distances2 = np.square(points[:, None, :] - self.features[None, :, :]).sum(axis=-1)
        weights = np.exp(-distances2 / (2.0 * self.bandwidth ** 2))
        return (weights @ self.eligible) / np.maximum(weights.sum(axis=1), 1e-12)
