"""Response-pattern clustering and critical-distribution set sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _kmeans(values: np.ndarray, clusters: int, seed: int, iterations: int = 100) -> np.ndarray:
    """Small deterministic NumPy k-means used to avoid an extra dependency."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) < 1:
        raise ValueError("values must be a non-empty [L,M] matrix")
    unique = np.unique(x, axis=0)
    k = min(int(clusters), len(unique), len(x))
    if k < 1:
        raise ValueError("clusters must be positive")
    rng = np.random.default_rng(seed)
    centers = unique[rng.choice(len(unique), size=k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iterations):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        updated = distances.argmin(axis=1)
        if np.array_equal(labels, updated) and _ > 0:
            break
        labels = updated
        for index in range(k):
            members = x[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return labels


@dataclass(frozen=True)
class ResponseClusterSampler:
    """Sample unique scenario sets by cycling over source-response clusters."""

    labels: np.ndarray

    @classmethod
    def fit(cls, source_response: np.ndarray, clusters: int, seed: int) -> "ResponseClusterSampler":
        response = np.asarray(source_response, dtype=float)
        if response.ndim != 2:
            raise ValueError("source_response must have shape [M,L]")
        return cls(_kmeans(response.T, clusters, seed))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        length = len(self.labels)
        if not 1 <= n <= length:
            raise ValueError("n must be between 1 and the candidate-pool size")
        groups = [np.flatnonzero(self.labels == label) for label in np.unique(self.labels)]
        order = rng.permutation(len(groups))
        shuffled = [rng.permutation(groups[index]).tolist() for index in order]
        result: list[int] = []
        cursor = 0
        while len(result) < n:
            group = shuffled[cursor % len(shuffled)]
            if group:
                result.append(int(group.pop()))
            cursor += 1
            if cursor > length * len(shuffled) * 2:
                break
        if len(result) < n:
            remaining = np.setdiff1d(np.arange(length), np.asarray(result), assume_unique=False)
            result.extend(int(v) for v in rng.choice(remaining, n - len(result), replace=False))
        return np.asarray(result, dtype=int)


def sample_uniform(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= n <= length:
        raise ValueError("n must be between 1 and the candidate-pool size")
    return np.asarray(rng.choice(length, size=n, replace=False), dtype=int)

