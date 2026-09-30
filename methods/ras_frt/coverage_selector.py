"""Risk, residual propagation, and coverage selection on one candidate bank."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class TargetOracle:
    """Reveal at most one target outcome for each selected candidate."""

    def __init__(self, labels: np.ndarray):
        self.__labels = labels.copy()
        self.queried: set[int] = set()

    def query(self, index: int) -> int | None:
        if index in self.queried:
            raise ValueError(f"repeated target query: {index}")
        self.queried.add(index)
        label = int(self.__labels[index])
        return None if label < 0 else label


def similarities(x: np.ndarray, responses: np.ndarray,
                 sigma_x: float, sigma_r: float | None) -> np.ndarray:
    if sigma_x <= 0 or sigma_r is not None and sigma_r <= 0:
        raise ValueError("similarity bandwidths must be positive")
    squared = cdist(x, x, "sqeuclidean") / (2 * sigma_x**2)
    if sigma_r is not None:
        squared += cdist(responses, responses, "sqeuclidean") / (
            2 * responses.shape[1] * sigma_r**2)
    return np.exp(-squared).astype(np.float32)


def history_rank(history_x: np.ndarray, history_risk: np.ndarray,
                 history_valid: np.ndarray, candidate_x: np.ndarray, k: int) -> np.ndarray:
    """Equal-weight source percentiles of k-neighbor collision/near-miss risk."""
    nearest = np.argpartition(cdist(candidate_x, history_x, "sqeuclidean"),
                              kth=k - 1, axis=1)[:, :k]
    hits = history_risk[nearest] * history_valid[nearest]
    valid = history_valid[nearest].sum(axis=1)
    per_source = np.divide(hits.sum(axis=1), valid,
                           out=np.zeros_like(valid, dtype=float), where=valid > 0)
    percentiles = []
    for source in range(history_risk.shape[1]):
        reference = np.sort(history_risk[history_valid[:, source], source])
        if len(reference):
            percentiles.append(np.searchsorted(reference, per_source[:, source],
                                                side="right") / len(reference))
    return np.mean(percentiles, axis=0) if percentiles else np.zeros(len(candidate_x))


def corrected_risk(prior: np.ndarray, similarity: np.ndarray,
                   selected: list[int], observed: list[int | None],
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


def select_sequence(method: str, oracle: TargetOracle, candidate_count: int,
                    prior: np.ndarray | None, similarity: np.ndarray | None,
                    budget: int, regularizer: float = 1.0,
                    coverage_weight: float = 0.2,
                    random_seed: int = 0) -> tuple[list[int], list[int | None], np.ndarray | None]:
    """Use labels only through one selected index per step."""
    rng = np.random.default_rng(random_seed)
    selected: list[int] = []
    observed: list[int | None] = []
    remaining = np.ones(candidate_count, dtype=bool)
    risk = None if prior is None else prior.copy()
    for _ in range(budget):
        if method == "Random":
            choice = int(rng.choice(np.flatnonzero(remaining)))
        else:
            if method in {"Residual risk-only", "RAS-FRT"}:
                risk = corrected_risk(prior, similarity, selected, observed, regularizer)
            score = risk.copy()
            if method == "RAS-FRT" and selected:
                gap = 1 - similarity[:, selected].max(axis=1)
                score = (1 - coverage_weight) * score + coverage_weight * gap
            score[~remaining] = -np.inf
            choice = int(np.argmax(score))
        selected.append(choice)
        remaining[choice] = False
        observed.append(oracle.query(choice))
    if method in {"Residual risk-only", "RAS-FRT"}:
        risk = corrected_risk(prior, similarity, selected, observed, regularizer)
    return selected, observed, risk
