"""Gap-aware adaptive action policy from AdaTE Eq. (9)-(10)."""

from __future__ import annotations
import numpy as np


def surrogate_gap(target_q: float,
                  mixed_q: float,
                  epsilon: float = 1e-12,
                  cap: float = 1e6) -> float:
    if mixed_q > epsilon:
        return float(abs(target_q - mixed_q) / mixed_q)
    if target_q <= epsilon:
        return 0.0
    return float(cap)


def gap_ucb_scores(target_values: np.ndarray, mixed_values: np.ndarray, natural_policy: np.ndarray,
                   visits: np.ndarray, exploration: float) -> np.ndarray:
    target_values, mixed_values, natural_policy, visits = map(
        lambda x: np.asarray(x, dtype=float),
        (target_values, mixed_values, natural_policy, visits))
    if not (target_values.shape == mixed_values.shape == natural_policy.shape == visits.shape):
        raise ValueError("all action arrays must share a shape")
    gaps = np.array([surrogate_gap(q, qa) for q, qa in zip(target_values, mixed_values)])
    return natural_policy * (gaps + exploration * np.sqrt(visits.sum()) / (1.0 + visits))


def select_gap_action(*args, rng: np.random.Generator, **kwargs) -> tuple[int, np.ndarray]:
    scores = gap_ucb_scores(*args, **kwargs)
    return int(rng.choice(np.flatnonzero(np.isclose(scores, scores.max())))), scores
