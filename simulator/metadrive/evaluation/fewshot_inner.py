"""Outcome scoring and paired statistics shared by Mining experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def valid_critical_score(outcome: Mapping[str, Any]) -> float:
    """Return the fixed formal score for one simulator outcome."""
    if not bool(outcome.get("is_valid_episode", False)):
        return 0.0
    if bool(outcome.get("valid_target_collision", False)):
        return 1.0
    if bool(outcome.get("valid_critical_near_miss", False)):
        return 0.5
    return 0.0


def summarize_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Summarize formal scores while retaining invalid-episode accounting."""
    scores = [valid_critical_score(outcome) for outcome in outcomes]
    total = max(len(outcomes), 1)
    invalid = sum(not bool(outcome.get("is_valid_episode", False)) for outcome in outcomes)
    return {
        "episodes": float(len(outcomes)),
        "valid_critical_score_mean": sum(scores) / total,
        "cumulative_valid_critical_score": sum(scores),
        "invalid_rate": invalid / total,
    }


def bootstrap_mean_interval(deltas: np.ndarray, *, seed: int) -> tuple[float, float]:
    """Return the paired mean's 95% interval using 5,000 bootstrap samples."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(5000)]
    )
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
