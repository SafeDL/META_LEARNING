"""Formal outcome scoring shared by Mining experiments."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


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
