"""Metrics and budget contracts for Inner few-shot adaptation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def valid_critical_score(outcome: Mapping[str, Any]) -> float:
    if not bool(outcome.get("is_valid_episode", False)):
        return 0.0
    if bool(outcome.get("valid_target_collision", False)):
        return 1.0
    if bool(outcome.get("valid_critical_near_miss", False)):
        return 0.5
    return 0.0


@dataclass(frozen=True)
class AdaptationQualityProtocol:
    """Fixed paired query set; support cost is reported, not equalized."""

    query_cases: int = 8
    support_shots: tuple[int, ...] = (0, 1, 2, 4)

    def total_episodes(self, shots: int) -> int:
        if shots not in self.support_shots:
            raise ValueError("unsupported support-shot count")
        return int(shots) + self.query_cases


@dataclass(frozen=True)
class BudgetEfficiencyProtocol:
    """Support and query episodes share one strict simulator budget."""

    total_episode_budget: int = 20
    support_shots: tuple[int, ...] = (0, 1, 2, 4)

    def query_cases(self, shots: int) -> int:
        if shots not in self.support_shots or not 0 <= shots < self.total_episode_budget:
            raise ValueError("support shots must leave a query episode within budget")
        return self.total_episode_budget - int(shots)


def summarize_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = [valid_critical_score(outcome) for outcome in outcomes]
    invalid = [not bool(outcome.get("is_valid_episode", False)) for outcome in outcomes]
    return {
        "episodes": float(len(outcomes)),
        "valid_critical_score_mean": sum(values) / max(len(values), 1),
        "cumulative_valid_critical_score": sum(values),
        "invalid_rate": sum(invalid) / max(len(invalid), 1),
    }
