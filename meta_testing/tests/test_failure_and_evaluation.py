from __future__ import annotations

from meta_testing.failure.metrics import FixedBudgetMetrics
from meta_testing.failure.signature import FailureSignatureBuilder


def test_signature_is_deterministic_and_fixed_budget_counts_unique_failures() -> None:
    builder = FailureSignatureBuilder()
    outcome = {"target_collision": True, "min_ttc": 1.0, "min_distance": 0.5, "max_closing_speed": 12.0}
    first = builder.from_outcome(outcome, "merge", "zone")
    assert first.is_valid_episode and first.is_failure and first == builder.from_outcome(outcome, "merge", "zone")
    metrics = FixedBudgetMetrics(total_budget=3)
    for _ in range(3):
        metrics.add(first)
    result = metrics.summary()
    assert result["cumulative_unique_failures"] == 1


def test_signature_distinguishes_candidate_and_conflict_zone() -> None:
    builder = FailureSignatureBuilder()
    outcome = {"target_collision": True, "min_ttc": 1.0, "min_distance": 0.5, "max_closing_speed": 12.0}
    left = builder.from_outcome(outcome, "merge", "merge:main", "main_conflict")
    right = builder.from_outcome(outcome, "merge", "merge:downstream", "downstream_merge")
    assert left.signature_id != right.signature_id
