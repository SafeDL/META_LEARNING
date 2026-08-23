from __future__ import annotations

import numpy as np

from meta_testing.failure.analyzer import analyze_rollout
from meta_testing.failure.criteria import FailureCriteria
from meta_testing.failure.inner_reward import InnerRiskReward
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


def test_failure_threshold_config_changes_failure_decision_and_inner_reward() -> None:
    broad = FailureCriteria.from_config(
        {"severity_thresholds": {"ttc_s": 5.0, "distance_m": 10.0, "closing_speed_mps": 20.0}, "severity_bins": 5}
    )
    strict = FailureCriteria.from_config(
        {"severity_thresholds": {"ttc_s": 1.0, "distance_m": 1.0, "closing_speed_mps": 20.0}, "severity_bins": 5}
    )
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 0.2, 0.05
    transitions = [{"info": {}, "trajectory_features": features}]
    _, broad_signature = analyze_rollout(transitions, "merge", "zone", "candidate", broad)
    _, strict_signature = analyze_rollout(transitions, "merge", "zone", "candidate", strict)
    assert broad_signature.is_failure
    assert not strict_signature.is_failure
    assert InnerRiskReward(broad)(features, {}, "gap_close", 1, 10) > InnerRiskReward(strict)(features, {}, "gap_close", 1, 10)
