from __future__ import annotations

import numpy as np

from mvr.failure.analyzer import analyze_rollout
from mvr.failure.criteria import FailureCriteria
from mvr.failure.inner_reward import InnerRiskReward
from mvr.failure.metrics import FixedBudgetMetrics
from mvr.failure.signature import FailureSignatureBuilder


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


def test_traffic_violation_invalidates_an_otherwise_critical_outcome() -> None:
    signature = FailureSignatureBuilder().from_outcome(
        {
            "target_collision": True,
            "adversary_traffic_violation": True,
            "min_ttc": 1.0,
            "min_distance": 0.5,
            "max_closing_speed": 12.0,
        },
        "cutin",
        "merge_window",
    )
    assert not signature.is_valid_episode
    assert not signature.is_failure


def test_traffic_violation_cannot_be_a_valid_target_collision() -> None:
    criteria = FailureCriteria(5.0, 10.0, 20.0, 5)
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 1.0 / 15.0, 1.0 / 100.0
    outcome, signature = analyze_rollout(
        [{"info": {"target_collision": True, "adversary_traffic_violation": True},
          "trajectory_features": features}],
        "cutin", "zone", "candidate", criteria,
    )
    assert outcome["target_collision"]
    assert not outcome["valid_target_collision"]
    assert not signature.is_failure


def test_event_time_traffic_validity_survives_post_impact_transient() -> None:
    signature = FailureSignatureBuilder().from_outcome(
        {
            "target_collision": True,
            "adversary_out_of_road": True,
            "event_kind": "collision",
            "event_semantic_valid": True,
            "event_traffic_valid": True,
            "valid_target_collision": True,
            "is_valid_episode": True,
            "min_ttc": 1.0,
            "min_distance": 0.5,
            "max_closing_speed": 12.0,
        },
        "cutin",
        "merge_window",
    )
    assert signature.is_valid_episode
    assert signature.is_failure


def test_failure_threshold_config_changes_failure_decision_and_inner_reward() -> None:
    broad = FailureCriteria.from_config(
        {"severity_thresholds": {"ttc_s": 5.0, "distance_m": 10.0, "closing_speed_mps": 20.0}, "severity_bins": 5}
    )
    strict = FailureCriteria.from_config(
        {"severity_thresholds": {"ttc_s": 1.0, "distance_m": 1.0, "closing_speed_mps": 20.0}, "severity_bins": 5}
    )
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 0.2, 0.05
    broad_transitions = [{"info": {
        "event_kind": "near_miss",
        "event_semantic_valid": True,
        "event_traffic_valid": True,
    }, "trajectory_features": features}]
    strict_transitions = [{"info": {}, "trajectory_features": features}]
    _, broad_signature = analyze_rollout(broad_transitions, "merge", "zone", "candidate", broad)
    _, strict_signature = analyze_rollout(strict_transitions, "merge", "zone", "candidate", strict)
    assert broad_signature.is_failure
    assert not strict_signature.is_failure
    assert InnerRiskReward(broad)(features, {}, "gap_close", 1, 10) > InnerRiskReward(strict)(features, {}, "gap_close", 1, 10)


def test_inner_reward_prefers_critical_interaction_over_direct_impact() -> None:
    criteria = FailureCriteria.from_config(
        {"severity_thresholds": {"ttc_s": 3.0, "distance_m": 5.0, "closing_speed_mps": 20.0}, "severity_bins": 5}
    )
    critical = np.zeros(12, dtype=np.float32)
    critical[8], critical[10] = 3.0 / 15.0, 5.0 / 100.0
    impact = np.zeros(12, dtype=np.float32)
    reward = InnerRiskReward(criteria)
    assert reward(critical, {}, "approach_conflict", 1, 10) > reward(impact, {}, "approach_conflict", 1, 10)


def test_inner_reward_bonus_requires_a_valid_critical_event() -> None:
    criteria = FailureCriteria.from_config(
        {"severity_thresholds": {"ttc_s": 3.0, "distance_m": 5.0, "closing_speed_mps": 20.0}, "severity_bins": 5}
    )
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 1.0 / 15.0, 1.0 / 100.0
    reward = InnerRiskReward(criteria)
    event = reward(features, {
        "valid_target_collision": True,
        "event_kind": "collision",
        "event_semantic_valid": True,
        "event_traffic_valid": True,
    }, "approach_conflict", 1, 10)
    invalid_event = reward(
        features,
        {
            "valid_target_collision": False,
            "event_kind": "collision",
            "event_semantic_valid": True,
            "event_traffic_valid": False,
            "adversary_traffic_violation": True,
        },
        "approach_conflict",
        1,
        10,
    )
    assert event > invalid_event


def test_inner_reward_has_no_positive_success_signal_without_consequence() -> None:
    reward = InnerRiskReward(FailureCriteria(3.0, 5.0, 20.0, 5))
    features = np.zeros(12, dtype=np.float32)
    assert reward(features, {}, "approach_conflict", 1, 10) <= 0.0
