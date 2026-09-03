from __future__ import annotations

import numpy as np
import pytest

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


def test_control_telemetry_does_not_invalidate_an_otherwise_critical_outcome() -> None:
    signature = FailureSignatureBuilder().from_outcome(
        {
            "target_collision": True,
            "traffic_max_abs_acceleration_mps2": 100.0,
            "min_ttc": 1.0,
            "min_distance": 0.5,
            "max_closing_speed": 12.0,
        },
        "cutin",
        "merge_window",
    )
    assert signature.is_valid_episode
    assert signature.is_failure


def test_control_telemetry_does_not_invalidate_a_semantically_valid_collision() -> None:
    criteria = FailureCriteria(5.0, 10.0, 20.0, 5)
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 1.0 / 15.0, 1.0 / 100.0
    outcome, signature = analyze_rollout(
        [{"info": {
            "target_collision": True,
            "event_kind": "collision",
            "event_semantic_valid": True,
            "event_execution_valid": True,
            "traffic_max_abs_acceleration_mps2": 100.0,
        },
          "trajectory_features": features}],
        "cutin", "zone", "candidate", criteria,
    )
    assert outcome["target_collision"]
    assert outcome["valid_target_collision"]
    assert signature.is_failure


def test_event_time_execution_validity_survives_post_impact_transient() -> None:
    signature = FailureSignatureBuilder().from_outcome(
        {
            "target_collision": True,
            "adversary_out_of_road": True,
            "event_kind": "collision",
            "event_semantic_valid": True,
            "event_execution_valid": True,
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


def test_analyzer_prefers_a_later_collision_over_an_earlier_near_miss() -> None:
    criteria = FailureCriteria(5.0, 10.0, 20.0, 5)
    features = np.zeros(12, dtype=np.float32)
    transitions = [
        {
            "info": {
                "event_kind": "near_miss",
                "event_semantic_valid": True,
                "event_execution_valid": True,
            },
            "trajectory_features": features,
        },
        {
            "info": {
                "target_collision": True,
                "event_kind": "collision",
                "event_semantic_valid": True,
                "event_execution_valid": True,
            },
            "trajectory_features": features,
        },
    ]
    outcome, signature = analyze_rollout(
        transitions, "cutin", "zone", "candidate", criteria
    )
    assert outcome["event_kind"] == "collision"
    assert outcome["valid_target_collision"]
    assert not outcome["valid_critical_near_miss"]
    assert signature.failure_type == "target_collision"


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
        "event_execution_valid": True,
    }, "trajectory_features": features}]
    strict_transitions = [{"info": {}, "trajectory_features": features}]
    _, broad_signature = analyze_rollout(broad_transitions, "merge", "zone", "candidate", broad)
    _, strict_signature = analyze_rollout(strict_transitions, "merge", "zone", "candidate", strict)
    assert broad_signature.is_failure
    assert not strict_signature.is_failure
    assert InnerRiskReward(broad)(features, {}) > InnerRiskReward(strict)(features, {})


def test_inner_reward_prefers_a_valid_target_collision_over_dense_criticality() -> None:
    criteria = FailureCriteria.from_config(
        {"severity_thresholds": {"ttc_s": 3.0, "distance_m": 5.0, "closing_speed_mps": 20.0}, "severity_bins": 5}
    )
    critical = np.zeros(12, dtype=np.float32)
    critical[8], critical[10] = 3.0 / 15.0, 5.0 / 100.0
    impact = np.zeros(12, dtype=np.float32)
    impact[8], impact[10] = 1.0 / 15.0, 1.0 / 100.0
    reward = InnerRiskReward(criteria)
    collision = reward(impact, {
        "valid_target_collision": True,
        "event_just_captured": True,
        "event_kind": "collision",
        "event_semantic_valid": True,
        "event_execution_valid": True,
    })
    assert collision > reward(critical, {})


def test_cutin_reward_does_not_reward_absolute_adversary_speed() -> None:
    criteria = FailureCriteria(5.0, 10.0, 20.0, 5)
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 0.2, 0.05
    shared = {
        "semantic_challenge_phase_active": True,
        "maneuver_reference_progress": 0.5,
        "maneuver_reference_lateral_error_m": 0.0,
        "maneuver_reference_heading_error_rad": 0.0,
    }
    slow = InnerRiskReward(criteria)(features, {**shared, "adversary_speed_mps": 2.0})
    fast = InnerRiskReward(criteria)(features, {**shared, "adversary_speed_mps": 19.0})
    assert slow == pytest.approx(fast)


def test_inner_reward_is_monotone_toward_contact_inside_challenge_corridor() -> None:
    reward = InnerRiskReward(FailureCriteria(5.0, 10.0, 20.0, 5))
    far = np.zeros(12, dtype=np.float32)
    far[8], far[10] = 4.0 / 15.0, 8.0 / 100.0
    close = np.zeros(12, dtype=np.float32)
    close[8], close[10] = 1.0 / 15.0, 2.0 / 100.0
    info = {"semantic_challenge_phase_active": True}
    assert reward(close, info) > reward(far, info)


def test_inner_reward_bonus_requires_a_valid_critical_event() -> None:
    criteria = FailureCriteria.from_config(
        {"severity_thresholds": {"ttc_s": 3.0, "distance_m": 5.0, "closing_speed_mps": 20.0}, "severity_bins": 5}
    )
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 1.0 / 15.0, 1.0 / 100.0
    reward = InnerRiskReward(criteria)
    event = reward(features, {
        "valid_target_collision": True,
        "event_just_captured": True,
        "event_kind": "collision",
        "event_semantic_valid": True,
        "event_execution_valid": True,
    })
    invalid_event = reward(
        features,
        {
            "valid_target_collision": False,
            "event_kind": "collision",
            "event_semantic_valid": True,
            "event_execution_valid": False,
            "wrong_route": True,
        },
    )
    assert event > invalid_event


def test_inner_reward_emits_near_miss_bonus_only_on_capture() -> None:
    reward = InnerRiskReward(FailureCriteria(3.0, 5.0, 20.0, 5))
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 1.0 / 15.0, 1.0 / 100.0
    event_info = {
        "event_kind": "near_miss",
        "event_semantic_valid": True,
        "event_execution_valid": True,
        "event_just_captured": True,
    }
    latched_info = {**event_info, "event_just_captured": False}
    assert reward(features, event_info) > reward(
        features, latched_info
    )


def test_inner_reward_uses_one_terminal_bonus_for_collision_and_near_miss() -> None:
    criteria = FailureCriteria(3.0, 5.0, 20.0, 5)
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10] = 1.0 / 15.0, 1.0 / 100.0
    collision = InnerRiskReward(criteria)(features, {
        "event_kind": "collision",
        "event_just_captured": True,
        "valid_target_collision": True,
        "event_semantic_valid": True,
        "event_execution_valid": True,
    })
    near_miss = InnerRiskReward(criteria)(features, {
        "event_kind": "near_miss",
        "event_just_captured": True,
        "event_semantic_valid": True,
        "event_execution_valid": True,
    })
    assert near_miss == pytest.approx(collision)


def test_inner_reward_uses_dense_signal_only_inside_the_challenge_corridor() -> None:
    reward = InnerRiskReward(FailureCriteria(3.0, 5.0, 20.0, 5))
    features = np.zeros(12, dtype=np.float32)
    assert reward(features, {"semantic_challenge_phase_active": False}) == pytest.approx(0.0)
    assert reward(features, {"semantic_challenge_phase_active": True}) > 0.0


def test_inner_reward_does_not_reward_absolute_adversary_speed_before_interaction() -> None:
    reward = InnerRiskReward(FailureCriteria(5.0, 10.0, 20.0, 5))
    features = np.ones(12, dtype=np.float32)
    stopped = reward(
        features,
        {
            "semantic_challenge_phase_active": False,
            "adversary_speed_mps": 0.0,
            "speed_limit_mps": 20.0,
        },
    )
    moving = reward(
        features,
        {
            "semantic_challenge_phase_active": False,
            "adversary_speed_mps": 10.0,
            "speed_limit_mps": 20.0,
        },
    )
    assert moving == pytest.approx(stopped)
