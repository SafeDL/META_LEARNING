from __future__ import annotations

from types import SimpleNamespace

import torch

from mvr.failure.criteria import FailureCriteria
from mvr.validation.stage1_preflight import (
    audit_event_bonus_once,
    audit_learning,
    audit_parameter_update,
    audit_preflight_gates,
    audit_replay_contract,
    audit_reachability,
    audit_reward,
    audit_training_signal,
)


def test_reward_audit_states_the_current_threshold_objective() -> None:
    criteria = FailureCriteria(5.0, 10.0, 20.0, 5)

    report = audit_reward(criteria)

    assert report["pass"]
    assert report["shape"] == "threshold_centered_gaussian"
    assert not report["severity_monotonic"]
    assert report["event_bonus_direction"]
    assert not audit_reward(criteria, objective="severity_monotonic")["pass"]


def test_preflight_checks_one_shot_bonus_and_raw_replay() -> None:
    bonus = audit_event_bonus_once(FailureCriteria(5.0, 10.0, 20.0, 5))
    rows = [
        SimpleNamespace(
            state=torch.zeros(11),
            action=torch.zeros(2),
            next_state=torch.ones(11),
            reward=-0.1,
            done=False,
        )
    ]

    assert bonus["pass"]
    assert bonus["bonus_steps"] == 1
    assert audit_replay_contract(rows)["pass"]


def test_preflight_checks_update_and_training_signal() -> None:
    before = {"weight": torch.zeros(2)}
    after = {"weight": torch.ones(2)}
    losses = [{"inner_actor_loss": 1.0, "inner_critic_loss": 2.0}]
    signal = {
        f"family:{family}": {"valid_event_episodes": 1}
        for family in ("merge", "cutin", "roundabout")
    }

    assert audit_parameter_update(before, after, losses)["pass"]
    assert audit_training_signal(signal)["pass"]
    assert audit_preflight_gates({"reward": {"pass": True}, "update": {"pass": False}})["failed_gates"] == ["update"]


def test_reachability_audit_requires_each_family_to_have_effective_residual() -> None:
    residuals = {
        "base": {"valid_rate": 1.0, "valid_critical_rate": 0.0, "median_min_ttc": 8.0, "median_min_distance": 20.0},
        "acceleration_brake": {"valid_rate": 1.0, "valid_critical_rate": 0.25, "median_min_ttc": 4.0, "median_min_distance": 8.0, "challenge_phase_rate": 0.4},
    }
    summary = {
        "paired_initial_conditions_verified": True,
        "by_family": {family: {"by_residual": residuals} for family in ("merge", "cutin", "roundabout")},
    }

    assert audit_reachability(summary)["pass"]
    summary["by_family"]["roundabout"]["by_residual"]["acceleration_brake"]["challenge_phase_rate"] = 0.0
    assert not audit_reachability(summary)["pass"]


def test_learning_audit_requires_two_families_and_positive_signal() -> None:
    def policy(offset: float) -> dict[str, object]:
        return {
            "by_family": {
                family: {
                    "median_min_ttc": 10.0 - offset,
                    "median_min_distance": 20.0 - offset,
                    "valid_event_count": offset,
                    "invalid_rate": 0.0,
                }
                for family in ("merge", "cutin", "roundabout")
            }
        }

    signal = {
        f"family:{family}": {"positive_reward_transition_fraction": 0.1}
        for family in ("merge", "cutin", "roundabout")
    }
    report = audit_learning({"base": policy(0.0), "random_residual": policy(1.0), "trained_inner": policy(2.0)}, signal)

    assert report["pass"]
