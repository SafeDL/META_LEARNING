from __future__ import annotations

import torch

from mvr.validation.stage1_acceptance import audit_stage1


def _policy(rate: float, by_family: dict[str, float]) -> dict[str, object]:
    return {
        "valid_rate": 1.0,
        "valid_critical_rate": rate,
        "violation_rates": {
            "non_target_collision": 0.0,
            "adversary_out_of_road": 0.0,
            "sut_out_of_road": 0.0,
            "wrong_route": 0.0,
            "adversary_traffic_violation": 0.0,
        },
        "by_family": {
            family: {"valid_critical_rate": value} for family, value in by_family.items()
        },
    }


def _coverage() -> dict[str, object]:
    return {
        "task_episode_counts": {f"task-{index}": 5 for index in range(36)},
        "family_episode_counts": {"merge": 60, "cutin": 60, "roundabout": 60},
        "geometry_episode_counts": {f"g{index}": 20 for index in range(9)},
        "sut_episode_counts": {f"sut-{index}": 45 for index in range(4)},
        "candidate_episode_counts": {f"candidate-{index}": 25 for index in range(7)},
        "option_episode_counts": {f"option-{index}": 60 for index in range(3)},
    }


def test_stage1_audit_requires_effect_on_two_families() -> None:
    validation = {
        "regime": "validation_sut_validation_geometry",
        "policies": [
            {"policy": "base", **_policy(0.10, {"merge": 0.0, "cutin": 0.10, "roundabout": 0.0})},
            {"policy": "random_residual", **_policy(0.15, {"merge": 0.10, "cutin": 0.15, "roundabout": 0.0})},
            {"policy": "trained_inner", **_policy(0.25, {"merge": 0.20, "cutin": 0.30, "roundabout": 0.0})},
        ],
    }
    result = audit_stage1(
        _coverage(), validation, {"model": {"weight": torch.ones(1)}}, pytest_passed=True, compileall_passed=True
    )
    assert result["pass"]
    assert result["gates"]["G4_learned_adversarial_effect"]["positive_family_count"] == 2


def test_stage1_audit_rejects_single_family_uplift() -> None:
    validation = {
        "policies": [
            {"policy": "base", **_policy(0.0, {"merge": 0.0, "cutin": 0.0, "roundabout": 0.0})},
            {"policy": "random_residual", **_policy(0.10, {"merge": 0.10, "cutin": 0.0, "roundabout": 0.0})},
            {"policy": "trained_inner", **_policy(0.20, {"merge": 0.0, "cutin": 0.40, "roundabout": 0.0})},
        ],
    }
    result = audit_stage1(
        _coverage(), validation, {"model": {"weight": torch.ones(1)}}, pytest_passed=True, compileall_passed=True
    )
    assert not result["gates"]["G4_learned_adversarial_effect"]["pass"]
