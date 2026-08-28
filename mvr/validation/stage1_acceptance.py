"""Machine-readable acceptance checks for the Formal Stage1 Inner run."""
from __future__ import annotations

from typing import Any, Mapping

import torch


FAMILIES = ("merge", "cutin", "roundabout")
VIOLATION_FIELDS = (
    "non_target_collision",
    "adversary_out_of_road",
    "sut_out_of_road",
    "wrong_route",
    "adversary_traffic_violation",
)


def _policy(validation: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in validation.get("policies", []):
        if row.get("policy") == name:
            return row
    raise ValueError(f"validation is missing policy {name!r}")


def _finite_checkpoint(checkpoint_state: Mapping[str, Any]) -> bool:
    tensors = [value for value in checkpoint_state.get("model", {}).values() if torch.is_tensor(value)]
    return bool(tensors) and all(torch.isfinite(value).all().item() for value in tensors)


def _coverage_pass(coverage: Mapping[str, Any]) -> bool:
    task_counts = coverage.get("task_episode_counts", {})
    family_counts = coverage.get("family_episode_counts", {})
    geometry_counts = coverage.get("geometry_episode_counts", {})
    sut_counts = coverage.get("sut_episode_counts", {})
    candidate_counts = coverage.get("candidate_episode_counts", {})
    option_counts = coverage.get("option_episode_counts", {})
    return bool(
        len(task_counts) == 36
        and all(int(value) == 5 for value in task_counts.values())
        and set(family_counts) == set(FAMILIES)
        and len(geometry_counts) == 9
        and len(sut_counts) == 4
        and len(candidate_counts) == 7
        and len(option_counts) == 3
    )


def _no_violations(policy: Mapping[str, Any]) -> bool:
    rates = policy.get("violation_rates", {})
    return all(float(rates.get(field, 1.0)) == 0.0 for field in VIOLATION_FIELDS)


def audit_stage1(
    coverage: Mapping[str, Any],
    validation: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    *,
    pytest_passed: bool,
    compileall_passed: bool,
) -> dict[str, Any]:
    """Evaluate the G1--G5 requirements declared in the Stage1 plan."""
    base = _policy(validation, "base")
    random = _policy(validation, "random_residual")
    trained = _policy(validation, "trained_inner")
    finite = _finite_checkpoint(checkpoint_state)
    family_uplifts = {
        family: float(trained["by_family"][family]["valid_critical_rate"])
        - max(
            float(base["by_family"][family]["valid_critical_rate"]),
            float(random["by_family"][family]["valid_critical_rate"]),
        )
        for family in FAMILIES
    }
    joint_effect = float(trained["valid_critical_rate"]) - max(
        float(base["valid_critical_rate"]),
        float(random["valid_critical_rate"]),
    )
    traffic_safe = float(trained["violation_rates"]["adversary_traffic_violation"]) <= float(
        random["violation_rates"]["adversary_traffic_violation"]
    )
    g1 = bool(pytest_passed and compileall_passed and finite)
    g2 = _coverage_pass(coverage)
    g3 = bool(
        float(trained["valid_rate"]) >= 0.80
        and traffic_safe
        and _no_violations(trained)
    )
    g4 = bool(
        joint_effect > 0.0
        and sum(value > 0.0 for value in family_uplifts.values()) >= 2
    )
    g5 = bool(joint_effect > 0.0)
    return {
        "stage": "formal_stage1",
        "regime": validation.get("regime"),
        "gates": {
            "G1_engineering": {
                "pass": g1,
                "pytest_passed": bool(pytest_passed),
                "compileall_passed": bool(compileall_passed),
                "checkpoint_reload_finite": finite,
            },
            "G2_coverage": {"pass": g2},
            "G3_semantic_traffic": {
                "pass": g3,
                "trained_valid_rate": float(trained["valid_rate"]),
                "trained_no_violations": _no_violations(trained),
                "traffic_violation_not_higher_than_random": traffic_safe,
            },
            "G4_learned_adversarial_effect": {
                "pass": g4,
                "overall_uplift_over_strongest_baseline": joint_effect,
                "family_uplifts_over_strongest_baseline": family_uplifts,
                "positive_family_count": sum(value > 0.0 for value in family_uplifts.values()),
            },
            "G5_joint_transfer": {
                "pass": g5,
                "G_joint": joint_effect,
            },
        },
        "pass": bool(g1 and g2 and g3 and g4 and g5),
    }
