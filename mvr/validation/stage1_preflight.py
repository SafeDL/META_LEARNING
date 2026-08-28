"""Pure, machine-readable checks used before a Formal Stage1 run.

The simulator-heavy parts of the preflight are deliberately kept in the
script layer.  This module only evaluates recorded rollouts and optimizer
artifacts, so the acceptance rules can be tested without MetaDrive.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from ..failure.criteria import FailureCriteria
from ..failure.inner_reward import InnerRiskReward
from ..state import PhysicalStateExtractor


FAMILIES = ("merge", "cutin", "roundabout")
VIOLATION_FIELDS = (
    "non_target_collision",
    "adversary_out_of_road",
    "sut_out_of_road",
    "wrong_route",
    "adversary_traffic_violation",
)


def _finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    try:
        return bool(np.isfinite(np.asarray(value, dtype=float)).all())
    except (TypeError, ValueError):
        return False


def audit_reward(
    criteria: FailureCriteria,
    *,
    objective: str = "threshold_proximity",
) -> dict[str, Any]:
    """Audit reward finiteness, semantic gating, and the declared shape.

    ``InnerRiskReward`` intentionally uses threshold-centred Gaussian
    shaping.  The audit reports that shape explicitly; callers that want a
    severity-monotone objective can pass ``severity_monotonic`` and receive a
    failing gate instead of silently accepting a different objective.
    """
    if objective not in {"threshold_proximity", "severity_monotonic"}:
        raise ValueError("unknown reward objective")
    reward = InnerRiskReward(criteria)
    ttc_grid = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
    distance_grid = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)

    def value(ttc: float, distance: float, info: Mapping[str, object] | None = None) -> float:
        features = np.zeros(12, dtype=np.float32)
        features[8], features[10], features[11] = ttc / 15.0, distance / 100.0, 1.0
        return reward(features, info or {})

    ttc_curve = [value(ttc, criteria.distance_m) for ttc in ttc_grid]
    distance_curve = [value(criteria.ttc_s, distance) for distance in distance_grid]
    invalid_info = {VIOLATION_FIELDS[-1]: True}
    valid_event = {
        "event_kind": "near_miss",
        "event_just_captured": True,
        "event_semantic_valid": True,
        "event_traffic_valid": True,
    }
    no_event = value(criteria.ttc_s, criteria.distance_m)
    captured = value(criteria.ttc_s, criteria.distance_m, valid_event)
    invalid_event = value(criteria.ttc_s, criteria.distance_m, {**valid_event, **invalid_info})
    finite = all(_finite(row) for row in (*ttc_curve, *distance_curve, no_event, captured, invalid_event))
    peak_ttc = ttc_grid[int(np.argmax(ttc_curve))]
    peak_distance = distance_grid[int(np.argmax(distance_curve))]
    threshold_shape = bool(
        abs(peak_ttc - criteria.ttc_s) <= max(criteria.ttc_s, 1.0)
        and abs(peak_distance - criteria.distance_m) <= max(criteria.distance_m, 1.0)
    )
    ttc_monotonic = all(left >= right for left, right in zip(ttc_curve, ttc_curve[1:]))
    distance_monotonic = all(left >= right for left, right in zip(distance_curve, distance_curve[1:]))
    shape_pass = threshold_shape if objective == "threshold_proximity" else ttc_monotonic and distance_monotonic
    return {
        "pass": bool(finite and no_event <= 0.0 and captured > no_event and invalid_event < captured and shape_pass),
        "objective": objective,
        "shape": "threshold_centered_gaussian",
        "finite": finite,
        "no_event_non_positive": bool(no_event <= 0.0),
        "event_bonus_direction": bool(captured > no_event),
        "invalid_event_penalized": bool(invalid_event < captured),
        "ttc_grid_s": list(ttc_grid),
        "ttc_reward": ttc_curve,
        "distance_grid_m": list(distance_grid),
        "distance_reward": distance_curve,
        "peak_ttc_s": peak_ttc,
        "peak_distance_m": peak_distance,
        "threshold_shape": threshold_shape,
        "severity_monotonic": bool(ttc_monotonic and distance_monotonic),
    }


def audit_event_bonus_once(criteria: FailureCriteria) -> dict[str, Any]:
    """Check that a latched near-miss cannot repeatedly earn its bonus."""
    reward = InnerRiskReward(criteria)
    features = np.zeros(12, dtype=np.float32)
    features[8], features[10], features[11] = criteria.ttc_s / 15.0, criteria.distance_m / 100.0, 1.0
    event = {
        "event_kind": "near_miss",
        "event_semantic_valid": True,
        "event_traffic_valid": True,
    }
    captured = reward(features, {**event, "event_just_captured": True})
    latched = [reward(features, {**event, "event_just_captured": False}) for _ in range(5)]
    return {
        "pass": bool(captured > latched[0] and all(value == latched[0] for value in latched)),
        "capture_reward": captured,
        "latched_rewards": latched,
        "bonus_steps": 1 if captured > latched[0] else 0,
    }


def audit_replay_contract(rows: Iterable[Any], *, state_dim: int = PhysicalStateExtractor.dimension) -> dict[str, Any]:
    """Validate the raw replay interface emitted by an Inner rollout."""
    rows = list(rows)
    failures: list[str] = []
    for index, row in enumerate(rows):
        def field(name: str) -> Any:
            if isinstance(row, Mapping):
                return row.get(name)
            return getattr(row, name)

        state = np.asarray(field("state"), dtype=float)
        action = np.asarray(field("action"), dtype=float)
        next_state = np.asarray(field("next_state"), dtype=float)
        reward = field("reward")
        done = field("done")
        if state.shape != (state_dim,):
            failures.append(f"row {index}: state shape {state.shape}")
        if action.shape != (2,):
            failures.append(f"row {index}: action shape {action.shape}")
        if next_state.shape != (state_dim,):
            failures.append(f"row {index}: next_state shape {next_state.shape}")
        if not _finite(state) or not _finite(action) or not _finite(next_state) or not _finite(reward):
            failures.append(f"row {index}: non-finite value")
        if not isinstance(done, (bool, np.bool_)):
            failures.append(f"row {index}: done is not bool")
    return {"pass": not failures and bool(rows), "rows": len(rows), "failures": failures}


def audit_parameter_update(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    losses: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Check finite SAC losses and non-zero parameter movement."""
    names = sorted(set(before) & set(after))
    deltas = [float(torch.linalg.vector_norm(after[name] - before[name]).detach().cpu()) for name in names]
    finite_losses = all(_finite(value) for loss in losses for value in loss.values())
    return {
        "pass": bool(names and finite_losses and any(delta > 0.0 for delta in deltas)),
        "finite_losses": finite_losses,
        "parameter_l2_change": float(np.sqrt(np.sum(np.square(deltas)))) if deltas else 0.0,
        "changed_parameter_tensors": int(sum(delta > 0.0 for delta in deltas)),
        "loss_updates": len(losses),
    }


def audit_training_signal(
    signal: Mapping[str, Mapping[str, Any]],
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    """Require a report bucket and at least one positive signal per family."""
    missing = [f"family:{family}" for family in families if f"family:{family}" not in signal]
    positive = {
        family: bool(
            signal.get(f"family:{family}", {}).get("valid_event_episodes", 0)
            or signal.get(f"family:{family}", {}).get("positive_reward_transition_fraction", 0.0) > 0.0
        )
        for family in families
    }
    return {
        "pass": bool(not missing and all(positive.values())),
        "missing_buckets": missing,
        "positive_signal_by_family": positive,
    }


def audit_reachability(summary: Mapping[str, Any], families: Sequence[str] = FAMILIES) -> dict[str, Any]:
    """Evaluate the no-RL action sweep without conflating it with performance."""
    by_family = summary.get("by_family", {})
    family_reports: dict[str, dict[str, Any]] = {}
    for family in families:
        family_summary = by_family.get(family, {})
        residuals = family_summary.get("by_residual", {})
        base = residuals.get("base", {})
        base_legal = float(base.get("valid_rate", 0.0)) >= 0.8
        challenge = max(
            float(row.get("challenge_phase_rate", 0.0))
            for name, row in residuals.items()
            if name != "base"
        ) if any(name != "base" for name in residuals) else 0.0
        critical = max(
            float(row.get("valid_critical_rate", 0.0))
            for name, row in residuals.items()
            if name != "base"
        ) if any(name != "base" for name in residuals) else 0.0
        base_ttc = float(base.get("median_min_ttc", np.inf))
        base_distance = float(base.get("median_min_distance", np.inf))
        risk_effect = any(
            float(row.get("valid_critical_rate", 0.0)) > float(base.get("valid_critical_rate", 0.0))
            or float(row.get("median_min_ttc", np.inf)) < base_ttc
            or float(row.get("median_min_distance", np.inf)) < base_distance
            for name, row in residuals.items()
            if name != "base"
        )
        family_reports[family] = {
            "pass": bool(base_legal and challenge > 0.0 and risk_effect),
            "base_legal": base_legal,
            "challenge_phase_rate": challenge,
            "valid_critical_rate": critical,
            "risk_effect": risk_effect,
        }
    return {
        "pass": bool(summary.get("paired_initial_conditions_verified", False) and all(
            report["pass"] for report in family_reports.values()
        )),
        "paired_initial_conditions_verified": bool(summary.get("paired_initial_conditions_verified", False)),
        "by_family": family_reports,
    }


def audit_learning(
    policies: Mapping[str, Mapping[str, Any]],
    signal: Mapping[str, Mapping[str, Any]],
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    """Evaluate the small shared-SAC pilot against paired baselines."""
    required = ("base", "random_residual", "trained_inner")
    missing = [name for name in required if name not in policies]
    if missing:
        return {"pass": False, "missing_policies": missing, "by_family": {}}
    by_family: dict[str, dict[str, Any]] = {}
    for family in families:
        values = {
            name: policies[name].get("by_family", {}).get(family, {})
            for name in required
        }
        trained = values["trained_inner"]
        baseline = values["base"], values["random_residual"]
        risk_effect = any(
            float(trained.get(metric, np.inf)) < min(float(row.get(metric, np.inf)) for row in baseline)
            for metric in ("median_min_ttc", "median_min_distance")
        )
        event_effect = float(trained.get("valid_event_count", 0.0)) > max(
            float(row.get("valid_event_count", 0.0)) for row in baseline
        )
        invalid_safe = float(trained.get("invalid_rate", 1.0)) <= max(
            float(row.get("invalid_rate", 1.0)) for row in baseline
        )
        by_family[family] = {
            "risk_effect": risk_effect,
            "event_effect": event_effect,
            "invalid_rate_not_higher": invalid_safe,
            "pass": bool(invalid_safe and (risk_effect or event_effect)),
        }
    signal_report = audit_training_signal(signal, families)
    return {
        "pass": bool(not missing and sum(row["pass"] for row in by_family.values()) >= 2 and signal_report["pass"]),
        "missing_policies": missing,
        "by_family": by_family,
        "training_signal": signal_report,
    }


def audit_nuisance_invariance(
    rows: Iterable[Mapping[str, Any]],
    families: Sequence[str] = ("merge", "roundabout"),
) -> dict[str, Any]:
    """Check that Cut-in-only onset has no effect on other families."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["family"]), str(row["case_id"])), []).append(row)
    checked, failures = 0, []
    for (family, case_id), group in groups.items():
        if family not in families:
            continue
        checked += 1
        baseline = group[0]
        for row in group[1:]:
            if row.get("trajectory_digest") != baseline.get("trajectory_digest") or row.get("outcome_digest") != baseline.get("outcome_digest"):
                failures.append(f"{family}:{case_id}")
    return {"pass": bool(checked and not failures), "checked_cases": checked, "failures": failures}


def audit_profile_effect(
    rows: Iterable[Mapping[str, Any]],
    profiles: Sequence[str],
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    """Require a complete profile comparison and an explicit neutral choice."""
    rows = list(rows)
    observed = {(str(row.get("family")), str(row.get("profile"))) for row in rows}
    expected = {(family, profile) for family in families for profile in profiles}
    missing = sorted(expected - observed)
    spreads = {}
    for family in families:
        family_rows = [row for row in rows if row.get("family") == family]
        ttc = [float(row["min_ttc"]) for row in family_rows if row.get("min_ttc") is not None]
        distance = [float(row["min_distance"]) for row in family_rows if row.get("min_distance") is not None]
        spreads[family] = {
            "min_ttc_spread": float(max(ttc) - min(ttc)) if ttc else 0.0,
            "min_distance_spread": float(max(distance) - min(distance)) if distance else 0.0,
        }
    return {
        "pass": not missing,
        "missing": missing,
        "profile_count": len(profiles),
        "by_family": spreads,
        "controlled_profile": profiles[0] if profiles else None,
    }


def audit_preflight_gates(gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Combine phase decisions without hiding skipped or failed phases."""
    normalized = {name: dict(value) for name, value in gates.items()}
    failed = [name for name, value in normalized.items() if not bool(value.get("pass", False))]
    return {"pass": not failed, "failed_gates": failed, "gates": normalized}
