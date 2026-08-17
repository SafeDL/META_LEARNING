"""Calibrated spatiotemporal near-miss measurements and classification."""
from __future__ import annotations

from typing import Any, Mapping
import math


CRITICAL_METRIC_SCHEMA = "spatiotemporal_near_miss_v2"
LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA = "logical_order_spatiotemporal_near_miss_v3"
SUPPORTED_CRITICAL_METRIC_SCHEMAS = frozenset({
    CRITICAL_METRIC_SCHEMA,
    LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA,
})


def is_strict_near_miss_schema(schema: str) -> bool:
    """Whether a schema uses calibrated collision-free near-miss termination."""
    return str(schema) in SUPPORTED_CRITICAL_METRIC_SCHEMAS


def conflict_entry_order_satisfied(required_order: str, first_entry_role: str | None) -> bool:
    """Check the additional v3 semantic condition without using observations.

    ``first_entry_role`` is recorded from route geometry during a rollout;
    neither this role nor the frozen target rule is an actor/critic/context
    feature.  Keeping the predicate pure makes the v3 rule testable and
    prevents a missing order observation from being silently accepted.
    """
    order = str(required_order)
    if order == "any":
        return True
    if order not in {"adversary_first", "sut_first"}:
        raise ValueError("required conflict-entry order is unsupported")
    return str(first_entry_role) == order.removesuffix("_first")


def critical_measurements(
    arrival: Mapping[str, float | str],
    *,
    pair_distance_m: float,
    ttc_s: float,
    closing_speed_mps: float,
    thresholds: Mapping[str, Any],
) -> dict[str, float | bool]:
    """Return raw continuous risk values plus the calibrated joint decision."""
    # Signed route ETA remains informative immediately after a vehicle passes
    # the conflict point. The legacy non-negative ETA collapses both vehicles
    # to zero and creates artificial millisecond-scale "near misses" downstream.
    adv_time_key = "adversary_signed_time_s" if "adversary_signed_time_s" in arrival else "adversary_time_s"
    sut_time_key = "sut_signed_time_s" if "sut_signed_time_s" in arrival else "sut_time_s"
    gap = abs(float(arrival[adv_time_key]) - float(arrival[sut_time_key]))
    joint_distance = max(
        abs(float(arrival["adversary_signed_distance_m"])),
        abs(float(arrival["sut_signed_distance_m"])),
    )
    limits = {
        "arrival": float(thresholds["arrival_gap_threshold_s"]),
        "joint": float(thresholds["joint_conflict_distance_threshold_m"]),
        "pair": float(thresholds["pair_distance_threshold_m"]),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in limits.values()):
        raise ValueError("critical thresholds must be finite positive values")
    margins = (
        1.0 - gap / limits["arrival"],
        1.0 - joint_distance / limits["joint"],
        1.0 - float(pair_distance_m) / limits["pair"],
    )
    return {
        "arrival_gap_abs_s": gap,
        "joint_conflict_distance_m": joint_distance,
        "pair_distance_m": float(pair_distance_m),
        "ttc_s": float(ttc_s),
        "closing_speed_mps": float(closing_speed_mps),
        "critical_margin": float(min(margins)),
        "spatiotemporal_near_miss_candidate": bool(min(margins) >= 0.0),
    }


def strict_near_miss_potential(
    measurements: Mapping[str, float | bool],
    thresholds: Mapping[str, Any],
    *,
    prospective_order_satisfied: bool,
) -> float:
    """Return a bounded smooth approach signal for the strict objective.

    This is not a success classifier: only ``critical_measurements`` can
    satisfy the hard calibrated conjunction and terminate as VCSR.  It exists
    solely to make all three continuous constraints visible to SAC before the
    final near-miss state is reached.
    """
    if not prospective_order_satisfied:
        return 0.0
    scales = (
        ("arrival_gap_abs_s", "arrival_gap_threshold_s", 4.0),
        ("joint_conflict_distance_m", "joint_conflict_distance_threshold_m", 6.0),
        ("pair_distance_m", "pair_distance_threshold_m", 2.0),
    )
    values = []
    for measurement_key, threshold_key, relaxation in scales:
        value = float(measurements[measurement_key])
        threshold = float(thresholds[threshold_key])
        if not math.isfinite(value) or not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("strict near-miss potential requires finite positive measurements and thresholds")
        values.append(math.exp(-max(0.0, value) / (relaxation * threshold)))
    return float(min(values))


def collision_risk_barrier(
    measurements: Mapping[str, float | bool],
    thresholds: Mapping[str, Any],
    *,
    safe_pair_distance_ratio: float,
) -> float:
    """Return a bounded pre-contact cost for strict near-miss shaping.

    VCSR correctly permits all pair distances below its calibrated upper
    bound, but a monotone dense proximity reward over that interval otherwise
    makes physical contact an attractive final step.  The barrier is zero in
    the outer, safe part of the same frozen band and rises continuously only
    inside a configured fraction of it.  It never classifies, terminates, or
    reads the hidden task-order rule.
    """
    ratio = float(safe_pair_distance_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("safe_pair_distance_ratio must lie strictly between 0 and 1")
    threshold = float(thresholds["pair_distance_threshold_m"])
    distance = float(measurements["pair_distance_m"])
    if not math.isfinite(threshold) or threshold <= 0.0 or not math.isfinite(distance):
        raise ValueError("collision-risk barrier requires finite positive pair-distance threshold")
    floor = ratio * threshold
    return float(min(max((floor - distance) / floor, 0.0), 1.0))
