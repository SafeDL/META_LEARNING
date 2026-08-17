"""Calibrated spatiotemporal near-miss measurements and classification."""
from __future__ import annotations

from typing import Any, Mapping
import math


CRITICAL_METRIC_SCHEMA = "spatiotemporal_near_miss_v2"


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
