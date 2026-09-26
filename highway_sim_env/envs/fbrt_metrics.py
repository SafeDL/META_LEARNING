"""Shared bumper-gap and TTC semantics for the FBRT v2 scenarios."""

from __future__ import annotations


def longitudinal_bumper_clearance(front_center_m: float, front_length_m: float,
                                  rear_center_m: float, rear_length_m: float) -> float:
    """Positive net longitudinal gap when `front` is ahead of `rear`."""
    return float(front_center_m - front_length_m / 2.0 -
                 (rear_center_m + rear_length_m / 2.0))


def time_to_collision(clearance_m: float, closing_speed_mps: float) -> float | None:
    """TTC under constant positive closing speed; negative/zero closure is undefined."""
    if clearance_m < 0:
        return 0.0
    if closing_speed_mps <= 0:
        return None
    return float(clearance_m / closing_speed_mps)

