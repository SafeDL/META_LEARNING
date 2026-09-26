"""Continuous, non-oracle vulnerability signal for Mining learning only."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class VulnerabilityResponseConfig:
    ttc_scale_s: float
    distance_scale_m: float
    noncritical_cap: float
    near_miss_floor: float
    proxy_event_threshold: float

    def __post_init__(self) -> None:
        if self.ttc_scale_s <= 0.0 or self.distance_scale_m <= 0.0:
            raise ValueError("response scales must be positive")
        if not 0.0 <= self.noncritical_cap < self.near_miss_floor <= 1.0:
            raise ValueError("response caps must separate noncritical and near-miss values")
        if not 0.0 < self.proxy_event_threshold <= 1.0:
            raise ValueError("proxy event threshold must lie in (0, 1]")


def _soft_component(value: object, scale: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return exp(-max(numeric, 0.0) / scale) if isfinite(numeric) else 0.0


def compute_vulnerability_response(
    outcome: Mapping[str, Any], status: str, config: VulnerabilityResponseConfig
) -> float | None:
    """Derive a bounded response from recorded challenge telemetry, never an oracle."""
    if status not in {"valid_event", "completed_noncritical"}:
        return None
    ttc = _soft_component(outcome.get("challenge_min_ttc"), config.ttc_scale_s)
    distance = _soft_component(
        outcome.get("challenge_min_distance"), config.distance_scale_m
    )
    soft = 1.0 - (1.0 - ttc) * (1.0 - distance)
    if bool(outcome.get("valid_target_collision", False)):
        return 1.0
    if bool(outcome.get("valid_critical_near_miss", False)):
        return min(1.0, max(config.near_miss_floor, soft))
    return min(config.noncritical_cap, soft)
