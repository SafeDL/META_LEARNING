"""Single source of truth for MVR failure thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FailureCriteria:
    ttc_s: float
    distance_m: float
    closing_speed_mps: float
    severity_bins: int

    def __post_init__(self) -> None:
        if min(self.ttc_s, self.distance_m, self.closing_speed_mps) <= 0.0:
            raise ValueError("failure thresholds must be positive")
        if self.severity_bins < 2:
            raise ValueError("failure severity_bins must be at least two")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FailureCriteria":
        thresholds = config["severity_thresholds"]
        return cls(
            float(thresholds["ttc_s"]),
            float(thresholds["distance_m"]),
            float(thresholds["closing_speed_mps"]),
            int(config["severity_bins"]),
        )


DEFAULT_FAILURE_CRITERIA = FailureCriteria(5.0, 10.0, 20.0, 5)
