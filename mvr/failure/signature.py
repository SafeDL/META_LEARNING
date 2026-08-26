from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
import math

from .criteria import DEFAULT_FAILURE_CRITERIA, FailureCriteria
from ..provenance import content_hash


FAILURE_SCHEMA = "failure_signature_v2"


@dataclass(frozen=True)
class FailureSignature:
    failure_type: str
    scenario_family: str
    conflict_zone_id: str | None
    severity_vector: tuple[int, int, int]
    is_valid_episode: bool
    is_failure: bool
    candidate_id: str | None = None
    severity_bins: int = DEFAULT_FAILURE_CRITERIA.severity_bins
    schema: str = FAILURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FAILURE_SCHEMA or not self.failure_type or not self.scenario_family:
            raise ValueError("invalid failure signature")
        if self.severity_bins < 2 or len(self.severity_vector) != 3 or any(
            not isinstance(value, int) or not 0 <= value < self.severity_bins
            for value in self.severity_vector
        ):
            raise ValueError("severity_vector must contain three valid bins")
        if self.is_failure and not self.is_valid_episode:
            raise ValueError("an invalid episode cannot be a counted failure")

    @property
    def signature_id(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class FailureSignatureBuilder:
    criteria: FailureCriteria = DEFAULT_FAILURE_CRITERIA

    def _bin(self, value: float, threshold: float, *, inverse: bool) -> int:
        if not math.isfinite(value):
            value = threshold
        normalized = min(max(float(value) / threshold, 0.0), 1.0)
        if inverse:
            normalized = 1.0 - normalized
        return min(self.criteria.severity_bins - 1, int(normalized * self.criteria.severity_bins))

    def from_outcome(
        self,
        outcome: Mapping[str, Any],
        scenario_family: str,
        conflict_zone_id: str | None,
        candidate_id: str | None = None,
    ) -> FailureSignature:
        invalid = any(
            bool(outcome.get(key, False))
            for key in (
                "non_target_collision",
                "adversary_out_of_road",
                "sut_out_of_road",
                "wrong_route",
                "adversary_traffic_violation",
            )
        )
        is_valid_episode = bool(outcome.get("is_valid_episode", not invalid)) and not invalid
        collision = bool(
            outcome.get(
                "valid_target_collision",
                outcome.get("target_collision", False) and not invalid,
            )
        ) and not invalid
        near_miss = bool(outcome.get("valid_critical_near_miss", outcome.get("valid_critical_strict", False)))
        is_failure = is_valid_episode and (collision or near_miss)
        failure_type = "target_collision" if collision else "valid_critical_near_miss" if near_miss else "none"
        severity = (
            self._bin(
                float(outcome.get("min_ttc", self.criteria.ttc_s)),
                self.criteria.ttc_s,
                inverse=True,
            ),
            self._bin(
                float(outcome.get("min_distance", self.criteria.distance_m)),
                self.criteria.distance_m,
                inverse=True,
            ),
            self._bin(
                float(outcome.get("max_closing_speed", outcome.get("closing_speed_mps", 0.0))),
                self.criteria.closing_speed_mps,
                inverse=False,
            ),
        )
        return FailureSignature(
            failure_type,
            str(scenario_family),
            conflict_zone_id,
            severity,
            is_valid_episode,
            is_failure,
            candidate_id,
            self.criteria.severity_bins,
        )
