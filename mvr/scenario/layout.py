"""Runtime-resolved physical layout for one outer scenario candidate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LaneIndex = tuple[Any, Any, int]
SCENARIO_CONTRACT_SCHEMA = "scenario_contract_v2"


@dataclass(frozen=True)
class TrafficBehaviorContract:
    """Non-learned traffic rules that constrain the adversary at runtime."""

    speed_limit_mps: float
    allowed_lane_numbers: tuple[int, ...]
    source_lane_number: int
    target_lane_number: int | None = None
    merge_window_s: tuple[float, float] | None = None
    crossing_boundary: str | None = None
    schema: str = SCENARIO_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCENARIO_CONTRACT_SCHEMA:
            raise ValueError("unsupported traffic behavior contract schema")
        if self.speed_limit_mps <= 0.0 or not self.allowed_lane_numbers:
            raise ValueError("traffic contract requires a speed limit and allowed lanes")
        if self.source_lane_number not in self.allowed_lane_numbers:
            raise ValueError("traffic contract source lane must be allowed")
        if self.target_lane_number is None:
            if self.merge_window_s is not None or self.crossing_boundary is not None:
                raise ValueError("lane-following contract cannot define a merge window")
            return
        if self.target_lane_number not in self.allowed_lane_numbers:
            raise ValueError("traffic contract target lane must be allowed")
        if self.target_lane_number == self.source_lane_number:
            raise ValueError("traffic contract target lane must differ from source")
        if self.merge_window_s is None or self.crossing_boundary is None:
            raise ValueError("lane-change contract requires a window and boundary")
        start, end = self.merge_window_s
        if not 0.0 <= start < end:
            raise ValueError("traffic contract merge window must be ordered and non-negative")


@dataclass(frozen=True)
class NativeNavigationContract:
    """Road-level route expected from native spawn-lane/destination navigation."""

    adversary_checkpoints: tuple[Any, ...]
    sut_checkpoints: tuple[Any, ...]
    sut_lane_stable: bool

    @staticmethod
    def checkpoints(route: tuple[LaneIndex, ...]) -> tuple[Any, ...]:
        return tuple((route[0][0], *(lane[1] for lane in route)))


@dataclass(frozen=True)
class ScenarioLayout:
    """Lane/route binding resolved from the *actual* generated road network.

    Candidate labels are intentionally task-level strings.  This object is the
    auditable bridge from that label to concrete MetaDrive lane indices and
    destinations; it is never a learned model input.
    """

    candidate: str
    conflict_zone_id: str
    adversary_lane: LaneIndex
    sut_lane: LaneIndex
    adversary_destination: Any
    sut_destination: Any
    adversary_route: tuple[LaneIndex, ...]
    sut_route: tuple[LaneIndex, ...]
    conflict_xy: tuple[float, float]
    traffic_contract: TrafficBehaviorContract
    native_navigation: NativeNavigationContract | None = None

    def __post_init__(self) -> None:
        if not self.candidate or not self.conflict_zone_id or not self.adversary_route or not self.sut_route:
            raise ValueError("scenario layout requires a candidate and non-empty routes")
        if self.adversary_route[0] != self.adversary_lane or self.sut_route[0] != self.sut_lane:
            raise ValueError("each runtime route must begin on its spawn lane")
        if not np.isfinite(np.asarray(self.conflict_xy, dtype=float)).all():
            raise ValueError("scenario layout conflict reference must be finite")
        if self.native_navigation is None:
            object.__setattr__(
                self,
                "native_navigation",
                NativeNavigationContract(
                    NativeNavigationContract.checkpoints(self.adversary_route),
                    NativeNavigationContract.checkpoints(self.sut_route),
                    len({lane[2] for lane in self.sut_route}) == 1,
                ),
            )
