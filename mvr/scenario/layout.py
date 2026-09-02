"""Runtime-resolved physical layout for one outer scenario candidate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LaneIndex = tuple[Any, Any, int]
SCENARIO_CONTRACT_SCHEMA = "scenario_contract"


@dataclass(frozen=True)
class TrafficBehaviorContract:
    """Non-learned traffic rules that constrain the adversary at runtime."""

    speed_limit_mps: float
    sut_nominal_speed_mps: float
    allowed_lane_numbers: tuple[int, ...]
    source_lane_number: int
    target_lane_number: int | None = None
    merge_window_m: tuple[float, float] | None = None
    crossing_boundary: str | None = None
    adversary_intent: str = "route_follow"
    sut_role: str = "route_following"
    completion_condition: str = "sut_route_destination"
    terminate_on_target_collision: bool = True
    min_completion_steps: int = 240
    schema: str = SCENARIO_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCENARIO_CONTRACT_SCHEMA:
            raise ValueError("unsupported traffic behavior contract schema")
        if min(self.speed_limit_mps, self.sut_nominal_speed_mps) <= 0.0 or not self.allowed_lane_numbers:
            raise ValueError("traffic contract requires positive speeds and allowed lanes")
        if not self.adversary_intent or not self.sut_role:
            raise ValueError("traffic contract requires explicit vehicle roles")
        if self.completion_condition != "sut_route_destination":
            raise ValueError("Stage 1 requires SUT route completion as the test condition")
        if not self.terminate_on_target_collision:
            raise ValueError("Stage 1 must terminate immediately on a target collision")
        if int(self.min_completion_steps) < 1:
            raise ValueError("traffic contract requires a positive completion-step budget")
        if self.source_lane_number not in self.allowed_lane_numbers:
            raise ValueError("traffic contract source lane must be allowed")
        if self.target_lane_number is None:
            if self.merge_window_m is not None or self.crossing_boundary is not None:
                raise ValueError("lane-following contract cannot define a merge window")
            return
        if self.target_lane_number not in self.allowed_lane_numbers:
            raise ValueError("traffic contract target lane must be allowed")
        if self.target_lane_number == self.source_lane_number:
            raise ValueError("traffic contract target lane must differ from source")
        if self.merge_window_m is None or self.crossing_boundary is None:
            raise ValueError("lane-change contract requires a window and boundary")
        start, end = self.merge_window_m
        if not 0.0 <= start < end:
            raise ValueError("traffic contract merge window must be ordered and non-negative")
        if self.adversary_intent == "cut_in_to_sut_lane" and end - start < 60.0:
            raise ValueError("cut-in contract requires a 60 m legal dashed corridor")


@dataclass(frozen=True)
class NativeNavigationContract:
    """Road-level route expected from native spawn-lane/destination navigation."""

    adversary_checkpoints: tuple[Any, ...]
    sut_checkpoints: tuple[Any, ...]
    sut_lane_sequence: tuple[int, ...]
    sut_lane_stable: bool

    @staticmethod
    def checkpoints(route: tuple[LaneIndex, ...]) -> tuple[Any, ...]:
        return tuple((route[0][0], *(lane[1] for lane in route)))

    def expected_sut_lane_number(self, road: tuple[Any, Any]) -> int:
        for index, lane_number in zip(zip(self.sut_checkpoints[:-1], self.sut_checkpoints[1:]), self.sut_lane_sequence):
            if tuple(road) == tuple(index):
                return int(lane_number)
        raise RuntimeError(f"SUT is outside its native route: {road!r}")

    def validate(self) -> None:
        if len(self.sut_checkpoints) < 2:
            raise ValueError("native navigation contract requires a SUT road route")
        if len(self.sut_lane_sequence) != len(self.sut_checkpoints) - 1:
            raise ValueError("SUT lane sequence must align with its road checkpoints")
        if self.sut_lane_stable and len(set(self.sut_lane_sequence)) != 1:
            raise ValueError("lane-stable SUT route cannot change lane numbers")


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
                    tuple(int(lane[2]) for lane in self.sut_route),
                    len({lane[2] for lane in self.sut_route}) == 1,
                ),
            )
        self.native_navigation.validate()
