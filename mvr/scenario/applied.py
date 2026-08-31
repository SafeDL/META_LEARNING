"""Auditable records produced by an executable scenario reset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .layout import LaneIndex
from .route_geometry import RoutePolyline


@dataclass(frozen=True)
class AppliedScenario:
    adversary_vehicle_id: str
    sut_vehicle_id: str
    adversary_lane: LaneIndex
    sut_lane: LaneIndex
    adversary_spawn_m: float
    sut_spawn_m: float
    adversary_speed_mps: float
    sut_speed_mps: float
    logical_parameters: Mapping[str, float]
    selected_candidate: str
    conflict_zone_id: str
    adversary_route: tuple[LaneIndex, ...]
    sut_route: tuple[LaneIndex, ...]
    normalized_continuous: tuple[float, ...]

    @property
    def adversary_distance_to_conflict_m(self) -> float:
        return float(self.logical_parameters["adversary_distance_to_conflict_m"])

    @property
    def sut_distance_to_conflict_m(self) -> float:
        return float(self.logical_parameters["sut_distance_to_conflict_m"])

    @property
    def maneuver_onset_progress(self) -> float:
        return float(self.logical_parameters["maneuver_onset_progress"])


@dataclass
class ExecutableEpisode:
    env: Any
    initial_observation: Any
    adversary: Any
    sut: Any
    sut_adapter: Any
    sut_profile: Any
    applied_scenario: AppliedScenario
    map_tokens: Any
    layout: Any
    adversary_route: RoutePolyline
    sut_route: RoutePolyline
    episode_seed: int | None = None
