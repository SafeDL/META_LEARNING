"""Auditable records produced by an executable scenario reset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    adversary_distance_to_conflict_m: float
    sut_distance_to_conflict_m: float
    adversary_speed_mps: float
    sut_speed_mps: float
    maneuver_onset_progress: float
    selected_candidate: str
    conflict_zone_id: str
    adversary_route: tuple[LaneIndex, ...]
    sut_route: tuple[LaneIndex, ...]
    normalized_continuous: tuple[float, ...]


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
