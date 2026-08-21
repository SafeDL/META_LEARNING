"""Auditable records produced by an executable scenario reset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .layout import LaneIndex


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
    selected_candidate: str
    selected_option: str
    adversary_route: tuple[LaneIndex, ...]
    sut_route: tuple[LaneIndex, ...]


@dataclass
class ExecutableEpisode:
    env: Any
    initial_observation: Any
    initial_info: Mapping[str, Any]
    adversary: Any
    sut: Any
    sut_adapter: Any
    sut_profile: Any
    applied_scenario: AppliedScenario
    map_tokens: Any
    layout: Any

