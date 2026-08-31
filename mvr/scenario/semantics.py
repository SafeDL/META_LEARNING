"""Stateful functional-scenario timing and event semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class ManeuverScheduleState:
    """Markov state shared by Cut-in, Merge, and Roundabout controllers."""

    maneuver_progress: float = 0.0
    maneuver_latched: bool = False
    challenge_phase_active: bool = False


class ScenarioActionAdapter:
    """Apply the fixed Logical Scenario maneuver schedule."""

    def __init__(self, episode: Any, family: str) -> None:
        self.episode = episode
        self.family = str(family)
        self.state = ManeuverScheduleState()

    def _route_progress(self) -> float:
        route = self.episode.adversary_route
        projection = route.projection(
            self.episode.adversary.position,
            self.episode.adversary.heading_theta,
        )
        return float(np.clip(projection.s_m / max(route.length_m, 1e-6), 0.0, 1.0))

    def _elapsed_seconds(self) -> float:
        seconds_per_step = float(self.episode.env.config["physics_world_step_size"])
        seconds_per_step *= int(self.episode.env.config["decision_repeat"])
        return float(self.episode.env.episode_step) * seconds_per_step

    def update(self) -> ManeuverScheduleState:
        """Advance a contract-defined maneuver without Inner action semantics."""
        self.state.maneuver_progress = self._route_progress()
        if self.family == "cutin" and not self.state.maneuver_latched:
            onset = float(
                self.episode.applied_scenario.logical_parameters["cutin_onset_time_s"]
            )
            if self._elapsed_seconds() >= onset:
                self.state.maneuver_latched = True
        return self.state

    def target_lane(self) -> Any:
        contract = self.episode.layout.traffic_contract
        number = (
            contract.target_lane_number
            if self.state.maneuver_latched and contract.target_lane_number is not None
            else contract.source_lane_number
        )
        current = self.episode.adversary.navigation.current_lane.index
        return self.episode.env.current_map.road_network.get_lane(
            (current[0], current[1], number)
        )

    def observe_semantics(self, challenge_phase_active: bool) -> None:
        self.state.challenge_phase_active = bool(challenge_phase_active)
        if self.family != "cutin" and self.state.challenge_phase_active:
            self.state.maneuver_latched = True


@dataclass(frozen=True)
class SemanticState:
    maneuver_started: bool
    maneuver_active: bool
    maneuver_completed: bool
    challenge_phase_active: bool
    target_lane_intrusion: bool
    semantic_valid: bool


class ScenarioSemanticMonitor:
    """Measure physical scenario semantics and freeze them at an event."""

    conflict_radius_m = 18.0
    merge_conflict_radius_m = 45.0

    def __init__(self, episode: Any, family: str, schedule: ScenarioActionAdapter) -> None:
        self.episode = episode
        self.family = str(family)
        self.schedule = schedule
        self._completed = False
        self._event_kind: str | None = None
        self._event_semantic_valid = False
        self._event_traffic_valid = False
        self._event_just_captured = False
        self._state = SemanticState(False, False, False, False, False, False)

    def _target_lane(self) -> Any | None:
        contract = self.episode.layout.traffic_contract
        if contract.target_lane_number is None:
            return None
        current = self.episode.adversary.navigation.current_lane.index
        return self.episode.env.current_map.road_network.get_lane(
            (current[0], current[1], contract.target_lane_number)
        )

    def _shared_conflict_active(self) -> bool:
        adv_s = self.episode.adversary_route.projection(
            self.episode.adversary.position, self.episode.adversary.heading_theta
        ).s_m
        sut_s = self.episode.sut_route.projection(
            self.episode.sut.position, self.episode.sut.heading_theta
        ).s_m
        adv_conflict = self.episode.adversary_route.conflict_s(self.episode.layout.conflict_xy)
        sut_conflict = self.episode.sut_route.conflict_s(self.episode.layout.conflict_xy)
        radius = (
            self.merge_conflict_radius_m
            if self.family == "merge"
            else self.conflict_radius_m
        )
        return bool(abs(adv_conflict - adv_s) <= radius and abs(sut_conflict - sut_s) <= radius)

    @staticmethod
    def _vehicle_overlaps_lane_corridor(vehicle: Any, lane: Any) -> bool:
        """Return whether the vehicle's oriented footprint overlaps ``lane``."""
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        lane_heading = float(
            lane.heading_theta_at(
                float(np.clip(longitudinal, 0.0, float(lane.length)))
            )
        )
        heading_delta = float(vehicle.heading_theta) - lane_heading
        half_length = 0.5 * float(getattr(vehicle, "LENGTH", 4.5))
        half_width = 0.5 * float(getattr(vehicle, "WIDTH", 2.0))
        footprint_longitudinal = (
            abs(np.cos(heading_delta)) * half_length
            + abs(np.sin(heading_delta)) * half_width
        )
        footprint_lateral = (
            abs(np.sin(heading_delta)) * half_length
            + abs(np.cos(heading_delta)) * half_width
        )
        return bool(
            float(longitudinal) + footprint_longitudinal > 0.0
            and float(longitudinal) - footprint_longitudinal < float(lane.length)
            and abs(float(lateral)) - footprint_lateral < 0.5 * float(lane.width)
        )

    def update(self) -> SemanticState:
        if self.family == "cutin":
            target = self._target_lane()
            if target is None:
                raise RuntimeError("cut-in semantic monitor requires a target lane")
            adversary_lateral = abs(float(target.local_coordinates(self.episode.adversary.position)[1]))
            sut_lateral = abs(float(target.local_coordinates(self.episode.sut.position)[1]))
            intrusion = self._vehicle_overlaps_lane_corridor(
                self.episode.adversary, target
            )
            target_conflict = intrusion and sut_lateral <= 0.5 * float(target.width) + 0.2
            started = bool(self.schedule.state.maneuver_latched)
            active = started and intrusion
            if adversary_lateral <= 0.4 * float(target.width):
                self._completed = True
            challenge = active and target_conflict
            state = SemanticState(
                started, active, self._completed, challenge, intrusion, active
            )
        else:
            if self.family == "merge":
                # Before both vehicles occupy the downstream lane, a lawful
                # branch/mainline conflict occurs on two converging lanes.
                # The route-progress window is tied to the verified merge
                # point; the event detector still requires physical close
                # distance and low TTC.
                challenge = self._shared_conflict_active()
            else:
                # Roundabout conflicts can occur at the entry mouth, before
                # either footprint reaches the first shared circulating lane.
                # The route-relative conflict window is the physical
                # corridor here; requiring both vehicles to overlap the
                # downstream lane would label a real entry collision invalid.
                challenge = self._shared_conflict_active()
            state = SemanticState(
                challenge, challenge, challenge, challenge, False, challenge
            )
        self._state = state
        self.schedule.observe_semantics(state.challenge_phase_active)
        return state

    @staticmethod
    def _traffic_valid(info: Mapping[str, Any]) -> bool:
        return not any(
            bool(info.get(key, False))
            for key in (
                "non_target_collision",
                "adversary_out_of_road",
                "sut_out_of_road",
                "wrong_route",
                "adversary_traffic_violation",
            )
        )

    def capture_event(self, kind: str | None, info: Mapping[str, Any]) -> bool:
        self._event_just_captured = False
        if kind is None:
            return False
        if kind not in {"collision", "near_miss"}:
            raise ValueError(f"unknown event kind {kind!r}")
        if self.family == "cutin":
            semantic = self._state.maneuver_active and self._state.target_lane_intrusion
            if kind == "near_miss":
                semantic = semantic and self._state.challenge_phase_active
        else:
            semantic = self._state.challenge_phase_active
        # A near-miss is recorded once but does not end the route-completion
        # test.  A later target collision is the decisive event and replaces
        # the earlier near-miss latch.
        if self._event_kind == "collision" or (
            self._event_kind == "near_miss" and kind == "near_miss"
        ):
            return False
        self._event_kind = kind
        self._event_semantic_valid = bool(semantic)
        self._event_traffic_valid = self._traffic_valid(info)
        self._event_just_captured = True
        return True

    def info(self) -> dict[str, Any]:
        return {
            "semantic_maneuver_started": self._state.maneuver_started,
            "semantic_maneuver_active": self._state.maneuver_active,
            "semantic_maneuver_completed": self._state.maneuver_completed,
            "semantic_challenge_phase_active": self._state.challenge_phase_active,
            "semantic_target_lane_intrusion": self._state.target_lane_intrusion,
            "semantic_valid": self._state.semantic_valid,
            "event_kind": self._event_kind,
            "event_semantic_valid": self._event_semantic_valid,
            "event_traffic_valid": self._event_traffic_valid,
            "event_just_captured": self._event_just_captured,
        }
