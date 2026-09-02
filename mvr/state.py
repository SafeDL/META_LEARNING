"""Explicit physical state features for the Inner controller."""
from __future__ import annotations

from typing import Any

import numpy as np

from .scenario.route_geometry import RoutePolyline

INNER_STATE_FIELDS = (
    "relative_longitudinal_m",
    "relative_lateral_m",
    "heading_difference_rad",
    "adversary_speed_mps",
    "sut_speed_mps",
    "closing_speed_mps",
    "pair_distance_m",
    "adversary_route_progress",
    "sut_route_progress",
    "conflict_timing_s",
    "challenge_phase_active",
    "adversary_route_lateral_m",
    "adversary_route_heading_error_rad",
    "maneuver_started",
    "cutin_reference_lateral_error_m",
    "cutin_reference_heading_error_rad",
    "cutin_reference_progress",
    "cutin_reference_length_m",
    "cutin_reference_curvature_m_inv",
    "cutin_reference_speed_limit_mps",
    "cutin_start_remaining_m",
    "cutin_start_time_remaining_s",
    "cutin_corridor_margin_m",
)


class PhysicalStateExtractor:
    """Normalize a fixed physical state; raw simulator observations are excluded."""

    dimension = len(INNER_STATE_FIELDS)
    scales = np.asarray(
        (100.0, 20.0, np.pi, 30.0, 30.0, 30.0, 100.0, 1.0, 1.0, 15.0, 1.0, 8.0, np.pi, 1.0,
         8.0, np.pi, 1.0, 60.0, 0.2, 20.0, 100.0, 5.0, 4.0),
        dtype=np.float32,
    )

    def __init__(self) -> None:
        self._adversary_route: RoutePolyline | None = None
        self._sut_route: RoutePolyline | None = None
        self._adversary_conflict_s = 0.0
        self._sut_conflict_s = 0.0

    @staticmethod
    def _heading(vehicle: Any) -> float:
        vector = np.asarray(getattr(vehicle, "heading", (1.0, 0.0)), dtype=float)
        return float(np.arctan2(vector[1], vector[0]))

    @staticmethod
    def _speed(vehicle: Any) -> float:
        return float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6

    @staticmethod
    def _eta(remaining_m: float, speed_mps: float) -> float:
        return float(np.clip(remaining_m / max(speed_mps, 0.25), -15.0, 15.0))

    def reset(
        self,
        env: Any,
        layout: Any,
        adversary_route: RoutePolyline | None = None,
        sut_route: RoutePolyline | None = None,
    ) -> None:
        self._adversary_route = adversary_route or RoutePolyline.from_env(
            env, {"route_id": "adversary", "lane_sequence": list(layout.adversary_route)}
        )
        self._sut_route = sut_route or RoutePolyline.from_env(
            env, {"route_id": "sut", "lane_sequence": list(layout.sut_route)}
        )
        self._adversary_conflict_s = self._adversary_route.conflict_s(layout.conflict_xy)
        self._sut_conflict_s = self._sut_route.conflict_s(layout.conflict_xy)

    def __call__(self, adversary: Any, sut: Any, schedule: Any | None = None) -> np.ndarray:
        if self._adversary_route is None or self._sut_route is None:
            raise RuntimeError("physical state extractor must be reset with an executable layout")
        adversary_position = np.asarray(adversary.position, dtype=float)
        sut_position = np.asarray(sut.position, dtype=float)
        relative = sut_position - adversary_position
        heading = self._heading(adversary)
        longitudinal = np.cos(heading) * relative[0] + np.sin(heading) * relative[1]
        lateral = -np.sin(heading) * relative[0] + np.cos(heading) * relative[1]
        adversary_speed, sut_speed = self._speed(adversary), self._speed(sut)
        distance = float(np.linalg.norm(relative))
        closing = float(-np.dot(relative, np.asarray(sut.velocity, dtype=float) - np.asarray(adversary.velocity, dtype=float)) / max(distance, 1e-6))
        adversary_projection = self._adversary_route.projection(adversary_position, heading)
        sut_projection = self._sut_route.projection(sut_position, self._heading(sut))
        schedule_state = getattr(schedule, "state", schedule)
        challenge_phase = 0.0 if schedule is None else float(
            schedule_state.challenge_phase_active
        )
        reference_lateral_error = 0.0
        reference_heading_error = 0.0
        reference_progress = 0.0
        reference_length = 0.0
        reference_curvature = 0.0
        reference_speed_limit = 0.0
        start_remaining = 0.0
        onset_remaining = 0.0
        corridor_margin = 0.0
        if getattr(schedule, "family", None) == "cutin":
            reference = schedule.cutin_reference()
            reference_lateral_error = reference.lateral_error_m
            reference_heading_error = reference.heading_error_rad
            reference_progress = reference.progress
            reference_length = reference.length_m
            reference_curvature = reference.curvature_m_inv
            reference_speed_limit = reference.speed_limit_mps
            start_remaining = reference.start_remaining_m
            contract = schedule.episode.layout.traffic_contract
            current = adversary.navigation.current_lane.index
            target_lane = schedule.episode.env.current_map.road_network.get_lane(
                (current[0], current[1], contract.target_lane_number)
            )
            _, target_lateral = target_lane.local_coordinates(adversary_position)
            onset = float(schedule.episode.applied_scenario.logical_parameters["cutin_start_time_s"])
            onset_remaining = max(0.0, onset - schedule._elapsed_seconds())
            corridor_margin = max(
                0.0, 0.5 * float(target_lane.width) - abs(target_lateral)
            )
        values = np.asarray((
            longitudinal,
            lateral,
            np.arctan2(np.sin(self._heading(sut) - heading), np.cos(self._heading(sut) - heading)),
            adversary_speed,
            sut_speed,
            closing,
            distance,
            adversary_projection.s_m / self._adversary_route.length_m,
            sut_projection.s_m / self._sut_route.length_m,
            self._eta(self._adversary_conflict_s - adversary_projection.s_m, adversary_speed)
            - self._eta(self._sut_conflict_s - sut_projection.s_m, sut_speed),
            challenge_phase,
            adversary_projection.lateral_m,
            adversary_projection.heading_error,
            0.0 if schedule is None else float(schedule_state.maneuver_latched),
            reference_lateral_error,
            reference_heading_error,
            reference_progress,
            reference_length,
            reference_curvature,
            reference_speed_limit,
            start_remaining,
            onset_remaining,
            corridor_margin,
        ), dtype=np.float32)
        return np.clip(np.nan_to_num(values / self.scales, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
