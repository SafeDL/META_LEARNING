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
)


class PhysicalStateExtractor:
    """Normalize a fixed physical state; raw simulator observations are excluded."""

    dimension = len(INNER_STATE_FIELDS)
    scales = np.asarray(
        (100.0, 20.0, np.pi, 30.0, 30.0, 30.0, 100.0, 1.0, 1.0, 15.0, 1.0),
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
        challenge_phase = 0.0 if schedule is None else float(schedule.challenge_phase_active)
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
        ), dtype=np.float32)
        return np.clip(np.nan_to_num(values / self.scales, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
