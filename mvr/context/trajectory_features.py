"""Extract the fixed 12-D evidence trajectory from real simulator state."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..scenario.route_geometry import RoutePolyline
from .trajectory_encoder import TRAJECTORY_FIELDS


class TrajectoryFeatureExtractor:
    """Vehicle/route measurements normalized into the encoder's public schema."""

    scales = np.asarray((100.0, 100.0, 30.0, 10.0, 30.0, 10.0, 200.0, 200.0, 15.0, 15.0, 100.0, 15.0), dtype=np.float32)

    def __init__(self) -> None:
        self._rows: list[np.ndarray] = []
        self._adversary_route: RoutePolyline | None = None
        self._sut_route: RoutePolyline | None = None
        self._adv_conflict_s = 0.0
        self._sut_conflict_s = 0.0
        self._previous_sut_speed: float | None = None

    def reset(
        self,
        env: Any,
        layout: Any,
        adversary_route: RoutePolyline | None = None,
        sut_route: RoutePolyline | None = None,
    ) -> None:
        self._rows.clear()
        self._adversary_route = adversary_route or RoutePolyline.from_env(
            env, {"route_id": "adversary", "lane_sequence": list(layout.adversary_route)}
        )
        self._sut_route = sut_route or RoutePolyline.from_env(
            env, {"route_id": "sut", "lane_sequence": list(layout.sut_route)}
        )
        self._adv_conflict_s = self._adversary_route.conflict_s(layout.conflict_xy)
        self._sut_conflict_s = self._sut_route.conflict_s(layout.conflict_xy)
        self._previous_sut_speed = None

    @staticmethod
    def _speed(vehicle: Any) -> float:
        return float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6

    @staticmethod
    def _heading(vehicle: Any) -> float:
        vector = np.asarray(getattr(vehicle, "heading", (1.0, 0.0)), dtype=float)
        return float(np.arctan2(vector[1], vector[0]))

    @staticmethod
    def _eta(remaining_m: float, speed_mps: float) -> float:
        return float(np.clip(remaining_m / max(speed_mps, 0.25), -15.0, 15.0))

    def step(self, env: Any, adversary: Any, sut: Any, info: Any) -> np.ndarray:
        del info
        if self._adversary_route is None or self._sut_route is None:
            raise RuntimeError("trajectory extractor must be reset with an executable layout")
        adv_position, sut_position = np.asarray(adversary.position, dtype=float), np.asarray(sut.position, dtype=float)
        relative = sut_position - adv_position
        heading = self._heading(adversary)
        c, s = np.cos(heading), np.sin(heading)
        rel_local = np.asarray((c * relative[0] + s * relative[1], -s * relative[0] + c * relative[1]))
        adv_velocity, sut_velocity = np.asarray(adversary.velocity, dtype=float), np.asarray(sut.velocity, dtype=float)
        distance = float(np.linalg.norm(relative))
        closing = float(-np.dot(relative, sut_velocity - adv_velocity) / max(distance, 1e-6))
        ttc = distance / closing if closing > 1e-4 else 15.0
        adv_speed, sut_speed = self._speed(adversary), self._speed(sut)
        dt = float(getattr(env, "config", {}).get("physics_world_step_size", 0.1))
        acceleration = 0.0 if self._previous_sut_speed is None else (sut_speed - self._previous_sut_speed) / max(dt, 1e-6)
        self._previous_sut_speed = sut_speed
        adv_projection = self._adversary_route.projection(adv_position, self._heading(adversary))
        sut_projection = self._sut_route.projection(sut_position, self._heading(sut))
        adv_eta = self._eta(self._adv_conflict_s - adv_projection.s_m, adv_speed)
        sut_eta = self._eta(self._sut_conflict_s - sut_projection.s_m, sut_speed)
        values = np.asarray((rel_local[0], rel_local[1], sut_speed - adv_speed, acceleration, sut_speed,
                             sut_projection.lateral_m, adv_projection.s_m, sut_projection.s_m, ttc,
                             abs(adv_eta - sut_eta), distance, adv_eta - sut_eta), dtype=np.float32)
        normalized = np.clip(np.nan_to_num(values / self.scales, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        self._rows.append(normalized)
        return normalized

    def finalize(self) -> torch.Tensor:
        if not self._rows:
            raise ValueError("trajectory extractor has no sampled steps")
        return torch.as_tensor(np.stack(self._rows), dtype=torch.float32)


assert len(TRAJECTORY_FIELDS) == 12
