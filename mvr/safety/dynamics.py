"""Shared physical action projection for all Cut-in vehicles."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


CUTIN_VEHICLE_CONFIG = {
    "max_engine_force": 825.0,
    "max_brake_force": 33.0,
}


@dataclass
class VehicleActionProjector:
    """Bound normalized MetaDrive controls without replacing Bullet physics."""

    vehicle: Any
    speed_limit_mps: float
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 6.0
    max_jerk_mps3: float = 2.0
    max_lateral_acceleration_mps2: float = 3.0
    max_steering_rate_per_s: float = 1.5
    _previous_acceleration_mps2: float = field(default=0.0, init=False)

    @property
    def previous_acceleration_mps2(self) -> float:
        """Return the acceleration used as the next jerk-limit reference."""
        return self._previous_acceleration_mps2

    @staticmethod
    def _step_seconds(vehicle: Any) -> float:
        config = vehicle.engine.global_config
        return float(config["physics_world_step_size"]) * int(config["decision_repeat"])

    @staticmethod
    def _speed_mps(vehicle: Any) -> float:
        return float(vehicle.speed_km_h) / 3.6

    def _steering_limit(self) -> float:
        speed = max(
            self._speed_mps(self.vehicle) + self.max_acceleration_mps2 * self._step_seconds(self.vehicle),
            self._speed_mps(self.vehicle) * 1.05,
            0.25,
        )
        max_steering_rad = np.deg2rad(float(self.vehicle.config["max_steering"]))
        wheelbase = max(0.6 * float(self.vehicle.LENGTH), 1.0)
        maximum = np.arctan(self.max_lateral_acceleration_mps2 * wheelbase / speed**2)
        return float(min(1.0, maximum / max(max_steering_rad, 1e-6)))

    def project(self, requested_action: Any) -> np.ndarray:
        action = np.asarray(requested_action, dtype=np.float32).reshape(-1)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("vehicle action projector requires a finite 2-D action")
        action = np.clip(action, -1.0, 1.0)
        dt = self._step_seconds(self.vehicle)
        current_steering = float(getattr(self.vehicle, "steering", 0.0))
        steering_delta = self.max_steering_rate_per_s * dt
        action[0] = np.clip(action[0], current_steering - steering_delta, current_steering + steering_delta)
        action[0] = np.clip(action[0], -self._steering_limit(), self._steering_limit())
        requested_acceleration = float(action[1]) * (
            self.max_acceleration_mps2 if action[1] >= 0.0 else self.max_deceleration_mps2
        )
        jerk_delta = self.max_jerk_mps3 * dt
        acceleration = float(np.clip(
            requested_acceleration,
            self._previous_acceleration_mps2 - jerk_delta,
            self._previous_acceleration_mps2 + jerk_delta,
        ))
        speed = self._speed_mps(self.vehicle)
        if acceleration < 0.0:
            acceleration = max(acceleration, -float(np.sqrt(2.0 * self.max_jerk_mps3 * speed)))
        if speed >= self.speed_limit_mps and acceleration > 0.0:
            acceleration = 0.0
        self._previous_acceleration_mps2 = acceleration
        action[1] = acceleration / (
            self.max_acceleration_mps2 if acceleration >= 0.0 else self.max_deceleration_mps2
        )
        return action.astype(np.float32)
