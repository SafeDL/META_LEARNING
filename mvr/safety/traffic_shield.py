"""Last-resort projection for a nominal action plus SAC residual."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..scenario.applied import ExecutableEpisode
from ..scenario.semantics import ScenarioActionAdapter


@dataclass(frozen=True)
class ShieldedAction:
    base_action: np.ndarray
    candidate_action: np.ndarray
    action: np.ndarray
    rejection_reason: str | None


@dataclass
class TrafficActionShield:
    """Project only physical/traffic hard constraints, never nominal driving."""

    episode: ExecutableEpisode
    schedule: ScenarioActionAdapter
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 6.0
    max_jerk_mps3: float = 4.0
    max_lateral_acceleration_mps2: float = 3.0
    _previous_speed_mps: float | None = field(default=None, init=False)
    _previous_acceleration_mps2: float | None = field(default=None, init=False)
    _rejections: Counter[str] = field(default_factory=Counter, init=False)
    _violations: Counter[str] = field(default_factory=Counter, init=False)
    _max_speed_mps: float = field(default=0.0, init=False)
    _max_abs_acceleration_mps2: float = field(default=0.0, init=False)
    _max_abs_jerk_mps3: float = field(default=0.0, init=False)
    _max_lateral_acceleration_mps2: float = field(default=0.0, init=False)

    @property
    def contract(self) -> Any:
        return self.episode.layout.traffic_contract

    @staticmethod
    def _speed_mps(vehicle: Any) -> float:
        return float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6

    @staticmethod
    def _step_seconds(env: Any) -> float:
        config = getattr(env, "config", {})
        return float(config.get("physics_world_step_size", 0.02)) * int(
            config.get("decision_repeat", 5)
        )

    @staticmethod
    def _vehicle_length(vehicle: Any) -> float:
        return float(getattr(vehicle, "LENGTH", 4.5))

    def _steering_limit(self) -> float:
        vehicle = self.episode.adversary
        speed = self._speed_mps(vehicle)
        # Projection happens before the simulator advances.  Reserve room for
        # the next decision interval so a throttle residual cannot make a
        # steering command legal at t but illegal at t + 1.
        speed = max(
            speed + self.max_acceleration_mps2 * self._step_seconds(self.episode.env),
            speed * 1.05,
        )
        if speed <= 0.25:
            return 1.0
        max_steering_rad = np.deg2rad(float(vehicle.config.get("max_steering", 40.0)))
        wheelbase = max(0.6 * self._vehicle_length(vehicle), 1.0)
        maximum = np.arctan(
            self.max_lateral_acceleration_mps2 * wheelbase / max(speed, 0.25) ** 2
        )
        return float(min(1.0, maximum / max(max_steering_rad, 1e-6)))

    def project(self, base_action: Any, candidate_action: Any) -> ShieldedAction:
        base = np.asarray(base_action, dtype=np.float32).reshape(-1)
        candidate = np.asarray(candidate_action, dtype=np.float32).reshape(-1)
        if base.shape != (2,) or candidate.shape != (2,):
            raise ValueError("traffic shield requires base and candidate 2-D actions")
        if not np.isfinite(base).all() or not np.isfinite(candidate).all():
            raise ValueError("traffic shield actions must be finite")
        action = np.clip(candidate, -1.0, 1.0)
        reason = None
        residual_changed_action = not np.allclose(base, candidate, atol=1e-6)
        # The Shield is not allowed to retune the native nominal controller.
        # A zero residual must reproduce the base action exactly.
        if not residual_changed_action:
            return ShieldedAction(base, candidate, action.astype(np.float32), None)
        steering_limit = self._steering_limit()
        if abs(float(action[0])) > steering_limit:
            action[0] = (
                base[0]
                if abs(float(base[0])) > steering_limit
                else np.clip(action[0], -steering_limit, steering_limit)
            )
            reason = "lateral_acceleration"
        if self._speed_mps(self.episode.adversary) >= self.contract.speed_limit_mps:
            if action[1] > 0.0:
                action[1] = 0.0
                reason = reason or "speed_limit"
        if reason is not None:
            self._rejections[reason] += 1
        return ShieldedAction(base, candidate, action.astype(np.float32), reason)

    def _legal_lane_lateral(self) -> float:
        vehicle = self.episode.adversary
        projection = self.episode.adversary_route.projection(
            vehicle.position, vehicle.heading_theta
        )
        return abs(float(projection.lateral_m))

    def observe(
        self,
        shielded: ShieldedAction,
        environment_info: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        vehicle = self.episode.adversary
        speed = self._speed_mps(vehicle)
        dt = self._step_seconds(self.episode.env)
        acceleration = (
            0.0 if self._previous_speed_mps is None else (speed - self._previous_speed_mps) / dt
        )
        jerk = (
            0.0
            if self._previous_acceleration_mps2 is None
            else (acceleration - self._previous_acceleration_mps2) / dt
        )
        self._previous_speed_mps = speed
        self._previous_acceleration_mps2 = acceleration
        max_steering_rad = np.deg2rad(float(vehicle.config.get("max_steering", 40.0)))
        wheelbase = max(0.6 * self._vehicle_length(vehicle), 1.0)
        lateral_acceleration = speed**2 * np.tan(float(shielded.action[0]) * max_steering_rad) / wheelbase
        self._max_speed_mps = max(self._max_speed_mps, speed)
        self._max_abs_acceleration_mps2 = max(self._max_abs_acceleration_mps2, abs(acceleration))
        self._max_abs_jerk_mps3 = max(self._max_abs_jerk_mps3, abs(jerk))
        self._max_lateral_acceleration_mps2 = max(
            self._max_lateral_acceleration_mps2, abs(lateral_acceleration)
        )
        if speed > self.contract.speed_limit_mps + 0.5:
            self._violations["speed_limit"] += 1
        # Attribute a violation only to an actuator command that still differs
        # from the native nominal action after projection.  A projected-away
        # residual must not relabel a lawful IDM turn as illegal.
        residual_changed_steering = not np.isclose(
            shielded.base_action[0], shielded.action[0], atol=1e-6
        )
        if (
            residual_changed_steering
            and abs(lateral_acceleration) > self.max_lateral_acceleration_mps2 + 1e-6
        ):
            self._violations["lateral_acceleration"] += 1
        if environment_info is not None and bool(environment_info.get("out_of_road", False)):
            self._violations["out_of_road"] += 1
        lateral = self._legal_lane_lateral()
        route_projection = self.episode.adversary_route.projection(
            vehicle.position, vehicle.heading_theta
        )
        if (
            residual_changed_steering
            and not self.episode.adversary_route.in_lane_change(route_projection.s_m)
            and lateral > 0.5 * float(vehicle.navigation.current_lane.width) + 0.2
        ):
            self._violations["lane_boundary_crossing"] += 1
        intervention = float(np.linalg.norm(shielded.candidate_action - shielded.action))
        return {
            "adversary_traffic_violation": bool(self._violations),
            "traffic_shield_rejected": shielded.rejection_reason is not None,
            "traffic_shield_rejection_reason": shielded.rejection_reason,
            "traffic_base_action": shielded.base_action.tolist(),
            "traffic_candidate_action": shielded.candidate_action.tolist(),
            "traffic_executed_action": shielded.action.tolist(),
            "traffic_shield_intervention_l2": intervention,
            "traffic_rejection_counts": dict(self._rejections),
            "traffic_violation_counts": dict(self._violations),
            "traffic_max_speed_mps": self._max_speed_mps,
            "traffic_max_abs_acceleration_mps2": self._max_abs_acceleration_mps2,
            "traffic_max_abs_jerk_mps3": self._max_abs_jerk_mps3,
            "traffic_max_lateral_acceleration_mps2": self._max_lateral_acceleration_mps2,
            "traffic_legal_lane_lateral_m": lateral,
            "traffic_route_transition_active": self.episode.adversary_route.in_lane_change(route_projection.s_m),
        }
