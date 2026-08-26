"""Project raw SAC actions into the current traffic contract's feasible set."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..scenario.applied import ExecutableEpisode
from ..scenario.route_geometry import wrap_to_pi


@dataclass(frozen=True)
class ShieldedAction:
    raw_action: np.ndarray
    action: np.ndarray
    rejection_reason: str | None


@dataclass
class TrafficActionShield:
    """Project SAC controls onto lawful lanes and physically feasible dynamics."""

    episode: ExecutableEpisode
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 6.0
    max_jerk_mps3: float = 4.0
    max_lateral_acceleration_mps2: float = 3.0
    min_merge_clearance_m: float = 2.0
    control_acceleration_scale_mps2: float = 20.0
    steering_residual_fraction: float = 0.35
    _previous_action: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32), init=False
    )
    _previous_speed_mps: float | None = field(default=None, init=False)
    _previous_command_acceleration_mps2: float = field(default=0.0, init=False)
    _filtered_acceleration_mps2: float | None = field(default=None, init=False)
    _lane_change_started: bool = field(default=False, init=False)
    _lane_change_completed: bool = field(default=False, init=False)
    _rejections: Counter[str] = field(default_factory=Counter, init=False)
    _violations: Counter[str] = field(default_factory=Counter, init=False)
    _max_speed_mps: float = field(default=0.0, init=False)
    _max_abs_acceleration_mps2: float = field(default=0.0, init=False)
    _max_abs_jerk_mps3: float = field(default=0.0, init=False)
    _max_lateral_acceleration_mps2: float = field(default=0.0, init=False)
    _max_abs_lane_lateral_m: float = field(default=0.0, init=False)

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

    def _route_lane_index(self) -> tuple[Any, Any, int]:
        vehicle = self.episode.adversary
        projection = self.episode.adversary_route.projection(
            vehicle.position, getattr(vehicle, "heading_theta", 0.0)
        )
        index = int(
            np.clip(
                np.searchsorted(self.episode.adversary_route.lane_end_s_m, projection.s_m),
                0,
                len(self.episode.adversary_route.lane_indices) - 1,
            )
        )
        return self.episode.adversary_route.lane_indices[index]

    def _lane(self, lane_number: int) -> Any:
        start, end, _ = self._route_lane_index()
        return self.episode.env.current_map.road_network.get_lane((start, end, lane_number))

    def _source_lane(self) -> Any:
        if self.contract.target_lane_number is None:
            return self.episode.adversary.navigation.current_lane
        return self._lane(self.contract.source_lane_number)

    def _expected_lane(self) -> Any:
        if self.contract.target_lane_number is not None and self._lane_change_completed:
            return self._lane(self.contract.target_lane_number)
        if self.contract.target_lane_number is not None:
            return self._lane(self.contract.source_lane_number)
        return self.episode.adversary.navigation.current_lane

    def _legal_lane_lateral(self, vehicle: Any) -> float:
        lanes = [self._expected_lane()]
        if (
            self.contract.target_lane_number is not None
            and self._lane_change_started
            and not self._lane_change_completed
        ):
            lanes.append(self._lane(self.contract.target_lane_number))
        return min(
            abs(float(lane.local_coordinates(vehicle.position)[1]))
            for lane in lanes
        )

    @staticmethod
    def _lane_follow_action(vehicle: Any, lane: Any) -> float:
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        lane_heading = float(lane.heading_theta_at(longitudinal))
        heading_error = wrap_to_pi(lane_heading - float(getattr(vehicle, "heading_theta", 0.0)))
        return float(np.clip(1.7 * heading_error + 0.30 * lateral, -1.0, 1.0))

    @staticmethod
    def _vehicle_length(vehicle: Any) -> float:
        return float(getattr(vehicle, "LENGTH", 4.5))

    def _target_lane_has_clearance(self, target_lane: Any) -> bool:
        adversary = self.episode.adversary
        sut = self.episode.sut
        adversary_s = float(target_lane.local_coordinates(adversary.position)[0])
        sut_s = float(target_lane.local_coordinates(sut.position)[0])
        separation = abs(sut_s - adversary_s) - 0.5 * (
            self._vehicle_length(adversary) + self._vehicle_length(sut)
        )
        return separation >= self.min_merge_clearance_m

    @staticmethod
    def _lane_curvature(lane: Any, longitudinal: float) -> float:
        offset = min(2.0, max(0.5, float(getattr(lane, "length", 4.0)) * 0.2))
        before = lane.heading_theta_at(max(0.0, longitudinal - offset))
        after = lane.heading_theta_at(longitudinal + offset)
        delta = (float(after) - float(before) + np.pi) % (2.0 * np.pi) - np.pi
        return abs(delta) / (2.0 * offset)

    def _roundabout_operational_speed_limit(self) -> float:
        """Reduce the posted limit only where the generated curve requires it."""
        route = self.episode.adversary_route
        projection = route.projection(
            self.episode.adversary.position,
            getattr(self.episode.adversary, "heading_theta", 0.0),
        )
        start = int(
            np.clip(
                np.searchsorted(route.lane_end_s_m, projection.s_m),
                0,
                len(route.lane_indices) - 1,
            )
        )
        curvatures: list[float] = []
        road_network = self.episode.env.current_map.road_network
        for lane_index in route.lane_indices[start : start + 3]:
            lane = road_network.get_lane(lane_index)
            length = float(getattr(lane, "length", 0.0))
            curvatures.extend(
                self._lane_curvature(lane, length * fraction)
                for fraction in (0.25, 0.5, 0.75)
            )
        curvature = max(curvatures, default=0.0)
        if curvature <= 1e-4:
            return self.contract.speed_limit_mps
        available_lateral_acceleration = max(
            self.max_lateral_acceleration_mps2 - 0.25,
            0.5,
        )
        curve_speed = np.sqrt(available_lateral_acceleration / curvature) * 0.95
        return float(min(self.contract.speed_limit_mps, curve_speed))

    def _longitudinal_action(self, raw: float) -> float:
        vehicle = self.episode.adversary
        dt = self._step_seconds(self.episode.env)
        speed_limit = self.contract.speed_limit_mps
        if self.episode.layout.conflict_zone_id.startswith("roundabout:"):
            speed_limit = self._roundabout_operational_speed_limit()
        change_limit = self.max_jerk_mps3 * dt
        requested_acceleration = np.clip(
            raw * self.max_acceleration_mps2,
            -self.max_deceleration_mps2,
            self.max_acceleration_mps2,
        )
        if self._speed_mps(vehicle) >= speed_limit:
            requested_acceleration = min(requested_acceleration, 0.0)
        estimated_acceleration = np.clip(
            requested_acceleration,
            self._previous_command_acceleration_mps2 - change_limit,
            self._previous_command_acceleration_mps2 + change_limit,
        )
        estimated_acceleration = np.clip(
            estimated_acceleration,
            -self.max_deceleration_mps2,
            self.max_acceleration_mps2,
        )
        action = estimated_acceleration / self.control_acceleration_scale_mps2
        self._previous_command_acceleration_mps2 = float(estimated_acceleration)
        return float(np.clip(action, -1.0, 1.0))

    def _steering_limit(self, vehicle: Any) -> float:
        speed = self._speed_mps(vehicle)
        if speed <= 0.25:
            return 1.0
        max_steering_rad = np.deg2rad(float(vehicle.config.get("max_steering", 40.0)))
        wheelbase = max(0.6 * self._vehicle_length(vehicle), 1.0)
        lateral_limit = np.arctan(
            (self.max_lateral_acceleration_mps2 - 0.25)
            * wheelbase
            / max(speed, 0.25) ** 2
        )
        return min(1.0, lateral_limit / max(max_steering_rad, 1e-6))

    def _project_lane_steering(self, raw_steering: float, lane: Any) -> float:
        """Keep SAC steering until it would cross the active lane's legal boundary."""
        vehicle = self.episode.adversary
        _, lateral = lane.local_coordinates(vehicle.position)
        lane_correction = self._lane_follow_action(vehicle, lane)
        steering_limit = self._steering_limit(vehicle)
        residual = raw_steering * self.steering_residual_fraction * steering_limit
        # Keep the learned lateral control in the lane interior.  The SAC
        # residual remains unrestricted in the middle of the lane; only near a
        # boundary do we suppress a residual that pushes away from the
        # lane-follow correction or scale it toward zero at the edge.
        if abs(lateral) > 0.35 * lane.width:
            if residual * lane_correction < 0.0:
                residual = 0.0
            boundary_margin = np.clip(
                (0.5 * lane.width - abs(lateral)) / (0.15 * lane.width), 0.0, 1.0
            )
            residual *= float(boundary_margin)
        steering = float(np.clip(
            lane_correction + residual,
            -steering_limit,
            steering_limit,
        ))
        return steering

    def project(self, raw_action: Any) -> ShieldedAction:
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw.shape != (2,) or not np.isfinite(raw).all():
            raise ValueError("traffic shield requires one finite steering/throttle action")
        raw = np.clip(raw, -1.0, 1.0)
        vehicle = self.episode.adversary
        projection = self.episode.adversary_route.projection(
            vehicle.position, getattr(vehicle, "heading_theta", 0.0)
        )
        source_lane = self._source_lane()
        lane = source_lane
        steering_input = float(raw[0])
        steering = self._project_lane_steering(steering_input, source_lane)
        rejection_reason = None
        if self.contract.target_lane_number is not None:
            target_lane = self._lane(self.contract.target_lane_number)
            in_window = self.contract.merge_window_s[0] <= projection.s_m <= self.contract.merge_window_s[1]
            request = abs(float(raw[0])) >= 0.1
            if self._lane_change_started:
                lane = target_lane
                steering = self._project_lane_steering(float(raw[0]), target_lane)
            elif request and not in_window:
                rejection_reason = "outside_merge_window"
            elif request and self.contract.crossing_boundary != "broken":
                rejection_reason = "solid_boundary"
            elif request and not self._target_lane_has_clearance(target_lane):
                rejection_reason = "occupied_target_lane"
            elif request and raw[0] * self._lane_follow_action(vehicle, target_lane) <= 0.0:
                rejection_reason = "wrong_lane_change_direction"
            elif request:
                self._lane_change_started = True
                lane = target_lane
                steering = self._project_lane_steering(float(raw[0]), target_lane)
            if rejection_reason is not None:
                self._rejections[rejection_reason] += 1
        action = np.asarray((steering, self._longitudinal_action(float(raw[1]))), dtype=np.float32)
        self._previous_action = action
        return ShieldedAction(raw, action, rejection_reason)

    def observe(
        self,
        shielded: ShieldedAction,
        environment_info: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        vehicle = self.episode.adversary
        speed = self._speed_mps(vehicle)
        dt = self._step_seconds(self.episode.env)
        raw_acceleration = (
            0.0 if self._previous_speed_mps is None else (speed - self._previous_speed_mps) / dt
        )
        previous_acceleration = self._filtered_acceleration_mps2
        acceleration = (
            raw_acceleration
            if previous_acceleration is None
            else 0.5 * previous_acceleration + 0.5 * raw_acceleration
        )
        jerk = 0.0 if previous_acceleration is None else (acceleration - previous_acceleration) / dt
        self._previous_speed_mps = speed
        self._filtered_acceleration_mps2 = acceleration
        max_steering_rad = np.deg2rad(float(vehicle.config.get("max_steering", 40.0)))
        wheelbase = max(0.6 * self._vehicle_length(vehicle), 1.0)
        lateral_acceleration = speed**2 * np.tan(float(shielded.action[0]) * max_steering_rad) / wheelbase
        self._max_speed_mps = max(self._max_speed_mps, speed)
        self._max_abs_acceleration_mps2 = max(self._max_abs_acceleration_mps2, abs(acceleration))
        self._max_abs_jerk_mps3 = max(self._max_abs_jerk_mps3, abs(jerk))
        self._max_lateral_acceleration_mps2 = max(self._max_lateral_acceleration_mps2, abs(lateral_acceleration))
        if speed > self.contract.speed_limit_mps + 0.5:
            self._violations["speed_limit"] += 1
        # The simulator's finite-difference velocity can contain collision,
        # curve-entry, and reset transients.  The projected command limits
        # above are the enforceable control contract; retain measured
        # acceleration/jerk as telemetry without invalidating a lawful event.
        if abs(lateral_acceleration) > self.max_lateral_acceleration_mps2 + 1e-6:
            self._violations["lateral_acceleration"] += 1
        if environment_info is not None and bool(environment_info.get("out_of_road", False)):
            self._violations["out_of_road"] += 1
        if self._lane_change_started and self.contract.target_lane_number is not None:
            target_lane = self._lane(self.contract.target_lane_number)
            if abs(float(target_lane.local_coordinates(vehicle.position)[1])) < 0.4 * target_lane.width:
                self._lane_change_completed = True
        expected_lane = self._expected_lane()
        lane_lateral = self._legal_lane_lateral(vehicle)
        self._max_abs_lane_lateral_m = max(
            self._max_abs_lane_lateral_m, abs(float(lane_lateral))
        )
        # Touching a line is not a violation; the vehicle centre must leave
        # the legal lane envelope before the event is rejected.  During a
        # lawful cut-in the expected lane switches only after the shield has
        # admitted the adjacent-lane manoeuvre.
        if abs(float(lane_lateral)) > 0.5 * float(expected_lane.width) + 0.15:
            self._violations["lane_boundary_crossing"] += 1
        return {
            "adversary_traffic_violation": bool(self._violations),
            "traffic_shield_rejected": shielded.rejection_reason is not None,
            "traffic_shield_rejection_reason": shielded.rejection_reason,
            "traffic_raw_action": shielded.raw_action.tolist(),
            "traffic_applied_action": shielded.action.tolist(),
            "traffic_lane_change_started": self._lane_change_started,
            "traffic_lane_change_completed": self._lane_change_completed,
            "traffic_rejection_counts": dict(self._rejections),
            "traffic_violation_counts": dict(self._violations),
            "traffic_max_speed_mps": self._max_speed_mps,
            "traffic_max_abs_acceleration_mps2": self._max_abs_acceleration_mps2,
            "traffic_max_abs_jerk_mps3": self._max_abs_jerk_mps3,
            "traffic_max_lateral_acceleration_mps2": self._max_lateral_acceleration_mps2,
            "traffic_expected_lane_lateral_m": float(lane_lateral),
            "traffic_max_abs_lane_lateral_m": self._max_abs_lane_lateral_m,
        }
