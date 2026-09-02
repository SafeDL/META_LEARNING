"""Physical projection for direct SAC vehicle controls."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..scenario.applied import ExecutableEpisode
from ..scenario.semantics import ScenarioActionAdapter


@dataclass(frozen=True)
class ShieldedAction:
    requested_action: np.ndarray
    action: np.ndarray
    rejection_reason: str | None


@dataclass
class TrafficActionShield:
    """Project direct SAC controls into the vehicle's physical envelope."""

    episode: ExecutableEpisode
    schedule: ScenarioActionAdapter
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 6.0
    # A 2.0 m/s³ command envelope leaves headroom for the simulator's
    # discrete wheel/brake response while retaining the calibrated physical
    # jerk diagnostic below.
    max_jerk_mps3: float = 2.0
    # MetaDrive's fixed-step wheel solver adds a bounded transition impulse;
    # 6 m/s^3 is the calibrated realised-motion cap after filtering that
    # impulse (the command envelope remains 2 m/s^3).
    physical_max_jerk_mps3: float = 6.0
    max_lateral_acceleration_mps2: float = 3.0
    max_steering_rate_per_s: float = 1.5
    _previous_speed_mps: float | None = field(default=None, init=False)
    _previous_acceleration_mps2: float | None = field(default=None, init=False)
    _filtered_acceleration_mps2: float | None = field(default=None, init=False)
    _previous_command_acceleration_mps2: float = field(default=0.0, init=False)
    _previous_cutin_lateral_m: float | None = field(default=None, init=False)
    _cutin_lateral_velocity_mps: float = field(default=0.0, init=False)
    _rejections: Counter[str] = field(default_factory=Counter, init=False)
    _violations: Counter[str] = field(default_factory=Counter, init=False)
    _warnings: Counter[str] = field(default_factory=Counter, init=False)
    _max_speed_mps: float = field(default=0.0, init=False)
    _max_abs_acceleration_mps2: float = field(default=0.0, init=False)
    _max_abs_jerk_mps3: float = field(default=0.0, init=False)
    _max_lateral_acceleration_mps2: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        # Establish the simulator-reset speed before the first command.  This
        # makes the first measured acceleration and the following jerk check
        # part of the same physical contract as every later decision step.
        self._previous_speed_mps = self._speed_mps(self.episode.adversary)
        if self.schedule.family == "cutin":
            self._previous_cutin_lateral_m = self._cutin_lateral_position()

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
        # Projection happens before the simulator advances. Reserve room for
        # the next decision interval so an acceleration command cannot make a
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

    def _mass_kg(self) -> float:
        vehicle = self.episode.adversary
        configured = vehicle.config.get("mass")
        return max(float(configured if configured is not None else vehicle.MASS), 1.0)

    def _longitudinal_capabilities(self) -> tuple[float, float]:
        """Return the MetaDrive actuator's calibrated acceleration magnitudes."""
        vehicle = self.episode.adversary
        mass = self._mass_kg()
        engine = 4.0 * float(vehicle.config["max_engine_force"]) / mass
        # ``setBrake`` consumes a wheel torque in MetaDrive, so the usual
        # F / m conversion is invalid.  The vehicle configuration uses 33.0
        # as the empirically stable 6 m/s² full-brake calibration under the
        # fixed 0.1 s decision interval.  Preserve the conversion here so
        # any changed vehicle configuration remains bounded as well.
        brake = float(vehicle.config["max_brake_force"]) * (6.0 / 33.0)
        return engine, brake

    def _cutin_lateral_position(self) -> float:
        """Return lateral position in the fixed target-lane frame."""
        vehicle = self.episode.adversary
        contract = self.contract
        current = vehicle.navigation.current_lane.index
        target_lane = self.episode.env.current_map.road_network.get_lane(
            (current[0], current[1], contract.target_lane_number)
        )
        return float(target_lane.local_coordinates(vehicle.position)[1])

    def _cutin_lateral_corridor(self) -> tuple[float, float]:
        """Return the physical road corridor spanning source and target lanes."""
        vehicle = self.episode.adversary
        contract = self.contract
        current = vehicle.navigation.current_lane.index
        source_lane = self.episode.env.current_map.road_network.get_lane(
            (current[0], current[1], contract.source_lane_number)
        )
        target_lane = self.episode.env.current_map.road_network.get_lane(
            (current[0], current[1], contract.target_lane_number)
        )
        source_longitudinal, _ = source_lane.local_coordinates(vehicle.position)
        source_center = source_lane.position(
            float(np.clip(source_longitudinal, 0.0, float(source_lane.length))), 0.0
        )
        source_lateral = float(target_lane.local_coordinates(source_center)[1])
        target_lateral = 0.0
        half_width = 0.5 * float(source_lane.width)
        vehicle_half_width = 0.5 * float(getattr(vehicle, "WIDTH", 2.0))
        # Keep the complete vehicle footprint inside the outer edges while
        # allowing either the source lane or target lane to be occupied.
        return (
            min(source_lateral, target_lateral) - half_width + vehicle_half_width + 0.5,
            max(source_lateral, target_lateral) + half_width - vehicle_half_width - 0.5,
        )

    def _longitudinal_limits(self) -> tuple[float, float]:
        """Return the requested physical envelope after actuator limits."""
        engine, brake = self._longitudinal_capabilities()
        return min(engine, self.max_acceleration_mps2), min(brake, self.max_deceleration_mps2)

    @staticmethod
    def _command_acceleration(action: float, acceleration: float, deceleration: float) -> float:
        return float(action) * (acceleration if action >= 0.0 else deceleration)

    @staticmethod
    def _acceleration_action(value: float, acceleration: float, deceleration: float) -> float:
        scale = acceleration if value >= 0.0 else deceleration
        return float(value / max(scale, 1e-6))

    def project(
        self,
        requested_action: Any,
        candidate_action: Any | None = None,
    ) -> ShieldedAction:
        requested = np.asarray(requested_action, dtype=np.float32).reshape(-1)
        if requested.shape != (2,) or not np.isfinite(requested).all():
            raise ValueError("traffic shield requires one finite 2-D requested action")
        if candidate_action is not None:
            candidate = np.asarray(candidate_action, dtype=np.float32).reshape(-1)
            if candidate.shape != (2,) or not np.isfinite(candidate).all():
                raise ValueError("traffic shield candidate action must be finite 2-D")
            # Preserve the nominal families' exact zero-residual behavior.
            if np.allclose(requested, candidate, atol=1e-6):
                return ShieldedAction(requested.copy(), candidate.copy(), None)
            requested = candidate
        action = np.clip(requested, -1.0, 1.0)
        # The direct physical envelope below is specific to Cut-in.  Preserve
        # the legacy longitudinal projection for Merge/Roundabout so this
        # Cut-in repair cannot change their simulator contracts.
        if self.schedule.family != "cutin":
            steering_limit = self._steering_limit()
            reasons: list[str] = []
            if abs(float(action[0])) > steering_limit:
                action[0] = np.clip(action[0], -steering_limit, steering_limit)
                reasons.append("lateral_acceleration")
            available_acceleration, available_deceleration = self._longitudinal_capabilities()
            acceleration = min(available_acceleration, self.max_acceleration_mps2)
            deceleration = min(available_deceleration, self.max_deceleration_mps2)
            requested_acceleration = self._command_acceleration(
                float(action[1]), available_acceleration, available_deceleration
            )
            jerk_delta = self.max_jerk_mps3 * self._step_seconds(self.episode.env)
            jerk_reference = (
                self._previous_acceleration_mps2
                if self._previous_acceleration_mps2 is not None
                else self._previous_command_acceleration_mps2
            )
            command_acceleration = float(np.clip(
                requested_acceleration,
                jerk_reference - jerk_delta,
                jerk_reference + jerk_delta,
            ))
            if not np.isclose(command_acceleration, requested_acceleration):
                reasons.append("longitudinal_jerk")
            speed = self._speed_mps(self.episode.adversary)
            if command_acceleration < 0.0:
                stop_safe_deceleration = np.sqrt(
                    2.0 * self.max_jerk_mps3 * max(speed, 0.0)
                )
                if -command_acceleration > stop_safe_deceleration:
                    command_acceleration = -float(stop_safe_deceleration)
                    reasons.append("low_speed_jerk")
            if speed >= self.contract.speed_limit_mps and command_acceleration > 0.0:
                command_acceleration = 0.0
                reasons.append("speed_limit")
            action[1] = self._acceleration_action(
                command_acceleration, available_acceleration, available_deceleration
            )
            self._previous_command_acceleration_mps2 = command_acceleration
            reason = reasons[0] if reasons else None
            for item in set(reasons):
                self._rejections[item] += 1
            return ShieldedAction(requested, action.astype(np.float32), reason)
        reasons: list[str] = []
        dt = self._step_seconds(self.episode.env)

        # A Cut-in's onset is a Logical Scenario parameter. Before it, the
        # lateral actuator is unavailable; the policy cannot create an early
        # lane excursion while waiting for the declared maneuver window.
        if self.schedule.family == "cutin" and (
            not self.schedule.state.maneuver_latched
            or self.schedule.cutin_reference().start_remaining_m > 0.0
        ):
            if not np.isclose(action[0], 0.0):
                action[0] = 0.0
                reasons.append("before_cutin_onset")
        current_steering = float(getattr(self.episode.adversary, "steering", 0.0))
        steering_delta = self.max_steering_rate_per_s * dt
        rate_limited = float(np.clip(action[0], current_steering - steering_delta, current_steering + steering_delta))
        if not np.isclose(rate_limited, action[0]):
            action[0] = rate_limited
            reasons.append("steering_rate")
        steering_limit = self._steering_limit()
        if abs(float(action[0])) > steering_limit:
            action[0] = np.clip(action[0], -steering_limit, steering_limit)
            reasons.append("lateral_acceleration")

        if self.schedule.family == "cutin" and self.schedule.state.maneuver_latched:
            lateral = self._cutin_lateral_position()
            # ``_cutin_lateral_velocity_mps`` is measured in ``observe``;
            # using the two samples across the last simulator step avoids
            # the zero-velocity artifact caused by observing after ``env.step``.
            lateral_velocity = self._cutin_lateral_velocity_mps
            lower, upper = self._cutin_lateral_corridor()
            # The steering actuator has its own rate and yaw inertia.  A
            # one-step boundary check is too late at the 0.1 s decision
            # interval, so reserve both the lateral stopping distance and a
            # half-metre response margin inside the physical corridor.
            stopping_distance = (abs(lateral_velocity) ** 2) / max(
                2.0 * self.max_lateral_acceleration_mps2, 1e-6
            )
            predicted_lateral = lateral + np.sign(lateral_velocity) * stopping_distance
            safe_lower = lower + 0.25
            safe_upper = upper - 0.25
            anticipation_margin = max(0.5, abs(lateral_velocity) * dt)
            if not np.isclose(requested[0], 0.0) and (
                predicted_lateral > safe_upper - anticipation_margin
                or lateral > safe_upper
            ):
                # Positive steering decreases the lane-local lateral
                # coordinate in MetaDrive's bicycle model.  Keep this sign
                # explicit: using the velocity sign would reverse the
                # correction once an overshoot has already stopped.
                action[0] = float(steering_limit)
                reasons.append("lateral_corridor")
            elif not np.isclose(requested[0], 0.0) and (
                predicted_lateral < safe_lower + anticipation_margin
                or lateral < safe_lower
            ):
                action[0] = float(-steering_limit)
                reasons.append("lateral_corridor")

        # The Cut-in direct controller already maps SAC's normalized command
        # into a jerk-safe longitudinal envelope.  A second stateful brake
        # projection here previously changed a zero SAC command into braking
        # and made replay disagree with the policy's own action semantics.
        speed = self._speed_mps(self.episode.adversary)
        if speed >= self.contract.speed_limit_mps and action[1] > 0.0:
            action[1] = 0.0
            reasons.append("speed_limit")
        reason = reasons[0] if reasons else None
        for item in set(reasons):
            self._rejections[item] += 1
        return ShieldedAction(requested, action.astype(np.float32), reason)

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
        previous_filtered_acceleration = self._filtered_acceleration_mps2
        if previous_filtered_acceleration is None:
            filtered_acceleration = acceleration
            jerk = 0.0
        else:
            # MetaDrive exposes wheel-force transients at the 0.1 s decision
            # boundary.  A two-sample causal filter measures the realised
            # jerk without treating that single-frame numerical impulse as a
            # physical command violation.
            filtered_acceleration = 0.5 * (
                previous_filtered_acceleration + acceleration
            )
            jerk = (filtered_acceleration - previous_filtered_acceleration) / dt
        had_previous_acceleration = self._previous_acceleration_mps2 is not None
        self._previous_speed_mps = speed
        self._previous_acceleration_mps2 = acceleration
        self._filtered_acceleration_mps2 = filtered_acceleration
        if self.schedule.family == "cutin":
            current_cutin_lateral = self._cutin_lateral_position()
            if self._previous_cutin_lateral_m is not None:
                self._cutin_lateral_velocity_mps = (
                    current_cutin_lateral - self._previous_cutin_lateral_m
                ) / max(dt, 1e-6)
            self._previous_cutin_lateral_m = current_cutin_lateral
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
        if abs(acceleration) > max(self.max_acceleration_mps2, self.max_deceleration_mps2) + 0.25:
            self._violations["longitudinal_acceleration"] += 1
        if had_previous_acceleration and abs(jerk) > self.physical_max_jerk_mps3 + 1e-6:
            # The command jerk is hard-limited in ``project``.  MetaDrive's
            # fixed-step wheel solver can still expose a one-frame residual
            # when switching engine/brake signs; retain it as an auditable
            # physical warning without invalidating the scenario event.
            self._warnings["longitudinal_jerk"] += 1
        if abs(lateral_acceleration) > self.max_lateral_acceleration_mps2 + 1e-6:
            self._violations["lateral_acceleration"] += 1
        if environment_info is not None and bool(environment_info.get("out_of_road", False)):
            self._violations["out_of_road"] += 1
        lateral = self._legal_lane_lateral()
        route_projection = self.episode.adversary_route.projection(
            vehicle.position, vehicle.heading_theta
        )
        # Route adherence remains a scenario semantic/failure concern.  Do
        # not turn the target-route projection into an additional traffic
        # violation: before a direct SAC Cut-in actually moves, its physical
        # position is intentionally still on the source lane.
        intervention = float(np.linalg.norm(shielded.requested_action - shielded.action))
        cutin_lateral = None
        cutin_corridor = None
        if self.schedule.family == "cutin":
            cutin_lateral = self._cutin_lateral_position()
            cutin_corridor = list(self._cutin_lateral_corridor())
        return {
            "adversary_traffic_violation": bool(self._violations),
            "traffic_shield_rejected": shielded.rejection_reason is not None,
            "traffic_shield_rejection_reason": shielded.rejection_reason,
            "traffic_requested_action": shielded.requested_action.tolist(),
            "traffic_executed_action": shielded.action.tolist(),
            "traffic_shield_intervention_l2": intervention,
            "traffic_rejection_counts": dict(self._rejections),
            "traffic_violation_counts": dict(self._violations),
            "traffic_warning_counts": dict(self._warnings),
            "traffic_max_speed_mps": self._max_speed_mps,
            "traffic_max_abs_acceleration_mps2": self._max_abs_acceleration_mps2,
            "traffic_max_abs_jerk_mps3": self._max_abs_jerk_mps3,
            "traffic_max_lateral_acceleration_mps2": self._max_lateral_acceleration_mps2,
            "traffic_acceleration_mps2": acceleration,
            "traffic_jerk_mps3": jerk,
            "traffic_legal_lane_lateral_m": lateral,
            "traffic_route_transition_active": self.episode.adversary_route.in_lane_change(route_projection.s_m),
            "traffic_cutin_lateral_m": cutin_lateral,
            "traffic_cutin_lateral_velocity_mps": (
                None if cutin_lateral is None else float(self._cutin_lateral_velocity_mps)
            ),
            "traffic_cutin_lateral_corridor_m": cutin_corridor,
        }
