"""Lane-stable native IDM implementation behind the black-box SUT boundary."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy

from ..safety.dynamics import VehicleActionProjector
from .base import ControllerProfile


class LaneStableNativeIDMPolicy(IDMPolicy):
    """Native IDM whose lane target follows native localization, never a lane change.

    MetaDrive navigation still selects and updates the road-level route.  This
    thin policy only bypasses ``move_to_next_road()`` because that helper can
    switch to a different lane at a road boundary even with lane changes
    disabled.  PID steering and IDM longitudinal control remain upstream.
    """

    NOMINAL_SPEED_MPS = IDMPolicy.NORMAL_SPEED / 3.6
    COMFORTABLE_LATERAL_ACCELERATION_MPS2 = 1.5

    def configure_dynamics(self, projector: VehicleActionProjector | None) -> None:
        self.action_projector = projector

    def configure_speed(
        self,
        speed_ratio: float,
        speed_limit_mps: float,
        nominal_speed_mps: float,
    ) -> None:
        self.speed_ratio = float(speed_ratio)
        self.speed_limit_mps = float(speed_limit_mps)
        self.scenario_nominal_speed_mps = float(nominal_speed_mps)
        self.nominal_target_speed_mps = 0.0
        self.curve_safe_speed_mps = 0.0

    @staticmethod
    def _curve_safe_speed(lane: Any, position: Any) -> float:
        radius = float(getattr(lane, "radius", math.inf))
        if math.isfinite(radius) and radius > 0.0:
            return math.sqrt(
                LaneStableNativeIDMPolicy.COMFORTABLE_LATERAL_ACCELERATION_MPS2 * radius
            )
        longitudinal, _ = lane.local_coordinates(position)
        lookahead = min(float(lane.length), float(longitudinal) + 8.0)
        distance = max(lookahead - float(longitudinal), 1e-3)
        heading_delta = math.atan2(
            math.sin(float(lane.heading_theta_at(lookahead)) - float(lane.heading_theta_at(longitudinal))),
            math.cos(float(lane.heading_theta_at(lookahead)) - float(lane.heading_theta_at(longitudinal))),
        )
        curvature = abs(heading_delta) / distance
        if curvature <= 1e-4:
            return math.inf
        return math.sqrt(
            LaneStableNativeIDMPolicy.COMFORTABLE_LATERAL_ACCELERATION_MPS2 / curvature
        )

    def _stable_target_lane(self) -> Any:
        navigation_lane = getattr(self.control_object.navigation, "current_lane", None)
        return navigation_lane if navigation_lane is not None else self.control_object.lane

    def _update_target_speed(self, target_lane: Any) -> None:
        curve_safe = self._curve_safe_speed(target_lane, self.control_object.position)
        nominal = min(
            self.NOMINAL_SPEED_MPS,
            self.scenario_nominal_speed_mps,
            float(self.speed_limit_mps),
            curve_safe,
        )
        self.nominal_target_speed_mps = float(nominal)
        self.curve_safe_speed_mps = float(curve_safe)
        self.target_speed = max(1.0, float(self.speed_ratio) * nominal * 3.6)
        self.NORMAL_SPEED = self.target_speed

    def act(self, *args: Any, **kwargs: Any) -> list[float]:
        del args, kwargs
        target_lane = self._stable_target_lane()
        self.routing_target_lane = target_lane
        self._update_target_speed(target_lane)
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        surrounding = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            target_lane,
            self.control_object.position,
            max_distance=self.MAX_LONG_DIST,
        )
        raw_action = [
            self.steering_control(target_lane),
            self.acceleration(surrounding.front_object(), surrounding.front_min_distance()),
        ]
        projector = getattr(self, "action_projector", None)
        action = raw_action if projector is None else projector.project(raw_action).tolist()
        self.action_info["raw_action"] = raw_action
        self.action_info["executed_action"] = action
        self.action_info["action"] = action
        return action


class IDMSUTAdapter:
    name = "idm"

    def reset(self, env: Any, task: Any, config: Mapping[str, Any], seed: int) -> None:
        del env, task, config, seed

    def attach(
        self,
        env: Any,
        vehicle: Any,
        profile: ControllerProfile,
        seed: int,
        speed_limit_mps: float,
        nominal_speed_mps: float,
    ) -> Any:
        """Attach lane-stable native IDM without touching navigation internals."""
        env.engine.traffic_manager.add_policy(
            vehicle.id,
            LaneStableNativeIDMPolicy,
            vehicle,
            int(seed),
        )
        policy = env.engine.get_policy(vehicle.id)
        policy.configure_speed(profile.speed_ratio, speed_limit_mps, nominal_speed_mps)
        policy.configure_dynamics(
            VehicleActionProjector(vehicle, speed_limit_mps)
            if float(vehicle.config.get("max_engine_force", 0.0)) == 825.0 else None
        )
        # Stage 1 uses lane-stable SUT routes.  Keep the tested controller
        # longitudinal-only so its response remains interpretable.
        policy.enable_lane_change = False
        policy.DISTANCE_WANTED = float(profile.distance_wanted_m)
        policy.TIME_WANTED = float(profile.time_headway_s)
        policy.ACC_FACTOR = float(profile.acceleration_factor)
        policy.DEACC_FACTOR = float(profile.deceleration_factor)
        return policy

    def observe(self, env: Any, vehicle: Any) -> Mapping[str, Any]:
        return {
            "position": np.asarray(vehicle.position, dtype=np.float32).copy(),
            "velocity": np.asarray(vehicle.velocity, dtype=np.float32).copy(),
            "speed_mps": float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6,
            "episode_step": int(getattr(env, "episode_step", 0)),
            "current_lane": tuple(vehicle.navigation.current_lane.index),
        }

    def step(self, observation: Mapping[str, Any]) -> Any:
        # MetaDrive invokes the attached traffic policy.
        return observation

    def metadata(self, profile: ControllerProfile) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "profile": profile.profile_id,
            "model_input_fields": (),
            "profile_is_model_input": False,
            "speed_semantics": "scenario_nominal_speed_ratio",
        }
