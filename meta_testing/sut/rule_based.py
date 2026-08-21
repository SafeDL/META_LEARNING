"""A deterministic yielding/gap-acceptance controller distinct from IDM."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .base import ControllerProfile


class RuleBasedSUTAdapter:
    name = "rule_based"

    def reset(self, env: Any, task: Any, config: Mapping[str, Any], seed: int) -> None:
        del env, task, config, seed

    def attach(self, env: Any, vehicle: Any, profile: ControllerProfile, seed: int) -> Any:
        from metadrive.policy.idm_policy import IDMPolicy

        class ProfileRulePolicy(IDMPolicy):
            """Route-following controller with deterministic conservative gaps.

            It intentionally disables IDM's opportunistic overtaking; the
            longitudinal response is a fixed gap rule, creating a behavior
            family that differs from the normal IDM lane-change policy.
            """

            def lane_change_policy(self, all_objects: Any) -> tuple[Any, float, Any]:
                surrounding = self.__class__.front_back(all_objects, self)
                return surrounding.front_object(), surrounding.front_min_distance(), self.routing_target_lane

            @staticmethod
            def front_back(all_objects: Any, policy: Any) -> Any:
                from metadrive.policy.idm_policy import FrontBackObjects
                return FrontBackObjects.get_find_front_back_objs(
                    all_objects, policy.routing_target_lane, policy.control_object.position, policy.MAX_LONG_DIST,
                    policy.control_object.navigation.current_ref_lanes,
                )

        env.engine.traffic_manager.add_policy(vehicle.id, ProfileRulePolicy, vehicle, int(seed))
        policy = env.engine.get_policy(vehicle.id)
        policy.target_speed = float(profile.target_speed_mps) * 3.6
        policy.enable_lane_change = bool(profile.enable_lane_change)
        policy.DISTANCE_WANTED = float(profile.yield_gap_m)
        policy.TIME_WANTED = max(0.5, float(profile.brake_gap_m) / max(profile.target_speed_mps, 1e-6))
        policy.DEACC_FACTOR = -max(2.0, float(profile.brake_gap_m) / 2.0)
        return policy

    def observe(self, env: Any, vehicle: Any) -> Mapping[str, Any]:
        return {
            "position": np.asarray(vehicle.position, dtype=np.float32).copy(),
            "velocity": np.asarray(vehicle.velocity, dtype=np.float32).copy(),
            "speed_mps": float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6,
            "episode_step": int(getattr(env, "episode_step", 0)),
        }

    def step(self, observation: Mapping[str, Any]) -> Any:
        return observation

    def metadata(self, profile: ControllerProfile) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "profile": profile.profile_id,
            "model_input_fields": (),
            "profile_is_model_input": False,
        }
