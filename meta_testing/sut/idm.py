"""IDM implementation behind the new black-box SUT boundary."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .base import ControllerProfile


class IDMSUTAdapter:
    name = "idm"

    def reset(self, env: Any, task: Any, config: Mapping[str, Any], seed: int) -> None:
        del env, task, config, seed

    def attach(self, env: Any, vehicle: Any, profile: ControllerProfile, seed: int) -> Any:
        from metadrive.policy.idm_policy import IDMPolicy

        env.engine.traffic_manager.add_policy(vehicle.id, IDMPolicy, vehicle, int(seed))
        policy = env.engine.get_policy(vehicle.id)
        policy.target_speed = float(profile.target_speed_mps) * 3.6
        policy.enable_lane_change = bool(profile.enable_lane_change)
        # IDM's gap parameters are in metres/seconds and are intentionally
        # profile-specific, but never exposed to learned modules as IDs.
        policy.DISTANCE_WANTED = float(profile.distance_wanted_m)
        policy.TIME_WANTED = float(profile.time_headway_s)
        return policy

    def observe(self, env: Any, vehicle: Any) -> Mapping[str, Any]:
        return {
            "position": np.asarray(vehicle.position, dtype=np.float32).copy(),
            "velocity": np.asarray(vehicle.velocity, dtype=np.float32).copy(),
            "speed_mps": float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6,
            "episode_step": int(getattr(env, "episode_step", 0)),
        }

    def step(self, observation: Mapping[str, Any]) -> Any:
        # MetaDrive invokes the attached traffic policy.  The method exists so
        # external adapters can share the same black-box protocol.
        return observation

    def metadata(self, profile: ControllerProfile) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "profile": profile.profile_id,
            "model_input_fields": (),
            "profile_is_model_input": False,
        }
