"""Adapters that give legacy, native, and external SUTs one runner interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from highway_env.vehicle.controller import MDPVehicle

from method_chains.failure_memory_regression.schema_v2 import BuildSpec
from sut_algorithms.highway_env.idm_profiles import SUTProfile, create_profiled_vehicle
from sut_algorithms.highway_env.mcts_cv import MCTSCVPolicy
from sut_algorithms.highway_env.ppo_ece import PPOPolicy
from sut_algorithms.highway_env.value_iteration import ValueIterationPolicy
from sut_algorithms.highway_env.regression_builds import make_native_vehicle


@dataclass
class PolicyAdapter:
    spec: BuildSpec
    policy: object | None = None

    def create_vehicle(self, env, lane, longitudinal_position: float,
                       initial_speed: float, target_speed: float,
                       lateral_offset_m: float = 0.0,
                       heading_offset_rad: float = 0.0):
        position = lane.position(longitudinal_position, lateral_offset_m)
        common = {"heading": lane.heading_at(longitudinal_position) + heading_offset_rad,
                  "speed": initial_speed,
                  "target_lane_index": env.ego_lane_index}
        if self.spec.adapter_kind == "legacy_profile":
            profile = SUTProfile(**(self.spec.profile or {"name": self.spec.build_id,
                                                          "controller": "IDM"}))
            fault = (self.spec.mutation or {}).get("legacy_fault")
            if fault:
                from method_chains.core_mine.local_fault_idm import LocalFault, LocalFaultIDMVehicle
                return LocalFaultIDMVehicle(env.road, position, profile=profile,
                                            fault=LocalFault(fault), **common)
            return create_profiled_vehicle(env.road, position, profile=profile, **common)
        if self.spec.adapter_kind == "native_vehicle":
            return make_native_vehicle(self.spec.build_id, env.road, position,
                                       target_speed=target_speed, **common)
        if self.spec.adapter_kind == "external_meta_policy":
            return MDPVehicle(env.road, position, target_speeds=env.action_type.target_speeds,
                              target_speed=initial_speed, **common)
        raise ValueError(f"unsupported adapter kind: {self.spec.adapter_kind}")

    def act(self, env, observation=None) -> int | None:
        if self.spec.adapter_kind != "external_meta_policy":
            return None
        if self.policy is None:
            if self.spec.family == "vi_ttc":
                self.policy = ValueIterationPolicy()
            elif self.spec.family == "mcts_cv":
                self.policy = MCTSCVPolicy()
                self.policy.reset()
            else:
                checkpoint = Path(self.spec.profile.get("checkpoint_path", "assets/ppo_ece/vd_1_5_trial_1.zip"))
                self.policy = PPOPolicy(checkpoint)
        obs = env.observation_type.observe() if observation is None else observation
        if self.spec.family in {"vi_ttc", "mcts_cv"}:
            return int(self.policy.act(env))
        self.policy.load()
        action, _ = self.policy.model.predict(obs, deterministic=True)
        return int(action)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


def adapter_for(spec: BuildSpec) -> PolicyAdapter:
    if spec.adapter_kind not in {"legacy_profile", "native_vehicle", "external_meta_policy"}:
        raise ValueError(f"unsupported build adapter: {spec.adapter_kind}")
    return PolicyAdapter(spec)
