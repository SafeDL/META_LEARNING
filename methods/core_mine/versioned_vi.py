"""Frozen same-family VI planner variants for configuration-regression testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from highway_env.envs.common.finite_mdp import compute_ttc_grid

from sut_algorithms.highway_env.value_iteration import ValueIterationPolicy


@dataclass
class VersionedVIPolicy(ValueIterationPolicy):
    collision_reward_scale: float = 1.0
    ttc_quantization: float = 1.0
    lane_change_cost_scale: float = 1.0

    def __post_init__(self) -> None:
        if (self.collision_reward_scale <= 0 or self.ttc_quantization <= 0
                or self.lane_change_cost_scale <= 0):
            raise ValueError("VI revision scales and TTC quantization must be positive")

    def act(self, env) -> int:
        grid = (env.ttc_observation() if self.ttc_quantization == 1.0 else
                compute_ttc_grid(env, time_quantization=self.ttc_quantization,
                                 horizon=6.0))
        speed_count, lane_count, _ = grid.shape
        values = np.zeros_like(grid, dtype=float)
        lane_reward = (np.arange(lane_count, dtype=float)
                       / max(lane_count - 1, 1) * env.config["right_lane_reward"])
        speed_reward = (np.arange(speed_count, dtype=float)
                        / max(speed_count - 1, 1) * env.config["high_speed_reward"])
        reward = (env.config["collision_reward"] * self.collision_reward_scale * grid
                  + lane_reward[None, :, None] + speed_reward[:, None, None])
        for _ in range(self.iterations):
            next_values = np.zeros_like(values)
            next_values[:, :, :-1] = values[:, :, 1:]
            values = reward + self.gamma * next_values
        speed = env.vehicle.speed_index
        lane = env.vehicle.lane_index[2]
        scores = [self._action_value(action, speed, lane, reward, values, env)
                  + ((self.lane_change_cost_scale - 1.0)
                     * env.config["lane_change_reward"] if action in (0, 2) else 0.0)
                  for action in range(5)]
        return int(np.argmax(scores))


VERSIONS = {
    "vi_reference": {"collision_reward_scale": 1.0, "ttc_quantization": 1.0,
                     "lane_change_cost_scale": 1.0},
    "vi_risk035": {"collision_reward_scale": 0.35, "ttc_quantization": 1.0,
                   "lane_change_cost_scale": 1.0},
    "vi_coarse2": {"collision_reward_scale": 1.0, "ttc_quantization": 2.0,
                   "lane_change_cost_scale": 1.0},
    "vi_lane10": {"collision_reward_scale": 1.0, "ttc_quantization": 1.0,
                  "lane_change_cost_scale": 10.0},
}


def version_policy(name: str) -> VersionedVIPolicy:
    return VersionedVIPolicy(name=name, **VERSIONS[name])
