"""Value iteration over Highway-env's finite TTC state model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from highway_sim_env.envs.external_cutin import ExternalCutInEnv


@dataclass
class ValueIterationPolicy:
    name: str = "vi_ttc"
    ego_kind: str = "mdp"
    gamma: float = 0.9
    iterations: int = 60

    def reset(self) -> None:
        return None

    def act(self, env: ExternalCutInEnv) -> int:
        grid = env.ttc_observation()
        speed_count, lane_count, _ = grid.shape
        values = np.zeros_like(grid, dtype=float)
        lane_reward = (
            np.arange(lane_count, dtype=float)
            / max(lane_count - 1, 1)
            * env.config["right_lane_reward"]
        )
        speed_reward = (
            np.arange(speed_count, dtype=float)
            / max(speed_count - 1, 1)
            * env.config["high_speed_reward"]
        )
        reward = (
            env.config["collision_reward"] * grid
            + lane_reward[None, :, None]
            + speed_reward[:, None, None]
        )
        for _ in range(self.iterations):
            next_values = np.zeros_like(values)
            next_values[:, :, :-1] = values[:, :, 1:]
            values = reward + self.gamma * next_values

        speed = env.vehicle.speed_index
        lane = env.vehicle.lane_index[2]
        scores = [
            self._action_value(action, speed, lane, reward, values, env)
            for action in range(5)
        ]
        return int(np.argmax(scores))

    def _action_value(
        self,
        action: int,
        speed: int,
        lane: int,
        reward: np.ndarray,
        values: np.ndarray,
        env: ExternalCutInEnv,
    ) -> float:
        next_speed = speed
        next_lane = lane
        if action == 0:
            next_lane -= 1
        elif action == 2:
            next_lane += 1
        elif action == 3:
            next_speed += 1
        elif action == 4:
            next_speed -= 1
        next_speed = int(np.clip(next_speed, 0, reward.shape[0] - 1))
        next_lane = int(np.clip(next_lane, 0, reward.shape[1] - 1))
        lane_change_penalty = (
            env.config["lane_change_reward"] if action in (0, 2) else 0.0
        )
        return float(
            reward[next_speed, next_lane, 0]
            + lane_change_penalty
            + self.gamma * values[next_speed, next_lane, 1]
        )

