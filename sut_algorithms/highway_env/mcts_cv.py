"""MCTS-style root sampling with a constant-velocity traffic model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from highway_env_benchmark.envs.external_cutin import ExternalCutInEnv


@dataclass
class MCTSCVPolicy:
    name: str = "mcts_cv"
    ego_kind: str = "mdp"
    rollouts_per_action: int = 24
    horizon: int = 6
    seed: int = 20_260_922
    rng: np.random.Generator = field(init=False, repr=False)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, env: ExternalCutInEnv) -> int:
        action_scores = np.asarray(
            [self._evaluate_action(action, env) for action in range(5)]
        )
        return int(np.argmax(action_scores))

    def _evaluate_action(self, action: int, env: ExternalCutInEnv) -> float:
        values = [
            self._rollout(action, env) for _ in range(self.rollouts_per_action)
        ]
        return float(np.mean(values))

    def _rollout(self, first_action: int, env: ExternalCutInEnv) -> float:
        lead = env.scheduled_vehicle
        lane = int(env.vehicle.lane_index[2])
        speed_index = int(env.vehicle.speed_index)
        speed = float(env.vehicle.index_to_speed(speed_index))
        gap = float(lead.position[0] - env.vehicle.position[0])
        lead_lane = int(lead.lane_index[2])
        value = 0.0

        for depth in range(self.horizon):
            action = first_action if depth == 0 else int(self.rng.integers(0, 5))
            lane, speed = self._apply_action(action, lane, speed)
            gap += lead.speed - speed
            collision_risk = lane == lead_lane and gap < 8.0
            step_value = 0.35 * speed / 40.0 - float(collision_risk)
            value += step_value * 0.9**depth
        return value

    @staticmethod
    def _apply_action(action: int, lane: int, speed: float) -> tuple[int, float]:
        if action == 0:
            lane = max(lane - 1, 0)
        elif action == 2:
            lane = min(lane + 1, 1)
        elif action == 3:
            speed = min(speed + 5.0, 40.0)
        elif action == 4:
            speed = max(speed - 5.0, 0.0)
        return lane, speed

