"""Continuous actions on the existing native driving development distribution."""

import gymnasium as gym
import numpy as np

from controllers.highway.adapted_agents import driving_reward
from controllers.highway.continuous_sac import continuous_control
from controllers.highway.decision_interface import public_world

from .training_env import DrivingEnv


class ContinuousDrivingEnv(DrivingEnv):
    def __init__(self, *, excluded_coordinates):
        super().__init__()
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.excluded_coordinates = frozenset(excluded_coordinates)
        self.training_kinds = {}

    def _observation(self, visible):
        return self.driver.observation(public_world(visible))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        scene = self.world.scenario
        if (
            scene.scenario_id.split(":")[1] != "policy_train"
            or scene.coordinate() in self.excluded_coordinates
        ):
            raise ValueError("Continuous SAC training encountered a reserved coordinate")
        self.training_kinds[scene.kind] = self.training_kinds.get(scene.kind, 0) + 1
        return self._observation(self.world.visible_world()), {}

    def step(self, action):
        before = self.world.ego.position[0]
        visible = self.world.visible_world()
        command = continuous_control(action, visible.ego)
        self.world.advance(command)
        visible = self.world.visible_world()
        ego = self.world.ego
        reward = driving_reward(ego.position[0] - before, command, visible.ego, visible)
        terminated = bool(ego.crashed or self.world.violations)
        truncated = self.world.terminal and not terminated
        if terminated:
            reward -= 10
        return (
            self._observation(visible),
            float(reward),
            terminated,
            bool(truncated),
            {"collision": bool(ego.crashed), "violations": tuple(sorted(self.world.violations))},
        )
