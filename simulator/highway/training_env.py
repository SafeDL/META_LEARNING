"""Common Gymnasium training distribution, actions, observation and reward."""

import gymnasium as gym
import numpy as np

from controllers.highway.learning import ACTION_INTERFACE, DiscreteDriver
from experiments.scenarios import sample
from simulator.highway.collision_env import CollisionWorld

EPISODE_DURATIONS = (10.0, 18.0)


def safety_penalty(world):
    """Shape braking and stopping using the same currently visible traffic state."""
    ego = world.ego
    penalty = 0.0
    for actor in world.vehicles:
        if actor.x <= ego.x or abs(actor.y - ego.y) >= 2.2:
            continue
        gap = actor.x - ego.x - (actor.length + ego.length) / 2
        closing = max(0, ego.speed - actor.speed)
        desired_gap = 2 + 1.4 * ego.speed + closing**2 / 12
        deficit = max(0, desired_gap - gap) / (5 + 1.4 * ego.speed)
        penalty += 0.5 * deficit**2
        # A stopped visible lead makes creeping into its remaining gap unproductive.
        proximity = np.clip((15 - gap) / 10, 0, 1)
        penalty += 0.08 * proximity * np.exp(-max(0, actor.speed)) * min(ego.speed, 5)
    return float(penalty)


class DrivingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(len(ACTION_INTERFACE["actions"]))
        self.observation_space = gym.spaces.Box(
            -1, 1, (ACTION_INTERFACE["observation_dimension"],), dtype=np.float32
        )
        self.world = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        kind = self.np_random.choice(
            ("front_brake", "cut_in", "cut_out", "lane_change_rear"), p=[0.4, 0.2, 0.2, 0.2]
        )
        # The separate generation namespace cannot intersect final/qualification coordinates.
        replicate = int(self.np_random.integers(1000000))
        scenario = sample(str(kind), 1, "policy_train", replicate=replicate)[0]
        if kind == "lane_change_rear":
            lanes = (0.0, 4.0)
        elif self.np_random.random() < (0.7 if kind == "front_brake" else 0.35):
            lanes = (4.0,)
        else:
            lanes = (0.0, 4.0, 8.0)
        duration = float(self.np_random.choice(EPISODE_DURATIONS))
        self.world = CollisionWorld(
            scenario,
            environment_seed=int(self.np_random.integers(1000000)),
            duration_s=duration,
            lane_centers=lanes,
        )
        self.driver = DiscreteDriver()
        return self.driver.observation(self.world.visible_world()), {}

    def step(self, action):
        before = self.world.ego.position[0]
        command = self.driver.command(int(action), self.world.visible_world())
        self.world.advance(command)
        ego = self.world.ego
        progress = ego.position[0] - before
        reward = 0.3 * progress / 6 + 0.01
        reward -= 0.002 * command.acceleration**2 / 64
        applied = command.limited(ego.speed, ego.LENGTH)
        lateral_acceleration = ego.speed**2 * np.tan(applied.steering) / ego.LENGTH
        reward -= 0.002 * (lateral_acceleration / 6) ** 2
        reward -= 0.03 * (max(0, ego.speed - 32) / 5) ** 2
        world = self.world.visible_world()
        reward -= safety_penalty(world)
        terminated = bool(ego.crashed or self.world.violations)
        truncated = self.world.terminal and not terminated
        if terminated:
            reward -= 10
        return (
            self.driver.observation(world),
            float(reward),
            terminated,
            bool(truncated),
            {"collision": bool(ego.crashed), "violations": tuple(sorted(self.world.violations))},
        )

    def close(self):
        self.world = None
