"""Episode-level failure reset curriculum using the existing driving reward."""

from collections import Counter
from copy import deepcopy

import gymnasium as gym

from controllers.highway.learning import DiscreteDriver
from experiments.protocol import episode_duration
from simulator.highway.collision_env import CollisionWorld
from simulator.highway.training_env import DrivingEnv


def physical_coordinate(scene):
    return tuple(sorted(scene.parameters.items()))


class RepairEnv(DrivingEnv):
    def __init__(self, current, historical, excluded, *, preparation=False):
        super().__init__()
        if not current or (not preparation and not historical):
            raise ValueError("Repair requires actual paid failure reset points")
        self.current = tuple(current)
        self.historical = tuple(historical)
        self.excluded = set(excluded)
        self.preparation = preparation
        self.episodes = Counter()
        self.transitions = Counter()
        self.reset_points = Counter()
        self.active_category = None

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        categories = ("ordinary", "current") if self.preparation else ("ordinary", "current", "historical")
        probabilities = (0.5, 0.5) if self.preparation else (0.5, 0.25, 0.25)
        category = str(self.np_random.choice(categories, p=probabilities))
        self.active_category = category
        self.episodes[category] += 1
        if category == "ordinary":
            # Compare actual parameters, independent of IDs and generation namespaces.
            while True:
                observation, info = super().reset()
                if physical_coordinate(self.world.scenario) not in self.excluded:
                    return observation, {**info, "curriculum": category}
        if self.world is not None:
            self.world.close()
        points = self.current if category == "current" else self.historical
        scene = points[int(self.np_random.integers(len(points)))]
        self.reset_points[scene.scenario_id] += 1
        self.world = CollisionWorld(scene, environment_seed=81001,
                                    duration_s=episode_duration(scene.kind))
        self.driver = DiscreteDriver()
        return self.driver.observation(self.world.visible_world()), {"curriculum": category}

    def step(self, action):
        self.transitions[self.active_category] += 1
        return super().step(action)

    def curriculum_state(self):
        return {"rng": deepcopy(self.np_random.bit_generator.state),
                "episodes": dict(self.episodes), "transitions": dict(self.transitions),
                "reset_points": dict(self.reset_points)}

    def restore_random_state(self, state):
        self.np_random.bit_generator.state = deepcopy(state)
