"""Shared tactical actions with persistent lane and zero-capable speed targets."""

import json
from pathlib import Path

import numpy as np

from simulator.highway.observation import policy_observation

from .control import Control, lane_steering

CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoints" / "collision"
ACTION_INTERFACE = {
    "actions": ["left", "keep", "right", "faster", "slower"],
    "speed_increment_mps": 5.0,
    "speed_gain_per_s": 2.0,
    "minimum_target_speed_mps": 0.0,
    "maximum_target_speed_mps": 35.0,
    "observation_dimension": 83,
}


class DiscreteDriver:
    def __init__(self):
        self.target_lane = None
        self.target_speed = None

    def observation(self, world):
        if self.target_lane is None:
            self.target_lane = min(
                range(len(world.lane_centers)),
                key=lambda i: abs(world.lane_centers[i] - world.ego.y),
            )
        if self.target_speed is None:
            self.target_speed = world.ego.speed
        visible = policy_observation(world, target_lane_y=world.lane_centers[self.target_lane])
        return np.append(visible, self.target_speed / 40).astype(np.float32)

    def command(self, action, world):
        if not 0 <= int(action) < 5 or int(action) != action:
            raise ValueError("Learning action must be one of the common five tactical intents")
        self.observation(world)
        if action in (3, 4):
            increment = ACTION_INTERFACE["speed_increment_mps"] * (1 if action == 3 else -1)
            self.target_speed = float(
                np.clip(
                    self.target_speed + increment,
                    ACTION_INTERFACE["minimum_target_speed_mps"],
                    ACTION_INTERFACE["maximum_target_speed_mps"],
                )
            )
        lateral = int(action) if action < 3 else 1
        settled = (
            abs(world.ego.y - world.lane_centers[self.target_lane]) < 0.3
            and abs(world.ego.heading) < 0.05
        )
        if settled:
            self.target_lane = int(
                np.clip(self.target_lane + lateral - 1, 0, len(world.lane_centers) - 1)
            )
        acceleration = ACTION_INTERFACE["speed_gain_per_s"] * (self.target_speed - world.ego.speed)
        return Control(
            float(np.clip(acceleration, -8, 3)),
            lane_steering(world.ego, world.lane_centers[self.target_lane]),
        )


class LearnedController:
    def __init__(self, sut):
        from stable_baselines3 import DQN, PPO

        root = CHECKPOINT_ROOT / sut.training_run
        manifest = json.loads((root / "training.json").read_text(encoding="utf-8"))
        if (
            manifest["training_seed"] != sut.training_seed
            or manifest["sut_id"] != sut.sut_id
            or manifest["status"] != "qualified"
        ):
            raise ValueError("Learning SUT requires its own independently qualified checkpoint")
        self.model = (PPO if sut.lineage == "ppo" else DQN).load(root / "best.zip", device="cpu")
        self.driver = DiscreteDriver()

    def act(self, world):
        action, _ = self.model.predict(self.driver.observation(world), deterministic=True)
        return self.driver.command(int(action), world)
