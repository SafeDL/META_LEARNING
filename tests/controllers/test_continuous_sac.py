from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import SAC

from controllers.highway.continuous_sac import continuous_control, create
from controllers.highway.control import steering_limit
from controllers.highway.decision_interface import public_world
from controllers.highway.learning import ACTION_INTERFACE, DiscreteDriver
from experiments.public_ads.continuous_sac import _save_checkpoint
from experiments.schema import Scenario
from simulator.highway.continuous_env import ContinuousDrivingEnv
from simulator.highway.observation import VehicleState, VisibleWorld
from simulator.highway.training_env import DrivingEnv


def test_continuous_action_endpoints_and_validation():
    ego = VehicleState("ego", 100, 4, 25, 0)
    first = continuous_control([-1, 1], ego)
    last = continuous_control([1, -1], ego)
    assert first.acceleration == -8 and last.acceleration == 3
    assert first.steering == steering_limit(25) and last.steering == -steering_limit(25)
    for bad in ([1], [1, float("nan")], [[1, 0]]):
        with pytest.raises(ValueError, match="two finite"):
            continuous_control(bad, ego)


def test_saved_sac_replays_identical_visible_controls(tmp_path):
    class Spaces(gym.Env):
        action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        observation_space = gym.spaces.Box(
            -1, 1, (ACTION_INTERFACE["observation_dimension"],), dtype=np.float32,
        )

    untrained = SAC("MlpPolicy", Spaces(), seed=1301, device="cpu")
    path = tmp_path / "policy.pt"
    _save_checkpoint(untrained, path)
    first, second = create(path), create(path)
    for time in (0, 0.2, 0.4):
        ego = VehicleState("ego", 100 + 25 * time, 4, 25, 0)
        lead = VehicleState("scripted_lead", 135 + 14 * time, 4, 14, 0)
        world = VisibleWorld(time, ego, (lead,))
        assert first.act(world) == second.act(world)
        assert first.policy.last_decision == second.policy.last_decision


def test_training_namespace_uses_the_same_observation_as_deployment(monkeypatch):
    ego = VehicleState("ego", 100, 4, 25, 0)
    world = VisibleWorld(0, ego, (VehicleState("lead", 135, 4, 14, 0),))
    scene = Scenario("front_brake:policy_train:1:000000", "front_brake", {})

    def reset(env, **kwargs):
        env.driver = DiscreteDriver()
        env.world = SimpleNamespace(scenario=scene, visible_world=lambda: world)

    monkeypatch.setattr(DrivingEnv, "reset", reset)
    env = ContinuousDrivingEnv(excluded_coordinates=[])
    observation, _ = env.reset()
    np.testing.assert_array_equal(observation, DiscreteDriver().observation(public_world(world)))
    assert env.training_kinds == {"front_brake": 1}
    env = ContinuousDrivingEnv(excluded_coordinates=[scene.coordinate()])
    with pytest.raises(ValueError, match="reserved coordinate"):
        env.reset()
