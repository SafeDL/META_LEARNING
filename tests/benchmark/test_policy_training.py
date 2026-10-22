import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
import torch

from controllers.highway.learning import DiscreteDriver
from simulation.highway.observation import VehicleState, VisibleWorld
from simulation.highway.training_env import DrivingEnv, EPISODE_DURATIONS, safety_penalty
from simulation.highway.collision_env import CollisionWorld
from benchmark.scenarios import sample
from benchmark.qualification import brake_scene


def test_discrete_policy_interface_includes_full_stop_and_road_boundaries():
    world = VisibleWorld(0, VehicleState("ego", 100, 4, 20, 0), (), (4.0,))
    driver = DiscreteDriver()
    for _ in range(4):
        driver.command(4, world)
    assert driver.target_speed == 0
    for action in (0, 1, 2, 4):
        command = driver.command(action, world)
        assert command.acceleration == -8
        assert command.steering == 0
    stopped = VisibleWorld(0, VehicleState("ego", 100, 4, 0, 0), (), (4.0,))
    assert driver.command(1, stopped).acceleration == 0
    # Target speed remains zero after a completed stop and a keep action.
    assert driver.observation(stopped)[-1] == 0
    braking_world = CollisionWorld(brake_scene("driver_stop"), duration_s=10, lane_centers=(4.0,))
    driver = DiscreteDriver()
    for step in range(50):
        braking_world.advance(driver.command(4 if step < 5 else 1, braking_world.visible_world()))
    assert braking_world.ego.speed < 0.001 and not braking_world.ego.crashed
    env = DrivingEnv()
    check_env(env, warn=True)
    a, _ = env.reset(seed=1101)
    b, _ = env.reset(seed=1101)
    np.testing.assert_array_equal(a, b)
    assert env.observation_space.contains(a)


def test_both_sb3_algorithms_train_against_the_shared_interface(tmp_path):
    torch.set_num_threads(1)
    for algorithm in (PPO, DQN):
        env = DrivingEnv()
        parameters = ({"n_steps": 32, "batch_size": 16, "n_epochs": 1}
                      if algorithm is PPO else
                      {"learning_starts": 8, "buffer_size": 200,
                       "batch_size": 16, "train_freq": 4})
        model = algorithm("MlpPolicy", env, seed=1101, device="cpu", verbose=0,
                          policy_kwargs={"net_arch": [16, 16]}, **parameters)
        model.learn(total_timesteps=64)
        destination = tmp_path / algorithm.__name__
        model.save(destination)
        restored = algorithm.load(destination, device="cpu")
        observation, _ = env.reset(seed=21001)
        action, _ = restored.predict(observation, deterministic=True)
        _, reward, terminated, truncated, _ = env.step(int(action))
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        env.close()


def test_keep_action_preserves_an_incomplete_lane_change_and_exposes_its_state():
    world = CollisionWorld(sample("front_brake", 1, "qualification")[0])
    driver = DiscreteDriver()
    driver.command(0, world.visible_world())  # initiate left, retain target speed
    assert driver.target_lane == 0
    command = driver.command(1, world.visible_world())  # keep both targets
    assert driver.target_lane == 0
    assert command.steering < 0
    assert driver.observation(world.visible_world())[-2] < 0
    assert driver.observation(world.visible_world()).shape == (83,)
    np.testing.assert_allclose(driver.observation(world.visible_world())[-1], world.ego.speed / 40)
    for _ in range(20):
        world.advance(driver.command(1, world.visible_world()))
    assert world.ego.position[1] < 0.5


def test_observable_safety_reward_favors_stopping_near_a_stationary_vehicle():
    def world(speed, lead_speed=0, lead_x=10):
        return VisibleWorld(0, VehicleState("ego", 0, 4, speed, 0),
                            (VehicleState("visible", lead_x, 4, lead_speed, 0),), (4.0,))
    assert safety_penalty(world(25, lead_speed=25, lead_x=70)) == 0
    assert safety_penalty(world(0)) == 0
    for speed in (0.2, 0.8, 5):
        # Per 0.2-second decision, progress contributes 0.01 * speed to reward.
        assert safety_penalty(world(speed)) > 0.01 * speed
    assert safety_penalty(world(25)) > safety_penalty(world(5))


def test_training_includes_long_episodes_and_truncates_at_the_world_horizon():
    assert max(EPISODE_DURATIONS) >= 18
    env = DrivingEnv()
    durations = set()
    for seed in range(20):
        env.reset(seed=seed)
        durations.add(env.world.duration_s)
    assert durations == set(EPISODE_DURATIONS)
    env.world = CollisionWorld(brake_scene("training_horizon"), duration_s=0.2, lane_centers=(4.0,))
    env.driver = DiscreteDriver()
    _, _, terminated, truncated, _ = env.step(1)
    assert truncated and not terminated
    env.close()
