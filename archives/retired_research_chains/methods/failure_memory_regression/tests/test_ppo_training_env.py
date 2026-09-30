"""PPO training actions must own the same physical steps as evaluation."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from highway_sim_env.envs.fbrt_training_env import FBRTTrainingEnv
from highway_sim_env.envs.fbrt_unified_env import FBRTUnifiedEnv
from methods.failure_memory_regression.bidirectional import compile_manifest
from sut_algorithms.highway_env.ppo_ece import PPO_CHECKPOINT
from sut_algorithms.highway_env.registry import build_spec_factory


SEED = 4179931


def _case() -> dict:
    return compile_manifest("development", resolution=3)[0]


def test_training_action_directly_changes_ego_and_uses_four_physics_frames():
    speeds = []
    for action in (3, 4):  # FASTER and SLOWER in the frozen DiscreteMetaAction map.
        env = FBRTTrainingEnv([_case()])
        try:
            env.reset(seed=1, options={"scenario_index": 0, "physics_seed": SEED})
            _, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)
            assert not terminated and not truncated
            assert info["decision_steps"] == 1 and info["physics_steps"] == 4
            assert len(env.core.control_actions) == 1
            assert env.core.adapter.policy is None
            speeds.append(float(env.core.vehicle.speed))
        finally:
            env.close()
    assert speeds[0] > speeds[1]


def test_training_action_trace_matches_frozen_ppo_evaluation():
    case = _case()
    model = PPO.load(PPO_CHECKPOINT, device="cpu")
    training = FBRTTrainingEnv([case])
    evaluation = FBRTUnifiedEnv(build_spec_factory("ppo_ref_v2"), case)
    try:
        observation, _ = training.reset(
            seed=1, options={"scenario_index": 0, "physics_seed": SEED})
        evaluator_observation, _ = evaluation.reset(seed=SEED)
        np.testing.assert_array_equal(observation, evaluator_observation)
        assert model.observation_space == training.observation_space
        assert model.action_space == training.action_space
        for decision in range(3):
            action = int(model.predict(observation, deterministic=True)[0])
            observation, _, terminated, truncated, info = training.step(action)
            for _ in range(4):
                evaluation._advance()
            assert not terminated and not truncated
            assert info["decision_steps"] == decision + 1
            assert info["physics_steps"] == (decision + 1) * 4
            assert training.core.trace == evaluation.trace
            assert training.core.control_actions == evaluation.control_actions
            np.testing.assert_array_equal(
                observation, evaluation.observation_type.observe())
    finally:
        training.close()
        evaluation.close()


def test_training_wrapper_satisfies_stable_baselines_gym_contract():
    env = FBRTTrainingEnv([_case()])
    try:
        check_env(env, warn=False)
    finally:
        env.close()

