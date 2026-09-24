from __future__ import annotations

import numpy as np

from highway_env_benchmark.envs.cutin_env import CutInScenario
from sut_algorithms.highway_env.idm_profiles import get_profile
from replications.adate_highway_env.adate.dense_env import DenseCutInEnv
from replications.adate_highway_env.adate.dense_protocol import (
    _scenario_pool,
    _scenario_probabilities,
    run_dense_batch,
)
from replications.adate_highway_env.adate.state_encoder import StateEncoder


def test_dense_action_reaches_vehicle_dynamics_and_snapshot_replays():
    scenario = CutInScenario(12.0, -6.0, "fast_intrusion")
    env = DenseCutInEnv(get_profile("SUT-A"), scenario)
    try:
        env.reset(seed=19)
        env.step(0)  # residual -2 m/s²
        snapshot = env.snapshot()
        before = env.state_features()
        env.restore(snapshot)
        after = env.state_features()
        trace = env.dense_trace()
        assert np.isclose(before["longitudinal_gap"], after["longitudinal_gap"])
        assert np.isclose(before["relative_speed"], after["relative_speed"])
        assert np.isclose(before["schedule_phase"], after["schedule_phase"])
        assert "npc_target_lane" in before
        assert np.all(trace.residual_action == -2.0)
        assert np.any(trace.npc_acceleration < -1.0)
    finally:
        env.close()


def test_dense_batch_rejects_target_leakage_before_source_collection(tmp_path):
    config = {
        "seed": 1,
        "batch_seeds": [1],
        "target_profiles": ["SUT-A"],
        "source_profiles": ["SUT-A"],
    }
    try:
        run_dense_batch(config, tmp_path, lambda *_: None)
    except ValueError as error:
        assert "held out" in str(error)
    else:
        raise AssertionError("target leakage must be rejected before source collection")


def test_passing_mode_controls_both_background_vehicles():
    scenario = CutInScenario(30.0, -2.0, "passing_cutin")
    env = DenseCutInEnv(get_profile("AV-Reference-IDM"), scenario)
    try:
        env.reset(seed=23)
        env.step(0)
        features = env.state_features()
        trace = env.dense_trace()
        assert features["leading_gap"] > 0.0
        assert features["leading_relative_speed"] < 0.0
        assert np.all(trace.residual_action == -6.0)
        assert np.allclose(trace.leading_acceleration, 0.0, atol=1e-6)
    finally:
        env.close()


def test_passing_encoder_matches_the_paper_state_variables():
    encoder = StateEncoder(
        representation="passing",
        position_bin=4.0,
        speed_bin=2.0,
        time_bin=0.2,
    )
    observation = {
        "npc_speed": 24.0,
        "leading_gap": 24.0,
        "leading_relative_speed": -3.0,
        "longitudinal_gap": 30.0,
        "relative_speed": -2.0,
        "time": 1.0,
    }
    assert encoder.encode(observation) == (12, 6, -2, 8, -1, 5)


def test_declared_initial_state_distribution_is_normalized():
    config = {
        "scenario_design": {
            "modes": ["passing_cutin"],
            "gaps": [10.0, 30.0],
            "gap_probabilities": [0.1, 0.9],
            "relative_speeds": [-5.0, 1.0],
            "relative_speed_probabilities": [0.2, 0.8],
        }
    }
    scenarios = _scenario_pool(config)
    probabilities = _scenario_probabilities(config, scenarios)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.allclose(probabilities, [0.02, 0.08, 0.18, 0.72])
