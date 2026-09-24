from __future__ import annotations

import numpy as np
import pytest

from highway_env_benchmark.data.generate_anchor_bank import generate_multifunction_anchor_bank
from highway_env_benchmark.envs.cutin_env import (
    CutInEnv,
    CutInScenario,
    run_cutin_episode,
    run_cutin_episode_with_trace,
)
from sut_algorithms.highway_env.idm_profiles import PROFILE_NAMES, get_profile


@pytest.mark.parametrize("sut_name", ("SUT-A", "SUT-D"))
def test_cutin_episode_returns_complete_safety_response(sut_name: str):
    result = run_cutin_episode(get_profile(sut_name), CutInScenario(8.0, -6.0), seed=11)
    assert 0.0 <= result.vulnerability <= 1.0
    assert result.min_distance >= 0.0
    assert result.collision or result.completed


def test_sut_bank_has_three_idm_and_three_fvdm_controllers():
    controllers = [get_profile(name).controller for name in PROFILE_NAMES]
    assert controllers.count("IDM") == controllers.count("FVDM") == 3


def test_multifunction_bank_is_balanced_and_longitudinal_modes_are_simulable():
    anchors, modes = generate_multifunction_anchor_bank(10, seed=11)
    expected_modes = {
        "fast_intrusion",
        "cutin_braking",
        "lead_braking",
        "stop_and_go",
        "slow_lead_following",
    }
    assert anchors.shape == (10, 2)
    assert set(modes) == expected_modes
    assert all((modes == mode).sum() == 2 for mode in expected_modes)
    for mode in (
        CutInEnv.LEAD_BRAKING,
        CutInEnv.STOP_AND_GO,
        CutInEnv.SLOW_LEAD_FOLLOWING,
    ):
        result = run_cutin_episode(
            get_profile("SUT-A"), CutInScenario(8.0, -6.0, mode), seed=11
        )
        assert 0.0 <= result.vulnerability <= 1.0
        assert result.collision or result.completed


def test_cutin_braking_applies_low_level_brake_for_one_second():
    profile = get_profile("SUT-A")
    braking_result, braking_trace = run_cutin_episode_with_trace(
        profile, CutInScenario(40.0, 0.0, CutInEnv.CUTIN_BRAKING), seed=20260912
    )
    _, baseline_trace = run_cutin_episode_with_trace(
        profile, CutInScenario(40.0, 0.0, CutInEnv.FAST_INTRUSION), seed=20260912
    )
    brake_indices = np.flatnonzero(
        np.isclose(
            braking_trace.acceleration, -CutInEnv.BRAKING_DECELERATION, atol=1e-9
        )
    )
    assert not braking_result.collision
    assert len(brake_indices) == 20
    assert np.isclose(
        braking_trace.time[brake_indices[-1]]
        - braking_trace.time[brake_indices[0]]
        + 1 / CutInEnv.default_config()["simulation_frequency"],
        CutInEnv.BRAKE_DURATION,
    )
    assert np.isclose(
        braking_trace.speed[brake_indices[0] - 1]
        - braking_trace.speed[brake_indices[-1]],
        CutInEnv.BRAKING_DECELERATION,
    )
    brake_start = CutInEnv.CUTIN_START + CutInEnv.DEFAULT_CUTIN_DURATION
    brake_end = brake_start + CutInEnv.BRAKE_DURATION
    baseline_start = np.flatnonzero(baseline_trace.time <= brake_start)[-1]
    baseline_end = np.flatnonzero(baseline_trace.time >= brake_end)[0]
    assert np.isclose(
        baseline_trace.speed[baseline_start] - baseline_trace.speed[baseline_end], 0.0
    )
    assert braking_trace.lateral_position.min() < 2.0
