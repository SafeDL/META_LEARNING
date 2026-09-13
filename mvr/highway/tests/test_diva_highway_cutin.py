from __future__ import annotations

import numpy as np
import pytest

from mvr.highway.data.generate_anchor_bank import generate_dual_mode_anchor_bank
from mvr.highway.envs.cutin_env import (
    CutInEnv,
    CutInScenario,
    run_cutin_episode,
    run_cutin_episode_with_trace,
)
from mvr.highway.sut.idm_profiles import PROFILE_NAMES, get_profile


@pytest.mark.parametrize("sut_name", ("SUT-A", "SUT-D"))
def test_cutin_episode_returns_complete_safety_response(sut_name: str):
    result = run_cutin_episode(get_profile(sut_name), CutInScenario(8.0, -6.0), seed=11)
    assert 0.0 <= result.vulnerability <= 1.0
    assert result.min_distance >= 0.0
    assert result.collision or result.completed


def test_sut_bank_has_three_idm_and_three_fvdm_controllers():
    controllers = [get_profile(name).controller for name in PROFILE_NAMES]
    assert controllers.count("IDM") == controllers.count("FVDM") == 3


def test_dual_mode_bank_is_balanced_and_modes_are_simulable():
    anchors, modes = generate_dual_mode_anchor_bank(8, seed=11)
    assert anchors.shape == (8, 2)
    assert (modes == "fast_intrusion").sum() == (modes == "cutin_braking").sum() == 4
    for mode in (CutInEnv.FAST_INTRUSION, CutInEnv.CUTIN_BRAKING):
        result = run_cutin_episode(
            get_profile("SUT-A"), CutInScenario(8.0, -6.0, mode), seed=11
        )
        assert 0.0 <= result.vulnerability <= 1.0


def test_cutin_braking_applies_low_level_brake_for_one_second():
    profile = get_profile("SUT-A")
    braking_result, braking_trace = run_cutin_episode_with_trace(
        profile, CutInScenario(40.0, 0.0, CutInEnv.CUTIN_BRAKING), seed=20260912
    )
    _, baseline_trace = run_cutin_episode_with_trace(
        profile, CutInScenario(40.0, 0.0, "single"), seed=20260912
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
