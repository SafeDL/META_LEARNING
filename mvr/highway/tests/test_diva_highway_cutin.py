from __future__ import annotations

import pytest

from mvr.highway.envs.cutin_env import CutInScenario, run_cutin_episode
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
