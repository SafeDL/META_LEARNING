from __future__ import annotations

from mvr.highway.envs.cutin_env import CutInScenario, run_cutin_episode
from mvr.highway.sut.idm_profiles import get_profile


def test_cutin_episode_returns_complete_safety_response():
    result = run_cutin_episode(get_profile("SUT-A"), CutInScenario(8.0, -6.0), seed=11)
    assert 0.0 <= result.vulnerability <= 1.0
    assert result.min_distance >= 0.0
    assert result.collision or result.completed
