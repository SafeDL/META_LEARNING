from __future__ import annotations

import pytest

from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.executor import ScenarioExecutor
from mvr.scenario.option import AdversarialOption
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.scenario.taskbook import load_taskbook
from mvr.scenario.registry import load_adapters


@pytest.mark.parametrize("family", ("merge", "cutin", "roundabout"))
def test_runtime_geometry_hash_and_sut_attachment(family: str) -> None:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.functional_scenario == family and row.geometry_split == "train"
    )
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task, NormalizedScenarioAction(0, (0.0,) * 4, AdversarialOption.GAP_CLOSE), episode_seed=999
    )
    try:
        assert episode.map_tokens.map_hash == task.geometry_hash
        assert episode.episode_seed == 999
        assert episode.sut_adapter.metadata(episode.sut_profile)["profile_is_model_input"] is False
    finally:
        episode.env.close()


def test_runtime_hash_mismatch_fails_fast() -> None:
    task = load_taskbook("mvr/configs/taskbook.json")[0]
    broken = type(task)(**{**task.to_dict(), "geometry_hash": "0" * 64})
    with pytest.raises(RuntimeError, match="map hash mismatch"):
        ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
            broken, NormalizedScenarioAction(0, (0.0,) * 4, AdversarialOption.GAP_CLOSE)
        )
