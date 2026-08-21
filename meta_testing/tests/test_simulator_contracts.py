from __future__ import annotations

import pytest

from meta_testing.scenario.adapters.cutin import CutInScenarioAdapter
from meta_testing.scenario.adapters.merge import MergeScenarioAdapter
from meta_testing.scenario.adapters.roundabout import RoundaboutScenarioAdapter
from meta_testing.scenario.task_spec import MetaTestTaskSpec


@pytest.mark.parametrize(
    ("family", "adapter"),
    (("merge", MergeScenarioAdapter()), ("cutin", CutInScenarioAdapter()), ("roundabout", RoundaboutScenarioAdapter())),
)
def test_headless_family_reset_and_outer_config_recording(family, adapter) -> None:
    task = MetaTestTaskSpec(f"{family}-test", "meta_train", "idm_cautious", family, f"{family}-map", "d" * 64, "template", f"{family}_v1", 1)
    config = {"route_or_conflict_candidate": "candidate", "option": "gap_close", "adversary_spawn_m": 8.0}
    env = adapter.build_env(task, config)
    try:
        _, info = adapter.reset(env, task, config, task.seed)
        adapter.validate_runtime(env, task, config)
        assert info["meta_testing_config"] == config
    finally:
        env.close()
