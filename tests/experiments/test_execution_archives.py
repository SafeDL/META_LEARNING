import json
from dataclasses import replace

import pytest

from experiments.executions import collect_bank
from experiments.protocol import DEFAULT
from experiments.scenarios import sample


def test_real_collision_bank_resumes_without_free_or_duplicate_physical_calls(tmp_path):
    scenes = sample("front_brake", 2, "development", replicate=17)
    destination = tmp_path / "physical"
    bank = collect_bank("history_idm_mobil", scenes, destination, workers=2)
    assert bank.manifest["attempted_calls"] == 2
    assert bank.manifest["successful_calls"] == 2
    assert bank.manifest["contract"]["episode_duration_s"] == DEFAULT.episode_duration_s
    for scene in scenes:
        execution = bank(scene)
        assert execution.feedback.scenario_id == scene.scenario_id
        assert execution.feedback.execution_s > 0
        assert (
            bool(any(row["collision"] for row in execution.trajectory))
            == execution.feedback.collision
        )
        assert execution.trajectory[-1]["time_s"] <= DEFAULT.episode_duration_s
    resumed = collect_bank("history_idm_mobil", scenes, destination, workers=2)
    assert resumed.manifest["attempted_calls"] == 2
    assert len(list((destination / "attempts").glob("*.json"))) == 2
    assert (
        sum(
            json.loads(line)["status"] == "scheduled"
            for line in (destination / "calls.jsonl").read_text().splitlines()
        )
        == 2
    )
    changed = [replace(scenes[0], parameters={**scenes[0].parameters, "gap_m": 65}), scenes[1]]
    with pytest.raises(ValueError, match="execution contract"):
        collect_bank("history_idm_mobil", changed, destination, workers=2)


def test_reused_coordinates_are_actual_historical_coordinates():
    history = sample("cut_in", 32, "history")
    reused = sample("cut_in", 32, "reuse")
    assert [s.coordinate() for s in history] == [s.coordinate() for s in reused]
    assert [s.scenario_id for s in history] != [s.scenario_id for s in reused]
    assert not {s.coordinate() for s in reused} & {
        s.coordinate() for s in sample("cut_in", 32, "new")
    }
