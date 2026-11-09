import json
import warnings
from dataclasses import asdict, replace

import pytest

from experiments.executions import collect_bank
from experiments.protocol import DEFAULT
from experiments.scenarios import sample


def test_solver_warnings_are_audited_for_success_and_failure_without_changing_exception(
    tmp_path, monkeypatch
):
    from experiments import executions

    scene = sample("car_following", 1, "history", replicate=19)[0]
    real_execute = executions.execute

    def observed_execute(policy, scenario, **options):
        warnings.simplefilter("always")
        for _ in range(2):
            warnings.warn("solver diagnostic", RuntimeWarning)
        return real_execute(policy, scenario, **{**options, "duration_s": 0.2})

    monkeypatch.setattr(executions, "execute", observed_execute)
    path = tmp_path / "success.json"
    execution = executions._execute_task("history_idm_mobil", scene, 81001, path, 1)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "complete" and execution.trajectory
    assert record["warnings"] == [
        {"category": "RuntimeWarning", "message": "solver diagnostic", "count": 2}
    ]

    def failed_execute(*args, **kwargs):
        warnings.warn("failed solver diagnostic", RuntimeWarning)
        raise ValueError("execution failed")

    monkeypatch.setattr(executions, "execute", failed_execute)
    path = tmp_path / "failure.json"
    with pytest.raises(ValueError, match="execution failed"):
        executions._execute_task("history_idm_mobil", scene, 81001, path, 2)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "failed" and "execution failed" in record["error"]
    assert record["warnings"] == [
        {"category": "RuntimeWarning", "message": "failed solver diagnostic", "count": 1}
    ]


def test_current_registration_preserves_frozen_ads_parameters_versions_and_weights():
    from controllers.highway.catalog import BY_ID, CURRENT_SUTS, HISTORY_SUTS, INDEPENDENT_SUTS
    from experiments.scenarios import TEST_SCENARIOS

    assert len(CURRENT_SUTS) == 16
    assert {s.role for s in CURRENT_SUTS} == {"history", "independent"}
    for sut in CURRENT_SUTS:
        current, original = asdict(sut), asdict(BY_ID[sut.sut_id])
        assert current.pop("supported_scenarios") == list(TEST_SCENARIOS)
        original.pop("supported_scenarios")
        assert current == original
    assert all(len(s.supported_scenarios) == 4 for s in HISTORY_SUTS)
    assert all(len(s.supported_scenarios) == 5 for s in INDEPENDENT_SUTS)


def test_current_following_bank_uses_explicit_registration_and_rejects_mixed_types(tmp_path):
    from experiments.scenarios import TEST_SCENARIOS

    scenes = sample("car_following", 2, "history", replicate=18)
    bank = collect_bank("history_idm_mobil", scenes, tmp_path, workers=2, current_registration=True)
    assert bank.manifest["contract"]["object"]["supported_scenarios"] == list(TEST_SCENARIOS)
    assert bank.manifest["successful_calls"] == 2
    resumed = collect_bank(
        "history_idm_mobil", scenes, tmp_path, workers=2, current_registration=True
    )
    assert resumed.manifest["attempted_calls"] == 2
    with pytest.raises(ValueError, match="execution contract"):
        collect_bank("history_idm_mobil", scenes, tmp_path, workers=2)
    with pytest.raises(ValueError, match="one registered functional scenario"):
        collect_bank(
            "history_idm_mobil",
            scenes + sample("front_brake", 1, "history"),
            tmp_path / "mixed",
            current_registration=True,
        )
    with pytest.raises(ValueError, match="one registered functional scenario"):
        collect_bank(
            "history_idm_mobil",
            sample("cut_in", 1, "history"),
            tmp_path / "unregistered",
            current_registration=True,
        )


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
