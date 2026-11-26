"""Paid responses survive an interruption without duplicate queries or label invention."""

from dataclasses import replace

import pytest

from experiments import query
from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.resume import ReplayInterrupted, ResumableLedger
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.storage import read_json, write_json


def replay_selector(candidates, oracle, *, budget):
    queries = []
    for index in range(budget):
        feedback = oracle.query(index, purpose="initial" if index == 0 else "target")
        queries.append({"index": index, "feedback": feedback.to_dict()})
    return {"selected_indices": list(range(budget)), "queries": queries}


@pytest.fixture
def task(tmp_path, monkeypatch):
    scenes = sample("car_following", 6, "new", replicate=954)
    physical, calls = {}, []
    monkeypatch.setattr(query, "trajectory_margin", lambda _: {"minimum_margin": 0.5})

    def execute(scene):
        index = scenes.index(scene)
        if index not in physical:
            calls.append(index)
            physical[index] = PaidExecution(
                QueryFeedback(scene.scenario_id, index == 2, SafetyMeasurements(1, 1), 0),
                [{"time_s": 0.0, "collision": index == 2, "fixture_index": index}],
            )
        return physical[index]

    def ledger(*, spec=None, candidates=None, stop_after=None, execute_fn=execute):
        return ResumableLedger(
            scenes if candidates is None else candidates,
            5,
            execute_fn,
            checkpoint_path=tmp_path / "paid.json.gz",
            specification={"model": "original"} if spec is None else spec,
            paid_execution=lambda index: physical[index],
            joint_feedback=True,
            stop_after=stop_after,
        )

    return scenes, physical, calls, ledger, tmp_path / "paid.json.gz"


def test_isolated_selector_recovery_has_the_same_feedback_and_only_new_calls_pay(task):
    scenes, physical, calls, make, path = task
    first = make(stop_after=2)
    with pytest.raises(ReplayInterrupted):
        run_isolated(replay_selector, {"candidates": scenes, "budget": 5}, first)
    assert calls == [0, 1] and len(first.records) == 2
    assert read_json(path)["status"] == "interrupted"
    resumed = make()
    result = run_isolated(replay_selector, {"candidates": scenes, "budget": 5}, resumed)
    resumed.finish()
    assert calls == [0, 1, 2, 3, 4] and len(resumed.records) == 5
    assert result["selected_indices"] == list(range(5))
    assert [r["feedback"] for r in result["queries"]] == [r["feedback"] for r in resumed.records]
    assert len(physical) == 5 and read_json(path)["status"] == "complete"
    assert read_json(path)["successful_paid_responses"] == 5


@pytest.mark.parametrize("change", ["model", "coordinate", "physical"])
def test_recovery_rejects_changed_task_population_or_paid_trajectory(task, change):
    scenes, physical, calls, make, _ = task
    first = make(stop_after=1)
    with pytest.raises(ReplayInterrupted):
        first.query(0, purpose="initial")
    if change == "model":
        with pytest.raises(ValueError, match="different task"):
            make(spec={"model": "changed"})
    elif change == "coordinate":
        changed = [
            replace(
                scenes[0],
                parameters={**scenes[0].parameters, "gap_m": scenes[0].parameters["gap_m"] + 0.1},
            ),
            *scenes[1:],
        ]
        with pytest.raises(ValueError, match="different task"):
            make(candidates=changed)
    else:
        physical[0].trajectory.append({"changed": True})
        with pytest.raises(ValueError, match="physical execution changed"):
            make()
    assert calls == [0]


def test_recovery_refuses_a_changed_query_order_before_disclosing_a_response(task):
    _, _, calls, make, _ = task
    first = make(stop_after=1)
    with pytest.raises(ReplayInterrupted):
        first.query(0, purpose="initial")
    restored = make()
    with pytest.raises(ValueError, match="different paid feedback sequence"):
        restored.query(1, purpose="initial")
    assert restored.replayed_count == 0 and calls == [0]


def test_failed_physical_calls_are_retained_and_cannot_be_filled_with_safe_labels(task):
    _, _, _, make, path = task

    def failure(_):
        raise RuntimeError("physical execution failed")

    ledger = make(execute_fn=failure)
    with pytest.raises(RuntimeError, match="physical execution failed"):
        ledger.query(0, purpose="initial")
    saved = read_json(path)
    assert saved["status"] == "failed" and saved["query_ledger"][0]["status"] == "failed"
    assert saved["successful_paid_responses"] == 0
    with pytest.raises(ValueError, match="Only completed paid responses"):
        make()


def test_saved_rss_values_cannot_change_without_invalidating_recovery(task):
    _, _, calls, make, path = task
    first = make(stop_after=1)
    with pytest.raises(ReplayInterrupted):
        first.query(0, purpose="initial")
    saved = read_json(path)
    saved["query_ledger"][0]["feedback"]["safety"]["rss_margin"] = -0.9
    write_json(path, saved)
    with pytest.raises(ValueError, match="integrity check"):
        make()
    assert calls == [0]


def _dynamic_selector(oracle, *, initial, proposed):
    queries = []
    for position, scene in enumerate([*initial, *proposed]):
        index = position if position < len(initial) else oracle.register(scene)
        feedback = oracle.query(index, purpose="initial" if position < len(initial) else "target")
        queries.append({"index": index, "feedback": feedback.to_dict()})
    return {
        "selected_indices": [q["index"] for q in queries],
        "queries": queries,
        "actual_queries": len(queries),
    }


def test_continuous_registration_discloses_no_label_and_only_executions_use_budget():
    scenes = sample("front_brake", 4, "new", replicate=27)

    def execute(scene):
        return PaidExecution(
            QueryFeedback(scene.scenario_id, False, SafetyMeasurements(10, 5), 1), []
        )

    ledger = QueryLedger(scenes[:2], 4, execute)
    result = run_isolated(_dynamic_selector, {"initial": scenes[:2], "proposed": scenes[2:]}, ledger)
    assert result["actual_queries"] == 4
    assert len(ledger.candidates) == 4
    assert len(ledger.records) == 4
    assert all(r["status"] == "complete" for r in ledger.records)
    assert [r["index"] for r in ledger.records] == [0, 1, 2, 3]
