import copy
import json
import time
from dataclasses import asdict, replace

import pytest

from controllers.highway.control import Control
from experiments import history, storage
from experiments.executions import ExecutionBank
from experiments.query import QueryLedger, run_isolated
from experiments.scenarios import DOMAINS, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.storage import read_json, write_json
from simulator.highway.collision_env import execute


class Brake:
    def act(self, world):
        return Control(-8, 0)


def trajectory_selector(oracle, *, candidates):
    queries = []
    for index in (0, 1):
        feedback = oracle.query(index, purpose="initial" if index == 0 else "target")
        assert feedback.scenario_id == candidates[index].scenario_id
        assert feedback.trajectory
        queries.append({"index": index, "feedback": feedback.to_dict()})
    return {"queries": queries, "selected_indices": [0, 1]}


def slow_selector(oracle, *, candidates):
    time.sleep(5)
    raise AssertionError("The resource monitor should terminate this process")


def test_query_worker_stops_at_registered_wall_limit_without_executing_candidates():
    scenes = sample("car_following", 1, "new", replicate=82)
    ledger = QueryLedger(scenes, 1, lambda _: pytest.fail("No paid execution is expected"))
    with pytest.raises(TimeoutError, match="wall limit"):
        run_isolated(slow_selector, {"candidates": scenes}, ledger, wall_limit_s=0.1)
    assert not ledger.records


def test_only_paid_executions_disclose_their_actual_trajectory_across_processes():
    scenes = sample("car_following", 3, "new", replicate=83)
    calls = []

    def execute_paid(scene):
        calls.append(scene.scenario_id)
        if scene == scenes[2]:
            raise AssertionError("Unqueried execution was read")
        return execute(Brake(), scene, duration_s=0.2)

    ledger = QueryLedger(scenes, 2, execute_paid, joint_feedback=True)
    assert not calls and not ledger.trajectories
    result = run_isolated(trajectory_selector, {"candidates": scenes}, ledger)
    assert calls == [s.scenario_id for s in scenes[:2]]
    for query, trace in zip(result["queries"], ledger.trajectories):
        assert query["feedback"]["trajectory"] == trace
        assert [f["time_s"] for f in trace] == pytest.approx([0, 0.05, 0.1, 0.15, 0.2])
        assert all("control" in f and "visible_ids" in f for f in trace)


def test_feedback_trace_cannot_mutate_evaluator_evidence():
    scene = sample("car_following", 1, "new", replicate=84)[0]
    execution = execute(Brake(), scene, duration_s=0.2)
    ledger = QueryLedger([scene], 1, lambda _: execution, joint_feedback=True)
    feedback = ledger.query(0)
    restored = QueryFeedback.from_dict(json.loads(json.dumps(feedback.to_dict())))
    assert restored == feedback
    original = execution.trajectory[0]["actors"][0]["x"]
    feedback.trajectory[0]["actors"][0]["x"] = original + 1000
    assert ledger.trajectories[0][0]["actors"][0]["x"] == original
    assert ledger.records[0]["feedback"]["trajectory"][0]["actors"][0]["x"] == original


def test_old_scalar_feedback_shape_and_trace_validation():
    feedback = QueryFeedback("scene", False, SafetyMeasurements(10, 5), 0)
    assert "trajectory" not in feedback.to_dict()
    assert QueryFeedback.from_dict(feedback.to_dict()) == feedback
    with pytest.raises(ValueError, match="timestamps"):
        replace(feedback, trajectory=[{"time_s": 1, "collision": False}] * 2)
    with pytest.raises(ValueError, match="disagree"):
        replace(feedback, trajectory=[{"time_s": 0, "collision": True}])


def test_completed_task_references_only_its_paid_physical_trajectory(tmp_path, monkeypatch):
    scene = sample("car_following", 1, "new", replicate=85)[0]
    actual = execute(Brake(), scene, duration_s=0.2)
    domain = DOMAINS[scene.kind]
    bank_root = tmp_path / "physical"
    manifest = {
        "format": "collision-execution-bank",
        "status": "complete",
        "scenarios": [asdict(scene)],
        "parameter_definition": domain.to_dict(),
        "contract": {"fixture": "shared-paid-archive"},
        "environment_seed": 81001,
    }
    write_json(bank_root / "manifest.json", manifest)
    write_json(
        bank_root / "executions/000000.json.gz",
        {
            "scenario": asdict(scene),
            "feedback": actual.feedback.to_dict(),
            "trajectory": actual.trajectory,
        },
    )
    bank = ExecutionBank(bank_root)
    ledger = QueryLedger(
        [scene], 1, bank, joint_feedback=True, domain=domain, allow_registration=False
    )
    ledger.query(0, purpose="initial")
    monkeypatch.setattr(history, "ROOT", tmp_path)
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    identity = dict(
        task_id="shared-archive-test",
        sut_id="fixture_ads",
        sut_version="fixture",
        family="rule",
        lineage="fixture",
        environment_seed=81001,
        completed_order=1,
        branch_id="shared-archive-test",
        execution_contract=manifest["contract"],
        model_snapshot={"frozen": True},
    )
    destination = tmp_path / "completed.json.gz"
    history.finalize_task(destination, ledger, execution_bank=bank, **identity)
    task = read_json(destination)
    assert len(task["executions"]) == len(task["attempts"]) == 1
    record = task["executions"][0]
    assert "trajectory" not in record and "trajectory" not in record["feedback"]
    assert record["execution_ref"] == "physical/executions/000000.json.gz"
    assert storage.read_trajectory(record) == actual.trajectory
    wrong = copy.deepcopy(record)
    wrong["scenario"]["scenario_id"] = "unpaid-scene"
    with pytest.raises(ValueError, match="another scenario"):
        storage.read_trajectory(wrong)
    history.finalize_task(destination, ledger, execution_bank=bank, **identity)
    sources = history.completed_archives(
        [destination], before_order=2, branch_id="shared-archive-test"
    )
    assert len(sources) == 1 and sources[0].role == "target"
    assert sources[0].scenarios == [scene] and sources[0].domain == domain
    task["executions"][0]["execution_ref"] = "unpaid.json.gz"
    write_json(destination, task)
    with pytest.raises(ValueError, match="different records"):
        history.finalize_task(destination, ledger, execution_bank=bank, **identity)
