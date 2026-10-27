import copy
import json

import numpy as np
import pytest

from experiments.collision_metrics import collision_recall, summarize
from experiments.discovery_config import development_tasks
from experiments.domain_comparison import METHODS, baseline_configuration, public_task
from experiments.domain_comparison_audit import check_record
from experiments.domain_comparison_report import gate, interaction, validate_rows
from experiments.initialization import initial_indices
from experiments.query import PaidExecution
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements


def rows(main=80, reference=60, *, empty=False):
    output = []
    for task in development_tasks():
        for method in METHODS:
            count = main if method == "gcrt" else reference if method == "indago" else 40
            output.append(
                {
                    **task,
                    "method": method,
                    "queries": 128,
                    "pool_unique_collisions": 0 if empty else 100,
                    "unique_collisions": 0 if empty else count,
                    "collision_recall": None if empty else count / 100,
                    "discovery_auc": 0 if empty else count / 2,
                    "selection_s": 0,
                    "adaptation_s": 0,
                }
            )
    return output


def test_frozen_baseline_configuration_survives_json_without_type_differences():
    configuration = baseline_configuration()
    assert json.loads(json.dumps(configuration)) == configuration
    assert configuration["samota"]["thresholds"] == [0.0, 0.0, 0.2]


def test_complete_domain_gate_includes_indago_and_uses_remaining_misses():
    data = rows()
    result = gate(data)
    assert result["strongest_reference"] == "indago"
    assert result["remaining_miss_reduction"] == pytest.approx(0.5)
    assert result["status"] == "passed"
    for row in data:
        if row["method"] == "indago":
            row["unique_collisions"], row["collision_recall"] = 80, 0.8
    assert gate(data)["status"] == "not_passed"


@pytest.mark.parametrize("empty", [False, True])
def test_no_headroom_and_no_collision_tasks_are_retained_without_false_admission(empty):
    data = rows(main=100, reference=100, empty=empty)
    result = gate(data)
    assert result["remaining_miss_reduction"] is None
    assert result["status"] == "not_passed"
    assert not result["feasibility"]["at_least_two_lineages_have_headroom"]
    if empty:
        assert result["main_recall"] is None


def test_matrix_validation_rejects_missing_duplicate_and_mismatched_denominator():
    data = rows()
    for corrupt in (data[:-1], data[:-1] + [data[0]]):
        with pytest.raises(ValueError, match="400-run"):
            validate_rows(corrupt)
    data[0]["pool_unique_collisions"] += 1
    with pytest.raises(ValueError, match="denominators"):
        validate_rows(data)


def test_interaction_is_reported_without_becoming_a_superadditivity_requirement():
    data = rows(main=80)
    values = {"H0F0": 40, "H1F0": 70, "H0F1": 65}
    for row in data:
        if row["method"] in values:
            row["unique_collisions"] = values[row["method"]]
    assert interaction(data, "unique_collisions")["mean"] == -15
    assert gate(data)["status"] == "passed"


def test_selector_dispatch_never_serializes_target_identity_or_exhaustive_outcomes(monkeypatch):
    import experiments.domain_comparison as domain

    monkeypatch.setattr(domain, "load_target", lambda path: ("kernel", {}))
    monkeypatch.setattr(
        domain,
        "load_sources",
        lambda kind: {
            "names": ["history"],
            "weights": np.ones(1),
            "means": np.zeros((1, 8)),
            "covariances": np.eye(8)[None],
        },
    )
    monkeypatch.setattr(
        domain, "build_proxy", lambda candidates, history: (np.ones(len(candidates)), {})
    )
    for module in (domain.indago_model, domain.graph_model):
        monkeypatch.setattr(
            module, "load", lambda path: ("fixed classifier", {"history_fingerprints": {}})
        )
    task = development_tasks()[0]
    scenes = sample("cut_in", 8, "development")
    for method in METHODS:
        _, public = public_task(task, method, scenes, [])
        assert not set(public) & {
            "target_id",
            "lineage",
            "family",
            "collision",
            "safety",
            "trajectory",
            "denominator",
        }
        assert set(public) <= {
            "candidates",
            "initial_indices",
            "budget",
            "seed",
            "pretrained",
            "sources",
            "mode",
            "device",
            "method",
            "proxy",
            "classifier",
        }


def random_audit_fixture():
    task = development_tasks()[0]
    candidates = sample("cut_in", 512, "development")
    initial = initial_indices(512, 8, task["algorithm_seed"])
    selected = initial_indices(512, 128, task["algorithm_seed"])
    executions = [
        PaidExecution(QueryFeedback(s.scenario_id, False, SafetyMeasurements(10, 10), 1), [])
        for s in candidates
    ]
    queries = [
        {"query": q, "index": index, "feedback": executions[index].feedback.to_dict()}
        for q, index in enumerate(selected, 1)
    ]
    records = [
        {
            "call": q,
            "index": index,
            "feedback": executions[index].feedback.to_dict(),
            "status": "complete",
            "account": "target",
            "scenario_id": candidates[index].scenario_id,
            "purpose": "initial" if q <= 8 else "target",
        }
        for q, index in enumerate(selected, 1)
    ]
    metrics = {
        **summarize(candidates, queries, [[] for _ in selected]),
        **collision_recall(candidates, [False] * 512, queries),
    }
    saved = {
        "specification": {
            "task": task,
            "method": "random",
            "protocol_sha256": "test",
            "candidate_ids": [s.scenario_id for s in candidates],
        },
        "initial_indices": initial,
        "query_ledger": records,
        "metrics": metrics,
        "result": {
            "queries": queries,
            "selected_indices": selected,
            "actual_queries": 128,
            "maximum_queries": 128,
            "termination": "budget_exhausted",
        },
    }
    return json.loads(json.dumps(saved)), task, candidates, executions


def test_raw_execution_audit_rejects_fabricated_feedback_metrics_and_duplicate_queries():
    saved, task, candidates, executions = random_audit_fixture()

    def check(value):
        return check_record(value, task, "random", candidates, executions, [False] * 512, "test")

    assert check(saved)["queries"] == 128
    feedback = copy.deepcopy(saved)
    feedback["result"]["queries"][0]["feedback"]["execution_s"] = 2
    metrics = copy.deepcopy(saved)
    metrics["metrics"]["collision_recall"] = 1
    duplicate = copy.deepcopy(saved)
    duplicate["result"]["queries"][1]["index"] = duplicate["result"]["queries"][0]["index"]
    for corrupt in (feedback, metrics, duplicate):
        with pytest.raises(ValueError):
            check(corrupt)


def test_saved_json_audit_preserves_nested_collision_behavior_and_feedback():
    saved, task, candidates, executions = random_audit_fixture()
    index = saved["result"]["selected_indices"][0]
    feedback = QueryFeedback(candidates[index].scenario_id, True, SafetyMeasurements(0, 0), 1)
    actors = [
        {"actor_id": name, "x": x, "y": 0, "heading": 0, "speed": speed, "length": 5, "width": 2}
        for name, x, speed in (("ego", 0, 10), ("lead", 3, 0))
    ]
    trajectory = [
        {
            "time_s": 3,
            "actors": actors,
            "collision": True,
            "collision_partner": "lead",
            "control": {"acceleration": 0, "steering": 0},
        }
    ]
    executions[index] = PaidExecution(feedback, trajectory)
    saved["result"]["queries"][0]["feedback"] = feedback.to_dict()
    saved["query_ledger"][0]["feedback"] = feedback.to_dict()
    labels = [i == index for i in range(512)]
    queries = saved["result"]["queries"]
    saved["metrics"] = {
        **summarize(
            candidates,
            queries,
            [executions[i].trajectory for i in saved["result"]["selected_indices"]],
        ),
        **collision_recall(candidates, labels, queries),
    }
    restored = json.loads(json.dumps(saved))
    assert (
        check_record(restored, task, "random", candidates, executions, labels, "test")["queries"]
        == 128
    )
