import copy
import hashlib
import json
import pickle
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import independent_validation as workflow
from experiments import independent_validation_audit as auditing
from experiments.collision_metrics import collision_recall, summarize
from experiments.discovery_config import write_json
from experiments.independent_models import available_jobs
from experiments.independent_protocol import (
    CONTROLS,
    METHODS,
    REUSED_METHODS,
    canonical_task,
    jobs,
    reuse_jobs,
    tasks,
)
from experiments.independent_validation_audit import (
    check_attempts,
    check_record,
    check_reuse_references,
)
from experiments.initialization import initial_indices
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.common.gp_classifier import ClassificationKernel


def test_reduced_matrix_keeps_all_main_comparisons_and_only_required_controls():
    plan = jobs()
    assert len(tasks()) == 400 and len(plan) == 1880
    full = Counter(method for task, method in plan if task["condition"] == "all_sources")
    strict = Counter(method for task, method in plan if task["condition"] == "strict_leave_family")
    assert all(full[m] == 200 for m in METHODS)
    assert strict == {"gcrt": 200, "scenario_fuzz": 200}
    assert all(full[m] == 40 for m in CONTROLS)
    assert "H0F0" not in full
    assert all(t["condition"] == "all_sources" for t, m in plan if m in CONTROLS)
    assert all(t["scenario"] == "cut_in_interaction" for t, m in plan if m in CONTROLS)
    assert len({(tuple(task.items()), method) for task, method in plan}) == 1880
    assert len(workflow.first_jobs()) == 18
    assert len(workflow.selected_reuse_jobs(True)) == 3
    assert {t["scenario"] for t, _ in workflow.first_jobs()} == {
        "cut_in_interaction",
        "front_brake",
    }
    assert len({t["target_id"] for t in tasks()}) == 8
    assert all(t["target_id"].startswith("independent_") for t in tasks())
    full_keys = {
        (tuple(task.items()), method) for task, method in plan if task["condition"] == "all_sources"
    }
    references = reuse_jobs()
    assert len(references) == 600
    assert all(
        method in REUSED_METHODS
        and (tuple(canonical_task(task, method).items()), method) in full_keys
        for task, method in references
    )
    models = available_jobs("classifiers")
    assert Counter(owner for owner, _, _ in models) == {"scenario_fuzz": 25, "indago": 5}
    assert all(held is None for owner, _, held in models if owner == "indago")


def test_main_public_inputs_exclude_target_identity_and_preserve_aggregate_information(monkeypatch):
    scenes = sample("cut_in_interaction", 512, "new")
    task = {
        **tasks()[0],
        "target_id": "private_target_identity_marker",
        "scenario": "cut_in_interaction",
    }
    history = [SimpleNamespace(source_id=name) for name in ("history_a", "history_b")]
    base = ClassificationKernel(9, deep=True).double()
    sources = {
        "names": ["history_a", "history_b"],
        "weights": np.array([0.5, 0.5]),
        "means": np.stack((np.zeros(512), np.ones(512))),
        "covariances": np.stack((0.5 * np.eye(512), 0.7 * np.eye(512))),
    }
    monkeypatch.setattr(workflow, "load_model", lambda _: (base, {}))
    monkeypatch.setattr(workflow, "load_sources", lambda *_: sources)
    _, explicit = workflow.public_task(task, "gcrt", scenes, history)
    _, aggregate = workflow.public_task(task, "H0F1", scenes, history)
    assert explicit["mode"] == "H1F1" and aggregate["mode"] == "H0F1"
    assert set(explicit) == {
        "candidates",
        "initial_indices",
        "budget",
        "pretrained",
        "sources",
        "mode",
        "device",
    }
    assert b"private_target_identity_marker" not in pickle.dumps(explicit)
    assert aggregate["sources"]["names"] == ["aggregate_history_teacher"]
    np.testing.assert_allclose(aggregate["sources"]["means"], 0.5)
    np.testing.assert_allclose(aggregate["sources"]["covariances"][0], 0.6 * np.eye(512) + 0.25)


@pytest.mark.parametrize("method", ("random", "av_bo_rf_ei", "samota"))
def test_source_agnostic_selectors_receive_identical_public_inputs_in_both_conditions(method):
    scenes = sample("cut_in", 512, "new")
    first = tasks()[0]
    second = {**first, "condition": "strict_leave_family"}
    a, public_a = workflow.public_task(first, method, scenes, [])
    b, public_b = workflow.public_task(
        second, method, scenes, [SimpleNamespace(secret="not_public")]
    )
    assert a is b and public_a == public_b
    assert "history" not in public_a and "target_id" not in public_a


@pytest.fixture
def canonical_run(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    scenes = sample("cut_in", 512, "new")
    task = {
        "target_id": "unit_fixture",
        "family": "rule",
        "lineage": "fixture",
        "scenario": "cut_in",
        "condition": "all_sources",
        "environment_seed": 81001,
        "algorithm_seed": 11,
    }
    feedbacks = [
        QueryFeedback(
            s.scenario_id, False, SafetyMeasurements(1, 10, ("road_departure",)), 1
        ).to_dict()
        for s in scenes
    ]
    private = {"candidate_ids": [s.scenario_id for s in scenes], "feedback": feedbacks}
    path = workflow.bank_audit_path(task)
    write_json(path, private)
    selected = initial_indices(512, 128, 11)
    queries = [
        {"query": n, "index": i, "feedback": feedbacks[i], "selection_s": 0, "adaptation_s": 0}
        for n, i in enumerate(selected, 1)
    ]
    records = [
        {
            "call": n,
            "index": i,
            "scenario_id": scenes[i].scenario_id,
            "account": "target",
            "status": "complete",
            "purpose": "initial" if n <= 8 else "target",
            "feedback": feedbacks[i],
        }
        for n, i in enumerate(selected, 1)
    ]
    metrics = {
        **summarize(scenes, queries, [[] for _ in selected]),
        **collision_recall(scenes, [False] * 512, queries),
    }
    saved = {
        "specification": {
            "task": task,
            "method": "random",
            "protocol_sha256": "fixture",
            "bank_contract": {"fixture": True},
            "candidate_ids": private["candidate_ids"],
            "bank_audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "initial_indices": selected[:8],
        "result": {
            "queries": queries,
            "selected_indices": selected,
            "actual_queries": 128,
            "maximum_queries": 128,
            "termination": "budget_exhausted",
        },
        "query_ledger": records,
        "metrics": metrics,
    }
    return (
        json.loads(json.dumps(saved)),
        task,
        SimpleNamespace(candidates=scenes, manifest={"contract": {"fixture": True}}),
        [SimpleNamespace(trajectory=[]) for _ in scenes],
        private,
    )


def test_final_audit_restores_serialized_feedback_and_retains_zero_collision_tasks(canonical_run):
    saved, task, bank, executions, private = canonical_run
    checked = check_record(saved, task, "random", bank, executions, private, "fixture")
    assert checked == {"queries": 128, "independent_main_choices": 0}
    assert (
        saved["metrics"]["unique_collisions"] == 0 and saved["metrics"]["collision_recall"] is None
    )


@pytest.mark.parametrize(
    "corruption", (None, "source_sha256", "public_inputs_sha256", "new_queries", "source_task")
)
@pytest.mark.parametrize("historical_paths", (False, True))
def test_reuse_references_require_an_audited_source_and_identical_inputs(
    canonical_run, monkeypatch, tmp_path, corruption, historical_paths
):
    saved, task, _, _, _ = canonical_run
    path = tmp_path / "experiment_outputs/source.json"
    write_json(path, saved)
    original_bytes = path.read_bytes()
    for module in (workflow, auditing):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "run_path", lambda *_: path)
    strict = {**task, "condition": "strict_leave_family"}
    reference = workflow.reference_record(strict, "random")
    source_path = "experiment_outputs/source.json"
    if historical_paths:
        source_path = "results/source.json"
        reference["source_path"] = source_path
        reference["selector"] = reference["selector"].replace("testing_methods.", "methods.", 1)
    assert path.read_bytes() == original_bytes
    index = {
        "status": "complete",
        "references": [reference],
        "new_queries": 0,
        "new_physical_calls": 0,
    }
    if corruption == "source_task":
        reference[corruption] = strict
    elif corruption == "new_queries":
        reference[corruption] = 128
    elif corruption:
        reference[corruption] = "wrong"
    if corruption:
        with pytest.raises(ValueError):
            check_reuse_references(index, [(strict, "random")], {source_path}, "fixture")
    else:
        assert check_reuse_references(index, [(strict, "random")], {source_path}, "fixture") == 1
        with pytest.raises(ValueError):
            check_reuse_references(index, [(strict, "random")], set(), "fixture")
        with pytest.raises(ValueError, match="source-agnostic"):
            workflow.reference_record(strict, "gcrt")


@pytest.mark.parametrize("corruption", ("feedback", "identity", "budget"))
def test_final_audit_rejects_wrong_paid_feedback_identity_or_budget(canonical_run, corruption):
    original, task, bank, executions, private = canonical_run
    saved = copy.deepcopy(original)
    if corruption == "feedback":
        saved["result"]["queries"][10]["feedback"]["collision"] = True
    elif corruption == "identity":
        saved["query_ledger"][10]["scenario_id"] = "another-coordinate"
    else:
        saved["result"]["queries"].pop()
    with pytest.raises(ValueError):
        check_record(saved, task, "random", bank, executions, private, "fixture")


def test_physical_cost_audit_rejects_a_hidden_failed_attempt(tmp_path):
    (tmp_path / "attempts").mkdir()
    scenes = sample("cut_in", 512, "new")
    identity = {
        "sut_id": "fixture",
        "environment_seed": 81001,
        "scenario_id": scenes[0].scenario_id,
    }
    write_json(
        tmp_path / "attempts/000001.json",
        {**identity, "call": 1, "status": "complete", "execution_s": 0.5, "wall_s": 0.7},
    )
    (tmp_path / "calls.jsonl").write_text(
        json.dumps(
            {"call": 1, "index": 0, "scenario_id": scenes[0].scenario_id, "status": "scheduled"}
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "sut_id": "fixture",
        "environment_seed": 81001,
        "attempted_calls": 1,
        "successful_calls": 1,
        "failed_or_interrupted_calls": 0,
        "interrupted_call_duration_unknown": 0,
        "physical_execution_s": 0.5,
        "total_successful_call_wall_s": 0.7,
        "sut_initialization_s": 0.2,
        "failed_call_wall_s": 0,
    }
    bank = SimpleNamespace(root=tmp_path, candidates=scenes, manifest=manifest)
    assert len(check_attempts(bank)) == 1
    write_json(
        tmp_path / "attempts/000002.json",
        {**identity, "call": 2, "status": "failed", "wall_s": 0.1},
    )
    with pytest.raises(ValueError, match="hides"):
        check_attempts(bank)
