"""Chronological execution on native unit banks, including real small model updates."""

import copy
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments import transfer_execution as execution
from experiments.accumulation import sequence as module
from experiments.storage import read_json, write_json
from testing_methods.history_transfer.training import Config

LIMITS = {"wall_s": 120, "child_memory_bytes": 8 * 1024**3}
TRAINING = Config(steps=2, initial_count=2, budget=4, historical_context=4, diagnostic_interval=2)


@pytest.fixture
def context(native_inputs, monkeypatch):
    native, history, models, _, _, settings, _ = native_inputs
    banks, suts = {}, {}
    for i in range(3):
        name = f"sequence_unit_{i}"
        bank = copy.copy(native)
        bank.manifest = {**native.manifest, "sut_id": name}
        bank.touched = []
        banks[name] = bank
        suts[name] = SimpleNamespace(
            sut_id=name, version="unit-v1", family="native_unit", lineage=f"unit_{i}", role="target"
        )
    monkeypatch.setattr(execution, "DEVELOPMENT_IDS", tuple(banks))
    return {
        "banks": banks,
        "suts": suts,
        "models": models,
        "history": history,
        "view_record_ids": {a.source_id: a.provenance["record_ids"] for a in history},
        "settings": settings,
    }


def sequence(context, regime):
    order = list(context["banks"])
    tasks = []
    for i, sut in enumerate(order):
        past = [] if regime == "fixed_history" else order[:i]
        tasks.append(
            {
                "ordinal": i + 1,
                "sut_id": sut,
                "available_past_target_ids": past,
                "source_count": 8 + len(past),
                "initialization_seed": 11,
                "initial_queries": 2,
                "total_budget": 4,
                "update_after_task": regime == "update_shared" and i == 1,
            }
        )
    return {
        "scenario": "car_following",
        "seed": 11,
        "regime": regime,
        "branch_id": f"unit:11:{regime}",
        "tasks": tasks,
    }


def run(root, context, regime, **kwargs):
    return module.execute_sequence(
        root,
        sequence(context, regime),
        context,
        settings=context["settings"],
        limits=LIMITS,
        scope="development",
        training_config=TRAINING,
        **kwargs,
    )


def test_append_only_uses_own_paid_past_and_preserves_initial_models(context, tmp_path):
    result = run(tmp_path, context, "append_sources")
    assert result["status"] == "complete"
    assert [r["actual_source_count"] for r in result["completed"]] == [8, 9, 10]
    assert result["updates"] == []
    assert len({tuple(r["model_versions"]) for r in result["completed"]}) == 1
    directory = module.directory(tmp_path, sequence(context, "append_sources"))
    for row in result["completed"]:
        archive = read_json(directory / row["archive"])
        assert len(archive["executions"]) == 4
        assert {r["query"] for r in archive["executions"]} == {1, 2, 3, 4}
        assert len({r["scenario"]["scenario_id"] for r in archive["executions"]}) == 4
    # Recovery reuses complete tasks, not extra logical queries or training.
    attempt_files = list(directory.glob("budget/*/*/*/*/attempts/*.json"))
    again = run(tmp_path, context, "append_sources")
    assert again == result
    assert list(directory.glob("budget/*/*/*/*/attempts/*.json")) == attempt_files


def test_fixed_history_archives_but_never_admits_target_feedback(context, tmp_path):
    result = run(tmp_path, context, "fixed_history")
    assert [r["actual_source_count"] for r in result["completed"]] == [8, 8, 8]
    assert all(r["source_ids"] == result["completed"][0]["source_ids"] for r in result["completed"])
    assert result["updates"] == []


def test_real_between_task_update_and_recovery_preserve_the_same_published_model(context, tmp_path):
    with pytest.raises(module.SequenceInterrupted):
        run(tmp_path, context, "update_shared", stop_after_tasks=2)
    directory = module.directory(tmp_path, sequence(context, "update_shared"))
    partial = read_json(directory / "progress.json")
    assert len(partial["completed"]) == 2 and len(partial["updates"]) == 1
    publication = partial["updates"][0]["publication"]
    assert publication["accepted"] is True
    assert len(publication["specification"]["new_source_ids"]) == 2
    update = directory / "models/after_02_tasks"
    before = {p.name: module._sha(p) for p in update.glob("*.pt")}
    result = run(tmp_path, context, "update_shared")
    assert result["status"] == "complete" and len(result["completed"]) == 3
    assert result["completed"][2]["model_versions"] == partial["updates"][0]["model_versions"]
    assert result["completed"][2]["model_versions"] != result["completed"][0]["model_versions"]
    assert {p.name: module._sha(p) for p in update.glob("*.pt")} == before
    actual = publication["specification"]["history_fingerprints"]
    assert len(actual) == 10
    assert not any("sequence_unit_2@" in name for name in actual)


def test_failed_model_update_keeps_the_prior_model_and_available_paid_sources(
    context, tmp_path, monkeypatch
):
    calls = []

    def failed_update(destination, history, models, **kwargs):
        calls.append((len(history), kwargs["new_source_ids"]))
        return models, {
            "accepted": False,
            "reason": "numerical_failure",
            "scope": "Explicit unit failure injection",
        }

    monkeypatch.setattr(module, "update_shared_model", failed_update)
    result = run(tmp_path, context, "update_shared")
    assert len(calls) == 1 and calls[0][0] == 10
    assert result["updates"][0]["publication"]["accepted"] is False
    assert result["completed"][2]["model_versions"] == result["completed"][0]["model_versions"]
    assert result["completed"][2]["actual_source_count"] == 10


def test_archive_rejects_foreign_branch_or_future_feedback(context, tmp_path):
    result = run(tmp_path, context, "append_sources")
    seq = sequence(context, "append_sources")
    directory = module.directory(tmp_path, seq)
    paths = [directory / r["archive"] for r in result["completed"]]
    with pytest.raises(ValueError, match="past"):
        module._past([paths[2]], seq, 2, context["suts"])
    foreign = read_json(paths[0])
    foreign["branch_id"] = "another_seed_or_regime"
    write_json(tmp_path / "foreign.json.gz", foreign)
    with pytest.raises(ValueError, match="past"):
        module._past([tmp_path / "foreign.json.gz"], seq, 2, context["suts"])


def test_interruption_within_a_target_is_replayed_before_that_target_can_become_history(
    context, tmp_path, monkeypatch
):
    actual = execution.run_budget
    interrupted = []

    def interrupt_once(*args, **kwargs):
        if not interrupted:
            interrupted.append(True)
            kwargs["stop_after"] = 2
        return actual(*args, **kwargs)

    monkeypatch.setattr(execution, "run_budget", interrupt_once)
    with pytest.raises(module.ReplayInterrupted):
        run(tmp_path, context, "append_sources")
    directory = module.directory(tmp_path, sequence(context, "append_sources"))
    assert not list((directory / "completed").glob("*.json.gz"))
    result = run(tmp_path, context, "append_sources")
    assert result["status"] == "complete"
    assert [r["successful_paid_feedback"] for r in result["completed"]] == [4, 4, 4]
    assert result["completed"][1]["actual_source_count"] == 9


def test_terminal_query_task_failure_is_not_retried_or_filled_as_safe_on_recovery(
    context, tmp_path, monkeypatch
):
    actual = execution._finalize
    failures = []

    def fail_first(case, *args, **kwargs):
        if case.sut_id == "sequence_unit_0":
            failures.append(case.sut_id)
            raise ValueError("Explicit unit terminal evaluator failure")
        return actual(case, *args, **kwargs)

    monkeypatch.setattr(execution, "_finalize", fail_first)
    first = run(tmp_path, context, "append_sources")
    assert first["completed"][0]["status"] == "failed"
    assert first["completed"][0]["successful_paid_feedback"] == 4
    assert first["completed"][1]["actual_source_count"] == 9
    again = run(tmp_path, context, "append_sources")
    assert again == first and failures == ["sequence_unit_0"]


def test_fixed_reference_reuses_exact_completed_native_results_without_reexecution(
    context, tmp_path
):
    # Only the approval is a unit dependency; results are actual native GP runs.
    result = run(tmp_path / "executed", context, "fixed_history")
    source = module.directory(tmp_path / "executed", sequence(context, "fixed_history"))
    approval = read_json(source / "approvals/budget/sequence_unit_0/car_following/11/full.json")
    case = execution.ApprovedCase(*approval["key"], specification=approval["specification"])
    main = tmp_path / "unit-main-reference"
    shutil.copytree(source, main)
    write_json(
        execution.approval_path(main, case),
        {"key": list(case.key), "specification": case.specification, "scope": "formal"},
    )
    destination = tmp_path / "reused"
    saved, failed = module._reuse_main(main, destination, case)
    assert not failed and len(saved["query_ledger"]) == 4
    for src, dst in zip(execution._paths(main, case), execution._paths(destination, case)):
        assert module._sha(src) == module._sha(dst)
    assert module._reuse_main(main, destination, case)[0] == saved
    assert result["completed"][0]["reused_main_run"] is False
    changed = replace(case, specification={**case.specification, "budget": 3})
    with pytest.raises(ValueError, match="differs"):
        module._reuse_main(main, tmp_path / "invalid-reference", changed)


def test_formal_fixed_history_requires_its_main_origin_before_any_artifact(context, tmp_path):
    with pytest.raises(ValueError, match="main root"):
        module.execute_sequence(
            tmp_path,
            sequence(context, "fixed_history"),
            context,
            settings=context["settings"],
            limits=LIMITS,
            scope="formal",
            training_config=TRAINING,
        )
    assert not list(tmp_path.iterdir())
