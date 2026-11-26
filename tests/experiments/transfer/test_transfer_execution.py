"""Approved execution must preserve method outputs and paid-data boundaries."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from controllers.highway.catalog import FORMAL_TARGET_SUTS
from experiments import collect_transfer_records as reader
from experiments import transfer_batches as batches
from experiments import transfer_execution as module
from experiments.collect_transfer_records import ApprovedCase
from experiments.history import model_version
from experiments.k_shot import split_candidates
from experiments.resume import ReplayInterrupted
from experiments.storage import read_json, write_json
from testing_methods.common.response import archive_fingerprints

LIMITS = {"wall_s": 120, "child_memory_bytes": 8 * 1024**3}


@pytest.fixture
def admitted(native_inputs, monkeypatch):
    monkeypatch.setattr(module, "DEVELOPMENT_IDS", ("executor_fixture",))
    monkeypatch.setattr(reader, "DEVELOPMENT_IDS", ("executor_fixture",))
    monkeypatch.setattr(reader, "ExecutionBank", lambda _: native_inputs[0])
    return native_inputs


@pytest.mark.parametrize("method", ["target_only", "pooled_history", "full"])
def test_prediction_enters_independent_reader_and_reuses_paid_receipts(
    tmp_path, admitted, method
):
    bank, history, models, sources, pooled, settings, _ = admitted
    used_models = (None, None) if method == "target_only" else models
    used_pooled = pooled if method == "pooled_history" else None
    used_sources = sources if method == "full" else None
    split = split_candidates(bank.candidates, 11, budget=4)
    case = module.prediction_case(
        bank,
        split,
        11,
        method,
        used_models,
        used_pooled,
        used_sources,
        settings=settings,
        checkpoints=(0, 2, 4),
        limits=LIMITS,
    )
    saved = module.run_prediction(
        tmp_path / "current",
        case,
        bank,
        used_models,
        used_pooled,
        used_sources,
        settings=settings,
        paid_execution=bank.read_paid,
        scope="development",
    )
    report = reader.collect_cases([case], tmp_path / "current", tmp_path, scope="development")
    assert report["status"] == "complete" and len(report["rows"]) == 3
    assert len(list((tmp_path / "current/approvals").rglob("*.json"))) == 1
    assert (
        module.run_prediction(
            tmp_path / "current",
            case,
            bank,
            used_models,
            used_pooled,
            used_sources,
            settings=settings,
            paid_execution=bank.read_paid,
            scope="development",
        )
        == saved
    )


@pytest.mark.parametrize(
    "method", ["target_only", "full", "random", "scenario_fuzz"]
)
def test_budget_preserves_actual_paid_selection_and_enters_independent_reader(
    tmp_path, admitted, method
):
    bank, history, models, sources, _, settings, classifier = admitted
    used_models = models if method == "full" else (None, None)
    used_sources = sources if method == "full" else None
    classifier_input = (
        None
        if method != "scenario_fuzz"
        else {
            "model": classifier,
            "construction": {
                "model_version": model_version(classifier),
                "history_fingerprints": archive_fingerprints(history),
            },
        }
    )
    case = module.budget_case(
        bank,
        11,
        method,
        used_models,
        used_sources,
        settings=settings,
        initial_count=3,
        budget=4,
        limits=LIMITS,
        classifier=classifier_input,
    )
    saved = module.run_budget(
        tmp_path / "current",
        case,
        bank,
        used_models,
        used_sources,
        settings=settings,
        paid_execution=bank.read_paid,
        classifier=classifier_input,
        scope="development",
    )
    assert saved["result"]["selected_indices"] == [q["index"] for q in saved["result"]["queries"]]
    assert len(saved["result"]["queries"]) == 4
    report = reader.collect_cases([case], tmp_path / "current", tmp_path, scope="development")
    assert report["status"] == "complete" and len(report["rows"]) == 1


def test_paid_only_prediction_and_budget_never_open_unqueried_truth(tmp_path, admitted):
    bank, _, _, _, _, settings, _ = admitted
    split = split_candidates(bank.candidates, 11, budget=4)
    case = module.prediction_case(
        bank,
        split,
        11,
        "target_only",
        (None, None),
        None,
        None,
        settings=settings,
        checkpoints=(0, 2, 4),
        limits=LIMITS,
        evaluate=False,
    )
    bank.touched = []
    saved = module.run_prediction(
        tmp_path,
        case,
        bank,
        (None, None),
        None,
        None,
        settings=settings,
        paid_execution=bank.read_paid,
        scope="development",
    )
    assert bank.touched == [bank.candidates[i].scenario_id for i in split["query_order"]]
    assert saved["metrics"] == [] and saved["new_physical_calls"] is None
    assert len(saved["result"]["prediction_snapshots"][-1]["probabilities"]) == 8
    case = module.budget_case(
        bank,
        11,
        "random",
        (None, None),
        None,
        settings=settings,
        initial_count=3,
        budget=4,
        limits=LIMITS,
        evaluate=False,
    )
    bank.touched = []
    saved = module.run_budget(
        tmp_path,
        case,
        bank,
        (None, None),
        None,
        settings=settings,
        paid_execution=bank.read_paid,
        scope="development",
    )
    assert len(bank.touched) == 4 and len(set(bank.touched)) == 4
    assert saved["metrics"] is None and "candidate_collisions" not in saved["evaluation"]


def test_recovery_reuses_paid_responses_and_rejects_setting_substitution(tmp_path, admitted):
    bank, _, _, _, _, settings, _ = admitted
    case = module.budget_case(
        bank,
        11,
        "target_only",
        (None, None),
        None,
        settings=settings,
        initial_count=3,
        budget=4,
        limits=LIMITS,
    )
    with pytest.raises(ReplayInterrupted):
        module.run_budget(
            tmp_path,
            case,
            bank,
            (None, None),
            None,
            settings=settings,
            paid_execution=bank.read_paid,
            scope="development",
            stop_after=3,
        )
    bank.touched = []
    saved = module.run_budget(
        tmp_path,
        case,
        bank,
        (None, None),
        None,
        settings=settings,
        paid_execution=bank.read_paid,
        scope="development",
    )
    assert len(bank.touched) == 1 + len(
        bank.candidates
    )  # One new logical query, then full evaluator metrics.
    assert saved["result"]["selected_indices"][:3] == case.specification["initial_indices"]
    changed = replace(
        settings, online=replace(settings.online, raw=replace(settings.online.raw, steps=3))
    )
    with pytest.raises(ValueError, match="pre-query"):
        module.run_budget(
            tmp_path,
            case,
            bank,
            (None, None),
            None,
            settings=changed,
            paid_execution=bank.read_paid,
            scope="development",
        )
    assert (
        len(
            list(
                (tmp_path / "budget/executor_fixture/car_following/11/target_only/attempts").glob(
                    "*.json"
                )
            )
        )
        == 2
    )


def test_formal_executor_requires_a_predeclared_task_before_querying(tmp_path, admitted):
    bank, _, _, _, _, settings, _ = admitted
    bank = copy.copy(bank)
    bank.manifest = {**bank.manifest, "sut_id": FORMAL_TARGET_SUTS[0].sut_id}
    case = module.budget_case(
        bank,
        11,
        "random",
        (None, None),
        None,
        settings=settings,
        initial_count=3,
        budget=4,
        limits=LIMITS,
    )

    bank.touched = []
    with pytest.raises(ValueError, match="before querying"):
        module.run_budget(
            tmp_path,
            case,
            bank,
            (None, None),
            None,
            settings=settings,
            paid_execution=bank.read_paid,
        )
    assert not bank.touched and not (tmp_path / "approvals").exists()


def test_no_historical_target_only_inputs_or_pooled_query_arm(admitted):
    bank, _, models, sources, _, settings, _ = admitted
    with pytest.raises(ValueError, match="historical"):
        module.budget_case(
            bank,
            11,
            "target_only",
            models,
            sources,
            settings=settings,
            initial_count=3,
            budget=4,
            limits=LIMITS,
        )
    with pytest.raises(ValueError, match="Pooled"):
        module.budget_case(
            bank,
            11,
            "pooled_history",
            (None, None),
            None,
            settings=settings,
            initial_count=3,
            budget=4,
            limits=LIMITS,
        )


def test_evaluator_failure_keeps_paid_budget_and_reports_missing_metrics(
    tmp_path, admitted, monkeypatch
):
    bank, _, _, _, _, settings, _ = admitted
    case = module.budget_case(
        bank,
        11,
        "random",
        (None, None),
        None,
        settings=settings,
        initial_count=3,
        budget=4,
        limits=LIMITS,
    )

    def fail(*args, **kwargs):
        raise ValueError("Measured evaluator failure fixture")

    monkeypatch.setattr(module, "_finalize", fail)
    with pytest.raises(ValueError, match="evaluator failure"):
        module.run_budget(
            tmp_path,
            case,
            bank,
            (None, None),
            None,
            settings=settings,
            paid_execution=bank.read_paid,
            scope="development",
        )
    directory = tmp_path / "budget/executor_fixture/car_following/11/random"
    assert read_json(directory / "paid_feedback.json.gz")["status"] == "complete"
    assert len(read_json(directory / "paid_feedback.json.gz")["query_ledger"]) == 4
    assert (
        read_json(next(directory.glob("failed_attempts_*.json")))["failure_stage"] == "evaluation"
    )
    report = reader.collect_cases(
        [replace(case, final_failure=True)], tmp_path, tmp_path, scope="development"
    )
    assert report["rows"][0]["status"] == "failed"


def test_executor_limits_are_forwarded_and_failures_preserved(tmp_path, admitted, monkeypatch):
    bank, _, _, _, _, settings, _ = admitted
    case = module.budget_case(
        bank,
        11,
        "random",
        (None, None),
        None,
        settings=settings,
        initial_count=3,
        budget=4,
        limits=LIMITS,
    )

    def fail(selector, public, ledger, **limits):
        assert limits == {"wall_limit_s": 120, "memory_limit_bytes": 8 * 1024**3}
        raise TimeoutError("Resource fixture")

    monkeypatch.setattr(module, "run_isolated", fail)
    with pytest.raises(TimeoutError):
        module.run_budget(
            tmp_path,
            case,
            bank,
            (None, None),
            None,
            settings=settings,
            paid_execution=bank.read_paid,
            scope="development",
        )
    attempt = read_json(
        next((tmp_path / "budget/executor_fixture/car_following/11/random/attempts").glob("*.json"))
    )
    assert attempt["status"] == "failed" and attempt["saved_responses_after_run"] == 0


def test_frozen_weight_flag_preserves_the_existing_control(tmp_path, admitted):
    bank, _, models, sources, _, settings, _ = admitted
    split = split_candidates(bank.candidates, 11, budget=4)
    case = module.prediction_case(
        bank,
        split,
        11,
        "full",
        models,
        None,
        sources,
        settings=settings,
        checkpoints=(0, 2, 4),
        limits=LIMITS,
        update_source_weights=False,
    )
    saved = module.run_prediction(
        tmp_path / "current",
        case,
        bank,
        models,
        None,
        sources,
        settings=settings,
        paid_execution=bank.read_paid,
        scope="development",
    )
    assert saved["result"]["source_weight_updates"] is False
    assert saved["result"]["physical_mass"] == pytest.approx(0.1)
    weights = saved["result"]["prediction_snapshots"][0]["component_weights"]
    assert all(s["component_weights"] == weights for s in saved["result"]["prediction_snapshots"])


@pytest.mark.parametrize("mode", ["prediction", "budget"])
def test_public_batch_gate_runs_before_settings_or_declarations(tmp_path, monkeypatch, mode):
    from experiments import history_transfer_experiment as formal

    def reject(_):
        raise ValueError("Formal prerequisites missing")

    monkeypatch.setattr(formal, "require_method_evaluation", reject)
    monkeypatch.setattr(
        batches, "load_settings", lambda _: pytest.fail("Read settings before study gate")
    )
    monkeypatch.setattr(
        batches, "prepare_context", lambda *a: pytest.fail("Read target data before study gate")
    )
    with pytest.raises(ValueError, match="prerequisites|deferred"):
        batches.run(mode, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode,expected", [("prediction", 24), ("budget", 48)])
def test_full_size_task_matrix_forwards_the_same_selected_settings(
    admitted, monkeypatch, mode, expected
):
    _, _, models, sources, pooled, settings, classifier = admitted
    from experiments.formal_scenarios import candidate_payload
    from experiments.schema import Scenario

    candidates = [
        Scenario(**r["scenario"])
        for r in candidate_payload("car_following_braking_onset", 2000, "target")["records"]
    ]
    banks = {
        s.sut_id: SimpleNamespace(candidates=candidates, manifest={"sut_id": s.sut_id})
        for s in FORMAL_TARGET_SUTS
    }
    context = {
        "models": models,
        "sources": sources,
        "pooled": pooled,
        "classifier": {"model": classifier},
    }
    observed = []

    def prediction(bank, split, seed, method, used_models, used_pooled, used_sources, **kwargs):
        assert kwargs["settings"] is settings and kwargs["checkpoints"] == batches.CHECKPOINTS
        assert len(split["query_order"]) == 200 and len(split["evaluation_indices"]) == 1800
        observed.append((method, used_models, used_pooled, used_sources))
        return ApprovedCase(
            "prediction", bank.manifest["sut_id"], "car_following", seed, method, {}
        )

    def budget(bank, seed, method, used_models, used_sources, **kwargs):
        assert kwargs["settings"] is settings and (kwargs["budget"], kwargs["initial_count"]) == (
            200,
            20,
        )
        assert (kwargs["classifier"] is context["classifier"]) == (method == "scenario_fuzz")
        observed.append((method, used_models, None, used_sources))
        return ApprovedCase("budget", bank.manifest["sut_id"], "car_following", seed, method, {})

    monkeypatch.setattr(execution := batches.execution, "prediction_case", prediction)
    monkeypatch.setattr(execution, "budget_case", budget)
    cases = batches.make_cases(mode, banks, context, settings, LIMITS)
    assert len(cases) == expected and len({c.key for c in cases}) == expected
    assert {c.seed for c in cases} == {11}
    assert ("pooled_history" in {c.method for c in cases}) == (mode == "prediction")
    for method, used_models, used_pooled, used_sources in observed:
        assert (used_models is models) == (method in ("full", "pooled_history"))
        assert (used_sources is sources) == (method == "full")
        if method == "target_only":
            assert used_models == (None, None) and used_pooled is None and used_sources is None


def test_cohort_declarations_are_complete_and_reject_omissions_before_writing(
    tmp_path, admitted, monkeypatch
):
    settings = admitted[5]
    cases = [
        ApprovedCase(
            "prediction",
            s.sut_id,
            "car_following",
            seed,
            m,
            {
                "evaluate_complete_bank": True,
                "execution_contract": module.execution_contract(LIMITS),
            },
        )
        for s in FORMAL_TARGET_SUTS
        for seed in batches.SEEDS
        for m in batches.PREDICTION_METHODS
    ]
    monkeypatch.setattr(batches, "_formal_context", lambda *args: {"settings": settings})
    checked = []
    monkeypatch.setattr(batches, "_check_formal_specification", lambda c, _: checked.append(c.key))
    with pytest.raises(ValueError, match="omits or duplicates"):
        batches.declare_batch(tmp_path, "prediction", "car_following", cases[:-1], settings, LIMITS)
    assert not list(tmp_path.iterdir()) and not checked
    write_json(
        tmp_path / batches.REGISTRY, {"scope": "Unit certificate placeholder; not formal admission"}
    )
    directory, protocol = batches.declare_batch(
        tmp_path, "prediction", "car_following", cases, settings, LIMITS
    )
    assert len(checked) == 24 and len(protocol["cases"]) == 24
    assert len(list((tmp_path / "approvals").rglob("*.json"))) == 24
    assert read_json(directory / "protocol.json") == protocol
