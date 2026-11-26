"""Registered-result export must retain real failures, missingness and paid receipts."""

import copy
import csv
import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch

from controllers.highway.control import Control
from experiments import collect_transfer_records as module
from experiments.costs import PeakMemory
from experiments.domain_risk import disclose_driving_quality, disclose_impact
from experiments.failure_metrics import summarize
from experiments.history_transfer_experiment import prediction_specification
from experiments.initialization import initial_indices
from experiments.k_shot import prediction_metrics, split_candidates
from experiments.resume import ReplayInterrupted, ResumableLedger
from experiments.scenarios import DOMAINS, sample
from experiments.storage import compact_feedback, compact_result, read_json, write_json
from simulator.highway.collision_env import execute
from testing_methods.baselines.response_baselines.collision import run as random_select
from testing_methods.history_transfer.adaptation import FitSettings, OnlineSettings
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


class Coast:
    def act(self, _):
        return Control(0.0, 0.0)


def test_formal_context_uses_registered_history_and_public_candidates(tmp_path, monkeypatch):
    from experiments import history, transfer_settings
    from experiments import history_transfer_experiment as study

    settings = transfer_settings.native_settings()
    models, archives, views, candidates = (object(), object()), object(), object(), object()
    view = {"selected_record_ids": {"history_ads": ["record"]}}
    source = {"specification": {"source": "unit fixture"}}
    pooled = object()
    calls = []
    monkeypatch.setattr(transfer_settings, "load_settings", lambda root: settings)
    monkeypatch.setattr(study, "models_and_history", lambda root, kind: (models, archives))
    monkeypatch.setattr(study, "registered_history_views", lambda root, kind, rows: (views, view))
    monkeypatch.setattr(
        study, "registered_populations", lambda root: {("target", "car_following"): candidates}
    )
    monkeypatch.setattr(study, "pooled_history", lambda rows, declaration: pooled)
    monkeypatch.setattr(history, "model_version", lambda model: str(models.index(model)))

    def build_source(model, rows, public, **kwargs):
        calls.append((model, rows, public, kwargs))
        return source

    monkeypatch.setattr(history, "build_source", build_source)
    monkeypatch.setattr(
        module,
        "read_json",
        lambda path: pytest.fail("Retired development or unused classifier data was read"),
    )
    cases = [
        module.ApprovedCase("prediction", "target_fixture", "car_following", 11, method, {})
        for method in study.PREDICTION_METHODS
    ]
    context = module._formal_context(tmp_path, cases)
    assert calls == [
        (
            models[1],
            archives,
            candidates,
            {
                "cache_directory": tmp_path / "sources/car_following",
                "view_record_ids": view["selected_record_ids"],
            },
        )
    ]
    assert context["prediction_inputs"]["car_following"] == {
        "models": models,
        "sources": source,
        "pooled": pooled,
    }
    assert context["inputs"]["car_following"]["full"]["source_snapshot"] == source["specification"]
    assert not context["classifiers"]
    assert not list(tmp_path.iterdir())


class NativeBank:
    def __init__(self, *, collision):
        self.candidates = sample("car_following", 8, "new", replicate=991)
        bounds = DOMAINS["car_following"].bounds
        initial = initial_indices(8, 2, 11)[0]
        for i, scene in enumerate(self.candidates):
            critical = collision and i == initial
            self.candidates[i] = replace(
                scene,
                parameters={
                    "ego_speed_mps": bounds["ego_speed_mps"][int(critical)],
                    "lead_speed_mps": bounds["lead_speed_mps"][int(not critical)],
                    "gap_m": bounds["gap_m"][int(not critical)],
                },
            )
        self.executions = {
            s.scenario_id: execute(Coast(), s, duration_s=0.4) for s in self.candidates
        }
        self.manifest = {
            "sut_id": "collector_fixture",
            "environment_seed": 81001,
            "contract": {"episode_duration_s": 0.4},
            "attempted_calls": 8,
            "successful_calls": 8,
            "failed_or_interrupted_calls": 0,
        }
        self.domain = None

    def __call__(self, scene):
        return self.executions[scene.scenario_id]

    @property
    def snapshot_fingerprint(self):
        return hashlib.sha256(
            json.dumps(
                [
                    {"feedback": p.feedback.to_dict(), "trajectory": p.trajectory}
                    for p in self.executions.values()
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()


@pytest.fixture(scope="module", params=[False, True])
def bank(request):
    torch.set_num_threads(1)
    return NativeBank(collision=request.param)


@pytest.fixture
def reader(bank, monkeypatch):
    monkeypatch.setattr(module, "DEVELOPMENT_IDS", ("collector_fixture",))
    monkeypatch.setattr(module, "ExecutionBank", lambda _: bank)


def prediction(bank, root, *, stop_after=None):
    split = split_candidates(bank.candidates, 11, budget=4)
    spec = prediction_specification(bank, split, 11, "target_only", None, {}, None, (0, 2, 4))
    case = module.ApprovedCase(
        "prediction", "collector_fixture", "car_following", 11, "target_only", copy.deepcopy(spec)
    )
    result_path, _, checkpoint = module._files(case, root)
    ledger = ResumableLedger(
        bank.candidates,
        4,
        bank,
        checkpoint_path=checkpoint,
        specification=spec,
        paid_execution=lambda i: bank(bank.candidates[i]),
        joint_feedback=True,
        allow_registration=False,
        stop_after=stop_after,
    )
    try:
        result = select(
            bank.candidates,
            ledger,
            query_order=split["query_order"],
            initial_indices=split["query_order"][:1],
            budget=4,
            snapshot_counts=(0, 2, 4),
            raw_kernel_type=SharedPrivateKernel,
        )
        ledger.finish()
    except ReplayInterrupted as error:
        write_json(
            result_path.with_suffix(".failure.json"),
            {
                "specification": spec,
                "query_ledger": compact_feedback(ledger.records),
                "error": repr(error),
            },
        )
        return case
    labels = [bank(s).feedback.collision for s in bank.candidates]
    metrics = []
    for snapshot in result["prediction_snapshots"]:
        values, calibration = prediction_metrics(
            snapshot["probabilities"], labels, split["candidate_ids"], split["evaluation_indices"]
        )
        metrics.append({"K": snapshot["paid_queries"], **values, "calibration": calibration})
    write_json(
        result_path,
        {
            "specification": spec,
            "result": compact_result(result),
            "metrics": metrics,
            "query_ledger": compact_feedback(ledger.records),
        },
    )
    return case


def budget(bank, root, seed=11, method="random"):
    spec = {
        "bank_contract": bank.manifest["contract"],
        "candidates": [asdict(s) for s in bank.candidates],
        "budget": 4,
        "initial_count": 2,
        "seed": seed,
        "task_snapshot": {
            "bank_snapshot": bank.snapshot_fingerprint,
            "method_contract": {"method": method},
        },
    }
    case = module.ApprovedCase(
        "budget", "collector_fixture", "car_following", seed, method, copy.deepcopy(spec)
    )
    result_path, _, checkpoint = module._files(case, root)
    ledger = ResumableLedger(
        bank.candidates,
        4,
        lambda s: disclose_driving_quality(disclose_impact(bank(s))),
        checkpoint_path=checkpoint,
        specification=spec,
        paid_execution=lambda i: bank(bank.candidates[i]),
        joint_feedback=True,
        allow_registration=False,
    )
    with PeakMemory() as memory:
        if method == "random":
            result = random_select(
                bank.candidates,
                ledger,
                initial_indices=initial_indices(8, 2, seed),
                budget=4,
                seed=seed,
            )
        else:
            assert method == "target_only"
            result = select(
                bank.candidates,
                ledger,
                initial_indices=initial_indices(8, 2, seed),
                budget=4,
                snapshot_counts=(0, 2, 4),
                raw_kernel_type=SharedPrivateKernel,
            )
    result["resources"] = memory.result
    ledger.finish()
    labels = [bank(s).feedback.collision for s in bank.candidates]
    metrics = summarize(
        bank.candidates,
        labels,
        result["queries"],
        ledger.trajectories,
        initial_count=2,
        query_ledger=ledger.records,
    )
    write_json(
        result_path,
        {
            "specification": spec,
            "result": compact_result(result),
            "metrics": metrics,
            "query_ledger": compact_feedback(ledger.records),
            "evaluation": {"candidate_collisions": labels},
        },
    )
    return case


def test_native_prediction_budget_and_csv_keep_zero_cases_and_physics_counts(
    bank, reader, tmp_path
):
    cases = [prediction(bank, tmp_path), budget(bank, tmp_path), budget(bank, tmp_path, 23)]
    report = module.collect_cases(cases, tmp_path, tmp_path / "physical", scope="development")
    assert report["status"] == "complete" and len(report["rows"]) == 5
    assert len(report["physical_banks"]) == 1
    assert report["physical_banks"][0]["successful_calls"] == 8
    assert sum(r["successful_logical_queries"] for r in report["costs"]) == 12
    assert all(r["additional_physical_calls"] == 0 for r in report["costs"])
    destination = tmp_path / "export"
    module.export_source_data(destination, report)
    with (destination / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5 and rows[-1]["budget_unique_collisions"] != ""
    with (destination / "discovery_curves.csv").open(encoding="utf-8", newline="") as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 8
    if not report["native_bank_profiles"][0]["collisions"]:
        assert all(r["collision_coverage"] == "" for r in curves)
        assert all(r["cumulative_collisions"] == "0" for r in curves)


def test_missing_task_prevents_all_truth_reading_and_final_export(
    bank, reader, tmp_path, monkeypatch
):
    case = prediction(bank, tmp_path)
    missing = replace(case, seed=23)
    monkeypatch.setattr(
        module, "ExecutionBank", lambda _: pytest.fail("Pending report read target truth")
    )
    report = module.collect_cases(
        [case, missing], tmp_path, tmp_path / "physical", scope="development"
    )
    assert report["status"] == "incomplete" and not report["rows"] and not report["costs"]
    with pytest.raises(ValueError, match="incomplete case inventory"):
        module.export_source_data(tmp_path / "invalid", report)
    assert not (tmp_path / "invalid").exists()


def test_interruption_is_pending_until_finalized_and_recovery_retains_attempt_history(
    bank, reader, tmp_path
):
    case = prediction(bank, tmp_path, stop_after=2)
    pending = module.collect_cases([case], tmp_path, tmp_path / "physical", scope="development")
    assert pending["status"] == "incomplete"
    recovered = prediction(bank, tmp_path)
    report = module.collect_cases([recovered], tmp_path, tmp_path / "physical", scope="development")
    assert report["inventory"][0]["prior_failure_files"] == 1
    assert report["costs"][0]["successful_logical_queries"] == 4
    assert all(r["status"] == "complete" for r in report["rows"])


def test_terminal_failure_keeps_all_checkpoints_missing_and_paid_costs(bank, reader, tmp_path):
    case = replace(prediction(bank, tmp_path, stop_after=2), final_failure=True)
    report = module.collect_cases([case], tmp_path, tmp_path / "physical", scope="development")
    assert [r["K"] for r in report["rows"]] == [0, 2, 4]
    assert all(
        r["status"] == "failed" and r["nll"] is None and r["failure_reason"] for r in report["rows"]
    )
    assert report["costs"][0]["successful_logical_queries"] == 2
    assert report["costs"][0]["selector_wall_s"] is None


def test_native_collected_rows_enter_the_actual_paired_summary_functions(bank, reader, tmp_path):
    from experiments.history_transfer_summary import budget_summary, prediction_summary

    cases = [
        prediction(bank, tmp_path),
        budget(bank, tmp_path),
        budget(bank, tmp_path, method="target_only"),
    ]
    report = module.collect_cases(cases, tmp_path, tmp_path / "physical", scope="development")
    common = {
        "ads": {"collector_fixture": "unit"},
        "scenes": ["car_following"],
        "seeds": [11],
        "candidate_count": 8,
        "replicates": 100,
    }
    predicted = prediction_summary(
        [r for r in report["rows"] if r["experiment"] == "prediction"],
        methods=["target_only"],
        checkpoints=(0, 2, 4),
        **common,
    )
    discovered = budget_summary(
        [r for r in report["rows"] if r["experiment"] == "budget"],
        methods=["target_only", "random"],
        budget=4,
        initial_count=2,
        **common,
    )
    assert predicted["experiment"] == "strict_k_shot"
    assert discovered["experiment"] == "budgeted_discovery"


@pytest.mark.parametrize(
    "change", ["missing_receipt", "receipt_hash", "metric", "case_owner", "physical_snapshot"]
)
def test_tampered_receipts_metrics_or_approvals_are_rejected(bank, reader, tmp_path, change):
    case = prediction(bank, tmp_path)
    result, _, receipt = module._files(case, tmp_path)
    if change == "missing_receipt":
        receipt.unlink()
    elif change == "receipt_hash":
        data = read_json(receipt)
        data["paid_feedback_hash"] = "changed"
        write_json(receipt, data)
    elif change == "metric":
        data = read_json(result)
        data["metrics"][0]["brier"] += 0.1
        write_json(result, data)
    elif change == "case_owner":
        case.specification["sut_id"] = "wrong_fixture"
    else:
        case.specification["bank_snapshot"] = "different_physical_bank"
    with pytest.raises(ValueError):
        module.collect_cases([case], tmp_path, tmp_path / "physical", scope="development")


def test_formal_guard_runs_before_any_target_truth_is_opened(tmp_path, monkeypatch):
    from experiments import history_transfer_experiment as entry

    cases = [
        module.ApprovedCase(mode, sut.sut_id, scene, seed, method, None)
        for mode, methods in [
            ("prediction", entry.PREDICTION_METHODS),
            ("budget", entry.BUDGET_METHODS),
        ]
        for sut in module.FORMAL_TARGET_SUTS
        for scene in entry.scenarios_for(tmp_path)
        for seed in entry.SEEDS
        for method in methods
    ]

    def closed(_):
        raise ValueError("Formal gate is closed")

    monkeypatch.setattr(entry, "require_method_evaluation", closed)
    monkeypatch.setattr(
        module, "ExecutionBank", lambda _: pytest.fail("Closed formal reader opened labels")
    )
    with pytest.raises(ValueError, match="Formal gate is closed"):
        module.collect_cases(cases, tmp_path, tmp_path / "physical")


def test_development_mode_cannot_read_a_formal_target(tmp_path):
    case = module.ApprovedCase("prediction", "target_idm_mobil", "car_following", 11, "full", None)
    with pytest.raises(ValueError, match="registered old ADS"):
        module.collect_cases([case], tmp_path, tmp_path / "physical", scope="development")


@pytest.mark.parametrize("change", ["raw_setting", "historical_access", "candidate_count"])
def test_formal_gp_specification_rejects_changed_settings_or_historical_target_only_access(change):
    from experiments.transfer_settings import MethodSettings

    settings = MethodSettings(
        OnlineSettings(FitSettings(10, 0.01), FitSettings(10, 0.003), -1.4),
        FitSettings(10, 0.003),
    )
    spec = {
        "candidates": [None] * 2000,
        "online_settings": settings.online.to_dict(),
        "method_contract": {"method": "target_only"},
        "device": "cpu",
        "checkpoints": [0, 5, 10, 20, 50, 100, 200],
        "update_source_weights": True,
        "model": None,
        "history_view": None,
        "source_snapshot": None,
        "model_construction": None,
    }
    case = module.ApprovedCase(
        "prediction", "target_idm_mobil", "car_following", 11, "target_only", spec
    )
    context = {
        "settings": settings,
        "method_contracts": {"target_only": {"method": "target_only"}},
        "inputs": {"car_following": {}},
    }
    module._check_formal_specification(case, context)
    if change == "raw_setting":
        spec["online_settings"]["raw"]["steps"] = 20
    elif change == "historical_access":
        spec["source_snapshot"] = {"source": "unit historical source"}
    else:
        spec["candidates"].pop()
    with pytest.raises(ValueError):
        module._check_formal_specification(case, context)


@pytest.mark.parametrize("experiment", ["prediction", "budget"])
@pytest.mark.parametrize("change", ["source_records", "source_code", "base_model", "mixture"])
def test_formal_ours_requires_the_selected_models_sources_and_mixture(experiment, change):
    from experiments.transfer_settings import MethodSettings

    settings = MethodSettings(
        OnlineSettings(FitSettings(10, 0.01), FitSettings(10, 0.003), -1.4),
        FitSettings(10, 0.003),
    )
    source = {
        "view_record_ids": {"unit_history": ["record"]},
        "source_contract": "unit source code",
        "candidates": [None] * 2000,
    }
    spec = {
        "candidates": [None] * 2000,
        "online_settings": settings.online.to_dict(),
        "update_source_weights": True,
        "source_snapshot": copy.deepcopy(source),
        "method_contract": {"method": "full"},
        "device": "cpu",
        "checkpoints": [0, 5, 10, 20, 50, 100, 200],
        "model": "base",
        "target_model_version": "base",
        "encoder_version": "encoder",
        "budget": 200,
        "initial_count": 20,
        "feedback_mode": "joint",
        "prior_mode": "sources",
        "historical_rss": True,
        "native_feedback": True,
    }
    context = {
        "settings": settings,
        "source_contract": "unit source code",
        "method_contracts": {"full": {"method": "full"}},
        "inputs": {
            "car_following": {
                "full": {
                    "model_versions": ["base", "encoder"],
                    "source_snapshot": copy.deepcopy(source),
                }
            }
        },
    }
    case = module.ApprovedCase(experiment, "target_idm_mobil", "car_following", 11, "full", spec)
    module._check_formal_specification(case, context)
    if change == "source_records":
        spec["source_snapshot"]["view_record_ids"]["unit_history"] = ["different record"]
    elif change == "source_code":
        spec["source_snapshot"]["source_contract"] = "changed source calculation"
    elif change == "base_model":
        spec["model"] = spec["target_model_version"] = "different base"
    else:
        spec["update_source_weights"] = False
    with pytest.raises(ValueError):
        module._check_formal_specification(case, context)


@pytest.mark.parametrize("method", ["random", "scenario_fuzz"])
def test_formal_baselines_cannot_substitute_configuration_or_history(method):
    construction = (
        {}
        if method == "random"
        else {
            "model_version": "unit classifier",
            "history_fingerprints": {"history": "fingerprint"},
        }
    )
    config = {} if method == "random" else {"unit_fixed_setting": 1}
    context = {
        "settings": None,
        "method_contracts": {method: {"method": method}},
        "baseline_configurations": {"scenario_fuzz": config},
        "inputs": {
            "car_following": {
                "full": {"history_view": {"history_fingerprints": {"history": "fingerprint"}}}
            }
        },
        "classifiers": {"car_following": {"model_version": "unit classifier"}},
    }
    spec = {
        "candidates": [None] * 2000,
        "task_snapshot": {"method_contract": {"method": method}},
        "budget": 200,
        "initial_count": 20,
        "configuration": config.copy(),
        "construction": copy.deepcopy(construction),
    }
    case = module.ApprovedCase("budget", "target_idm_mobil", "car_following", 11, method, spec)
    module._check_formal_specification(case, context)
    spec["construction"]["model_version"] = "different history"
    with pytest.raises(ValueError):
        module._check_formal_specification(case, context)
