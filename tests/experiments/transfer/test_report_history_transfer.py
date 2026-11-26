"""Main report wiring uses native unit traces; admission stubs are explicit."""

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments import collect_transfer_records as reader
from experiments import report_history_transfer as module
from experiments import transfer_batches as batches
from experiments import transfer_execution as execution
from experiments.k_shot import METRICS, split_candidates
from experiments.resume import ReplayInterrupted
from experiments.storage import read_json, write_json
from tests.fixtures.predictions import DESIGN, prediction_rows

LIMITS = {"wall_s": 120, "child_memory_bytes": 8 * 1024**3}


@pytest.fixture
def native_runs(native_inputs, tmp_path, monkeypatch):
    bank, _, models, sources, pooled, settings, _ = native_inputs
    monkeypatch.setattr(execution, "DEVELOPMENT_IDS", (bank.manifest["sut_id"],))
    monkeypatch.setattr(reader, "DEVELOPMENT_IDS", (bank.manifest["sut_id"],))
    monkeypatch.setattr(reader, "ExecutionBank", lambda _: bank)
    cases = []
    for method in ("target_only", "pooled_history", "full"):
        model_pair = (None, None) if method == "target_only" else models
        history = pooled if method == "pooled_history" else None
        source = sources if method == "full" else None
        case = execution.prediction_case(
            bank,
            split_candidates(bank.candidates, 11, budget=4),
            11,
            method,
            model_pair,
            history,
            source,
            settings=settings,
            checkpoints=(0, 2, 4),
            limits=LIMITS,
        )
        if method == "target_only":
            with pytest.raises(ReplayInterrupted):
                execution.run_prediction(
                    tmp_path,
                    case,
                    bank,
                    model_pair,
                    history,
                    source,
                    settings=settings,
                    paid_execution=bank.read_paid,
                    scope="development",
                    stop_after=2,
                )
        execution.run_prediction(
            tmp_path,
            case,
            bank,
            model_pair,
            history,
            source,
            settings=settings,
            paid_execution=bank.read_paid,
            scope="development",
        )
        cases.append(case)
    for method in ("target_only", "random"):
        case = execution.budget_case(
            bank,
            11,
            method,
            (None, None),
            None,
            settings=settings,
            initial_count=2,
            budget=4,
            limits=LIMITS,
        )
        execution.run_budget(
            tmp_path,
            case,
            bank,
            (None, None),
            None,
            settings=settings,
            paid_execution=bank.read_paid,
            scope="development",
        )
        cases.append(case)
    records = reader.collect_cases(cases, tmp_path, tmp_path, scope="development")
    statistics = {
        "ads": {bank.manifest["sut_id"]: "explicit_native_unit_fixture"},
        "scenarios": ["car_following"],
        "feedback_seeds": [11],
        "candidate_count": 8,
        "prediction": {
            "methods": ["target_only", "pooled_history", "full"],
            "checkpoints": [0, 2, 4],
        },
        "budget": {"methods": ["target_only", "random"], "queries": 4, "common_initial": 2},
        "statistics": {"replicates": 20},
    }
    return cases, records, statistics


def test_native_saved_records_enter_both_registered_estimators_and_pooled_pairing(native_runs):
    _, records, statistics = native_runs
    prediction = module.summarize_audited(records, statistics, "prediction")
    budget = module.summarize_audited(records, statistics, "budget")
    assert len(records["rows"]) == 11
    assert (
        prediction["scopes"]["all"]["design"]["bootstrap_unit"]
        == "ADS only; no feedback-seed resampling"
    )
    comparison = prediction["comparison_with_pooled"]["scopes"]["all"]
    assert comparison["design"]["baseline"] == "pooled_history"
    for index, k in enumerate((0, 2, 4)):
        values = {r["method"]: r["nll"] for r in records["rows"] if r.get("K") == k}
        assert comparison["paired"]["gain"][2][index][METRICS.index("nll")] == pytest.approx(
            values["pooled_history"] - values["full"]
        )
    assert budget["budget"] == 4 and budget["initial_count"] == 2
    assert budget["curves"]["all"]["design"]["registered_ads"] == 1


def test_complete_native_attempt_chain_counts_interruption_without_recharging_queries(
    native_runs, tmp_path
):
    cases, records, _ = native_runs
    attempts = module.attempt_costs(tmp_path, cases)
    assert len(attempts) == 6
    assert [a["status"] for a in attempts[:2]] == ["interrupted", "complete"]
    assert [
        (a["saved_responses_before_run"], a["saved_responses_after_run"]) for a in attempts[:2]
    ] == [(0, 2), (2, 4)]
    assert sum(c["successful_logical_queries"] for c in records["costs"]) == 20
    assert sum(a["attempt_wall_s"] for a in attempts) > attempts[-1]["attempt_wall_s"]
    assert attempts[0]["selector_wall_s"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("saved_responses_before_run", 1),
        ("saved_responses_after_run", 5),
        ("wall_s", -1),
        ("status", "running"),
        ("specification", {"changed": True}),
    ],
)
def test_cost_report_rejects_changed_attempts(native_runs, tmp_path, field, value):
    cases, _, _ = native_runs
    path = execution._paths(tmp_path, cases[0])[0].parent / "attempts"
    path = sorted(path.glob("target_only_*.json"))[0]
    changed = read_json(path)
    changed[field] = value
    write_json(path, changed)
    with pytest.raises(ValueError, match="attempt"):
        module.attempt_costs(tmp_path, cases)


def test_formal_report_refuses_missing_admission_before_reading_declarations(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Declarations or target labels were read before admission")

    monkeypatch.setattr(batches, "approved_cases", forbidden)
    with pytest.raises(FileNotFoundError, match="method_settings"):
        module.report(root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_pending_inventory_never_exports_final_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module, "load_acceptance", lambda root: {"scope": "explicit unit admission stub"}
    )
    monkeypatch.setattr(module, "register_statistics", lambda root: {})
    monkeypatch.setattr(batches, "approved_cases", lambda root, mode: [])
    monkeypatch.setattr(
        reader,
        "collect_cases",
        lambda *args: {"status": "incomplete", "inventory": [{"status": "pending"}]},
    )

    def forbidden(*args):
        pytest.fail("A partial inventory must not produce a statistical summary")

    monkeypatch.setattr(module, "summarize_audited", forbidden)
    result = module.report("prediction", tmp_path)
    assert result["status"] == "incomplete"
    assert not (tmp_path / "reports/main_results/prediction/completion.json").exists()


def test_full_export_wiring_uses_native_audited_records_with_explicit_gate_stubs(
    native_runs, tmp_path, monkeypatch
):
    cases, records, statistics = native_runs
    monkeypatch.setattr(
        module, "load_acceptance", lambda root: {"scope": "explicit unit admission stub"}
    )
    monkeypatch.setattr(module, "register_statistics", lambda root: statistics)
    monkeypatch.setattr(
        batches, "approved_cases", lambda root, mode: [c for c in cases if c.experiment == mode]
    )
    monkeypatch.setattr(reader, "collect_cases", lambda *args: records)
    for name in ("reports/statistics_protocol.json", batches.ACCEPTANCE, batches.REGISTRY):
        write_json(
            tmp_path / name, {"scope": "Explicit unit dependency stub; not formal stage evidence"}
        )
    source = cases[2].specification["source_snapshot"]
    key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    write_json(
        tmp_path / "reports/source_construction/car_following" / f"{key}.json",
        {
            "specification": source,
            "scope": "source_construction",
            "resources": {"wall_s": 0.1, "sampled_peak_cpu_bytes": 1024},
            "note": "Explicit unit cost fixture",
        },
    )
    monkeypatch.setattr("experiments.transfer_admission.require_experiment", lambda mode: None)
    result = module.report("all", tmp_path)
    assert result["cases"] == 5 and result["execution_attempts"] == 6
    assert result["physical_bank_successful_calls"] == 8
    assert result["successful_logical_queries"] == 20
    assert result["new_physical_executions"] == 0
    folder = tmp_path / "reports/main_results/all"
    for name, digest in result["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    for name in ("metrics.csv", "discovery_curves.csv", "costs.csv", "execution_attempts.csv"):
        assert (folder / name).exists()


def test_approved_case_loader_reconstructs_the_required_specification(tmp_path, monkeypatch):
    monkeypatch.setattr(batches, "scenarios_for", lambda root: ("unit_scene",))
    write_json(tmp_path / batches.REGISTRY, {"scope": "unit fixture"})
    case = reader.ApprovedCase(
        "prediction", "unit_ads", "unit_scene", 11, "full", {"fixture": True}
    )
    protocol = {
        "entry_source_sha256": batches._sha(batches.__file__),
        "settings_sha256": batches._sha(tmp_path / batches.REGISTRY),
        "execution_contract": execution.execution_contract(LIMITS),
        "cases": [
            {
                "key": list(case.key),
                "specification_sha256": hashlib.sha256(
                    json.dumps(case.specification, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
            }
        ],
    }
    folder = tmp_path / "reports/execution_tasks/prediction/unit_scene"
    write_json(folder / "protocol.json", protocol)
    write_json(
        folder / "progress.json",
        {"protocol": protocol, "completed": [{"key": list(case.key), "status": "failed"}]},
    )
    write_json(
        execution.approval_path(tmp_path, case),
        {"key": list(case.key), "scope": "formal", "specification": case.specification},
    )
    loaded = batches.approved_cases(tmp_path, "prediction")
    assert loaded == [replace(case, final_failure=True)]


def test_shared_cost_receipt_is_required_and_deduplicated(tmp_path):
    source = {"scope": "Explicit unit source specification"}
    case = reader.ApprovedCase(
        "prediction",
        "unit_ads",
        "unit_scene",
        11,
        "full",
        {
            "source_snapshot": source,
        },
    )
    with pytest.raises(FileNotFoundError):
        module.shared_construction(tmp_path, [case])
    key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "reports/source_construction/unit_scene" / f"{key}.json"
    receipt = {
        "specification": source,
        "scope": "source_construction",
        "resources": {"wall_s": 2.0, "sampled_peak_cpu_bytes": 1024},
    }
    write_json(path, receipt)
    rows = module.shared_construction(
        tmp_path, [case, replace(case, experiment="budget"), replace(case, seed=23)]
    )
    assert len(rows) == 1 and rows[0]["resources"]["wall_s"] == 2
    receipt["specification"] = {"scope": "Changed unit source"}
    write_json(path, receipt)
    with pytest.raises(ValueError, match="snapshot"):
        module.shared_construction(tmp_path, [case])


def test_direct_pooled_comparison_keeps_only_common_valid_seed_cells():
    # Explicit numerical fixture: target-only is defined for both seeds while
    # Pooled's seed 23 failed; Ours' comparison must use only seed 11.
    rows = prediction_rows()
    spec = {
        "ads": DESIGN["ads"],
        "scenarios": DESIGN["scenes"],
        "feedback_seeds": DESIGN["seeds"],
        "candidate_count": 10,
        "prediction": {"methods": ["target_only", "pooled_history", "full"], "checkpoints": [0, 1]},
        "statistics": {"replicates": 20},
    }
    combined = []
    for row in rows:
        row = {"experiment": "prediction", **row}
        combined.append(row)
        if row["method"] == "target_only":
            pooled = copy.deepcopy(row)
            pooled["method"] = "pooled_history"
            if row["seed"] == 23:
                pooled.update(status="failed", failure_reason="Explicit unit optimizer failure")
                pooled.update({m: None for m in METRICS})
            combined.append(pooled)
    result = module.summarize_audited({"status": "complete", "rows": combined}, spec, "prediction")
    pooled = result["comparison_with_pooled"]["scopes"]["all"]
    assert np.asarray(pooled["paired"]["valid_seed_runs"])[2, 0, 0] == 4
