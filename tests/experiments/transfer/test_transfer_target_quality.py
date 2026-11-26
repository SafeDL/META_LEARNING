"""Target truth stays sealed until configuration selection and complete collection."""

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import audit_transfer_targets as module
from experiments.history_transfer_experiment import RUN_SCENARIOS
from experiments.rss import measurement_contract
from experiments.storage import read_json, write_json


def test_default_target_quality_scope_is_four_independent_scene_libraries():
    assert RUN_SCENARIOS == ("car_following_braking_onset", "cut_in_onset", "cut_out_rear_gap", "side_encroachment")
    assert len(RUN_SCENARIOS) * len(module.FORMAL_TARGET_SUTS) == 32
    assert module.TARGET_COUNT == 2000


def test_unsealed_settings_prevent_even_target_metadata_access(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Target access must follow settings sealing")

    monkeypatch.setattr(module, "target_banks", forbidden)
    root = tmp_path / "not_created"
    with pytest.raises(FileNotFoundError):
        module.audit(root)
    assert not root.exists()


def test_missing_later_bank_prevents_reading_any_target_trace(native_inputs, tmp_path, monkeypatch):
    bank = native_inputs[0]
    systems = (SimpleNamespace(sut_id="executor_fixture"), SimpleNamespace(sut_id="missing_unit"))
    monkeypatch.setattr(module, "load_settings", lambda root: None)
    monkeypatch.setattr(module, "scenarios_for", lambda root: ("car_following",))
    monkeypatch.setattr(module, "FORMAL_TARGET_SUTS", systems)
    monkeypatch.setattr(module, "TARGET_COUNT", 8)
    monkeypatch.setattr(
        module,
        "registered_populations",
        lambda root: {("target", "car_following"): bank.candidates},
    )
    monkeypatch.setattr(module, "require_qualified", lambda sut: None)
    monkeypatch.setattr(
        module, "execution_contract", lambda *args, **kwargs: bank.manifest["contract"]
    )

    def bank_loader(path):
        if path.name == "missing_unit":
            raise FileNotFoundError("unfinished unit bank")
        return bank

    monkeypatch.setattr(module, "ExecutionBank", bank_loader)
    monkeypatch.setattr(
        module, "audit_bank", lambda bank: pytest.fail("No truth before all banks exist")
    )
    touched = list(bank.touched)
    with pytest.raises(FileNotFoundError, match="unfinished unit"):
        module.audit(tmp_path)
    assert bank.touched == touched
    assert not (tmp_path / "reports/target_quality").exists()


def test_native_clock_collision_geometry_and_derived_rss_are_measured(native_inputs):
    bank = native_inputs[0]
    result = module.audit_bank(bank)
    assert result["profile"]["records"] == result["rss"]["measured_records"] == 8
    expected = sum(
        any(frame["collision"] for frame in bank.read_paid(i).trajectory) for i in range(8)
    )
    assert result["profile"]["collisions"] == expected
    assert result["profile"]["noncollision_early_ends"] == 0
    assert result["rss"]["stored_rss_records"] == 0
    assert all(np.isfinite(result["rss"][field]) for field in ("minimum", "maximum", "mean"))


def test_inconsistent_stored_rss_is_not_accepted_as_a_native_response(native_inputs):
    bank = copy.deepcopy(native_inputs[0])
    paid = bank.read_paid(0)
    paid.feedback = replace(
        paid.feedback,
        safety=replace(paid.feedback.safety, rss_margin=123456.0),
        rss_contract=measurement_contract(),
    )
    with pytest.raises(ValueError, match="RSS disagrees"):
        module.audit_bank(bank)


def test_quality_records_reuse_only_the_same_settings_and_bank(native_inputs, tmp_path, monkeypatch):
    bank = native_inputs[0]
    monkeypatch.setattr(module, "load_settings", lambda root: None)
    monkeypatch.setattr(module, "scenarios_for", lambda root: ("car_following",))
    monkeypatch.setattr(module, "FORMAL_TARGET_SUTS", (SimpleNamespace(sut_id="executor_fixture"),))
    monkeypatch.setattr(module, "TARGET_COUNT", 8)
    monkeypatch.setattr(
        module, "target_banks", lambda root: {("car_following", "executor_fixture"): bank}
    )
    write_json(tmp_path / module.REGISTRY, {"scope": "explicit untrained unit configuration"})
    write_json(tmp_path / "reports/protocol.json", {"scope": "native eight-record unit quality"})
    result = module.audit(tmp_path)
    assert result["status"] == "complete" and result["records"] == 8
    assert result["formal_target_labels_read"]
    monkeypatch.setattr(
        module, "audit_bank", lambda bank: pytest.fail("Reuse completed identical evidence")
    )
    assert module.audit(tmp_path) == result
    write_json(tmp_path / module.REGISTRY, {"scope": "changed unit configuration"})
    with pytest.raises(ValueError, match="Registered.*changed|registered.*changed|Registered"):
        module.audit(tmp_path)


def test_verified_target_quality_binds_native_files(native_inputs, tmp_path, monkeypatch):
    native = native_inputs[0]
    path = tmp_path / "native_unit_bank"
    write_json(
        path / "manifest.json",
        {
            **native.manifest,
            "format": "collision-execution-bank",
            "status": "complete",
            "scenarios": [asdict(s) for s in native.candidates],
        },
    )
    for i, scene in enumerate(native.candidates):
        paid = native.read_paid(i)
        write_json(
            path / "executions" / f"{i:06d}.json.gz",
            {
                "scenario": asdict(scene),
                "feedback": paid.feedback.to_dict(),
                "trajectory": paid.trajectory,
            },
        )
    loader = module.ExecutionBank
    monkeypatch.setattr(module, "load_settings", lambda root: None)
    monkeypatch.setattr(module, "scenarios_for", lambda root: ("car_following",))
    monkeypatch.setattr(module, "FORMAL_TARGET_SUTS", (SimpleNamespace(sut_id="executor_fixture"),))
    monkeypatch.setattr(module, "TARGET_COUNT", 8)
    monkeypatch.setattr(
        module, "target_banks", lambda root: {("car_following", "executor_fixture"): loader(path)}
    )
    write_json(tmp_path / module.REGISTRY, {"scope": "unit configuration; not formal release"})
    write_json(tmp_path / "reports/protocol.json", {"scope": "unit native quality"})
    summary = module.audit(tmp_path)
    verified = module.verify(tmp_path)
    assert verified["summary"] == summary
    assert len(verified["files"]) == 12
    saved = read_json(path / "executions/000000.json.gz")
    saved["feedback"]["collision"] = not saved["feedback"]["collision"]
    write_json(path / "executions/000000.json.gz", saved)
    with pytest.raises(ValueError, match="another input or settings"):
        module.verify(tmp_path)
