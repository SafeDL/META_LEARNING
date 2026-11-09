"""Frozen evidence and maintained source are checked independently."""

import gzip
import json
from dataclasses import asdict

import pytest

from experiments import source_registration


def registration(tmp_path, monkeypatch):
    path = tmp_path / "source_reorganization.json"
    path.write_text(json.dumps({"relocated_source_files": {}}), encoding="utf-8")
    monkeypatch.setattr(source_registration, "SOURCE_REORGANIZATION", path)


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("simulation/highway/observation.py", "simulator/highway/observation.py"),
        ("methods/gcrt/training.py", "testing_methods/history_transfer/training.py"),
        ("methods/common/response.py", "testing_methods/common/response.py"),
    ],
)
def test_reorganization_accepts_only_reviewed_source_changes(
    tmp_path, monkeypatch, old_name, new_name
):
    registration(tmp_path, monkeypatch)
    path = source_registration.SOURCE_REORGANIZATION
    record = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "relocated_source_files": {
            old_name: {"before_sha256": "original", "after_sha256": "reviewed"}
        }
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    original = {old_name: "original", "checkpoint": "unchanged"}
    current = {new_name: "reviewed", "checkpoint": "unchanged"}
    assert source_registration.source_contract_matches(original, current)
    assert source_registration.specification_matches(
        {"source_files": original, "budget": 128}, {"source_files": current, "budget": 128}
    )
    assert not source_registration.specification_matches(
        {"source_files": original, "budget": 128}, {"source_files": current, "budget": 64}
    )
    for changed in (
        {**current, new_name: "unreviewed"},
        {**current, "checkpoint": "retrained"},
        {**current, "extra.py": "new"},
    ):
        assert not source_registration.source_contract_matches(original, changed)


def test_path_only_training_changes_require_the_recorded_digest_pair(tmp_path, monkeypatch):
    registration(tmp_path, monkeypatch)
    maintenance = source_registration.SOURCE_REORGANIZATION
    maintenance.write_text(json.dumps({"relocated_source_files": {}}), encoding="utf-8")
    path = tmp_path / "reorganization.json"
    path.write_text(
        json.dumps(
            {
                "source_changes": {
                    "controllers/highway/train.py": {
                        "before_sha256": "before",
                        "after_sha256": "after",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(source_registration, "ARTIFACT_MAINTENANCE", path)
    recorded = {"controllers/highway/train.py": "before", "checkpoint": "unchanged"}
    current = {**recorded, "controllers/highway/train.py": "after"}
    assert source_registration.source_contract_matches(recorded, current)
    assert not source_registration.source_contract_matches(
        recorded, {**current, "controllers/highway/train.py": "unreviewed"}
    )
    assert not source_registration.source_contract_matches(
        recorded, {**current, "checkpoint": "retrained"}
    )


def test_code_cleanup_accepts_only_the_reviewed_source_pair(tmp_path, monkeypatch):
    registration(tmp_path, monkeypatch)
    maintenance = source_registration.SOURCE_REORGANIZATION
    maintenance.write_text(json.dumps({"relocated_source_files": {}}), encoding="utf-8")
    path = tmp_path / "cleanup.json"
    path.write_text(
        json.dumps(
            {"source_changes": {"training.py": {"before_sha256": "old", "after_sha256": "new"}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(source_registration, "CODE_MAINTENANCE", path)
    recorded = {"training.py": "old", "model.py": "unchanged"}
    assert source_registration.source_contract_matches(recorded, {**recorded, "training.py": "new"})
    for changed in (
        {**recorded, "training.py": "unreviewed"},
        {"training.py": "new", "model.py": "modified"},
        {**recorded, "training.py": "new", "extra.py": "added"},
    ):
        assert not source_registration.source_contract_matches(recorded, changed)


def test_reviewed_intermediate_source_requires_exact_path_digest_and_unchanged_other_inputs(
    tmp_path, monkeypatch
):
    registration(tmp_path, monkeypatch)
    path = tmp_path / "changes.json"
    path.write_text(
        json.dumps(
            {
                "source_changes": {
                    "selection.py": {
                        "before_sha256": "first",
                        "after_sha256": "third",
                        "reviewed_sha256": ["second"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(source_registration, "CODE_MAINTENANCE", path)
    current = {"selection.py": "third", "model.py": "same"}
    for prior in ("first", "second"):
        recorded = {"selection.py": prior, "model.py": "same"}
        assert source_registration.source_contract_matches(recorded, current)
        for changed in (
            {**current, "selection.py": "unreviewed"},
            {**current, "model.py": "different"},
            {**current, "extra.py": "added"},
            {"selection.py": "third"},
        ):
            assert not source_registration.source_contract_matches(recorded, changed)
    assert not source_registration.source_contract_matches(
        {"selection.py": "unknown", "model.py": "same"}, current
    )


@pytest.mark.parametrize("root", ("results", "experiment_outputs", "artifacts/runs"))
def test_completed_bank_survives_relocation_without_rewriting_or_executing(
    tmp_path, monkeypatch, root
):
    from experiments import executions
    from experiments.scenarios import sample
    from experiments.schema import QueryFeedback, SafetyMeasurements

    scenes = sample("car_following", 1, "history")
    identity = "history_idm_mobil"
    if root == "artifacts/runs":
        qualification = f"{root}/qualification/reports/{identity}/qualification.json"
    else:
        qualification = f"{root}/benchmark/collision/qualification/{identity}/qualification.json"
    manifest = {
        "format": "collision-execution-bank",
        "sut_id": identity,
        "environment_seed": 81001,
        "qualification": qualification,
        "contract": {"fixture": True},
        "scenarios": [asdict(scenes[0])],
        "status": "complete",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "executions").mkdir()
    execution = tmp_path / "executions/000000.json.gz"
    feedback = QueryFeedback(scenes[0].scenario_id, False, SafetyMeasurements(2, 10), 0.1)
    execution.write_bytes(
        gzip.compress(
            json.dumps(
                {"scenario": asdict(scenes[0]), "feedback": feedback.to_dict(), "trajectory": []}
            ).encode()
        )
    )
    before = path.read_bytes(), execution.read_bytes()
    monkeypatch.setattr(
        executions,
        "require_qualified",
        lambda _: source_registration.maintained_path(qualification),
    )
    monkeypatch.setattr(executions, "execution_contract", lambda *_, **kw: {"fixture": True})

    def forbid_execution(*_, **kw):
        pytest.fail("A complete moved bank must not start an executor")

    monkeypatch.setattr(executions, "ProcessPoolExecutor", forbid_execution)
    bank = executions.collect_bank(identity, scenes, tmp_path)
    assert bank(scenes[0]).feedback == feedback
    assert (path.read_bytes(), execution.read_bytes()) == before
    with pytest.raises(ValueError, match="different scene/controller"):
        executions.collect_bank(identity, scenes, tmp_path, environment_seed=81002)
