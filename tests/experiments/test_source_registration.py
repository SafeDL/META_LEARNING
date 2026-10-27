"""Frozen evidence and maintained source are checked independently."""

import hashlib
import json
import zipfile

import pytest

from experiments import source_registration


def registration(tmp_path, monkeypatch):
    folder = tmp_path / "model_construction"
    folder.mkdir()
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"budget": 128}', encoding="utf-8")
    original = b"original source\n"
    with zipfile.ZipFile(folder / "frozen_sources.zip", "w") as archive:
        archive.writestr("method.py", original)
    maintained = {"method.py": "maintained-source-digest"}
    (folder / "source_maintenance.json").write_text(
        json.dumps(
            {
                "frozen_protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "maintained_source_files": maintained,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(source_registration, "REGISTRATION", folder)
    return {"method.py": hashlib.sha256(original).hexdigest()}, maintained


def test_maintenance_does_not_relabel_the_original_source(tmp_path, monkeypatch):
    frozen, maintained = registration(tmp_path, monkeypatch)
    source_registration.validate_sources(frozen, maintained)
    with pytest.raises(ValueError, match="maintained implementation"):
        source_registration.validate_sources(frozen, {"method.py": "unreviewed-change"})


def test_original_source_corruption_is_rejected(tmp_path, monkeypatch):
    frozen, maintained = registration(tmp_path, monkeypatch)
    with zipfile.ZipFile(source_registration.REGISTRATION / "frozen_sources.zip", "w") as archive:
        archive.writestr("method.py", b"changed source")
    with pytest.raises(ValueError, match="originally registered source"):
        source_registration.validate_sources(frozen, maintained)


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("simulation/highway/observation.py", "simulator/highway/observation.py"),
        ("methods/baselines/fsbo/method.py", "testing_methods/common/fsbo/method.py"),
        ("methods/gcrt/domain_training.py", "testing_methods/history_transfer/domain_training.py"),
        ("methods/common/response.py", "testing_methods/common/response.py"),
    ],
)
def test_reorganization_accepts_only_reviewed_source_changes(
    tmp_path, monkeypatch, old_name, new_name
):
    registration(tmp_path, monkeypatch)
    path = source_registration.REGISTRATION / "source_maintenance.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["source_reorganization"] = {
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
