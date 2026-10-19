"""Keep old artifacts strict while recognizing the frozen optional interfaces."""

from zipfile import ZipFile

import pytest

from methods.common.snapshots import ADDED_SOURCES, verify_source


def test_reviewed_api_extension_checks_exact_bytes_and_rejects_unlisted_changes(
    tmp_path,
):
    relative = "methods/baselines/lrt/selection.py"
    paths = {relative, *ADDED_SOURCES, "methods/baselines/lrt/value.py"}
    snapshot = tmp_path / "results/baselines/lrt/by_family/frozen_inputs.zip"
    snapshot.parent.mkdir(parents=True)
    with ZipFile(snapshot, "w") as archive:
        for name in paths:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"reviewed\n")
            archive.writestr(name, b"reviewed\n")
    assert verify_source(tmp_path, relative, b"original\n") is True
    assert verify_source(tmp_path, relative, b"reviewed\n") is False
    with pytest.raises(ValueError, match="Frozen source"):
        verify_source(tmp_path, relative, None)
    with pytest.raises(ValueError, match="Frozen source"):
        verify_source(tmp_path, "methods/baselines/lrt/value.py", b"original\n")
    (tmp_path / relative).write_bytes(b"unexpected edit\n")
    with pytest.raises(ValueError, match="Reviewed interface"):
        verify_source(tmp_path, relative, b"original\n")


def test_retained_baseline_and_lrt_freeze_audits_accept_only_reviewed_interfaces():
    from methods.baselines.dart.evaluation.freeze import verify_freeze
    from methods.baselines.lrt.evaluation.protocol import freeze

    report = verify_freeze()
    assert "methods/baselines/dart/encoder.py" in report["reviewed_api_extensions"]
    freeze()


def test_new_terminal_timing_revision_keeps_older_api_checks_strict(tmp_path):
    relative = "methods/baselines/dart/encoder.py"
    old = tmp_path / "results/baselines/lrt/by_family/frozen_inputs.zip"
    old.parent.mkdir(parents=True)
    with ZipFile(old, "w") as archive:
        for name in {relative, *ADDED_SOURCES}:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"reviewed old\n")
            archive.writestr(name, b"reviewed old\n")
    updated = "benchmark/timing.py"
    (tmp_path / updated).write_bytes(b"reviewed stopping\n")
    new = tmp_path / "results/baselines/lrt/adaptive/reviewed_interfaces.zip"
    new.parent.mkdir(parents=True)
    with ZipFile(new, "w") as archive:
        for name in ADDED_SOURCES:
            archive.writestr(name, (tmp_path / name).read_bytes())
    assert verify_source(tmp_path, relative, b"original\n")
    (tmp_path / updated).write_bytes(b"unreviewed edit\n")
    with pytest.raises(ValueError, match="Reviewed interface"):
        verify_source(tmp_path, relative, b"original\n")
