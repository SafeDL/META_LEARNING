import json

import pytest

from benchmark import run_collision as pipeline


def configure(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT", tmp_path / "pipeline")
    monkeypatch.setattr(pipeline, "STATE_PATH", tmp_path / "pipeline/state.json")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "method_contract", lambda: {"method": "fixture-v1"})
    monkeypatch.setattr(pipeline, "training_contract", lambda: {"training": "fixture-v1"})


@pytest.mark.parametrize("exit_code", [0, 7])
def test_pipeline_records_native_exit_even_when_a_child_writes_to_stderr(tmp_path, monkeypatch, exit_code):
    configure(tmp_path, monkeypatch)
    (tmp_path / "native_child.py").write_text(
        f"import sys\nprint('diagnostic on stderr', file=sys.stderr)\nsys.exit({exit_code})\n", encoding="utf-8")
    runner = pipeline.Pipeline()
    if exit_code:
        with pytest.raises(RuntimeError, match="native exit 7"):
            runner.command("fixture", "native_child")
    else:
        runner.command("fixture", "native_child")
    saved = json.loads(pipeline.STATE_PATH.read_text())["stages"]["fixture"]
    assert saved["native_exit_code"] == exit_code
    assert saved["status"] == ("complete" if exit_code == 0 else "failed")
    assert "diagnostic on stderr" in (tmp_path / "pipeline/fixture_1.log").read_text()
    runner.close()


def test_resume_reuses_completed_stage_only_with_identical_source_contract(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    (tmp_path / "native_child.py").write_text("print('one paid construction stage')\n", encoding="utf-8")
    runner = pipeline.Pipeline()
    runner.command("fixture", "native_child")
    log = tmp_path / "pipeline/fixture_1.log"
    original = log.read_bytes()
    runner.close()
    resumed = pipeline.Pipeline(resume=True)
    resumed.command("fixture", "native_child")
    resumed.close()
    assert log.read_bytes() == original and not (tmp_path / "pipeline/fixture_2.log").exists()
    monkeypatch.setattr(pipeline, "method_contract", lambda: {"method": "changed"})
    with pytest.raises(ValueError, match="identical code"):
        pipeline.Pipeline(resume=True)


def test_an_active_pipeline_holds_an_exclusive_execution_ledger(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    owner = pipeline.Pipeline()
    with pytest.raises(RuntimeError, match="already owns"):
        pipeline.Pipeline(resume=True)
    owner.close()
    resumed = pipeline.Pipeline(resume=True)
    resumed.close()
