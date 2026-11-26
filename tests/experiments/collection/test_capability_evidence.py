"""Course/criterion changes and corrupt archives must close physical collection."""

import gzip
import json
from dataclasses import replace

import pytest

from controllers.highway.catalog import BY_ID
from controllers.highway.contracts import runtime_contract
from controllers.highway.control import Control
from experiments import executions, qualification
from experiments.qualification import Course, brake_scene


class Coast:
    def act(self, _):
        return Control(0, 0)


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    courses = (Course("short", brake_scene("short"), 0.4, (4.0,), min_progress_m=9),)
    monkeypatch.setattr(qualification, "COURSES", courses)
    monkeypatch.setattr(executions, "ROOT", tmp_path)
    monkeypatch.setattr(executions, "QUALIFICATION_ROOT", tmp_path)
    sut = BY_ID["independent_idm_mobil"]
    root = tmp_path / sut.sut_id
    result = qualification.qualify(Coast, save_root=root)
    (root / "qualification.json").write_text(
        json.dumps(
            {"parameters": sut.parameters, "runtime_contract": runtime_contract(sut), **result}
        ),
        encoding="utf-8",
    )
    return root, sut, courses


def test_gate_accepts_native_saved_evidence_and_replay_preserves_originals(accepted):
    root, sut, _ = accepted
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.name != "course_evidence.json"}
    executions.require_qualified(sut)
    result = qualification.replay_courses(root)
    assert result["qualified"] and result["mode"] == "native_saved_command_replay"
    assert before == {p.name: p.read_bytes() for p in root.iterdir() if p.name in before}
    executions.require_qualified(sut)


@pytest.mark.parametrize("change", ["duration", "road", "scenario", "criterion", "source"])
def test_course_or_criteria_change_invalidates_saved_acceptance(accepted, monkeypatch, change):
    _, sut, (course,) = accepted
    changes = {
        "duration": {"duration_s": 0.8},
        "road": {"lane_centers": (0.0, 4.0)},
        "scenario": {"scenario": brake_scene("changed", gap=60)},
        "criterion": {"min_progress_m": 11},
    }
    if change == "source":
        original = qualification.course_contract
        monkeypatch.setattr(
            qualification,
            "course_contract",
            lambda courses=None: {**original(courses), "criteria_source_sha256": "changed"},
        )
    else:
        monkeypatch.setattr(qualification, "COURSES", (replace(course, **changes[change]),))
    with pytest.raises(ValueError, match="criteria changed"):
        executions.require_qualified(sut)


def test_missing_or_changed_trace_invalidates_gate_and_cannot_be_recertified(accepted):
    root, sut, _ = accepted
    path = root / "short.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload["trajectory"][3]["actors"][0]["x"] += 0.01
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    with pytest.raises(ValueError, match="trajectory changed"):
        executions.require_qualified(sut)
    with pytest.raises(ValueError, match="replay differs"):
        qualification.replay_courses(root)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        executions.require_qualified(sut)


def test_replay_does_not_accept_a_different_initial_scene(accepted, monkeypatch):
    root, _, (course,) = accepted
    monkeypatch.setattr(
        qualification, "COURSES", (replace(course, scenario=brake_scene("different", gap=60)),)
    )
    with pytest.raises(ValueError, match="initial state differs"):
        qualification.replay_courses(root)


def test_shortened_archive_cannot_pass_by_keeping_the_old_summary(accepted):
    root, _, _ = accepted
    path = root / "short.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload["trajectory"] = payload["trajectory"][:5]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    with pytest.raises(ValueError, match="Truncated"):
        qualification.replay_courses(root)
