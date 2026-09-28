"""The release calibrator must never select from tester performance."""

from __future__ import annotations

import json

import pytest

from methods.failure_memory_regression.nl_calibration import BUILDS, choose
from sut_algorithms.highway_env.registry import build_spec_factory


def test_progress_rule_ignores_tester_scores(tmp_path):
    case = {"scenario_id": "development-only"}
    (tmp_path / "scenario_manifest.jsonl").write_text(json.dumps(case) + "\n",
                                                       encoding="utf-8")
    protocol = {"build_fingerprints": {build: build_spec_factory(build).fingerprint
                                       for build in BUILDS}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    speeds = {"nl_v0": 10.0, "nl_eff_c1": 12.0, "nl_eff_c2": 11.0,
              "nl_eff_c3": 9.0}
    rows = [{"scenario_id": case["scenario_id"], "build_id": build,
             "build_fingerprint": protocol["build_fingerprints"][build],
             "ego_collision": False, "completed": True, "inconclusive": False,
             "progress_rate_mps": speeds[build],
             "tester_discovery": 1000 if build == "nl_eff_c3" else 0}
            for build in BUILDS]
    (tmp_path / "physical_episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = choose(tmp_path)
    assert result["selected"] == "nl_eff_c1"
    assert result["selection_used_tester_results"] is False


def test_calibration_refuses_inconclusive_episode(tmp_path):
    case = {"scenario_id": "development-only"}
    (tmp_path / "scenario_manifest.jsonl").write_text(json.dumps(case) + "\n",
                                                       encoding="utf-8")
    protocol = {"build_fingerprints": {build: build_spec_factory(build).fingerprint
                                       for build in BUILDS}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    rows = [{"scenario_id": case["scenario_id"], "build_id": build,
             "build_fingerprint": protocol["build_fingerprints"][build],
             "ego_collision": False, "completed": True, "inconclusive": False,
             "progress_rate_mps": 10.0} for build in BUILDS]
    rows[-1]["inconclusive"] = True
    (tmp_path / "physical_episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="inconclusive"):
        choose(tmp_path)
