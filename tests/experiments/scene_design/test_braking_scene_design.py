"""Accepted native records remain usable after promoting the scene to formal sampling."""

import pytest

from experiments.scene_design.check_braking import verify_development
from experiments.formal_scenarios import BRAKING_KIND, FREE_BOUNDS, development_coordinates


def test_accepted_scene_keeps_all_ads_and_the_same_coordinate_avoidance_certificate():
    report = verify_development()
    assert sum(row["collisions"] for row in report["rows"]) == 222
    certificate = report["joint_reference_certificate"]
    assert certificate["scenario"]["parameters"] == {
        "ego_speed_mps": 24, "closing_speed_mps": 7, "gap_m": 18, "deceleration_mps2": 8,
    }
    case_id = certificate["scenario"]["scenario_id"]
    for row in report["rows"]:
        assert next(r for r in row["rows"] if r["scenario_id"] == case_id)["collision"]
    assert certificate["summary"]["minimum_body_clearance_m"] > 0.16
    assert certificate["summary"]["violations"] == []
    assert len(development_coordinates(BRAKING_KIND)) == 383


def test_changed_scene_cannot_reuse_the_accepted_development_gate(monkeypatch):
    monkeypatch.setitem(FREE_BOUNDS, BRAKING_KIND, {**FREE_BOUNDS[BRAKING_KIND], "gap_m": (24, 60)})
    with pytest.raises(ValueError, match="does not match the current scene"):
        verify_development()
