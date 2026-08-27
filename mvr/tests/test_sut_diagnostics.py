from __future__ import annotations

from mvr.scripts.diagnose_sut import collect


def test_sut_only_lane_stability_diagnostic_passes_all_families() -> None:
    report = collect()
    assert report["passed"]
    for family in report["families"]:
        assert family["route_completion"]
        assert not family["out_of_road"]
        assert family["routing_target_lane_mismatch"] == 0
        assert family["lateral_rms_m"] <= 0.30
        assert not family["sustained_steering_sign_oscillation"]
        assert family["records"]
