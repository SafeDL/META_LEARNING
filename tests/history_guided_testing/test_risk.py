import math

import numpy as np
import pytest

from methods.history_guided_testing.risk import Body, rectangle_ttc, longitudinal_drac, pair_risk, PassiveRiskObserver


def body(x=0, y=0, vx=0, vy=0, heading=0):
    return Body(np.array([x, y], float), np.array([vx, vy], float), heading)


def test_analytic_collinear():
    ego, lead = body(vx=25), body(x=25, vx=20)
    assert rectangle_ttc(ego, lead) == pytest.approx(4)
    assert longitudinal_drac(ego, lead)[0] == pytest.approx(25 / 40)
    r = pair_risk(ego, lead)
    assert r["body_distance"] == pytest.approx(20)
    assert r["risk"] == pytest.approx(np.linalg.norm([1.5 / 5.5, .625 / 3.625, 1 / 21]) / math.sqrt(3))


def test_overlap_no_conflict_and_lateral_closing():
    assert rectangle_ttc(body(), body(x=1)) == 0
    assert math.isinf(rectangle_ttc(body(vx=10), body(x=20, y=4, vx=0)))
    assert longitudinal_drac(body(vx=10), body(x=20, y=4))[0] == 0
    assert longitudinal_drac(body(vy=10), body(x=20))[0] == 0
    assert pair_risk(body(vx=10), body(x=1))["risk"] == pytest.approx(1)


def test_rotation_translation_invariance():
    a, b = body(vx=25), body(x=25, vx=20)
    theta = .77
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    def transform(x):
        return Body(rot @ x.position + [100, -50], rot @ x.velocity, x.heading + theta)
    assert pair_risk(a, b)["risk"] == pytest.approx(pair_risk(transform(a), transform(b), road_heading=theta)["risk"], abs=1e-12)


def test_monotonic_approach_and_gap():
    assert pair_risk(body(vx=25), body(x=15, vx=20))["risk"] > pair_risk(body(vx=25), body(x=25, vx=20))["risk"]
    assert pair_risk(body(vx=30), body(x=25, vx=20))["risk"] > pair_risk(body(vx=25), body(x=25, vx=20))["risk"]


def test_lateral_crossing_and_prediction_horizon():
    a, b = body(), body(y=10, vy=-2)
    assert rectangle_ttc(a, b) == pytest.approx(4)
    assert math.isinf(rectangle_ttc(a, b, horizon=3))
    assert longitudinal_drac(a, b)[0] == 0


def test_invalid_is_not_safe():
    with pytest.raises(ValueError):
        pair_risk(body(vx=np.nan), body())
    obs = PassiveRiskObserver({"ttc_scale_s": 1.5, "drac_scale_mps2": 3., "gap_scale_m": 1., "ttc_prediction_horizon_s": 6})
    assert obs.summary()["risk"] is None
    assert obs.summary()["valid_risk"] is False


def test_synchronous_fusion_not_independent_extrema():
    pairs = [pair_risk(body(vx=25), body(x=25, vx=20)), pair_risk(body(vx=20), body(x=6, vx=25))]
    synchronous = max(r["risk"] for r in pairs)
    independent = np.linalg.norm(np.max([r["components"] for r in pairs], axis=0)) / np.sqrt(3)
    assert independent > synchronous
