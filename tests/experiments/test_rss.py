"""Fixed Intel parameters and ADS-centred RSS distance margins."""

import math
from dataclasses import replace

import pytest

from experiments.rss import (
    DEFAULT,
    frame_margin,
    lateral_distance,
    longitudinal_distance,
    measurement_contract,
    pair_margin,
    rss_values,
    trajectory_margin,
)


def car(name="ego", *, x=0, y=0, speed=20, heading=0):
    return dict(actor_id=name, x=x, y=y, speed=speed, heading=heading, length=5, width=2)


def test_distance_matches_published_braking_equations():
    assert longitudinal_distance(20, 20) == pytest.approx(20 + 1.75 + 23.5**2 / 8 - 400 / 16)
    assert longitudinal_distance(20, 20, rear_is_ego=False) == pytest.approx(
        40 + 7 + 27**2 / 8 - 400 / 16
    )
    assert lateral_distance(0, 0) == pytest.approx(0.725)
    assert longitudinal_distance(20, 0) > longitudinal_distance(20, 20)


def test_outward_lateral_motion_has_no_fictitious_inward_stopping_distance():
    assert lateral_distance(-3, 3) == pytest.approx(DEFAULT.lateral_buffer_m)
    assert lateral_distance(-3, 3) < lateral_distance(0, 0) < lateral_distance(3, -3)


def test_rear_impact_risk_and_role_symmetry():
    ego, rear = car(speed=15), car("rear", x=-20, speed=30)
    assert pair_margin(ego, rear)["margin"] < 0
    assert pair_margin(ego, rear)["margin"] == pair_margin(rear, ego)["margin"]
    assert pair_margin(ego, {**rear, "x": -200})["margin"] < 0
    assert pair_margin(ego, {**rear, "x": -300})["margin"] > 0


def test_intel_response_roles_follow_the_evaluated_ads_not_the_front_rear_order():
    ego = car()
    ahead = pair_margin(ego, car("lead", x=80))
    behind = pair_margin(ego, car("rear", x=-80))
    assert ahead["required_center_separation_m"]["actor_ahead"] == pytest.approx(70.78125)
    assert behind["required_center_separation_m"]["actor_behind"] == pytest.approx(118.125)
    assert ahead["margin"] > 0 > behind["margin"]
    lower = pair_margin(ego, car("side", y=4, heading=-0.05))
    upper = pair_margin(ego, car("side", y=-4, heading=0.05))
    assert lower["margin"] == pytest.approx(upper["margin"])


def test_parallel_separation_and_lateral_approach():
    ego = car()
    parallel = car("adjacent", x=0, y=4)
    approach = car("adjacent", x=0, y=4, heading=-0.1)
    assert pair_margin(ego, parallel)["margin"] > 0
    assert pair_margin(ego, approach)["margin"] < 0


def test_equal_speed_close_following_is_dangerous_despite_infinite_ttc():
    assert pair_margin(car(), car("lead", x=15))["margin"] < 0
    assert pair_margin(car(), car("lead", x=100))["margin"] > 0


def test_velocity_and_response_change_margin_at_identical_clearance():
    ego, lead = car(), car("lead", x=50)
    slow = pair_margin(ego, lead)["margin"]
    fast = pair_margin({**ego, "speed": 25}, lead)["margin"]
    delayed = pair_margin(ego, lead, replace(DEFAULT, ego_response_s=1.5))["margin"]
    assert fast < slow
    assert delayed < slow


def test_normalization_preserves_close_following_information_and_zero_distance():
    ego = car()
    close = pair_margin(ego, car("lead", x=15))["margin"]
    farther = pair_margin(ego, car("lead", x=25))["margin"]
    assert -1 < close < farther < 0
    stopped = pair_margin(
        car(speed=0), car("lead", x=10, speed=0), replace(DEFAULT, ego_response_s=0)
    )
    assert stopped["margin"] == pytest.approx(1)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_order_crossings_are_continuous(axis):
    ego, other = car(), car("other", speed=23, heading=0.05)
    before = pair_margin(ego, {**other, axis: -1e-8})["margin"]
    after = pair_margin(ego, {**other, axis: 1e-8})["margin"]
    assert abs(after - before) <= 3e-8


def test_translation_and_actor_order_invariance():
    ego, lead, rear = car(), car("lead", x=100), car("rear", x=-15, speed=30)
    frames = [dict(time_s=0, actors=[lead, ego, rear])]
    value = trajectory_margin(frames)
    shifted = [{**a, "x": a["x"] + 234, "y": a["y"] - 20} for a in frames[0]["actors"]]
    assert trajectory_margin([dict(time_s=0, actors=shifted)])["risk"] == value["risk"]
    assert value["critical_actor"] == "rear"


def test_sampled_minimum_and_empty_traffic():
    a = dict(time_s=0, actors=[car(), car("lead", x=100)])
    b = dict(time_s=1, actors=[car(), car("lead", x=15)])
    value = trajectory_margin([a, b])
    assert value["first_negative_time_s"] == 1
    assert value["minimum_margin"] == value["series"][1]["margin"]
    assert trajectory_margin([dict(time_s=0, actors=[car()])])["risk"] == -math.inf
    with pytest.raises(ValueError, match="nonnegative road"):
        pair_margin(car(), car("oncoming", heading=math.pi))


def test_npc_npc_collision_does_not_create_an_ads_safety_violation():
    ego, a, b = car(), car("a", x=1000), car("b", x=1000.1)
    assert pair_margin(a, b)["margin"] < 0
    only_a = frame_margin(dict(time_s=0, actors=[ego, a]))
    both = frame_margin(dict(time_s=0, actors=[ego, a, b]))
    assert both["margin"] == only_a["margin"] > 0
    assert {pair["actor_id"] for pair in both["pairs"]} == {"a", "b"}


def test_one_contract_and_signed_rss_values_preserve_the_physical_boundary():
    from dataclasses import FrozenInstanceError

    assert measurement_contract()["parameters"] == DEFAULT.__dict__
    with pytest.raises(FrozenInstanceError):
        DEFAULT.ego_response_s = 0.2
    assert rss_values([-1, -0.2, 0, 0.5, 100]).tolist() == [-1, -0.2, 0, 0.5, 100]
    for invalid in (math.inf, math.nan, -1.01):
        with pytest.raises(ValueError, match="normalized margins"):
            rss_values(invalid)
