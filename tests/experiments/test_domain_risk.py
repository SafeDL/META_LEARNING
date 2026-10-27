import math
from dataclasses import replace

import pytest

from experiments.domain_risk import disclose_impact
from experiments.query import PaidExecution
from experiments.schema import QueryFeedback, SafetyMeasurements


def execution(collision, frames):
    return PaidExecution(QueryFeedback("scene", collision, SafetyMeasurements(0, 0), 1), frames)


def frame(ego_speed, other_speed, other_heading=0, *, collision=True):
    return {
        "collision": collision,
        "collision_partner": "other" if collision else None,
        "actors": [
            {"actor_id": "ego", "speed": ego_speed, "heading": 0},
            {"actor_id": "other", "speed": other_speed, "heading": other_heading},
        ],
    }


@pytest.mark.parametrize(
    "speeds, expected",
    [
        ((30, 0, 0), 15),
        ((30, 20, 0), 5),
        ((10, 10, math.pi / 2), math.sqrt(50)),
        ((10, 10, math.pi), 10),
    ],
)
def test_impact_uses_first_contact_relative_vector_and_equal_mass_formula(speeds, expected):
    paid = execution(True, [frame(30, 30, collision=False), frame(*speeds), frame(0, 0)])
    disclosed = disclose_impact(paid)
    assert disclosed.feedback.safety.impact_delta_v_mps == pytest.approx(expected)
    assert paid.feedback.safety.impact_delta_v_mps is None
    assert disclosed.trajectory is paid.trajectory
    assert QueryFeedback.from_dict(disclosed.feedback.to_dict()) == disclosed.feedback


def test_noncollision_discloses_no_impact_and_mismatched_or_unidentified_collision_fails():
    assert (
        disclose_impact(
            execution(False, [frame(20, 20, collision=False)])
        ).feedback.safety.impact_delta_v_mps
        is None
    )
    with pytest.raises(ValueError, match="paid trajectory"):
        disclose_impact(execution(False, [frame(20, 10)]))
    bad = {**frame(20, 10), "collision_partner": None}
    with pytest.raises(ValueError, match="collision partner"):
        disclose_impact(execution(True, [bad]))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(SafetyMeasurements(0, 0), impact_delta_v_mps=math.nan)
