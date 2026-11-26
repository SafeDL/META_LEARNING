import math

import numpy as np
import pytest

from experiments.domain_risk import disclose_driving_quality
from experiments.query import PaidExecution
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.scenario_fuzz.driving_quality import (
    penalties,
    trajectory_score,
    understeer_level,
)


def trace(speed=20, frames=201):
    return [
        {
            "time_s": i / 20,
            "collision": False,
            "actors": [
                {"actor_id": "ego", "x": speed * i / 20, "y": 4, "heading": 0, "speed": speed},
                {
                    "actor_id": "other",
                    "x": speed * i / 20 + 150,
                    "y": 4,
                    "heading": 0,
                    "speed": speed,
                },
            ],
            "control": {"steering": 0},
        }
        for i in range(frames)
    ]


def test_constant_velocity_straight_motion_has_no_driving_deductions():
    score, deductions = trajectory_score(trace())
    assert score == 0
    assert all(value == 0 for value in deductions.values())
    paid = PaidExecution(
        QueryFeedback("scene", False, SafetyMeasurements(145, math.inf), 1), trace()
    )
    disclosed = disclose_driving_quality(paid)
    assert disclosed.feedback.safety.driving_quality_score == 0
    assert paid.feedback.safety.driving_quality_score is None
    assert QueryFeedback.from_dict(disclosed.feedback.to_dict()) == disclosed.feedback


def test_hard_braking_uses_original_kmh_per_frame_threshold_and_minimum_distance_term():
    speed = np.linspace(120, 0, 60)
    zeros = np.zeros(60)
    score, deductions = penalties(speed, speed, zeros, zeros, zeros, 5)
    assert deductions["hard_braking"] == 59
    assert deductions["minimum_distance"] == pytest.approx(0.2)
    assert score == pytest.approx(-59.2)
    assert understeer_level(0) == 0


def test_crash_stops_before_native_successor_scoring_and_short_safe_trace_is_rejected():
    paid = PaidExecution(QueryFeedback("crash", True, SafetyMeasurements(0, 0), 1), [])
    assert disclose_driving_quality(paid).feedback.safety.driving_quality_score is None
    with pytest.raises(ValueError, match="20 Hz"):
        trajectory_score(trace(frames=3))
