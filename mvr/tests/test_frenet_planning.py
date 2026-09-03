from __future__ import annotations

import numpy as np
import pytest

from mvr.scenario.frenet import (
    ANCHOR_PROGRESS,
    FrenetManeuverContract,
    FrenetPathPlanner,
    decode_frenet_path,
    quintic_smoothstep,
)
from mvr.scenario.route_geometry import RoutePolyline


def _straight_route(length_m: float = 120.0) -> RoutePolyline:
    x = np.linspace(0.0, length_m, 121, dtype=float)
    points = np.column_stack((x, np.zeros_like(x)))
    return RoutePolyline(
        (("a", "b", 0),),
        points,
        x,
        (length_m,),
    )


def _contract(start_lateral_m: float = 3.5) -> FrenetManeuverContract:
    return FrenetManeuverContract(
        _straight_route(),
        10.0,
        start_lateral_m,
        0.0,
        30.0,
        60.0,
        -0.5,
        4.0,
        20.0,
        True,
    )


def test_seventh_degree_path_satisfies_all_eight_constraints() -> None:
    contract = _contract()
    path = decode_frenet_path(contract, (0.25, 0.8, -0.7))
    for s_m, expected in (
        (contract.start_s_m, contract.start_lateral_m),
        (path.end_s_m, contract.end_lateral_m),
    ):
        _, lateral, slope, second = path.evaluate(s_m)
        assert lateral == pytest.approx(expected, abs=1e-9)
        assert slope == pytest.approx(0.0, abs=1e-9)
        assert second == pytest.approx(0.0, abs=1e-9)
    early = path.evaluate(contract.start_s_m + ANCHOR_PROGRESS[0] * path.length_m)[1]
    late = path.evaluate(contract.start_s_m + ANCHOR_PROGRESS[1] * path.length_m)[1]
    assert early != pytest.approx(late)


def test_zero_shape_action_exactly_recovers_quintic_lane_change() -> None:
    contract = _contract()
    path = decode_frenet_path(contract, (0.0, 0.0, 0.0))
    for progress in np.linspace(0.0, 1.0, 21):
        lateral = path.evaluate(
            contract.start_s_m + progress * path.length_m
        )[1]
        expected = contract.start_lateral_m + (
            contract.end_lateral_m - contract.start_lateral_m
        ) * float(quintic_smoothstep(progress))
        assert lateral == pytest.approx(expected, abs=1e-8)


def test_length_and_control_points_are_significant_and_deterministic() -> None:
    contract = _contract()
    shortest = decode_frenet_path(contract, (-1.0, 0.0, 0.0))
    longest = decode_frenet_path(contract, (1.0, 0.0, 0.0))
    assert shortest.length_m == pytest.approx(30.0)
    assert longest.length_m == pytest.approx(60.0)
    early_low = decode_frenet_path(contract, (0.0, -1.0, 0.0))
    early_high = decode_frenet_path(contract, (0.0, 1.0, 0.0))
    s_early = contract.start_s_m + ANCHOR_PROGRESS[0] * early_low.length_m
    displacement = abs(
        early_high.evaluate(s_early)[1] - early_low.evaluate(s_early)[1]
    )
    assert displacement >= 0.5
    np.testing.assert_allclose(
        early_high.coefficients,
        decode_frenet_path(contract, (0.0, 1.0, 0.0)).coefficients,
    )


def test_non_replan_steps_lock_shape_but_accept_longitudinal_action() -> None:
    contract = _contract()
    planner = FrenetPathPlanner(contract)
    first = planner.apply(np.asarray((1.0, 0.5, -0.5, -1.0)), True, 10.0)
    second = planner.apply(np.asarray((-1.0, -1.0, 1.0, 0.75)), True, 11.0)
    np.testing.assert_allclose(second[:3], first[:3])
    assert second[3] == pytest.approx(0.75)


def test_planner_projects_endpoint_ahead_of_current_vehicle() -> None:
    contract = _contract()
    planner = FrenetPathPlanner(contract)
    action = planner.apply(np.asarray((-1.0, 0.0, 0.0, 0.0)), True, 45.0)
    decoded = decode_frenet_path(contract, action[:3])
    assert decoded.end_s_m >= 57.0 - 1e-6
