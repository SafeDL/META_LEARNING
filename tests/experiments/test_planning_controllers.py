import numpy as np
import pytest

from controllers.highway.catalog import BY_ID
from controllers.highway.optimization import OptimizationController
from controllers.highway.search import SearchController
from simulator.highway.observation import VehicleState, VisibleWorld


def free_world():
    return VisibleWorld(0, VehicleState("ego", 100, 4, 25, 0), (), (0.0, 4.0, 8.0))


@pytest.mark.parametrize("route", ["linear_mpc", "nonlinear_trajectory"])
def test_optimization_routes_solve_real_constrained_horizon(route):
    sut = BY_ID[f"history_{route}"]
    controller = OptimizationController(route, sut.parameters)
    command = controller.act(free_world())
    assert command.acceleration > 0
    assert -0.45 <= command.steering <= 0.45
    assert any(d["success"] for d in controller.last_diagnostics["solves"])
    assert controller.horizon == 15
    assert controller.last_diagnostics["route"] == route


def test_linear_mpc_rear_constraint_blocks_an_unsafe_lane():
    sut = BY_ID["history_linear_mpc"]
    controller = OptimizationController(sut.lineage, sut.parameters)
    world = VisibleWorld(
        0,
        VehicleState("ego", 100, 4, 25, 0),
        (
            VehicleState("lead", 125, 4, 12, 0),
            VehicleState("rear", 89, 0, 38, 0),
            VehicleState("adjacent_front", 122, 0, 20, 0),
        ),
        (0.0, 4.0),
    )
    command = controller.act(world)
    assert command.acceleration < 0
    assert controller.last_diagnostics.get("selected_lane", 1) == 1


def test_uct_expands_a_tree_and_backpropagates_all_iterations():
    sut = BY_ID["history_uct"]
    controller = SearchController(sut.lineage, sut.parameters)
    command = controller.act(free_world())
    diagnostics = controller.last_diagnostics
    assert diagnostics["maximum_tree_depth"] >= 2
    assert diagnostics["root_visits"] == sut.parameters["iterations"]
    assert sum(c["visits"] for c in diagnostics["root_children"]) == sut.parameters["iterations"]
    assert diagnostics["nodes"] > len(diagnostics["root_children"]) + 1
    assert np.isfinite(command.acceleration)


def test_beam_search_is_deterministic_and_retains_multistep_sequences():
    sut = BY_ID["history_beam"]
    first = SearchController(sut.lineage, sut.parameters)
    second = SearchController(sut.lineage, sut.parameters, seed=999)
    assert first.act(free_world()) == second.act(free_world())
    assert len(first.last_diagnostics["best_sequence"]) == sut.parameters["horizon_steps"]
    assert all(
        n > sut.parameters["beam_width"] for n in first.last_diagnostics["expanded_per_depth"][1:]
    )
