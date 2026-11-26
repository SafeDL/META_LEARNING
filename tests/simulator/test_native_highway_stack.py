"""Independent regression oracles for the native simulator/controller contract."""

import json
import math
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import RoadObject

from controllers.highway.catalog import BY_ID
from controllers.highway.contracts import runtime_contract
from controllers.highway.control import Control
from controllers.highway.factory import create
from controllers.highway.optimization import OptimizationController
from controllers.highway.planning import bicycle_rollout
from controllers.highway.rule import RuleController
from controllers.highway.search import SearchController, transition
from experiments.qualification import brake_scene
from experiments.scenarios import sample
from simulator.highway.collision_env import (
    ENV_ID,
    CollisionWorld,
    ForwardVehicle,
    ScheduledVehicle,
    vehicle_state,
)
from simulator.highway.observation import VehicleState, VisibleWorld


@pytest.mark.parametrize("route", ["idm_mobil", "fvdm_lane"])
def test_first_lane_change_brakes_for_the_still_occupied_original_lane(route):
    ego = VehicleState("ego", 100, 4, 25, 0)
    lead = VehicleState("lead", 125, 4, 12, 0)
    controller = RuleController(route, BY_ID[f"independent_{route}"].parameters)
    command = controller.act(VisibleWorld(0, ego, (lead,)))
    assert controller.target_lane != 1
    assert command.acceleration == -8
    assert command.steering != 0


def test_fvdm_optimal_velocity_is_monotone_in_spacing_and_has_no_ego_speed_argument():
    controller = RuleController("fvdm_lane", BY_ID["independent_fvdm_lane"].parameters)
    values = [controller.optimal_velocity(gap) for gap in np.linspace(0, 200, 100)]
    assert values[0] == 0
    assert np.all(np.diff(values) >= 0)
    assert values[-1] == pytest.approx(controller.parameters["desired_speed_mps"])
    # Exact formula oracle with independently substituted fixed model parameters.
    desired, headway = 29, 1.8
    s0 = 2 + headway * desired / 2
    expected = desired * (math.tanh((30 - s0) / 10) + math.tanh(s0 / 10)) / (1 + math.tanh(s0 / 10))
    assert controller.optimal_velocity(30) == pytest.approx(expected)


def test_idm_acceleration_matches_independently_substituted_equation():
    p = BY_ID["history_idm_mobil"].parameters
    controller = RuleController("idm_mobil", p)
    ego, lead = VehicleState("ego", 0, 4, 20, 0), VehicleState("lead", 50, 4, 18, 0)
    wanted = 2 + max(0, 20 * 1.6 + 20 * 2 / (2 * math.sqrt(2 * 3)))
    expected = 2 * (1 - (20 / 30) ** 4 - (wanted / 45) ** 2)
    assert controller.acceleration(ego, lead) == pytest.approx(expected)


def test_registered_environment_uses_native_environment_and_physics_for_every_actor(monkeypatch):
    counts = {}
    for cls, name in [
        (AbstractEnv, "step"),
        (AbstractEnv, "_simulate"),
        (Road, "act"),
        (Road, "step"),
        (Vehicle, "step"),
        (RoadObject, "handle_collisions"),
    ]:
        original = getattr(cls, name)
        key = cls.__name__ + "." + name

        def wrapper(*args, _original=original, _key=key, **kwargs):
            counts[_key] = counts.get(_key, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(cls, name, wrapper)
    env = gym.make(
        ENV_ID, scenario=sample("lane_change_rear", 1, "qualification")[0], duration_s=0.4
    )
    env.reset(seed=123)
    for _ in range(2):
        env.step(np.array([-1.0, 0.0], dtype=np.float32))
    world = env.unwrapped
    assert isinstance(world, AbstractEnv)
    assert counts["AbstractEnv.step"] == counts["AbstractEnv._simulate"] == 2
    assert counts["Road.act"] == counts["Road.step"] == 8
    assert counts["Vehicle.step"] == 8 * 4  # ego plus THREE native background vehicles
    assert counts["RoadObject.handle_collisions"] == 8 * 6
    assert world.steps == world.tick == 8 and world.decisions == 2
    assert world.time == pytest.approx(0.4)
    assert world.trace[-1]["requested_control"]["acceleration"] == -8
    env.close()


@pytest.mark.parametrize("kind", ["car_following", "front_brake", "cut_out", "lane_change_rear"])
@pytest.mark.parametrize(
    "route",
    ["idm_mobil", "fvdm_lane", "linear_mpc", "nonlinear_trajectory", "uct", "beam", "ppo", "dqn"],
)
def test_every_target_ads_and_original_scenario_calls_native_environment_step(
    kind, route, monkeypatch
):
    from simulator.highway.collision_env import execute

    calls = []
    original = AbstractEnv.step

    def observed(env, action):
        calls.append(type(env).__name__)
        return original(env, action)

    monkeypatch.setattr(AbstractEnv, "step", observed)
    paid = execute(
        create(f"independent_{route}"), sample(kind, 1, "qualification")[0], duration_s=0.4
    )
    assert calls == ["CollisionWorld", "CollisionWorld"]
    assert paid.trajectory[-1]["time_s"] == 0.4


def test_background_consumes_native_pending_impacts_and_records_actual_partner():
    world = CollisionWorld(brake_scene("impact"), duration_s=1)
    lead = world.actors["lead"]
    assert ScheduledVehicle.step is ForwardVehicle.step
    lead.impact = np.array([0.1, 0.0])
    lead._impact_partner = "ego"
    lead.step(0.05)
    assert lead.crashed and lead.impact is None and lead._crash_partner == "ego"
    # Direct native callback provides the actor ID even with no nearest-distance heuristic.
    lead.crashed = False
    lead.position = world.ego.position.copy()
    world.ego.handle_collisions(lead, 0.05)
    assert world.ego.crashed and world.ego._crash_partner == "lead"


@pytest.mark.parametrize("sut_id", ["independent_idm_mobil", "independent_ppo"])
def test_collection_gate_rejects_old_or_changed_runtime_qualifications(
    tmp_path, monkeypatch, sut_id
):
    from experiments import executions

    sut = BY_ID[sut_id]
    monkeypatch.setattr(executions, "ROOT", tmp_path)
    monkeypatch.setattr(executions, "QUALIFICATION_ROOT", tmp_path)
    path = tmp_path / sut_id / "qualification.json"
    path.parent.mkdir()
    report = {"qualified": True, "parameters": sut.parameters}
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="current native highway-env contract"):
        executions.require_qualified(sut)
    report["runtime_contract"] = runtime_contract(sut)
    report["courses"] = []
    monkeypatch.setattr(executions, "verify_course_evidence", lambda _: {"courses": []})
    path.write_text(json.dumps(report), encoding="utf-8")
    assert executions.require_qualified(sut) == str(path.relative_to(executions.ROOT))
    report["runtime_contract"]["source_files"]["simulator/highway/collision_env.py"] = "changed"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="current native highway-env contract"):
        executions.require_qualified(sut)


def test_random_bicycle_predictions_match_native_execution_including_saturation_and_full_stop():
    generator = np.random.default_rng(71031)
    for _ in range(32):
        world = CollisionWorld(brake_scene("model"), duration_s=1)
        world.road.vehicles = [world.ego]
        world.actors = {"ego": world.ego}
        world.ego.speed = float(generator.uniform(0, 38))
        world.ego.heading = float(generator.uniform(-0.15, 0.15))
        controls = generator.uniform([-8, -0.45], [3, 0.45], (5, 2))
        initial = vehicle_state("ego", world.ego)
        prediction = bicycle_rollout(initial, controls)
        for step, (acceleration, steering) in enumerate(controls):
            # Native env may flag road departure; use its native physics directly
            # so all five input intervals remain comparable to the model oracle.
            world.ego.act({"acceleration": acceleration, "steering": steering})
            for _ in range(4):
                world.road.act()
                world.road.step(0.05)
            actual = [*world.ego.position, world.ego.speed, world.ego.heading]
            np.testing.assert_allclose(actual, prediction[step], rtol=0, atol=2e-12)


@pytest.mark.parametrize("route", ["linear_mpc", "nonlinear_trajectory"])
def test_optimizers_reject_a_native_infeasible_candidate_even_if_solver_reports_success(
    route, monkeypatch
):
    from types import SimpleNamespace

    controller = OptimizationController(route, BY_ID[f"history_{route}"].parameters)
    world = VisibleWorld(
        0, VehicleState("ego", 0, 4, 25, 0), (VehicleState("stopped", 20, 4, 0, 0),), (4.0,)
    )
    fake = SimpleNamespace(x=np.zeros(2 * controller.horizon), fun=0, nit=1, message="success")
    method = "_linear_solve" if route == "linear_mpc" else "_nonlinear_solve"
    monkeypatch.setattr(controller, method, lambda *args: (fake, True))
    assert controller.act(world).acceleration == -8
    assert controller.last_diagnostics["emergency_brake"]
    assert not controller.last_diagnostics["solves"][0]["native_prediction_feasible"]


def test_search_checks_subdecision_collisions():
    # An object travels through the ego between 0 and .2s, but is separate at .2s.
    world = VisibleWorld(
        0, VehicleState("ego", 0, 4, 0, 0), (VehicleState("fast", -10, 4, 100, 0),), (4.0,)
    )
    _, reward, terminal = transition(world, (0, 0), 25)
    assert terminal and reward < -1000


@pytest.mark.parametrize("route", ["uct", "beam"])
def test_search_chooses_delayed_reward_over_greedy_reward(route, monkeypatch):
    from controllers.highway import search

    monkeypatch.setattr(search, "actions", lambda world: [(0.0, 0), (1.0, 0)])

    def toy_transition(world, action, desired):
        if world.time_s == 0:
            return replace(world, time_s=1, ego=replace(world.ego, x=action[0])), action[0], False
        return replace(world, time_s=2), (10.0 if world.ego.x == 0 else -100.0), True

    monkeypatch.setattr(search, "transition", toy_transition)
    controller = SearchController(
        route,
        {
            "horizon_steps": 2,
            "iterations": 128,
            "exploration": 1.4,
            "desired_speed_mps": 25,
            "beam_width": 4,
        },
    )
    world = VisibleWorld(0, VehicleState("ego", 0, 4, 25, 0), (), (4.0,))
    action = controller._uct(world) if route == "uct" else controller._beam(world)
    assert action == (0.0, 0)


def test_uct_discounted_backpropagation_matches_an_exact_two_edge_return(monkeypatch):
    from controllers.highway import search

    monkeypatch.setattr(search, "actions", lambda world: [(0.0, 0)])

    def toy_transition(world, action, desired):
        return replace(world, time_s=world.time_s + 1), (2.0 if world.time_s == 0 else 3.0), False

    monkeypatch.setattr(search, "transition", toy_transition)
    controller = SearchController(
        "uct", {"horizon_steps": 2, "iterations": 8, "exploration": 1.4, "desired_speed_mps": 25}
    )
    controller._uct(VisibleWorld(0, VehicleState("ego", 0, 4, 25, 0), (), (4.0,)))
    child = controller.last_diagnostics["root_children"][0]
    assert child["visits"] == 8
    assert child["mean_return"] == pytest.approx(2 + 0.98 * 3)


def test_planners_accept_nonzero_heading_linearization_at_its_reference():
    controller = OptimizationController("linear_mpc", BY_ID["history_linear_mpc"].parameters)
    ego = VehicleState("ego", 100, 4, 25, 0.15)
    reference = np.tile([-1.0, -0.01], controller.horizon)
    base, jac = controller._linear_model(ego, reference)
    exact = bicycle_rollout(ego, reference.reshape(-1, 2))
    np.testing.assert_allclose(base + np.einsum("nij,j->ni", jac, reference), exact, atol=1e-10)


@pytest.mark.parametrize(
    "kind", ["front_brake_single_lane", "stopped_vehicle", "cut_in_interaction"]
)
def test_supplementary_scenarios_are_deterministic_and_have_no_initial_overlap(kind):
    scene = sample(kind, 1, "qualification")[0]
    a, b = CollisionWorld(scene), CollisionWorld(scene)
    for world in (a, b):
        world.advance(Control(-8, 0))
    assert a.trace == b.trace
    assert all(
        not s.polygon().intersects(t.polygon())
        for i, s in enumerate(a.states())
        for t in a.states()[i + 1 :]
    )
    if kind in ("front_brake_single_lane", "stopped_vehicle"):
        assert a.lane_centers == (4.0,)
        assert len(a.road.network.graph["0"]["1"]) == 1
