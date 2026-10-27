from dataclasses import replace

import numpy as np
import pytest

from controllers.highway.catalog import BY_ID
from controllers.highway.control import Control
from controllers.highway.rule import RuleController
from experiments.scenarios import sample
from simulator.highway.collision_env import CollisionWorld, execute
from simulator.highway.observation import VehicleState, VisibleWorld


class FullBrake:
    def act(self, world):
        return Control(-8, 0)


def test_shared_actuator_can_stop_and_decisions_are_5_hz():
    scene = sample("front_brake", 1, "qualification")[0]
    scene = replace(
        scene,
        parameters={**scene.parameters, "ego_speed_mps": 18, "gap_m": 65, "closing_speed_mps": -4},
    )
    world = CollisionWorld(scene, duration_s=4)
    while not world.terminal:
        world.advance(Control(-8, 0))
    assert world.ego.speed == 0
    assert min(row["actors"][0]["speed"] for row in world.trace) >= 0
    assert world.tick == 80
    assert world.decisions == 20
    assert not world.ego.crashed
    assert all(row["control"]["acceleration"] >= -8 for row in world.trace)
    assert np.diff([r["time_s"] for r in world.trace]) == pytest.approx(0.05)


@pytest.mark.parametrize("kind", ["front_brake", "cut_in", "cut_out", "lane_change_rear"])
def test_four_scenarios_have_deterministic_world_and_continuous_background(kind):
    scene = sample(kind, 1, "qualification")[0]
    first = CollisionWorld(scene, duration_s=4)
    second = CollisionWorld(scene, duration_s=4)
    while not first.terminal:
        first.advance(Control(-8, 0))
        second.advance(Control(-8, 0))
    assert first.trace == second.trace
    assert first.ego.speed == second.ego.speed
    lead_y = np.array([row["actors"][1]["y"] for row in first.trace])
    assert np.abs(np.diff(lead_y)).max() < 0.25
    if kind == "lane_change_rear":
        assert {r for r in first.actors} == {"ego", "lead", "rear", "adjacent_front"}
        assert first.lane_centers == (0.0, 4.0)


def test_cutout_obstacle_is_initially_hidden_and_then_revealed_geometrically():
    scene = sample("cut_out", 1, "qualification")[0]
    scene = replace(
        scene,
        parameters={
            "ego_speed_mps": 18,
            "gap_m": 30,
            "lead_speed_mps": 20,
            "obstacle_distance_m": 160,
            "event_time_s": 0.5,
            "lateral_duration_s": 1.8,
        },
    )
    world = CollisionWorld(scene, duration_s=4)
    assert "obstacle" not in world.trace[0]["visible_ids"]
    while not world.terminal:
        world.advance(Control(-8, 0))
    assert world.revealed_at > 0.5
    assert world.revealed_at < 2.3
    assert "obstacle" in world.trace[-1]["visible_ids"]


@pytest.mark.parametrize("route", ["idm_mobil", "fvdm_lane"])
def test_rule_controllers_use_front_and_rear_constraints(route):
    sut = BY_ID[f"history_{route}"]
    controller = RuleController(route, sut.parameters)
    ego = VehicleState("ego", 0, 4, 25, 0)
    slow = VehicleState("lead", 25, 4, 12, 0)
    rear = VehicleState("rear", -9, 0, 38, 0)
    front = VehicleState("adjacent_front", 15, 0, 20, 0)
    world = VisibleWorld(0, ego, (slow, rear, front), (0.0, 4.0))
    command = controller.act(world)
    assert controller.target_lane == 1
    assert command.acceleration < 0
    assert command.steering == 0
    safe_world = VisibleWorld(3, ego, (slow,), (0.0, 4.0))
    controller.act(safe_world)
    assert controller.target_lane == 0
    # Full braking is available without using a target-speed floor.
    standstill = VehicleState("stopped", 7, 4, 0, 0)
    assert controller.acceleration(ego, standstill) == -8


def test_query_feedback_and_private_trace_are_separate():
    scene = sample("front_brake", 1, "qualification")[0]
    execution = execute(FullBrake(), scene, duration_s=0.4)
    assert execution.feedback.execution_s > 0
    assert isinstance(execution.feedback.collision, bool)
    assert execution.trajectory[0]["actors"][0]["actor_id"] == "ego"
    assert not hasattr(execution.feedback, "trajectory")
