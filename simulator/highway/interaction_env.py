"""Paired cut-in interaction with optional constant-speed adjacent traffic."""

import math
import time

from experiments.query import PaidExecution
from experiments.scenarios import validate
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario

from .collision_env import CollisionWorld

GUARD_FRONT_GAP_M = 10.0
GUARD_REAR_GAP_M = 10.0


class InteractionWorld(CollisionWorld):
    def __init__(self, scenario, *, environment_seed=81001, duration_s=10):
        validate(scenario)
        if scenario.kind != "cut_in_interaction":
            raise ValueError("InteractionWorld requires the paired cut-in task")
        base = Scenario(
            scenario.scenario_id,
            "cut_in",
            scenario.parameters,
            {"side": scenario.categories["side"]},
        )
        super().__init__(base, environment_seed=environment_seed, duration_s=duration_s)
        self.scenario = scenario
        if scenario.categories["adjacent_space"] == "restricted":
            lane_y = 8 if scenario.categories["side"] == "left" else 0
            speed = scenario.parameters["ego_speed_mps"]
            self.add("guard_front", 105 + GUARD_FRONT_GAP_M, lane_y, speed)
            self.add("guard_rear", 95 - GUARD_REAR_GAP_M, lane_y, speed)
            self.road.vehicles = list(self.actors.values())
        self.trace = []
        self.min_clearance = math.inf
        self.min_ttc = math.inf
        self._record()


def execute(policy, scenario, *, environment_seed=81001, duration_s=10):
    started = time.perf_counter()
    world = InteractionWorld(scenario, environment_seed=environment_seed, duration_s=duration_s)
    while not world.terminal:
        world.advance(policy.act(world.visible_world()))
    feedback = QueryFeedback(
        scenario.scenario_id,
        bool(world.ego.crashed),
        SafetyMeasurements(world.min_clearance, world.min_ttc, tuple(sorted(world.violations))),
        time.perf_counter() - started,
    )
    return PaidExecution(feedback, world.trace)
