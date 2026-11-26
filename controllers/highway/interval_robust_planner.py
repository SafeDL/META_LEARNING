"""Pinned rl-agents interval planner on a shadow road of currently visible actors."""

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.uncertainty.prediction import IntervalVehicle

from simulator.highway.timing import DECISION_HZ, PHYSICS_HZ

from .control import Control
from .decision_interface import DecisionInterface
from .public_agents import CACHE, META_SPEEDS, RL_AGENTS_REV

SUT_ID = "interval_robust_planner"
UPSTREAM = CACHE / "upstream" / f"rl-agents-{RL_AGENTS_REV}"
CONFIG = UPSTREAM / "scripts/configs/HighwayEnv/agents/IntervalRobustPlannerAgent"


@lru_cache(maxsize=1)
def _implementation():
    paths = (str(UPSTREAM), str(CACHE / "dependencies"))
    sys.path[:0] = paths
    try:
        from rl_agents.agents.robust import robust
    finally:
        for path in paths:
            sys.path.remove(path)
    if not Path(robust.__file__).resolve().is_relative_to(UPSTREAM.resolve()):
        raise ValueError("Interval planner was loaded from an unregistered upstream revision")
    return robust.IntervalRobustPlannerAgent


def upstream_assets():
    paths = [
        CONFIG / "baseline.json", CONFIG / "opd.json",
        *sorted((UPSTREAM / "rl_agents").rglob("*.py")),
    ]
    return {
        path.relative_to(UPSTREAM).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


class CompatibleIntervalVehicle(IntervalVehicle):
    def handle_collisions(self, other, dt):
        if isinstance(other, MDPVehicle):
            return super().handle_collisions(other, dt)
        return Vehicle.handle_collisions(self, other, dt)


class VisibleRoad(HighwayEnv):
    def __init__(self, world):
        self.visible = world
        super().__init__({
            "lanes_count": len(world.lane_centers),
            "simulation_frequency": PHYSICS_HZ,
            "policy_frequency": DECISION_HZ,
            "action": {"type": "DiscreteMetaAction", "target_speeds": list(META_SPEEDS)},
        })

    def _reset(self):
        network = RoadNetwork()
        for y in self.visible.lane_centers:
            network.add_lane("0", "1", StraightLane([-500, y], [2500, y]))
        self.road = Road(network=network, np_random=self.np_random)
        ego = self.visible.ego
        self.controlled_vehicles = [
            MDPVehicle(self.road, [ego.x, ego.y], ego.heading, ego.speed, target_speeds=META_SPEEDS)
        ]
        self.road.vehicles = [self.vehicle]

    def refresh(self, world):
        ego = world.ego
        self.vehicle.position[:] = ego.x, ego.y
        self.vehicle.speed, self.vehicle.heading = ego.speed, ego.heading
        self.vehicle.on_state_update()
        neighbors = []
        for actor in world.vehicles:
            proxy = IDMVehicle(
                self.road, [actor.x, actor.y], actor.heading, actor.speed,
                target_speed=actor.speed, enable_lane_change=False,
            )
            proxy.LENGTH, proxy.WIDTH = actor.length, actor.width
            neighbors.append(proxy)
        self.road.vehicles = [self.vehicle, *neighbors]
        self.time = world.time_s

    def simplify(self):
        copied = super().simplify()
        if any(isinstance(item, IntervalVehicle) for item in copied.road.vehicles):
            copied.road.vehicles.remove(copied.vehicle)
            copied.road.vehicles.append(copied.vehicle)
        return copied

    def change_vehicles(self, vehicle_class_path):
        if vehicle_class_path == "highway_env.vehicle.uncertainty.prediction.IntervalVehicle":
            vehicle_class_path = f"{__name__}.CompatibleIntervalVehicle"
        copied = super().change_vehicles(vehicle_class_path)
        for original, clone in zip(self.road.vehicles, copied.road.vehicles, strict=True):
            clone.LENGTH, clone.WIDTH = original.LENGTH, original.WIDTH
            if isinstance(clone, IntervalVehicle):
                clone.enable_lane_change = original.enable_lane_change
                clone.data = {}
        return copied


class IntervalRobustPlanner:
    def __init__(self):
        self.agent = None
        self.last_decision = None

    def act(self, world):
        if self.agent is None:
            self.env = VisibleRoad(world)
            config = json.loads((CONFIG / "baseline.json").read_text(encoding="utf-8"))
            config["sub_agent_path"] = str(CONFIG / "opd.json")
            self.agent = _implementation()(self.env, config)
            self.agent.seed(1301)
        self.env.refresh(world)
        action = int(self.agent.act(self.env.observation_type.observe()))
        self.env.action_type.act(action)
        self.last_decision = {"action": action}
        return Control(**self.env.vehicle.action).limited(world.ego.speed, world.ego.length)


def create():
    return DecisionInterface(IntervalRobustPlanner())
