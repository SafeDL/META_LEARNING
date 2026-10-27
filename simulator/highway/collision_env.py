"""HighwayEnv bicycle dynamics with one ego owner and a shared visible interface."""

import math
import time

import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle

from controllers.highway.control import Control
from experiments.query import PaidExecution
from experiments.scenarios import validate
from experiments.schema import QueryFeedback, SafetyMeasurements
from simulator.highway.observation import VehicleState, observe

PHYSICS_HZ = 20
DECISION_HZ = 5
DT = 1 / PHYSICS_HZ
LANES = (0.0, 4.0, 8.0)


class ForwardVehicle(Vehicle):
    MIN_SPEED = 0
    MAX_SPEED = 40

    def step(self, dt):
        self.action["acceleration"] = max(self.action["acceleration"], -self.speed / dt)
        super().step(dt)
        self.speed = float(np.clip(self.speed, 0, self.MAX_SPEED))


class ScheduledVehicle(Vehicle):
    """Background-only continuous motion; no reference to ego or its controller."""

    def __init__(
        self,
        road,
        x,
        y,
        speed,
        *,
        event_time=math.inf,
        deceleration=0,
        ramp=1,
        destination_y=None,
        duration=2,
    ):
        super().__init__(road, [x, y], speed=speed)
        self.elapsed = 0.0
        self.longitudinal_speed = speed
        self.initial_y = y
        self.event_time = event_time
        self.deceleration = deceleration
        self.ramp = ramp
        self.destination_y = destination_y
        self.duration = duration

    def step(self, dt):
        if self.crashed:
            super().step(dt)
            return
        self.elapsed += dt
        t = max(0, self.elapsed - self.event_time)
        a = -self.deceleration * min(1, t / self.ramp)
        next_speed = max(0, self.longitudinal_speed + a * dt)
        self.position[0] += (self.longitudinal_speed + next_speed) / 2 * dt
        self.longitudinal_speed = next_speed
        lateral_speed = 0.0
        if self.destination_y is not None:
            s = min(1, t / self.duration)
            blend = 10 * s**3 - 15 * s**4 + 6 * s**5
            delta = self.destination_y - self.initial_y
            self.position[1] = self.initial_y + delta * blend
            lateral_speed = delta * 30 * s**2 * (1 - s) ** 2 / self.duration
        self.heading = math.atan2(lateral_speed, max(next_speed, 1e-8))
        self.speed = math.hypot(next_speed, lateral_speed)
        self.on_state_update()


def vehicle_state(name, vehicle):
    return VehicleState(
        name,
        float(vehicle.position[0]),
        float(vehicle.position[1]),
        float(vehicle.speed),
        float(vehicle.heading),
        float(vehicle.LENGTH),
        float(vehicle.WIDTH),
    )


def collision_ttc(a, b):
    """Constant-velocity time to overlap for the current oriented rectangles."""
    interval_low, interval_high = 0.0, math.inf
    rel_position = np.array([b.x - a.x, b.y - a.y])
    rel_velocity = np.array([b.vx - a.vx, b.vy - a.vy])
    axes = [np.array([math.cos(h), math.sin(h)]) for h in (a.heading, b.heading)]
    axes += [np.array([-v[1], v[0]]) for v in axes]
    for axis in axes:
        radius = 0.0
        for v in (a, b):
            forward = np.array([math.cos(v.heading), math.sin(v.heading)])
            lateral = np.array([-forward[1], forward[0]])
            radius += abs(axis @ forward) * v.length / 2 + abs(axis @ lateral) * v.width / 2
        position, speed = float(axis @ rel_position), float(axis @ rel_velocity)
        if abs(speed) < 1e-12:
            if abs(position) > radius:
                return math.inf
            continue
        times = sorted(((-radius - position) / speed, (radius - position) / speed))
        interval_low = max(interval_low, times[0])
        interval_high = min(interval_high, times[1])
        if interval_low > interval_high:
            return math.inf
    return interval_low if interval_high >= 0 else math.inf


class CollisionWorld:
    def __init__(self, scenario, *, environment_seed=81001, duration_s=10, lane_centers=None):
        validate(scenario)
        self.scenario = scenario
        self.duration_s = duration_s
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                lanes=3, start=0, length=2000, speed_limit=40
            ),
            np_random=np.random.default_rng(environment_seed),
        )
        p = scenario.parameters
        ego = ForwardVehicle(self.road, [100.0, 4.0], speed=p["ego_speed_mps"])
        self.actors = {"ego": ego}
        self.ego = ego
        self.tick = 0
        self.command = Control(0, 0)
        self.decisions = 0
        self.trace = []
        self.event_log = []
        self.revealed_at = None
        self.response_at = None
        self.min_clearance = math.inf
        self.min_ttc = math.inf
        self.collision_partner = None
        self.violations = set()
        if scenario.kind == "front_brake":
            self.add(
                "lead",
                100 + p["gap_m"] + 5,
                4,
                p["ego_speed_mps"] - p["closing_speed_mps"],
                event_time=p["event_time_s"],
                deceleration=p["deceleration_mps2"],
                ramp=p["brake_ramp_s"],
            )
        elif scenario.kind == "cut_in":
            y = 0 if scenario.categories["side"] == "left" else 8
            self.add(
                "cut_in",
                100 + p["gap_m"] + 5,
                y,
                p["ego_speed_mps"] - p["closing_speed_mps"],
                event_time=p["event_time_s"],
                destination_y=4,
                duration=p["lateral_duration_s"],
            )
        elif scenario.kind == "cut_out":
            self.add(
                "lead",
                100 + p["gap_m"] + 5,
                4,
                p["lead_speed_mps"],
                event_time=p["event_time_s"],
                destination_y=0,
                duration=p["lateral_duration_s"],
            )
            self.actors["obstacle"] = Obstacle(self.road, [100 + p["obstacle_distance_m"], 4])
        elif scenario.kind == "lane_change_rear":
            self.add("lead", 100 + p["lead_gap_m"] + 5, 4, p["lead_speed_mps"])
            self.add("rear", 100 - p["rear_gap_m"] - 5, 0, p["rear_speed_mps"])
            self.add("adjacent_front", 100 + p["adjacent_front_gap_m"] + 5, 0, p["ego_speed_mps"])
            # Right lane is unavailable in this functional scenario.
        self.lane_centers = (
            tuple(lane_centers)
            if lane_centers is not None
            else (0.0, 4.0)
            if scenario.kind == "lane_change_rear"
            else LANES
        )
        self.road.vehicles = [v for v in self.actors.values() if isinstance(v, Vehicle)]
        self.road.objects = [v for v in self.actors.values() if isinstance(v, Obstacle)]
        self._record()

    def add(self, role, x, y, speed, **schedule):
        self.actors[role] = ScheduledVehicle(self.road, x, y, speed, **schedule)

    def states(self):
        return tuple(vehicle_state(name, vehicle) for name, vehicle in self.actors.items())

    def visible_world(self):
        return observe(self.tick * DT, self.states(), lane_centers=self.lane_centers)

    def advance(self, command):
        if self.tick % (PHYSICS_HZ // DECISION_HZ):
            raise ValueError("Ego decisions are accepted only at the shared 5 Hz ticks")
        self.command = command.limited(self.ego.speed, self.ego.LENGTH)
        self.decisions += 1
        if self.command.acceleration < -0.5 and self.response_at is None:
            event_time = self.scenario.parameters.get("event_time_s", 0)
            if self.tick * DT >= event_time:
                self.response_at = self.tick * DT
        for _ in range(PHYSICS_HZ // DECISION_HZ):
            # Road.act() is never called: exactly one owner issues ego commands.
            applied = self.command.limited(self.ego.speed, self.ego.LENGTH)
            self.ego.act({"acceleration": applied.acceleration, "steering": applied.steering})
            self.road.step(DT)
            self.tick += 1
            self._record()
            if self.terminal:
                break

    @property
    def terminal(self):
        return bool(
            self.ego.crashed
            or self.tick * DT >= self.duration_s - 1e-9
            or "road_departure" in self.violations
        )

    def _record(self):
        states = self.states()
        visible = self.visible_world()
        timestamp = round(self.tick * DT, 6)
        if "obstacle" in {s.actor_id for s in visible.vehicles} and self.revealed_at is None:
            self.revealed_at = timestamp
            self.event_log.append({"time_s": timestamp, "event": "obstacle_visible"})
        event_time = self.scenario.parameters.get("event_time_s", 0)
        if self.tick and timestamp - DT < event_time <= timestamp:
            self.event_log.append({"time_s": timestamp, "event": "background_event_started"})
        ego = states[0]
        distances = [
            (ego.polygon().distance(actor.polygon()), actor.actor_id) for actor in states[1:]
        ]
        self.min_clearance = min(self.min_clearance, min(d[0] for d in distances))
        self.min_ttc = min(self.min_ttc, *(collision_ttc(ego, actor) for actor in states[1:]))
        if self.ego.crashed and self.collision_partner is None:
            self.collision_partner = min(distances)[1]
        if not min(self.lane_centers) - 1 <= ego.y <= max(self.lane_centers) + 1:
            self.violations.add("road_departure")
        self.trace.append(
            {
                "time_s": timestamp,
                "actors": [s.__dict__.copy() for s in states],
                "visible_ids": [s.actor_id for s in visible.vehicles],
                "control": {
                    "acceleration": float(self.ego.action["acceleration"]),
                    "steering": float(self.ego.action["steering"]),
                },
                "requested_control": self.command.__dict__.copy(),
                "collision": bool(self.ego.crashed),
                "collision_partner": self.collision_partner,
            }
        )


def execute(policy, scenario, *, environment_seed=81001, duration_s=10):
    started = time.perf_counter()
    world = CollisionWorld(scenario, environment_seed=environment_seed, duration_s=duration_s)
    while not world.terminal:
        world.advance(policy.act(world.visible_world()))
    feedback = QueryFeedback(
        scenario.scenario_id,
        bool(world.ego.crashed),
        SafetyMeasurements(world.min_clearance, world.min_ttc, tuple(sorted(world.violations))),
        time.perf_counter() - started,
    )
    return PaidExecution(feedback, world.trace)
