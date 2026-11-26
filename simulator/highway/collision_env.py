"""Native highway-env lifecycle, dynamics and collisions for parameterized scenes.

All actors receive acceleration/steering commands. Scripts never assign poses
after reset. AbstractEnv.step/_simulate and native collision resolution are kept.
"""

import math
import time

import gymnasium as gym
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle

from controllers.highway.control import Control
from experiments.query import PaidExecution
from experiments.scenarios import validate
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from simulator.highway.observation import VehicleState, observe
from simulator.highway.timing import DECISION_HZ, PHYSICS_HZ, PHYSICS_PER_DECISION
from simulator.highway.timing import PHYSICS_DT as DT

LANES = (0.0, 4.0, 8.0)
ENV_ID = "response-highway-v2"
ENVIRONMENT_VERSION = "native-highway-v2"


def default_lane_centers(kind):
    if kind in ("stopped_vehicle", "front_brake_single_lane", "car_following_braking_onset"):
        return (4.0,)
    return (0.0, 4.0) if kind in ("lane_change_rear", "cut_out_reveal", "cut_out_restricted", "cut_out_rear_gap") else LANES


class ForwardVehicle(Vehicle):
    """Native integration and impacts; common forward-only physical actuator."""

    MIN_SPEED = 0
    MAX_SPEED = 40

    def act(self, action=None):
        if action is not None:
            self._requested_action = dict(action)
        # Clipping a past tick must not permanently change the held request.
        super().act(dict(getattr(self, "_requested_action", self.action)))

    def step(self, dt):
        pending = self.impact is not None
        was_crashed = self.crashed
        command = Control(**self.action).limited(self.speed, self.LENGTH)
        self.action = {
            "acceleration": max(command.acceleration, -self.speed / dt),
            "steering": command.steering,
        }
        self.action["acceleration"] = min(
            self.action["acceleration"], (self.MAX_SPEED - self.speed) / dt
        )
        super().step(dt)
        if not was_crashed and self.crashed and pending:
            self._crash_partner = getattr(self, "_impact_partner", None)

    def handle_collisions(self, other, dt=0):
        """Observe the native callback without changing its collision criterion."""
        before = [
            (v, bool(v.crashed), None if v.impact is None else v.impact.copy())
            for v in (self, other)
        ]
        super().handle_collisions(other, dt)
        for actor, was_crashed, impact in before:
            partner = other if actor is self else self
            if actor.impact is not None and (
                impact is None or not np.array_equal(impact, actor.impact)
            ):
                actor._impact_partner = getattr(partner, "actor_id", None)
            if not was_crashed and actor.crashed:
                actor._crash_partner = getattr(partner, "actor_id", None)


class ScheduledVehicle(ForwardVehicle):
    """Scene intent controller; position/heading are integrated by Vehicle.step.

    A fifth-order lateral reference is tracked with steering. Optional IDM
    following prevents unrelated background rear-ends. No tested policy is read.
    """

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
        reactive=False,
    ):
        super().__init__(road, [x, y], speed=speed)
        self.initial_y, self.cruise_speed = y, speed
        self.event_time, self.deceleration, self.ramp = event_time, deceleration, ramp
        self.destination_y, self.duration, self.reactive = destination_y, duration, reactive
        self.longitudinal_acceleration = 0.0

    @property
    def elapsed(self):
        return self.road.world.tick * DT

    @property
    def longitudinal_speed(self):
        return float(self.speed * math.cos(self.heading))

    def act(self, action=None):
        if action is not None:
            raise ValueError("Background intent has exactly one owner")
        t = max(0.0, self.elapsed - self.event_time)
        fraction = float(self.elapsed >= self.event_time) if self.ramp == 0 else min(1, t / self.ramp)
        acceleration = -self.deceleration * fraction
        target_y, lateral_speed = self.initial_y, 0.0
        if self.destination_y is not None:
            s = min(1, t / self.duration)
            delta = self.destination_y - self.initial_y
            target_y += delta * (10 * s**3 - 15 * s**4 + 6 * s**5)
            lateral_speed = delta * 30 * s**2 * (1 - s) ** 2 / self.duration
        heading_ref = np.clip(
            math.atan2(lateral_speed + 2.0 * (target_y - self.position[1]), max(self.speed, 2)),
            -0.45,
            0.45,
        )
        steering = math.atan(self.LENGTH * 3 * (heading_ref - self.heading) / max(self.speed, 2))
        if self.reactive:
            fronts = [
                v
                for v in self.road.vehicles
                if v is not self
                and v.position[0] > self.position[0]
                and abs(v.position[1] - self.position[1]) < 2.2
            ]
            free = 2 * (1 - (self.speed / max(self.cruise_speed, 1)) ** 4)
            if fronts:
                front = min(fronts, key=lambda v: v.position[0])
                gap = max(
                    0.1, front.position[0] - self.position[0] - (self.LENGTH + front.LENGTH) / 2
                )
                wanted = 2 + max(
                    0,
                    1.5 * self.speed + self.speed * (self.speed - front.speed) / (2 * math.sqrt(6)),
                )
                free -= 2 * (wanted / gap) ** 2
            acceleration = min(acceleration, free)
        command = Control(float(acceleration), steering).limited(self.speed, self.LENGTH)
        self.longitudinal_acceleration = command.acceleration
        super().act(command.__dict__.copy())


class RecordingRoad(Road):
    """Record after the unmodified native Road.step finishes a physics tick."""

    def step(self, dt):
        if self.world.tick % PHYSICS_PER_DECISION == 0:
            self.world.command = Control(**self.world.ego._requested_action).limited()
            self.world.decisions += 1
        super().step(dt)
        self.world.tick += 1
        self.world._record()


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
    """Diagnostic constant-velocity TTC; it does not set the collision label."""
    low, high = 0.0, math.inf
    position = np.array([b.x - a.x, b.y - a.y])
    velocity = np.array([b.vx - a.vx, b.vy - a.vy])
    axes = [np.array([math.cos(h), math.sin(h)]) for h in (a.heading, b.heading)]
    axes += [np.array([-v[1], v[0]]) for v in axes]
    for axis in axes:
        radius = 0.0
        for v in (a, b):
            forward = np.array([math.cos(v.heading), math.sin(v.heading)])
            lateral = np.array([-forward[1], forward[0]])
            radius += abs(axis @ forward) * v.length / 2 + abs(axis @ lateral) * v.width / 2
        p, speed = float(axis @ position), float(axis @ velocity)
        if abs(speed) < 1e-12:
            if abs(p) > radius:
                return math.inf
            continue
        times = sorted(((-radius - p) / speed, (radius - p) / speed))
        low, high = max(low, times[0]), min(high, times[1])
        if low > high:
            return math.inf
    return low if high >= 0 else math.inf


class CollisionWorld(AbstractEnv):
    """Parameterized environment following the official AbstractEnv interface."""

    def configure(self, config):
        requested = {**self.default_config(), **self.config, **(config or {})}
        if (
            requested["simulation_frequency"] != PHYSICS_HZ
            or requested["policy_frequency"] != DECISION_HZ
        ):
            raise ValueError("All ADS use common 20 Hz physics and 5 Hz decisions")
        super().configure(config)

    @classmethod
    def default_config(cls):
        return {
            **super().default_config(),
            "simulation_frequency": PHYSICS_HZ,
            "policy_frequency": DECISION_HZ,
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": [-8, 3],
                "steering_range": [-0.45, 0.45],
                "speed_range": [0, 40],
            },
            "observation": {"type": "Kinematics", "vehicles_count": 9},
            "offscreen_rendering": True,
            "screen_width": 1000,
            "screen_height": 260,
        }

    def __init__(
        self,
        scenario=None,
        *,
        environment_seed=81001,
        duration_s=None,
        lane_centers=None,
        config=None,
        render_mode=None,
    ):
        if scenario is None:
            scenario = Scenario(
                "native-default",
                "car_following",
                {"ego_speed_mps": 25, "lead_speed_mps": 20, "gap_m": 50},
                {},
            )
        validate(scenario)
        if duration_s is None:
            from experiments.protocol import episode_duration

            duration_s = episode_duration(scenario.kind)
        if duration_s <= 0 or not math.isclose(
            duration_s * DECISION_HZ, round(duration_s * DECISION_HZ)
        ):
            raise ValueError(
                "Episode duration must be a positive multiple of the decision interval"
            )
        self.scenario, self.environment_seed, self.duration_s = (
            scenario,
            environment_seed,
            duration_s,
        )
        default_lanes = default_lane_centers(scenario.kind)
        if scenario.kind == "side_encroachment":
            default_lanes = (0.0, 4.0) if scenario.categories["side"] == "left" else (4.0, 8.0)
        self.lane_centers = tuple(default_lanes if lane_centers is None else lane_centers)
        if (
            not self.lane_centers
            or tuple(sorted(set(self.lane_centers))) != self.lane_centers
            or 4.0 not in self.lane_centers
        ):
            raise ValueError("Sorted, unique lane centers must include the ego spawn lane at y=4")
        super().__init__(config=config, render_mode=render_mode)

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=self.environment_seed if seed is None else seed, options=options)

    @property
    def ego(self):
        return self.vehicle

    def _reset(self):
        network = RoadNetwork()
        for i, y in enumerate(self.lane_centers):
            network.add_lane(
                "0",
                "1",
                StraightLane(
                    [0, y],
                    [2000, y],
                    width=4,
                    line_types=[
                        LineType.CONTINUOUS_LINE if i == 0 else LineType.STRIPED,
                        LineType.CONTINUOUS_LINE
                        if i == len(self.lane_centers) - 1
                        else LineType.STRIPED,
                    ],
                    speed_limit=40,
                ),
            )
        self.road = RecordingRoad(network=network, np_random=self.np_random)
        self.road.world = self
        self.vehicle = ForwardVehicle(
            self.road, [100.0, 4.0], speed=self.scenario.parameters["ego_speed_mps"]
        )
        self.vehicle.actor_id = "ego"
        self.actors = {"ego": self.vehicle}
        self.tick, self.decisions = 0, 0
        self.command = Control(0, 0)
        self.trace, self.event_log = [], []
        self.revealed_at = self.response_at = self.collision_partner = None
        self.min_clearance = self.min_ttc = math.inf
        self.violations = set()
        self._create_traffic()
        self.road.vehicles = [v for v in self.actors.values() if isinstance(v, Vehicle)]
        self.road.objects = [v for v in self.actors.values() if isinstance(v, Obstacle)]
        states = self.states()
        if any(
            a.polygon().intersects(b.polygon())
            for i, a in enumerate(states)
            for b in states[i + 1 :]
        ):
            raise ValueError("A scene cannot start with intersecting bodies")
        self._record()

    def _create_traffic(self):
        p, kind = self.scenario.parameters, self.scenario.kind
        if kind in ("front_brake", "front_brake_single_lane"):
            self.add(
                "lead",
                105 + p["gap_m"],
                4,
                p["ego_speed_mps"] - p["closing_speed_mps"],
                event_time=p["event_time_s"],
                deceleration=p["deceleration_mps2"],
                ramp=p["brake_ramp_s"],
            )
        elif kind == "car_following":
            self.add("lead", 105 + p["gap_m"], 4, p["lead_speed_mps"])
        elif kind == "car_following_braking_onset":
            self.add(
                "lead", 105 + p["gap_m"], 4,
                p["ego_speed_mps"] - p["closing_speed_mps"],
                event_time=0.0,
                deceleration=p["deceleration_mps2"], ramp=0,
            )
        elif kind in ("cut_in", "cut_in_interaction", "cut_in_onset"):
            y = 0 if self.scenario.categories["side"] == "left" else 8
            self.add(
                "cut_in",
                105 + p["gap_m"],
                y,
                p["ego_speed_mps"] - p["closing_speed_mps"],
                event_time=0.0 if kind == "cut_in_onset" else p["event_time_s"],
                destination_y=4,
                duration=p["lateral_duration_s"],
            )
            if (
                kind == "cut_in_onset" or (kind == "cut_in_interaction"
                and self.scenario.categories["adjacent_space"] == "restricted")
            ):
                other_y = 8 if y == 0 else 0
                self.add("guard_front", 115, other_y, p["ego_speed_mps"])
                self.add("guard_rear", 85, other_y, p["ego_speed_mps"])
        elif kind in ("cut_out", "cut_out_reveal", "cut_out_restricted", "cut_out_rear_gap"):
            reveal = kind != "cut_out"
            gap = 10 if kind == "cut_out_rear_gap" else p["gap_m"]
            self.add(
                "lead",
                (106.5 if reveal else 105) + gap,
                4,
                p["ego_speed_mps"] if reveal else p["lead_speed_mps"],
                event_time=0.0 if reveal else p["event_time_s"],
                destination_y=0,
                duration=p["lateral_duration_s"],
            )
            if reveal:
                self.actors["lead"].LENGTH = 8.0
                self.actors["lead"].WIDTH = 2.5
            obstacle = Obstacle(self.road, [100 + p["obstacle_distance_m"], 4])
            obstacle.actor_id = "obstacle"
            self.actors["obstacle"] = obstacle
            if kind == "cut_out_restricted":
                self.add("guard_rear", 85, 0, p["ego_speed_mps"])
            elif kind == "cut_out_rear_gap":
                self.add("guard_rear", 100 - p["rear_gap_m"], 0, p["ego_speed_mps"])
        elif kind == "side_encroachment":
            y = 0 if self.scenario.categories["side"] == "left" else 8
            self.add(
                "intruder", 100 + p["longitudinal_offset_m"], y,
                p["ego_speed_mps"] + p["relative_speed_mps"],
                event_time=0.0, destination_y=4, duration=p["lateral_duration_s"],
            )
        elif kind == "lane_change_rear":
            self.add("lead", 105 + p["lead_gap_m"], 4, p["lead_speed_mps"])
            self.add("rear", 95 - p["rear_gap_m"], 0, p["rear_speed_mps"], reactive=True)
            self.add("adjacent_front", 105 + p["adjacent_front_gap_m"], 0, p["ego_speed_mps"])
        elif kind == "stopped_vehicle":
            self.add("stopped", 105 + p["gap_m"], 4, 0)
        else:
            raise ValueError(f"Unsupported native scenario: {kind}")

    def add(self, role, x, y, speed, **schedule):
        actor = ScheduledVehicle(self.road, x, y, speed, **schedule)
        actor.actor_id = role
        self.actors[role] = actor

    def states(self):
        return tuple(vehicle_state(name, vehicle) for name, vehicle in self.actors.items())

    def visible_world(self):
        return observe(self.tick * DT, self.states(), lane_centers=self.lane_centers)

    def advance(self, command):
        if self.terminal:
            raise RuntimeError("Reset a finished environment before stepping")
        self.command = command.limited()
        action = np.array(
            [2 * (self.command.acceleration + 8) / 11 - 1, self.command.steering / 0.45],
            dtype=np.float64,
        )
        return self.step(action)

    def _reward(self, action):
        return float(not self.ego.crashed)

    def _is_terminated(self):
        return bool(self.ego.crashed or self.violations)

    def _is_truncated(self):
        return bool(self.tick * DT >= self.duration_s - 1e-9)

    @property
    def terminal(self):
        return self._is_terminated() or self._is_truncated()

    def _record(self):
        states, visible = self.states(), self.visible_world()
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
        self.min_clearance = min(
            self.min_clearance, min((d[0] for d in distances), default=math.inf)
        )
        self.min_ttc = min(
            self.min_ttc, min((collision_ttc(ego, a) for a in states[1:]), default=math.inf)
        )
        if self.ego.crashed and self.collision_partner is None:
            self.collision_partner = getattr(self.ego, "_crash_partner", None)
            self.event_log.append(
                {
                    "time_s": timestamp,
                    "event": "native_ego_collision",
                    "partner": self.collision_partner,
                }
            )
        half_width = (
            abs(math.sin(ego.heading)) * ego.length + abs(math.cos(ego.heading)) * ego.width
        ) / 2
        if (
            ego.y - half_width < min(self.lane_centers) - 2
            or ego.y + half_width > max(self.lane_centers) + 2
        ):
            self.violations.add("road_departure")
        if (
            self.tick
            and self.ego.action["acceleration"] < -0.5
            and self.response_at is None
            and timestamp - DT >= event_time
        ):
            self.response_at = round(timestamp - DT, 6)
        self.trace.append(
            {
                "time_s": timestamp,
                "actors": [s.__dict__.copy() for s in states],
                "visible_ids": [s.actor_id for s in visible.vehicles],
                "control": {k: float(v) for k, v in self.ego.action.items()},
                "requested_control": self.command.__dict__.copy(),
                "collision": bool(self.ego.crashed),
                "collision_partner": self.collision_partner,
                "background_controls": {
                    name: {
                        "acceleration_mps2": float(actor.action["acceleration"]),
                        "steering_rad": float(actor.action["steering"]),
                        "scripted": not bool(actor.crashed),
                    }
                    for name, actor in self.actors.items()
                    if name != "ego" and isinstance(actor, Vehicle)
                },
            }
        )


if ENV_ID not in gym.envs.registry:
    gym.register(ENV_ID, entry_point="simulator.highway.collision_env:CollisionWorld")


def execute(policy, scenario, *, environment_seed=81001, duration_s=None):
    started = time.perf_counter()
    env = gym.make(
        ENV_ID, scenario=scenario, environment_seed=environment_seed, duration_s=duration_s
    )
    world = env.unwrapped
    try:
        env.reset(seed=environment_seed)
        while not world.terminal:
            world.advance(policy.act(world.visible_world()))
        feedback = QueryFeedback(
            scenario.scenario_id,
            bool(world.ego.crashed),
            SafetyMeasurements(world.min_clearance, world.min_ttc, tuple(sorted(world.violations))),
            time.perf_counter() - started,
        )
        return PaidExecution(feedback, world.trace)
    finally:
        env.close()
