"""Immutable visible-world values, with no simulator objects or future schedules."""

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Polygon


@dataclass(frozen=True)
class VehicleState:
    actor_id: str
    x: float
    y: float
    speed: float
    heading: float
    length: float = 5.0
    width: float = 2.0

    @property
    def vx(self):
        return self.speed * math.cos(self.heading)

    @property
    def vy(self):
        return self.speed * math.sin(self.heading)

    def polygon(self):
        corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
        corners = corners * [self.length / 2, self.width / 2]
        c, s = math.cos(self.heading), math.sin(self.heading)
        corners = corners @ np.array([[c, s], [-s, c]]) + [self.x, self.y]
        return Polygon(corners)


@dataclass(frozen=True)
class VisibleWorld:
    time_s: float
    ego: VehicleState
    vehicles: tuple[VehicleState, ...]
    lane_centers: tuple[float, ...] = (0.0, 4.0, 8.0)
    speed_limit_mps: float = 40.0


def observe(time_s, states, *, sensor_range_m=180.0, lane_centers=(0.0, 4.0, 8.0)):
    """An actor is visible if at least one edge/corner sight line is unobstructed."""
    ego, others = states[0], tuple(states[1:])
    visible = []
    origin = (
        ego.x + ego.length / 2 * math.cos(ego.heading),
        ego.y + ego.length / 2 * math.sin(ego.heading),
    )
    polygons = {v.actor_id: v.polygon() for v in others}
    for actor in others:
        distance = math.hypot(actor.x - ego.x, actor.y - ego.y)
        if distance > sensor_range_m:
            continue
        vertices = np.asarray(polygons[actor.actor_id].exterior.coords)[:-1]
        targets = np.concatenate((vertices, (vertices + np.roll(vertices, 1, axis=0)) / 2))
        for point in targets:
            ray = LineString([origin, point])
            blocked = any(
                ray.crosses(polygon) or ray.within(polygon)
                for name, polygon in polygons.items()
                if name != actor.actor_id
            )
            if not blocked:
                visible.append(actor)
                break
    return VisibleWorld(float(time_s), ego, tuple(visible), tuple(lane_centers))


def policy_observation(world, *, target_lane_y=None, max_vehicles=8):
    """Common PPO/DQN observation: ego row then nearest visible actors with a mask."""
    output = np.zeros((max_vehicles + 1, 9), dtype=np.float32)
    ego = world.ego
    output[0] = [
        1,
        0,
        ego.y / 12,
        ego.vx / 40,
        ego.vy / 10,
        math.sin(ego.heading),
        math.cos(ego.heading),
        (min(world.lane_centers) - ego.y) / 12,
        (max(world.lane_centers) - ego.y) / 12,
    ]
    ordered = sorted(
        world.vehicles, key=lambda v: ((v.x - ego.x) ** 2 + (v.y - ego.y) ** 2, v.actor_id)
    )
    for i, actor in enumerate(ordered[:max_vehicles], 1):
        output[i] = [
            1,
            (actor.x - ego.x) / 180,
            (actor.y - ego.y) / 12,
            (actor.vx - ego.vx) / 40,
            (actor.vy - ego.vy) / 10,
            math.sin(actor.heading),
            math.cos(actor.heading),
            0,
            0,
        ]
    target = ego.y if target_lane_y is None else target_lane_y
    return np.r_[
        np.clip(output, -1, 1).ravel(), np.float32(np.clip((target - ego.y) / 12, -1, 1))
    ].astype(np.float32)
