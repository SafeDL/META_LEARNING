"""Visible-state constant-velocity prediction and common bicycle rollout."""

import math

import numpy as np

from simulator.highway.observation import VehicleState

from .control import MAX_LATERAL_ACCELERATION, MAX_STEERING, lane_steering


def bicycle_rollout(ego, controls, *, dt=0.2):
    """Vectorized rollouts, including batches used by optimization derivatives."""
    controls = np.asarray(controls)
    shape = controls.shape[:-2]
    state = np.broadcast_to([ego.x, ego.y, ego.speed, ego.heading], (*shape, 4)).copy()
    output = []
    for u in np.moveaxis(controls, -2, 0):
        for _ in range(round(dt / 0.05)):
            acceleration, steering = u[..., 0], u[..., 1]
            x, y, v, heading = np.moveaxis(state, -1, 0)
            steering_limit = np.minimum(
                MAX_STEERING, np.arctan(MAX_LATERAL_ACCELERATION * ego.length / np.maximum(v**2, 1))
            )
            steering = np.clip(steering, -steering_limit, steering_limit)
            beta = np.arctan(0.5 * np.tan(steering))
            state = np.stack(
                (
                    x + v * np.cos(heading + beta) * 0.05,
                    y + v * np.sin(heading + beta) * 0.05,
                    np.clip(v + acceleration * 0.05, 0, 40),
                    heading + 2 * v / ego.length * np.sin(beta) * 0.05,
                ),
                -1,
            )
        output.append(state)
    return np.stack(output, axis=-2)


def predicted_actors(world, horizon, dt):
    times = np.arange(1, horizon + 1) * dt
    if not world.vehicles:
        return np.empty((horizon, 0, 2))
    return np.array([[[v.x + v.vx * t, v.y + v.vy * t] for v in world.vehicles] for t in times])


def advance_visible(world, acceleration, target_lane, *, duration_s=0.6):
    """Search transition; backgrounds use their current velocities, never schedules."""
    from simulator.highway.observation import VisibleWorld

    ego = world.ego
    dt = 0.2
    for _ in range(round(duration_s / dt)):
        steering = lane_steering(ego, world.lane_centers[target_lane])
        row = bicycle_rollout(ego, np.array([[acceleration, steering]]), dt=dt)[0]
        ego = VehicleState(
            ego.actor_id,
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            ego.length,
            ego.width,
        )
    actors = tuple(
        VehicleState(
            v.actor_id,
            v.x + v.vx * duration_s,
            v.y + v.vy * duration_s,
            v.speed,
            v.heading,
            v.length,
            v.width,
        )
        for v in world.vehicles
    )
    return VisibleWorld(
        world.time_s + duration_s, ego, actors, world.lane_centers, world.speed_limit_mps
    )


def overlap_margin(ego, actors):
    """Separating-axis rectangle margin; negative means overlapping projections."""
    margins = []
    for other in actors:
        delta = np.array([other.x - ego.x, other.y - ego.y])
        separation = []
        for heading in (ego.heading, other.heading):
            for angle in (heading, heading + math.pi / 2):
                axis = np.array([math.cos(angle), math.sin(angle)])
                radius = 0
                for v in (ego, other):
                    radius += (
                        abs(math.cos(angle - v.heading)) * v.length
                        + abs(math.sin(angle - v.heading)) * v.width
                    ) / 2
                separation.append(abs(axis @ delta) - radius)
        margins.append(max(separation))
    return min(margins, default=math.inf)
