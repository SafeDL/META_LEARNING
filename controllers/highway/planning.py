"""Visible-state constant-velocity prediction and common bicycle rollout."""

import math

import numpy as np

from simulator.highway.observation import VehicleState
from simulator.highway.timing import DECISION_DT, PHYSICS_DT

from .control import MAX_LATERAL_ACCELERATION, MAX_STEERING, lane_steering


def bicycle_rollout(ego, controls, *, dt=DECISION_DT, return_substeps=False):
    """Vectorized rollouts, including batches used by optimization derivatives."""
    controls = np.asarray(controls)
    shape = controls.shape[:-2]
    state = np.broadcast_to([ego.x, ego.y, ego.speed, ego.heading], (*shape, 4)).copy()
    output = []
    for u in np.moveaxis(controls, -2, 0):
        for _ in range(round(dt / PHYSICS_DT)):
            acceleration, steering = np.clip(u[..., 0], -8, 3), u[..., 1]
            x, y, v, heading = np.moveaxis(state, -1, 0)
            steering_limit = np.minimum(
                MAX_STEERING, np.arctan(MAX_LATERAL_ACCELERATION * ego.length / np.maximum(v**2, 1))
            )
            steering = np.clip(steering, -steering_limit, steering_limit)
            beta = np.arctan(0.5 * np.tan(steering))
            state = np.stack(
                (
                    x + v * np.cos(heading + beta) * PHYSICS_DT,
                    y + v * np.sin(heading + beta) * PHYSICS_DT,
                    np.clip(v + acceleration * PHYSICS_DT, 0, 40),
                    heading + 2 * v / ego.length * np.sin(beta) * PHYSICS_DT,
                ),
                -1,
            )
            if return_substeps:
                output.append(state)
        if not return_substeps:
            output.append(state)
    return np.stack(output, axis=-2)


def predicted_actors(world, horizon, dt):
    times = np.arange(1, horizon + 1) * dt
    if not world.vehicles:
        return np.empty((horizon, 0, 2))
    return np.array([[[v.x + v.vx * t, v.y + v.vy * t] for v in world.vehicles] for t in times])


def advance_visible(world, acceleration, target_lane, *, duration_s=0.6, return_path=False):
    """Search transition; backgrounds use their current velocities, never schedules."""
    from simulator.highway.observation import VisibleWorld

    ego = world.ego
    dt = DECISION_DT
    path = []
    elapsed = 0.0
    for _ in range(round(duration_s / dt)):
        steering = lane_steering(ego, world.lane_centers[target_lane])
        rows = bicycle_rollout(
            ego, np.array([[acceleration, steering]]), dt=dt, return_substeps=True
        )
        for row in rows:
            elapsed += PHYSICS_DT
            next_ego = VehicleState(ego.actor_id, *(float(v) for v in row), ego.length, ego.width)
            actors = tuple(
                VehicleState(
                    v.actor_id,
                    v.x + v.vx * elapsed,
                    v.y + v.vy * elapsed,
                    v.speed,
                    v.heading,
                    v.length,
                    v.width,
                )
                for v in world.vehicles
            )
            path.append(
                VisibleWorld(
                    world.time_s + elapsed,
                    next_ego,
                    actors,
                    world.lane_centers,
                    world.speed_limit_mps,
                )
            )
        ego = path[-1].ego
    return path if return_path else path[-1]


def trajectory_feasibility(world, controls, *, dt=DECISION_DT):
    """Check native-equivalent prediction at 20 Hz, including rotated bodies.

    This checks the declared constant-velocity prediction, not unknown future
    traffic actions. It is not a guarantee against every actual future collision.
    """
    rows = bicycle_rollout(world.ego, controls, dt=dt, return_substeps=True)
    minimum = math.inf
    for k, row in enumerate(rows, 1):
        ego = VehicleState(
            world.ego.actor_id, *(float(v) for v in row), world.ego.length, world.ego.width
        )
        actors = tuple(
            VehicleState(
                v.actor_id,
                v.x + v.vx * k * PHYSICS_DT,
                v.y + v.vy * k * PHYSICS_DT,
                v.speed,
                v.heading,
                v.length,
                v.width,
            )
            for v in world.vehicles
        )
        margin = overlap_margin(ego, actors)
        minimum = min(minimum, margin)
        half_width = (
            abs(math.sin(ego.heading)) * ego.length + abs(math.cos(ego.heading)) * ego.width
        ) / 2
        if (
            margin <= 0
            or ego.y - half_width < min(world.lane_centers) - 2
            or ego.y + half_width > max(world.lane_centers) + 2
        ):
            return False, minimum
    return True, minimum


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
