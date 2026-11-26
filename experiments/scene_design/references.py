"""Causal reference manoeuvres used to certify scene avoidability."""

import math

import numpy as np
from scipy.optimize import Bounds, minimize

from controllers.highway.control import Control, MAX_STEERING, lane_steering
from controllers.highway.planning import bicycle_rollout
from simulator.highway.timing import DECISION_DT

LATERAL_ACCELERATION = 5.5
ROAD_MARGIN = 0.05
FINAL_HEADING = 0.3


def plan_stop(ego, lane_centers):
    steps = max(1, math.ceil(ego.speed / (8 * DECISION_DT) - 1e-9))
    speeds = np.maximum(0, ego.speed - 8 * DECISION_DT * np.arange(steps))
    limits = np.minimum(
        MAX_STEERING,
        np.arctan(LATERAL_ACCELERATION * ego.length / np.maximum(speeds**2, 1)),
    )
    lower, upper = min(lane_centers) - 2, max(lane_centers) + 2
    epsilon = 1e-5
    cached_u, cached = None, None

    def evaluate(u):
        nonlocal cached_u, cached
        if cached_u is None or not np.array_equal(u, cached_u):
            batch = np.concatenate((u[None], u[None] + epsilon * np.eye(steps)))
            controls = np.stack((np.full(batch.shape, -8.0), batch * limits), axis=-1)
            path = bicycle_rollout(ego, controls, return_substeps=True)
            y, heading = path[..., 1], path[..., 3]
            half_width = (
                ego.length * np.abs(np.sin(heading))
                + ego.width * np.abs(np.cos(heading))
            ) / 2
            constraints = np.concatenate((
                y - half_width - lower - ROAD_MARGIN,
                upper - ROAD_MARGIN - y - half_width,
                FINAL_HEADING - path[:, -1, 3:4],
                FINAL_HEADING + path[:, -1, 3:4],
            ), axis=1)
            final_heading = heading[:, -1]
            front = path[:, -1, 0] + (
                ego.length * np.abs(np.cos(final_heading))
                + ego.width * np.abs(np.sin(final_heading))
            ) / 2
            value = front - ego.x - ego.length / 2
            cached = (
                value[0], (value[1:] - value[0]) / epsilon,
                constraints[0], ((constraints[1:] - constraints[0]) / epsilon).T,
            )
            cached_u = u.copy()
        return cached

    best = None
    for frequency in (1, 2, 3, 4):
        for amplitude in (0.4, 0.8):
            initial = amplitude * np.sin(
                2 * np.pi * frequency * (np.arange(steps) + 0.5) / steps
            )
            result = minimize(
                lambda u: evaluate(u)[:2], initial, jac=True, method="SLSQP",
                bounds=Bounds(-np.ones(steps), np.ones(steps)),
                constraints=[{
                    "type": "ineq", "fun": lambda u: evaluate(u)[2],
                    "jac": lambda u: evaluate(u)[3],
                }], options={"maxiter": 150, "ftol": 1e-10},
            )
            value, _, constraints, _ = evaluate(result.x)
            if result.success and constraints.min() >= -1e-6:
                if best is None or value < best[0]:
                    best = (value, result.x.copy())
    if best is None:
        raise ValueError("No legal ego stopping plan was found")
    controls = [Control(-8, float(s)) for s in best[1] * limits]
    return controls, {
        "inputs": {"ego": ego.__dict__, "lane_centers": list(lane_centers)},
        "objective": "minimize ego stopping front position; no other actor predictions",
        "max_lateral_acceleration_mps2": LATERAL_ACCELERATION,
        "minimum_planned_road_margin_m": ROAD_MARGIN,
        "decision_dt_s": DECISION_DT,
        "stopping_front_progress_m": float(best[0]),
        "controls": [c.__dict__ for c in controls],
    }


class EgoStoppingReference:
    def __init__(self):
        self.started = None
        self.controls = None
        self.specification = None

    def act(self, world):
        if self.started is None:
            if not any(v.x > world.ego.x for v in world.vehicles):
                return Control(0, 0)
            self.controls, self.specification = plan_stop(world.ego, world.lane_centers)
            self.started = world.time_s
        index = round((world.time_s - self.started) / DECISION_DT)
        return self.controls[index] if index < len(self.controls) else Control(-8, 0)


class EscapeReference:
    """Coast into the lane away from a slower adjacent vehicle.

    The target is chosen from visible geometry and velocity at the first
    observation. No scene parameters, future scripts or tested ADS are read.
    """

    def __init__(self):
        self.target_y = None

    def act(self, world):
        if self.target_y is None:
            adjacent = [v for v in world.vehicles if abs(v.y-world.ego.y) > 2]
            actor = min(adjacent, key=lambda v: (v.vx, abs(v.x-world.ego.x)), default=None)
            if actor is None:
                self.target_y = world.ego.y
            else:
                self.target_y = max(world.lane_centers) if actor.y < world.ego.y else min(world.lane_centers)
        return Control(0, lane_steering(world.ego, self.target_y))


class BrakeLaneReference:
    """Brake and enter the other lane when a front vehicle is visible.

    This intentionally cautious reference uses no hidden obstacle information.
    It proves existence of an executable trajectory, not an ADS baseline score.
    """

    def __init__(self):
        self.target_y = None

    def act(self, world):
        if self.target_y is None:
            if not any(v.x > world.ego.x for v in world.vehicles):
                return Control(0, 0)
            self.target_y = max(world.lane_centers,key=lambda y:abs(y-world.ego.y))
        return Control(-8,lane_steering(world.ego,self.target_y))


class CoastLaneReference:
    """Enter the other visible lane at constant speed, without hidden truth.

    Actual clearance to every actor must be checked in the native trajectory;
    the reference does not assume that this manoeuvre is always feasible.
    """

    def __init__(self):
        self.target_y = None

    def act(self, world):
        if self.target_y is None:
            if not any(v.x > world.ego.x for v in world.vehicles):
                return Control(0, 0)
            self.target_y = max(world.lane_centers,key=lambda y:abs(y-world.ego.y))
        return Control(0,lane_steering(world.ego,self.target_y))
