"""Receding-horizon linearized QP and constrained nonlinear trajectory control."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from simulator.highway.observation import VehicleState
from simulator.highway.timing import DECISION_DT, PHYSICS_DT, PHYSICS_PER_DECISION

from .control import MAX_STEERING, Control, lane_steering, steering_limit
from .planning import bicycle_rollout, predicted_actors, trajectory_feasibility


class OptimizationController:
    def __init__(self, route, parameters):
        if route not in ("linear_mpc", "nonlinear_trajectory"):
            raise ValueError(route)
        self.route = route
        self.parameters = dict(parameters)
        self.dt = DECISION_DT
        self.horizon = round(parameters["horizon_s"] / self.dt)
        self.warm = {}
        self.last_diagnostics = {}

    def _linear_model(self, ego, reference=None):
        """Affine first-order model about the current warm-start trajectory.

        Linearize the same clipped bicycle dynamics used in execution, including
        nonzero headings and current speed. This avoids the old fixed-speed model.
        """
        reference = np.zeros(2 * self.horizon) if reference is None else np.asarray(reference)
        epsilon = 1e-5
        eye = np.eye(len(reference)) * epsilon
        batch = np.concatenate((reference[None], reference + eye, reference - eye))
        paths = bicycle_rollout(ego, batch.reshape(-1, self.horizon, 2), dt=self.dt)
        jac = np.moveaxis(
            (paths[1 : 1 + len(reference)] - paths[1 + len(reference) :]) / (2 * epsilon), 0, -1
        )
        return paths[0] - np.einsum("nij,j->ni", jac, reference), jac

    def _initial_controls(self, world, lane):
        # A constant steering angle for an entire horizon is a continuous turn,
        # not a lane-change reference. Track the requested lane in the warm start.
        ego, controls = world.ego, []
        for _ in range(self.horizon):
            steer = np.clip(
                lane_steering(ego, world.lane_centers[lane]),
                -0.9 * steering_limit(ego.speed, ego.length),
                0.9 * steering_limit(ego.speed, ego.length),
            )
            controls.append([0.0, float(steer)])
            row = bicycle_rollout(ego, np.array([controls[-1]]), dt=self.dt)[0]
            ego = VehicleState(ego.actor_id, *(float(v) for v in row), ego.length, ego.width)
        return np.asarray(controls).ravel()

    def _loss(self, states, controls, target_y, actors):
        p = self.parameters
        loss = (
            0.8 * (states[..., 2] - p["desired_speed_mps"]) ** 2
            + 2 * (states[..., 1] - target_y) ** 2
            + 80 * states[..., 3] ** 2
        )
        loss = loss.sum(-1) + p["effort_weight"] * (controls[..., 0] ** 2).sum(-1)
        loss += 100 * (controls[..., 1] ** 2).sum(-1)
        if actors.shape[1]:
            dx = actors[:, :, 0] - states[..., :, None, 0]
            dy = actors[:, :, 1] - states[..., :, None, 1]
            front = (dx > 0) & (np.abs(dy) < 2.2)
            deficit = np.maximum(0, 7 + p["headway_s"] * states[..., :, None, 2] - dx)
            loss += 0.3 * (front * deficit**2).sum(axis=(-2, -1))
        return loss

    def _linear_solve(self, world, lane, initial):
        n = self.horizon
        base, jac = self._linear_model(world.ego, initial)
        target_y = world.lane_centers[lane]
        actors = predicted_actors(world, n, self.dt)
        # A fixed maneuver corridor makes the obstacle constraints affine.
        t = np.arange(1, n + 1) * self.dt
        s = np.minimum(t / 2.5, 1)
        reference_y = world.ego.y + (target_y - world.ego.y) * (10 * s**3 - 15 * s**4 + 6 * s**5)
        matrices, lower, upper = [], [], []
        for k in range(n):
            matrices.extend((jac[k, 2], jac[k, 1], jac[k, 3]))
            lower.extend(
                (-base[k, 2], min(world.lane_centers) - 0.5 - base[k, 1], -0.6 - base[k, 3])
            )
            upper.extend(
                (40 - base[k, 2], max(world.lane_centers) + 0.5 - base[k, 1], 0.6 - base[k, 3])
            )
            for j, actor in enumerate(world.vehicles):
                if (
                    abs(actors[k, j, 1] - reference_y[k])
                    > (world.ego.width + actor.width) / 2 + 0.3
                ):
                    continue
                margin = (world.ego.length + actor.length) / 2 + 1
                matrices.append(jac[k, 0])
                if actor.x >= world.ego.x:
                    lower.append(-np.inf)
                    upper.append(actors[k, j, 0] - margin - base[k, 0])
                else:
                    lower.append(actors[k, j, 0] + margin - base[k, 0])
                    upper.append(np.inf)

        def objective(u):
            states = base + np.einsum("nij,j->ni", jac, u)
            controls = u.reshape(n, 2)
            value = self._loss(states, controls, target_y, actors)
            residual = np.column_stack(
                (
                    np.zeros(n),
                    4 * (states[:, 1] - target_y),
                    1.6 * (states[:, 2] - self.parameters["desired_speed_mps"]),
                    160 * states[:, 3],
                )
            )
            # The front mask follows the fixed corridor, preserving a convex QP.
            if actors.shape[1]:
                front = (actors[:, :, 0] > base[:, None, 0]) & (
                    np.abs(actors[:, :, 1] - reference_y[:, None]) < 2.2
                )
                deficit = np.maximum(
                    0,
                    7
                    + self.parameters["headway_s"] * states[:, None, 2]
                    - actors[:, :, 0]
                    + states[:, None, 0],
                )
                # Replace the dynamic-mask headway term with this fixed convex term.
                dx = actors[:, :, 0] - states[:, None, 0]
                dynamic = (dx > 0) & (np.abs(actors[:, :, 1] - states[:, None, 1]) < 2.2)
                value += 0.3 * ((front - dynamic.astype(float)) * deficit**2).sum()
                residual[:, 0] += 0.6 * (front * deficit).sum(1)
                residual[:, 2] += 0.6 * self.parameters["headway_s"] * (front * deficit).sum(1)
            gradient = np.einsum("ni,nij->j", residual, jac)
            effort = controls * [2 * self.parameters["effort_weight"], 200]
            return float(value), gradient + effort.ravel()

        constraint = LinearConstraint(np.asarray(matrices), lower, upper)
        # Both solvers expose the full common input range. The prediction applies
        # the same speed-dependent actuator clipping as native execution.
        reference_path = bicycle_rollout(world.ego, np.asarray(initial).reshape(n, 2), dt=self.dt)
        # Respect the physical envelope along the current reference, instead of
        # letting a local affine model exploit the actuator's saturated region.
        speeds = np.r_[world.ego.speed, reference_path[:-1, 2]]
        limits = np.array([steering_limit(v, world.ego.length) for v in speeds])
        lower_bound = np.column_stack((np.full(n, -8.0), -limits)).ravel()
        upper_bound = np.column_stack((np.full(n, 3.0), limits)).ravel()
        result = minimize(
            objective,
            np.clip(initial, lower_bound, upper_bound),
            jac=True,
            method="SLSQP",
            bounds=Bounds(lower_bound, upper_bound),
            constraints=[constraint],
            options={"maxiter": 80, "ftol": 1e-5},
        )
        feasible = (np.asarray(matrices) @ result.x >= np.asarray(lower) - 1e-4).all()
        feasible &= (np.asarray(matrices) @ result.x <= np.asarray(upper) + 1e-4).all()
        return result, bool(feasible and result.success)

    def _nonlinear_solve(self, world, lane, initial):
        n = self.horizon
        actors = predicted_actors(world, n, self.dt)
        dense_actors = predicted_actors(world, n * PHYSICS_PER_DECISION, PHYSICS_DT)
        target_y = world.lane_centers[lane]

        def evaluate_batch(u):
            dense = bicycle_rollout(
                world.ego, u.reshape(-1, n, 2), dt=self.dt, return_substeps=True
            )
            states = dense[..., 3::4, :]
            value = self._loss(states, u.reshape(-1, n, 2), target_y, actors)
            c, s = np.abs(np.cos(dense[..., 3])), np.abs(np.sin(dense[..., 3]))
            ego_half_y = (s * world.ego.length + c * world.ego.width) / 2
            ego_half_x = (c * world.ego.length + s * world.ego.width) / 2
            constraints = [
                dense[..., 1] - ego_half_y - min(world.lane_centers) + 2,
                max(world.lane_centers) + 2 - dense[..., 1] - ego_half_y,
            ]
            if actors.shape[1]:
                for j, actor in enumerate(world.vehicles):
                    # Enclose the SUM of rotated rectangular bodies, at every
                    # physics tick; outside this ellipse implies separation.
                    actor_half_x = (
                        abs(np.cos(actor.heading)) * actor.length
                        + abs(np.sin(actor.heading)) * actor.width
                    ) / 2
                    actor_half_y = (
                        abs(np.sin(actor.heading)) * actor.length
                        + abs(np.cos(actor.heading)) * actor.width
                    ) / 2
                    dx = (dense[..., 0] - dense_actors[:, j, 0]) / (ego_half_x + actor_half_x + 1)
                    dy = (dense[..., 1] - dense_actors[:, j, 1]) / (ego_half_y + actor_half_y + 0.3)
                    constraints.append(dx**2 / 2 + dy**2 / 2 - 1)
            return value, np.concatenate(constraints, axis=-1)

        cached_u, cached = None, None

        def evaluate(u):
            nonlocal cached_u, cached
            if cached_u is None or not np.array_equal(u, cached_u):
                delta = 1e-5
                batch = np.concatenate((u[None], u[None] + delta * np.eye(len(u))))
                value, constraint = evaluate_batch(batch)
                cached = (
                    value[0],
                    (value[1:] - value[0]) / delta,
                    constraint[0],
                    ((constraint[1:] - constraint[0]) / delta).T,
                )
                cached_u = u.copy()
            return cached

        steering_limit = MAX_STEERING
        lower_bound = np.tile([-8, -steering_limit], n)
        upper_bound = np.tile([3, steering_limit], n)
        result = minimize(
            lambda u: evaluate(u)[:2],
            np.clip(initial, lower_bound, upper_bound),
            jac=True,
            method="SLSQP",
            bounds=Bounds(lower_bound, upper_bound),
            constraints=[
                {"type": "ineq", "fun": lambda u: evaluate(u)[2], "jac": lambda u: evaluate(u)[3]}
            ],
            options={"maxiter": 60, "ftol": 1e-4},
        )
        return result, bool(result.success and evaluate(result.x)[2].min() >= -1e-4)

    def act(self, world):
        choices, diagnostics = [], []
        current = min(
            range(len(world.lane_centers)), key=lambda i: abs(world.lane_centers[i] - world.ego.y)
        )
        for lane in range(max(0, current - 1), min(len(world.lane_centers), current + 2)):
            initial = self.warm[lane] if lane in self.warm else self._initial_controls(world, lane)
            solve = self._linear_solve if self.route == "linear_mpc" else self._nonlinear_solve
            result, feasible = solve(world, lane, initial)
            model_feasible, minimum_margin = trajectory_feasibility(
                world, result.x.reshape(-1, 2), dt=self.dt
            )
            diagnostics.append(
                {
                    "lane": lane,
                    "success": bool(feasible and model_feasible),
                    "solver_feasible": feasible,
                    "native_prediction_feasible": model_feasible,
                    "minimum_prediction_margin_m": minimum_margin,
                    "iterations": int(result.nit),
                    "message": str(result.message),
                }
            )
            if feasible and model_feasible:
                self.warm[lane] = np.r_[result.x[2:], result.x[-2:]]
                choices.append((float(result.fun) + 0.2 * (lane != current), lane, result.x))
        self.last_diagnostics = {"route": self.route, "solves": diagnostics}
        if not choices:
            self.last_diagnostics["emergency_brake"] = True
            return Control(-8, lane_steering(world.ego, world.lane_centers[current]))
        _, lane, controls = min(choices, key=lambda c: (c[0], c[1]))
        self.last_diagnostics["selected_lane"] = lane
        return Control(float(controls[0]), float(controls[1])).limited()
