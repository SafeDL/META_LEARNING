"""Receding-horizon linearized QP and constrained nonlinear trajectory control."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .control import MAX_LATERAL_ACCELERATION, Control, lane_steering
from .planning import bicycle_rollout, predicted_actors


class OptimizationController:
    def __init__(self, route, parameters):
        if route not in ("linear_mpc", "nonlinear_trajectory"):
            raise ValueError(route)
        self.route = route
        self.parameters = dict(parameters)
        self.dt = 0.2
        self.horizon = round(parameters["horizon_s"] / self.dt)
        self.warm = {}
        self.last_diagnostics = {}

    def _linear_model(self, ego):
        n, dt = self.horizon, self.dt
        speed = max(ego.speed, 1)
        step_a = np.eye(4)
        step_a[0, 2] = 0.05
        step_a[1, 3] = speed * 0.05
        step_b = np.zeros((4, 2))
        step_b[2, 0] = 0.05
        step_b[1, 1] = speed * 0.05 / 2
        step_b[3, 1] = speed * 0.05 / ego.length
        a, b = np.eye(4), np.zeros((4, 2))
        for _ in range(round(dt / 0.05)):
            a, b = step_a @ a, step_a @ b + step_b
        state = np.array([ego.x, ego.y, ego.speed, ego.heading])
        jacobian = np.zeros((4, 2 * n))
        states, jacobians = [], []
        for step in range(n):
            state = a @ state
            jacobian = a @ jacobian
            jacobian[:, 2 * step : 2 * step + 2] += b
            states.append(state.copy())
            jacobians.append(jacobian.copy())
        return np.array(states), np.array(jacobians)

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
        base, jac = self._linear_model(world.ego)
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
        steering_limit = np.arctan(MAX_LATERAL_ACCELERATION * world.ego.length / 40**2)
        lower_bound = np.tile([-8, -steering_limit], n)
        upper_bound = np.tile([3, steering_limit], n)
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
        target_y = world.lane_centers[lane]

        def evaluate_batch(u):
            states = bicycle_rollout(world.ego, u.reshape(-1, n, 2), dt=self.dt)
            value = self._loss(states, u.reshape(-1, n, 2), target_y, actors)
            constraints = [
                states[..., 1] - min(world.lane_centers) + 0.5,
                max(world.lane_centers) + 0.5 - states[..., 1],
            ]
            if actors.shape[1]:
                for j, actor in enumerate(world.vehicles):
                    # An enclosing ellipse conservatively separates the rectangles.
                    dx = (states[..., 0] - actors[:, j, 0]) / (
                        (world.ego.length + actor.length) / 2 + 1
                    )
                    dy = (states[..., 1] - actors[:, j, 1]) / (
                        (world.ego.width + actor.width) / 2 + 0.3
                    )
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

        steering_limit = np.arctan(MAX_LATERAL_ACCELERATION * world.ego.length / 40**2)
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
            initial = self.warm.get(
                lane, np.tile([0, lane_steering(world.ego, world.lane_centers[lane])], self.horizon)
            )
            solve = self._linear_solve if self.route == "linear_mpc" else self._nonlinear_solve
            result, feasible = solve(world, lane, initial)
            diagnostics.append(
                {
                    "lane": lane,
                    "success": feasible,
                    "iterations": int(result.nit),
                    "message": str(result.message),
                }
            )
            if feasible:
                self.warm[lane] = np.r_[result.x[2:], result.x[-2:]]
                choices.append((float(result.fun) + 0.2 * (lane != current), lane, result.x))
        self.last_diagnostics = {"route": self.route, "solves": diagnostics}
        if not choices:
            self.last_diagnostics["emergency_brake"] = True
            return Control(-8, lane_steering(world.ego, world.lane_centers[current]))
        _, lane, controls = min(choices, key=lambda c: (c[0], c[1]))
        self.last_diagnostics["selected_lane"] = lane
        return Control(float(controls[0]), float(controls[1])).limited()
