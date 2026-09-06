"""Scenario-neutral Frenet path planning for the Inner SAC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .route_geometry import RoutePolyline, wrap_to_pi


def quintic_smoothstep(progress: float | np.ndarray) -> float | np.ndarray:
    value = np.asarray(progress, dtype=float)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


ANCHOR_PROGRESS = (1.0 / 3.0, 2.0 / 3.0)
MIN_PATH_LENGTH_M = 30.0
MAX_PATH_LENGTH_M = 60.0
REFERENCE_LATERAL_ACCELERATION_MPS2 = 0.6
REFERENCE_MAX_DECELERATION_MPS2 = 6.0
REFERENCE_MAX_JERK_MPS3 = 1.5
SHAPE_PROGRESS_SCALE = 0.15
SPEED_PROFILE_SPACING_M = 1.0


def _constraint_matrix() -> np.ndarray:
    rows: list[list[float]] = []
    for x, derivative in (
        (0.0, 0), (0.0, 1), (0.0, 2),
        (1.0, 0), (1.0, 1), (1.0, 2),
        (ANCHOR_PROGRESS[0], 0), (ANCHOR_PROGRESS[1], 0),
    ):
        row = []
        for power in range(8):
            if power < derivative:
                row.append(0.0)
            else:
                coefficient = 1.0
                for offset in range(derivative):
                    coefficient *= power - offset
                row.append(coefficient * x ** (power - derivative))
        rows.append(row)
    return np.asarray(rows, dtype=float)


FRENET_CONSTRAINT_INVERSE = np.linalg.inv(_constraint_matrix())


@dataclass(frozen=True)
class FrenetManeuverContract:
    """Adapter-provided legal route corridor for one maneuver candidate."""

    spine: RoutePolyline
    start_s_m: float
    start_lateral_m: float
    end_lateral_m: float
    min_length_m: float
    max_length_m: float
    corridor_lower_m: float
    corridor_upper_m: float
    speed_limit_mps: float
    monotonic_lateral: bool

    def __post_init__(self) -> None:
        values = np.asarray((
            self.start_s_m, self.start_lateral_m, self.end_lateral_m,
            self.min_length_m, self.max_length_m, self.corridor_lower_m,
            self.corridor_upper_m, self.speed_limit_mps,
        ), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Frenet maneuver contract must be finite")
        if not 0.0 <= self.start_s_m < self.spine.length_m:
            raise ValueError("Frenet maneuver start lies outside its spine")
        if not 0.0 < self.min_length_m <= self.max_length_m:
            raise ValueError("Frenet path-length interval is invalid")
        if self.start_s_m + self.min_length_m > self.spine.length_m + 1e-6:
            raise ValueError("Frenet spine is too short for its minimum path")
        if not self.corridor_lower_m < self.corridor_upper_m:
            raise ValueError("Frenet corridor must be ordered")


@dataclass(frozen=True)
class FrenetPath:
    """One uniquely decoded seventh-degree lateral path."""

    contract: FrenetManeuverContract
    length_m: float
    beta_early: float
    beta_late: float
    coefficients: np.ndarray

    @property
    def end_s_m(self) -> float:
        return self.contract.start_s_m + self.length_m

    def evaluate(self, s_m: float) -> tuple[float, float, float, float]:
        progress = float(np.clip(
            (float(s_m) - self.contract.start_s_m) / max(self.length_m, 1e-6),
            0.0,
            1.0,
        ))
        powers = np.asarray([progress**index for index in range(8)], dtype=float)
        first = np.asarray([
            0.0 if index == 0 else index * progress ** (index - 1)
            for index in range(8)
        ], dtype=float)
        second = np.asarray([
            0.0 if index < 2 else index * (index - 1) * progress ** (index - 2)
            for index in range(8)
        ], dtype=float)
        lateral = float(np.dot(self.coefficients, powers))
        slope = float(np.dot(self.coefficients, first) / self.length_m)
        second_derivative = float(
            np.dot(self.coefficients, second) / self.length_m**2
        )
        return progress, lateral, slope, second_derivative

    def curvature_m_inv(self, s_m: float) -> float:
        _, _, slope, lateral_second = self.evaluate(s_m)
        lateral_curvature = lateral_second / max((1.0 + slope**2) ** 1.5, 1e-6)
        return float(self.contract.spine.curvature_at_s(float(s_m)) + lateral_curvature)

    def speed_profile(
        self,
        current_s_m: float,
        *,
        max_deceleration_mps2: float = REFERENCE_MAX_DECELERATION_MPS2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return spatial samples, local curve caps, and a braking envelope."""
        start = float(np.clip(current_s_m, 0.0, self.contract.spine.length_m))
        end = max(start, self.end_s_m)
        count = max(2, int(np.ceil((end - start) / SPEED_PROFILE_SPACING_M)) + 1)
        positions = np.linspace(start, end, num=count, dtype=float)
        local_limits = np.asarray([
            min(
                self.contract.speed_limit_mps,
                float(np.sqrt(
                    REFERENCE_LATERAL_ACCELERATION_MPS2
                    / max(abs(self.curvature_m_inv(float(s_m))), 1e-5)
                )),
            )
            for s_m in positions
        ], dtype=float)
        allowed = local_limits.copy()
        for index in range(len(positions) - 2, -1, -1):
            distance = float(positions[index + 1] - positions[index])
            reachable = np.sqrt(
                allowed[index + 1] ** 2
                + 2.0 * float(max_deceleration_mps2) * distance
            )
            allowed[index] = min(allowed[index], reachable)
        return positions, local_limits, allowed


@dataclass(frozen=True)
class FrenetReferenceState:
    """Reference geometry observed by the common low-level tracker."""

    progress: float
    lateral_error_m: float
    heading_error_rad: float
    desired_lateral_m: float
    length_m: float
    curvature_m_inv: float
    curvature_speed_limit_mps: float
    speed_limit_mps: float
    start_remaining_m: float
    active_lambda_length: float
    active_beta_early: float
    active_beta_late: float
    blend_progress: float
    replan_due: bool
    path_projection_scale: float
    path_speed_feasible: bool


def _path_is_monotonic(path: FrenetPath, tolerance: float = 1e-9) -> bool:
    first = np.polynomial.polynomial.polyder(path.coefficients)
    second = np.polynomial.polynomial.polyder(first)
    points = [0.0, 1.0]
    if len(second) > 1:
        for root in np.roots(second[::-1]):
            if abs(float(root.imag)) <= 1e-8 and 0.0 < float(root.real) < 1.0:
                points.append(float(root.real))
    derivatives = np.polynomial.polynomial.polyval(points, first)
    if path.contract.end_lateral_m >= path.contract.start_lateral_m:
        return bool(np.min(derivatives) >= -tolerance)
    return bool(np.max(derivatives) <= tolerance)


def _build_path(
    contract: FrenetManeuverContract,
    length: float,
    beta_early: float,
    beta_late: float,
) -> FrenetPath:
    base_progress = np.asarray([
        float(quintic_smoothstep(value)) for value in ANCHOR_PROGRESS
    ])
    progress = np.clip(
        base_progress
        + SHAPE_PROGRESS_SCALE * np.asarray((beta_early, beta_late), dtype=float),
        0.0,
        1.0,
    )
    if contract.monotonic_lateral:
        progress[1] = max(progress[0], progress[1])
    effective_beta = (progress - base_progress) / SHAPE_PROGRESS_SCALE
    anchors = contract.start_lateral_m + (
        contract.end_lateral_m - contract.start_lateral_m
    ) * progress
    targets = np.asarray((
        contract.start_lateral_m, 0.0, 0.0,
        contract.end_lateral_m, 0.0, 0.0,
        anchors[0], anchors[1],
    ), dtype=float)
    return FrenetPath(
        contract,
        float(length),
        float(effective_beta[0]),
        float(effective_beta[1]),
        FRENET_CONSTRAINT_INVERSE @ targets,
    )


def decode_frenet_path(
    contract: FrenetManeuverContract,
    normalized: np.ndarray | tuple[float, float, float],
) -> FrenetPath:
    """Map normalized length/anchor actions to one feasible path."""
    action = np.asarray(normalized, dtype=float).reshape(-1)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("Frenet path action must contain length and two anchors")
    action = np.clip(action, -1.0, 1.0)
    length = contract.min_length_m + 0.5 * (action[0] + 1.0) * (
        contract.max_length_m - contract.min_length_m
    )
    requested = _build_path(contract, length, float(action[1]), float(action[2]))
    if not contract.monotonic_lateral or _path_is_monotonic(requested):
        return requested
    lower, upper = 0.0, 1.0
    for _ in range(40):
        scale = 0.5 * (lower + upper)
        candidate = _build_path(
            contract,
            length,
            scale * requested.beta_early,
            scale * requested.beta_late,
        )
        if _path_is_monotonic(candidate):
            lower = scale
        else:
            upper = scale
    return _build_path(
        contract,
        length,
        lower * requested.beta_early,
        lower * requested.beta_late,
    )


def _speed_feasible(
    path: FrenetPath,
    current_s_m: float,
    current_speed_mps: float,
    current_acceleration_mps2: float,
    decision_seconds: float,
) -> bool:
    """Check the path under maximum discrete jerk-limited braking."""
    positions, local_limits, _ = path.speed_profile(current_s_m)
    position = float(positions[0])
    speed = max(0.0, float(current_speed_mps))
    acceleration = float(current_acceleration_mps2)
    dt = float(decision_seconds)
    while position < float(positions[-1]) - 1e-6 and speed > 1e-6:
        limit = float(np.interp(position, positions, local_limits))
        if speed > limit + 1e-6:
            return False
        acceleration = max(
            -REFERENCE_MAX_DECELERATION_MPS2,
            acceleration - REFERENCE_MAX_JERK_MPS3 * dt,
        )
        next_speed = max(0.0, speed + acceleration * dt)
        position += max(0.0, 0.5 * (speed + next_speed) * dt)
        speed = next_speed
    if speed <= 1e-6:
        return True
    return bool(speed <= float(local_limits[-1]) + 1e-6)


class FrenetPathPlanner:
    """Apply low-frequency path actions and expose a blended reference."""

    replan_interval_steps = 5

    def __init__(self, contract: FrenetManeuverContract) -> None:
        self.contract = contract
        self._active_action = np.zeros(3, dtype=np.float32)
        initial = decode_frenet_path(contract, self._active_action)
        self._previous_path = initial
        self._target_path = initial
        self._maneuver_steps = 0
        self._blend_step = self.replan_interval_steps
        self._locked = False
        self._path_projection_scale = 1.0
        self._path_speed_feasible = True

    @property
    def active_action(self) -> np.ndarray:
        return self._active_action.copy()

    @property
    def replan_due(self) -> bool:
        return self._maneuver_steps % self.replan_interval_steps == 0

    @property
    def blend_progress(self) -> float:
        return float(np.clip(
            self._blend_step / self.replan_interval_steps, 0.0, 1.0
        ))

    def lock(self) -> None:
        """Freeze path replanning after the candidate route is reached."""
        self._locked = True

    def apply(
        self,
        raw_action: np.ndarray,
        maneuver_active: bool,
        current_s_m: float,
        current_speed_mps: float,
        current_acceleration_mps2: float,
        decision_seconds: float,
    ) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Inner SAC must emit one finite four-dimensional action")
        action = np.clip(action, -1.0, 1.0)
        if maneuver_active and not self._locked and self.replan_due:
            minimum_endpoint = min(
                self.contract.start_s_m + self.contract.max_length_m,
                float(current_s_m) + 12.0,
            )
            minimum_length = max(
                self.contract.min_length_m,
                minimum_endpoint - self.contract.start_s_m,
            )
            requested_length = self.contract.min_length_m + 0.5 * (
                float(action[0]) + 1.0
            ) * (self.contract.max_length_m - self.contract.min_length_m)
            feasible_length = float(np.clip(
                requested_length, minimum_length, self.contract.max_length_m
            ))
            interval = self.contract.max_length_m - self.contract.min_length_m
            action[0] = 0.0 if interval <= 1e-6 else (
                2.0 * (feasible_length - self.contract.min_length_m) / interval - 1.0
            )
            requested = action[:3].copy()
            requested[0] = 0.0 if interval <= 1e-6 else (
                2.0 * (feasible_length - self.contract.min_length_m) / interval - 1.0
            )
            requested_path = decode_frenet_path(self.contract, requested)
            requested[1:] = (
                requested_path.beta_early, requested_path.beta_late
            )
            selected = requested_path
            projection_scale = 1.0
            feasible = _speed_feasible(
                selected, current_s_m, current_speed_mps,
                current_acceleration_mps2, decision_seconds,
            )
            if not feasible:
                neutral = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
                neutral_path = decode_frenet_path(self.contract, neutral)
                if _speed_feasible(
                    neutral_path, current_s_m, current_speed_mps,
                    current_acceleration_mps2, decision_seconds,
                ):
                    lower, upper = 0.0, 1.0
                    for _ in range(30):
                        scale = 0.5 * (lower + upper)
                        candidate_action = neutral + scale * (requested - neutral)
                        candidate = decode_frenet_path(
                            self.contract, candidate_action
                        )
                        if _speed_feasible(
                            candidate, current_s_m, current_speed_mps,
                            current_acceleration_mps2, decision_seconds,
                        ):
                            lower = scale
                        else:
                            upper = scale
                    projection_scale = lower
                    selected = decode_frenet_path(
                        self.contract, neutral + lower * (requested - neutral)
                    )
                    feasible = True
                else:
                    projection_scale = 0.0
                    selected = neutral_path
            self._previous_path = self._blended_path()
            interval = self.contract.max_length_m - self.contract.min_length_m
            effective_lambda = 0.0 if interval <= 1e-6 else (
                2.0 * (selected.length_m - self.contract.min_length_m) / interval
                - 1.0
            )
            self._active_action = np.asarray((
                effective_lambda, selected.beta_early, selected.beta_late,
            ), dtype=np.float32)
            self._target_path = selected
            self._path_projection_scale = float(projection_scale)
            self._path_speed_feasible = bool(feasible)
            self._blend_step = 0
        effective = np.concatenate((self._active_action, action[3:4])).astype(
            np.float32
        )
        if maneuver_active:
            self._maneuver_steps += 1
            self._blend_step = min(
                self._blend_step + 1, self.replan_interval_steps
            )
        return effective

    def reference_points(self, samples: int = 81) -> list[list[float]]:
        """Return world-space samples of the currently effective path."""
        if samples < 2:
            raise ValueError("reference path rendering requires at least two samples")
        path = self._blended_path()
        points = []
        for s_m in np.linspace(
            self.contract.start_s_m, path.end_s_m, num=samples, dtype=float
        ):
            _, lateral, _, _ = path.evaluate(float(s_m))
            points.append(
                self.contract.spine.position_at_s(float(s_m), lateral).tolist()
            )
        return points

    def _blended_path(self) -> FrenetPath:
        weight = float(quintic_smoothstep(self.blend_progress))
        coefficients = (
            (1.0 - weight) * self._previous_path.coefficients
            + weight * self._target_path.coefficients
        )
        length = (
            (1.0 - weight) * self._previous_path.length_m
            + weight * self._target_path.length_m
        )
        return FrenetPath(
            self.contract,
            float(length),
            float((1.0 - weight) * self._previous_path.beta_early + weight * self._target_path.beta_early),
            float((1.0 - weight) * self._previous_path.beta_late + weight * self._target_path.beta_late),
            coefficients,
        )

    def reference(self, position: Any, heading: float) -> FrenetReferenceState:
        path = self._blended_path()
        projection = self.contract.spine.projection(position, heading)
        progress, desired_lateral, slope, lateral_second = path.evaluate(
            projection.s_m
        )
        tangent = self.contract.spine.tangent_at_s(projection.s_m)
        route_heading = float(np.arctan2(tangent[1], tangent[0]))
        desired_heading = route_heading + float(np.arctan(slope))
        heading_error = wrap_to_pi(float(heading) - desired_heading)
        curvature = path.curvature_m_inv(projection.s_m)
        _, local_limits, allowed = path.speed_profile(projection.s_m)
        return FrenetReferenceState(
            progress=progress,
            lateral_error_m=float(projection.lateral_m - desired_lateral),
            heading_error_rad=heading_error,
            desired_lateral_m=desired_lateral,
            length_m=path.length_m,
            curvature_m_inv=float(curvature),
            curvature_speed_limit_mps=float(local_limits[0]),
            speed_limit_mps=float(allowed[0]),
            start_remaining_m=float(self.contract.start_s_m - projection.s_m),
            active_lambda_length=float(self._active_action[0]),
            active_beta_early=float(self._active_action[1]),
            active_beta_late=float(self._active_action[2]),
            blend_progress=self.blend_progress,
            replan_due=self.replan_due,
            path_projection_scale=self._path_projection_scale,
            path_speed_feasible=self._path_speed_feasible,
        )
