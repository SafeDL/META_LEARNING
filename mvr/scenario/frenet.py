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


@dataclass(frozen=True)
class FrenetReferenceState:
    """Reference geometry observed by the common low-level tracker."""

    progress: float
    lateral_error_m: float
    heading_error_rad: float
    desired_lateral_m: float
    length_m: float
    curvature_m_inv: float
    speed_limit_mps: float
    start_remaining_m: float
    active_lambda_length: float
    active_beta_early: float
    active_beta_late: float
    blend_progress: float
    replan_due: bool


def _base_anchor(contract: FrenetManeuverContract, progress: float) -> float:
    weight = float(quintic_smoothstep(progress))
    return float(
        contract.start_lateral_m
        + (contract.end_lateral_m - contract.start_lateral_m) * weight
    )


def _anchor_scale(contract: FrenetManeuverContract, base: float) -> float:
    available = min(
        base - contract.corridor_lower_m,
        contract.corridor_upper_m - base,
    )
    transition = abs(contract.end_lateral_m - contract.start_lateral_m)
    desired = 0.15 * transition if transition > 1e-6 else 0.35
    return float(max(0.0, min(desired, 0.8 * available)))


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
    bases = [_base_anchor(contract, value) for value in ANCHOR_PROGRESS]
    anchors = [
        float(np.clip(
            base + beta * _anchor_scale(contract, base),
            contract.corridor_lower_m,
            contract.corridor_upper_m,
        ))
        for base, beta in zip(bases, action[1:])
    ]
    if contract.monotonic_lateral:
        lower = min(contract.start_lateral_m, contract.end_lateral_m)
        upper = max(contract.start_lateral_m, contract.end_lateral_m)
        anchors = [float(np.clip(value, lower, upper)) for value in anchors]
        if contract.start_lateral_m <= contract.end_lateral_m:
            anchors[1] = max(anchors[0], anchors[1])
        else:
            anchors[1] = min(anchors[0], anchors[1])
    targets = np.asarray((
        contract.start_lateral_m, 0.0, 0.0,
        contract.end_lateral_m, 0.0, 0.0,
        anchors[0], anchors[1],
    ), dtype=float)
    coefficients = FRENET_CONSTRAINT_INVERSE @ targets
    return FrenetPath(
        contract, float(length), float(action[1]), float(action[2]), coefficients,
    )


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
            self._previous_path = self._blended_path()
            self._active_action = action[:3].copy()
            self._target_path = decode_frenet_path(
                self.contract, self._active_action
            )
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
        path_curvature = lateral_second / max((1.0 + slope**2) ** 1.5, 1e-6)
        curvature = self.contract.spine.curvature_at_s(projection.s_m) + path_curvature
        preview_start = max(projection.s_m, self.contract.start_s_m)
        preview_end = max(preview_start, path.end_s_m)
        preview_curvature = [abs(curvature)]
        for preview_s in np.linspace(preview_start, preview_end, num=33):
            _, _, preview_slope, preview_second = path.evaluate(float(preview_s))
            lateral_curvature = preview_second / max(
                (1.0 + preview_slope**2) ** 1.5, 1e-6
            )
            preview_curvature.append(abs(
                self.contract.spine.curvature_at_s(float(preview_s))
                + lateral_curvature
            ))
        speed_limit = min(
            self.contract.speed_limit_mps,
            float(np.sqrt(
                REFERENCE_LATERAL_ACCELERATION_MPS2
                / max(max(preview_curvature), 1e-5)
            )),
        )
        return FrenetReferenceState(
            progress=progress,
            lateral_error_m=float(projection.lateral_m - desired_lateral),
            heading_error_rad=heading_error,
            desired_lateral_m=desired_lateral,
            length_m=path.length_m,
            curvature_m_inv=float(curvature),
            speed_limit_mps=speed_limit,
            start_remaining_m=float(self.contract.start_s_m - projection.s_m),
            active_lambda_length=float(self._active_action[0]),
            active_beta_early=float(self._active_action[1]),
            active_beta_late=float(self._active_action[2]),
            blend_progress=self.blend_progress,
            replan_due=self.replan_due,
        )
