"""Deterministic, balanced scene sampling for Inner Stage 1 pretraining."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..evaluation.baselines import low_discrepancy_samples
from ..scenario.parameter_space import NormalizedScenarioAction, ParameterSpace
from ..scenario.task_spec import logical_parameter_names, ScenarioMiningTaskSpec


@dataclass(frozen=True)
class PretrainSceneSampler:
    """Cover task-local candidates in a near-conflict arrival-time band.

    The sampler is deliberately independent of the frozen Outer policy.  Task
    rank is used only to choose a reproducible point in the coverage sequence;
    calibrated family offsets may be supplied by the Stage 1 casebook.
    Candidate-relative spawn fractions are coupled only when their executable
    ranges can place both vehicles in a bounded interaction window; otherwise
    ordinary low-discrepancy sampling is retained.  The learned model does not
    receive a family label from this sampler.
    """

    tasks: tuple[ScenarioMiningTaskSpec, ...]
    episodes_per_task: int
    seed: int = 0
    eta_offsets_s: Mapping[str, float] | None = None

    # The fallback offset leaves a lawful base gap while giving the residual
    # enough pre-conflict runway to create or relieve interaction pressure.
    interaction_eta_offset_s = 1.5
    nominal_speed_mps = {"cutin": 8.3, "merge": 8.3, "roundabout": 4.5}

    def __post_init__(self) -> None:
        if not self.tasks or self.episodes_per_task < 1:
            raise ValueError("Stage 1 sampler requires tasks and positive episodes_per_task")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Stage 1 sampler task ids must be unique")
        offsets = dict(self.eta_offsets_s or {})
        unknown = set(offsets) - set(self.nominal_speed_mps)
        if unknown:
            raise ValueError(f"unknown interaction ETA families: {sorted(unknown)}")
        if any(not np.isfinite(value) for value in offsets.values()):
            raise ValueError("interaction ETA offsets must be finite")
        object.__setattr__(self, "eta_offsets_s", offsets)

    def __call__(
        self,
        task: ScenarioMiningTaskSpec,
        episode_index: int,
        candidates: Sequence[object],
        space: ParameterSpace,
    ) -> NormalizedScenarioAction:
        if task not in self.tasks:
            raise ValueError(f"task {task.task_id!r} is not in the Stage 1 sampler")
        if not 0 <= episode_index < self.episodes_per_task:
            raise ValueError("episode index is outside the configured task budget")
        rank = self.tasks.index(task)
        candidate_index = (rank + episode_index) % len(candidates)
        sequence_index = rank * self.episodes_per_task + episode_index
        rows = low_discrepancy_samples(
            self.seed,
            space.continuous_dim,
            len(self.tasks) * self.episodes_per_task,
        )
        controls = np.asarray(rows[sequence_index], dtype=np.float32)
        candidate = candidates[candidate_index]
        controls = self._domain_controls(task, controls)
        if getattr(task, "functional_scenario", None) != "cutin":
            controls = self._interaction_aligned_controls(task, candidate, space, controls)
        # The first two controls remain candidate-relative spawn fractions.
        # The executor maps them into the selected route's exact feasible
        # interval; re-encoding through global metre bounds would silently
        # change x0.
        return NormalizedScenarioAction(candidate_index, controls)

    @staticmethod
    def _domain_controls(
        task: ScenarioMiningTaskSpec,
        controls: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(controls, dtype=np.float32).copy()
        bounds = getattr(task, "logical_domain_bounds", {})
        family = getattr(task, "functional_scenario", None)
        if family is None:
            return values
        names = logical_parameter_names(family)
        mask = getattr(task, "logical_parameter_mask", (True,) * len(values))
        for index in range(len(values)):
            if not bool(mask[index]):
                values[index] = 0.0
                continue
            lower, upper = (float(value) for value in bounds.get(names[index], (-1.0, 1.0)))
            values[index] = lower + 0.5 * (float(values[index]) + 1.0) * (upper - lower)
        return values

    @staticmethod
    def _allowed_controls(task: ScenarioMiningTaskSpec, index: int) -> tuple[float, float]:
        names = logical_parameter_names(task.functional_scenario)
        mask = getattr(task, "logical_parameter_mask", (True,) * len(names))
        if not bool(mask[index]):
            return 0.0, 0.0
        bounds = getattr(task, "logical_domain_bounds", {})
        return tuple(float(value) for value in bounds.get(names[index], (-1.0, 1.0)))

    def _interaction_aligned_controls(
        self,
        task: ScenarioMiningTaskSpec,
        candidate: object,
        space: ParameterSpace,
        controls: np.ndarray,
    ) -> tuple[float, ...]:
        """Couple the two spawns only when the candidate admits a safe band."""
        required = (
            "adversary_distance_min_m",
            "adversary_distance_available_m",
            "sut_distance_min_m",
            "sut_distance_available_m",
        )
        family = getattr(task, "functional_scenario", None)
        if (
            family not in self.nominal_speed_mps
            or len(controls) < 4
            or any(not hasattr(candidate, name) for name in required)
        ):
            return tuple(float(value) for value in controls)
        speed_bounds = (
            space.bounds["adversary_initial_speed_mps"],
            space.bounds["sut_initial_speed_mps"],
        )
        speed_controls = (float(controls[2]), float(controls[3]))
        speeds = tuple(
            lower + 0.5 * (control + 1.0) * (upper - lower)
            for control, (lower, upper) in zip(speed_controls, speed_bounds)
        )
        if any(speed <= 0.0 for speed in speeds):
            return tuple(float(value) for value in controls)
        adv_min = float(getattr(candidate, "adversary_distance_min_m"))
        adv_max = float(getattr(candidate, "adversary_distance_available_m"))
        sut_min = float(getattr(candidate, "sut_distance_min_m"))
        sut_max = float(getattr(candidate, "sut_distance_available_m"))
        distance_ranges = []
        for index, (lower, upper) in enumerate(((adv_min, adv_max), (sut_min, sut_max))):
            action_lower, action_upper = self._allowed_controls(task, index)
            distance_ranges.append((
                lower + 0.5 * (action_lower + 1.0) * (upper - lower),
                lower + 0.5 * (action_upper + 1.0) * (upper - lower),
            ))
        offset = float(self.eta_offsets_s.get(family, self.interaction_eta_offset_s))
        eta_lower = max(distance_ranges[0][0] / speeds[0], distance_ranges[1][0] / speeds[1] - offset)
        eta_upper = min(distance_ranges[0][1] / speeds[0], distance_ranges[1][1] / speeds[1] - offset)
        if eta_lower > eta_upper:
            return tuple(float(value) for value in controls)
        domain_lower, domain_upper = self._allowed_controls(task, 0)
        fraction = (float(controls[0]) - domain_lower) / max(domain_upper - domain_lower, 1e-6)
        eta = eta_lower + fraction * (eta_upper - eta_lower)
        distances = (speeds[0] * eta, speeds[1] * (eta + offset))
        ranges = ((adv_min, adv_max), (sut_min, sut_max))
        result = controls.copy()
        for index, (distance, (lower, upper)) in enumerate(zip(distances, ranges)):
            result[index] = 2.0 * (distance - lower) / max(upper - lower, 1e-6) - 1.0
        for index, (lower, upper) in enumerate(speed_bounds, start=2):
            result[index] = 2.0 * (speeds[index - 2] - lower) / (upper - lower) - 1.0
        for index in range(len(result)):
            lower, upper = self._allowed_controls(task, index)
            result[index] = float(np.clip(result[index], lower, upper))
        return tuple(float(value) for value in result)
