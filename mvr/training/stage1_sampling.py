"""Deterministic, balanced scene sampling for Inner Stage 1 pretraining."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..evaluation.baselines import low_discrepancy_samples
from ..scenario.parameter_space import NormalizedScenarioAction, ParameterSpace
from ..scenario.task_spec import ScenarioMiningTaskSpec


@dataclass(frozen=True)
class PretrainSceneSampler:
    """Cover candidates/options while stratifying conflict-relative x0.

    The sampler is deliberately independent of the frozen Outer policy.  Task
    rank is used only to choose a reproducible point in the coverage sequence;
    it is not provided to a learned model.
    """

    tasks: tuple[ScenarioMiningTaskSpec, ...]
    episodes_per_task: int
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.tasks or self.episodes_per_task < 1:
            raise ValueError("Stage 1 sampler requires tasks and positive episodes_per_task")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Stage 1 sampler task ids must be unique")

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
        option_index = (rank + episode_index) % len(space.options)
        sequence_index = rank * self.episodes_per_task + episode_index
        rows = low_discrepancy_samples(
            self.seed,
            space.continuous_dim,
            len(self.tasks) * self.episodes_per_task,
        )
        controls = np.asarray(rows[sequence_index], dtype=np.float32)
        # The first two controls are candidate-relative spawn fractions.  The
        # executor maps them into the selected route's exact feasible interval;
        # re-encoding through global metre bounds would silently change x0.
        return NormalizedScenarioAction(candidate_index, controls, space.options[option_index])
