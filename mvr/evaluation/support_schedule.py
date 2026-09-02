"""Deterministic, task-local support schedules for fixed-query few-shot tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..scenario.parameter_space import NormalizedScenarioAction, ParameterSpace
from ..scenario.task_spec import ScenarioMiningTaskSpec
from ..training.stage1_sampling import PretrainSceneSampler


@dataclass
class FixedQuerySupportSchedule:
    """Supply a nested sequence of support scenes followed by one fixed query.

    The first K support scenes are identical for every comparison at that K
    prefix.  They are sampled from the task-local Logical domain and are never
    allowed to equal the held-out query action.
    """

    task: ScenarioMiningTaskSpec
    query: NormalizedScenarioAction
    support_shots: int
    max_support: int
    seed: int
    _sampler: PretrainSceneSampler = field(init=False, repr=False)
    _actions: list[NormalizedScenarioAction] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.support_shots <= self.max_support:
            raise ValueError("support shots must lie within the declared maximum")
        self._sampler = PretrainSceneSampler((self.task,), self.max_support, self.seed)

    def __call__(
        self,
        task: ScenarioMiningTaskSpec,
        episode_index: int,
        candidates: Sequence[object],
        space: ParameterSpace,
    ) -> NormalizedScenarioAction:
        if task != self.task:
            raise ValueError("support schedule received a different task")
        if episode_index >= self.support_shots:
            return self.query
        action = self._sampler(task, episode_index, candidates, space)
        if (
            action.candidate_index == self.query.candidate_index
            and np.allclose(action.continuous, self.query.continuous, atol=1e-8, rtol=0.0)
        ):
            action = NormalizedScenarioAction(
                (action.candidate_index + 1) % len(candidates), action.continuous,
            )
        self._actions.append(action)
        return action

    def provenance(self) -> list[dict[str, object]]:
        if len(self._actions) != self.support_shots:
            raise RuntimeError("support action provenance is incomplete")
        return [
            {
                "candidate_index": int(action.candidate_index),
                "continuous": [float(value) for value in action.continuous],
            }
            for action in self._actions
        ]
