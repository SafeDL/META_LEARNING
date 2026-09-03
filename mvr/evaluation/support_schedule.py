"""Deterministic task-local support schedules for few-shot validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..scenario.parameter_space import NormalizedScenarioAction, ParameterSpace
from ..scenario.task_spec import ScenarioMiningTaskSpec
from ..training.stage1_sampling import PretrainSceneSampler


def _same_action(
    left: NormalizedScenarioAction, right: NormalizedScenarioAction
) -> bool:
    return bool(
        left.candidate_index == right.candidate_index
        and np.allclose(left.continuous, right.continuous, atol=1e-8, rtol=0.0)
    )


@dataclass
class NestedSupportSchedule:
    """Supply one nested support prefix disjoint from an entire query pool."""

    task: ScenarioMiningTaskSpec
    max_support: int
    seed: int
    forbidden: Sequence[NormalizedScenarioAction] = ()
    _sampler: PretrainSceneSampler = field(init=False, repr=False)
    _actions: list[NormalizedScenarioAction] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.max_support < 1:
            raise ValueError("support schedule requires at least one support scene")
        self._sampler = PretrainSceneSampler(
            (self.task,), max(16, self.max_support * 8), self.seed
        )

    def __call__(
        self,
        task: ScenarioMiningTaskSpec,
        episode_index: int,
        candidates: Sequence[object],
        space: ParameterSpace,
    ) -> NormalizedScenarioAction:
        if task != self.task:
            raise ValueError("support schedule received a different task")
        if not 0 <= episode_index < self.max_support:
            raise ValueError("support schedule was asked for a query episode")
        if episode_index < len(self._actions):
            return self._actions[episode_index]
        for sampler_index in range(episode_index, self.max_support * 16):
            action = self._sampler(task, sampler_index, candidates, space)
            disallowed = tuple(self.forbidden) + tuple(self._actions)
            if not any(_same_action(action, other) for other in disallowed):
                self._actions.append(action)
                return action
        raise RuntimeError("could not construct support scenes disjoint from query pool")

    def provenance(self) -> list[dict[str, object]]:
        if len(self._actions) != self.max_support:
            raise RuntimeError("support action provenance is incomplete")
        return [
            {
                "candidate_index": int(action.candidate_index),
                "continuous": [float(value) for value in action.continuous],
            }
            for action in self._actions
        ]
