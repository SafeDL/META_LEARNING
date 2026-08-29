"""Task-local, base-safe concrete scenarios for few-shot Inner evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..scenario.parameter_space import NormalizedScenarioAction, ParameterSpace
from ..scenario.task_spec import ScenarioMiningTaskSpec


@dataclass(frozen=True)
class HeadroomCasebook:
    """A casebook that has challenge coverage without a Base critical event."""

    actions: Mapping[str, tuple[NormalizedScenarioAction, ...]]
    metadata: Mapping[str, Any]

    def action_for(self, task_id: str, case_index: int) -> NormalizedScenarioAction:
        values = self.actions[str(task_id)]
        return values[int(case_index)]

    def sampler(
        self, tasks: Sequence[ScenarioMiningTaskSpec], cases_per_task: int
    ) -> "HeadroomSampler":
        return HeadroomSampler(self, tuple(tasks), int(cases_per_task))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "schema": "meta_headroom_casebook",
            "metadata": dict(self.metadata),
            "tasks": {
                task_id: [
                    {"candidate_index": action.candidate_index, "continuous": list(action.continuous)}
                    for action in values
                ]
                for task_id, values in sorted(self.actions.items())
            },
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HeadroomCasebook":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "meta_headroom_casebook":
            raise ValueError("invalid headroom casebook schema")
        actions = {
            str(task_id): tuple(NormalizedScenarioAction(
                int(row["candidate_index"]), tuple(float(value) for value in row["continuous"])
            ) for row in rows)
            for task_id, rows in payload.get("tasks", {}).items()
        }
        return cls(actions, dict(payload.get("metadata", {})))


@dataclass(frozen=True)
class HeadroomSampler:
    casebook: HeadroomCasebook
    tasks: tuple[ScenarioMiningTaskSpec, ...]
    cases_per_task: int

    def __post_init__(self) -> None:
        if self.cases_per_task < 1:
            raise ValueError("headroom case count must be positive")
        for task in self.tasks:
            if len(self.casebook.actions.get(task.task_id, ())) < self.cases_per_task:
                raise ValueError(f"headroom casebook lacks {task.task_id!r}")

    def __call__(
        self,
        task: ScenarioMiningTaskSpec,
        case_index: int,
        candidates: Sequence[object],
        space: ParameterSpace,
    ) -> NormalizedScenarioAction:
        action = self.casebook.action_for(task.task_id, case_index)
        action.validate(space.continuous_dim)
        if action.candidate_index >= len(candidates):
            raise ValueError("headroom candidate is not executable")
        return action


def is_base_safe_headroom(outcome: Mapping[str, Any], challenge_steps: int) -> bool:
    """Accept only lawful, challenge-active Base episodes with risk headroom."""
    return bool(
        outcome.get("is_valid_episode", False)
        and not outcome.get("is_failure", False)
        and int(challenge_steps) > 0
    )
