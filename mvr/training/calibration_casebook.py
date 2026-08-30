"""Calibration-SUT headroom cases with explicit, non-transferable provenance."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..scenario.parameter_space import NormalizedScenarioAction, ParameterSpace
from ..scenario.task_spec import ScenarioMiningTaskSpec


@dataclass(frozen=True)
class CalibrationCase:
    action: NormalizedScenarioAction
    calibration_task_id: str
    calibration_sut_ref: str
    calibration_case_id: int

    def provenance(self) -> dict[str, Any]:
        return {
            "calibration_task_id": self.calibration_task_id,
            "calibration_sut_ref": self.calibration_sut_ref,
            "calibration_case_id": self.calibration_case_id,
        }


@dataclass(frozen=True)
class CalibrationCasebook:
    """Case actions screened on validation SUTs, never declared test-SUT safe."""

    cases: Mapping[str, tuple[CalibrationCase, ...]]
    metadata: Mapping[str, Any]

    def case_for(self, task_id: str, case_index: int) -> CalibrationCase:
        return self.cases[str(task_id)][int(case_index)]

    def sampler(
        self, tasks: Sequence[ScenarioMiningTaskSpec], cases_per_task: int
    ) -> "CalibrationCaseSampler":
        return CalibrationCaseSampler(self, tuple(tasks), int(cases_per_task))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "schema": "calibration_sut_casebook",
            "metadata": dict(self.metadata),
            "tasks": {
                task_id: [
                    {
                        "candidate_index": case.action.candidate_index,
                        "continuous": list(case.action.continuous),
                        **case.provenance(),
                    }
                    for case in values
                ]
                for task_id, values in sorted(self.cases.items())
            },
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationCasebook":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "calibration_sut_casebook":
            raise ValueError("invalid calibration casebook schema")
        cases = {
            str(task_id): tuple(
                CalibrationCase(
                    NormalizedScenarioAction(
                        int(row["candidate_index"]),
                        tuple(float(value) for value in row["continuous"]),
                    ),
                    str(row["calibration_task_id"]),
                    str(row["calibration_sut_ref"]),
                    int(row["calibration_case_id"]),
                )
                for row in rows
            )
            for task_id, rows in payload.get("tasks", {}).items()
        }
        return cls(cases, dict(payload.get("metadata", {})))


@dataclass(frozen=True)
class CalibrationCaseSampler:
    casebook: CalibrationCasebook
    tasks: tuple[ScenarioMiningTaskSpec, ...]
    cases_per_task: int

    def __post_init__(self) -> None:
        if self.cases_per_task < 1:
            raise ValueError("calibration case count must be positive")
        for task in self.tasks:
            if len(self.casebook.cases.get(task.task_id, ())) < self.cases_per_task:
                raise ValueError(f"calibration casebook lacks {task.task_id!r}")

    def __call__(
        self,
        task: ScenarioMiningTaskSpec,
        case_index: int,
        candidates: Sequence[object],
        space: ParameterSpace,
    ) -> NormalizedScenarioAction:
        action = self.casebook.case_for(task.task_id, case_index).action
        action.validate(space.continuous_dim)
        if action.candidate_index >= len(candidates):
            raise ValueError("calibration case candidate is not executable")
        return action


def is_calibration_headroom(outcome: Mapping[str, Any], challenge_steps: int) -> bool:
    return bool(
        outcome.get("is_valid_episode", False)
        and not outcome.get("is_failure", False)
        and int(challenge_steps) > 0
    )
