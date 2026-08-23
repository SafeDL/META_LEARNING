"""Explicit SUT/geometry OOD regimes for fixed-budget evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..scenario.task_spec import ScenarioMiningTaskSpec


@dataclass(frozen=True)
class EvaluationRegime:
    name: str
    sut_split: str
    geometry_split: str


REGIMES = (
    EvaluationRegime("R1_seen_sut_seen_geometry", "train", "train"),
    EvaluationRegime("R2_unseen_sut_seen_geometry", "test", "train"),
    EvaluationRegime("R3_seen_sut_unseen_geometry", "train", "test"),
    EvaluationRegime("R4_unseen_sut_unseen_geometry", "test", "test"),
)


def select_regime_tasks(
    tasks: Iterable[ScenarioMiningTaskSpec], regime: EvaluationRegime, family: str = "all"
) -> list[ScenarioMiningTaskSpec]:
    selected = [
        task for task in tasks
        if task.sut_split == regime.sut_split
        and task.geometry_split == regime.geometry_split
        and task.functional_split == "train"
        and (family == "all" or task.functional_scenario == family)
    ]
    if not selected:
        raise ValueError(f"taskbook has no tasks for {regime.name}")
    return selected
