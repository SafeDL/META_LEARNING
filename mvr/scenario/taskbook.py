"""Persist and validate task splits without putting SUT identities in models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .task_spec import ScenarioMiningTaskSpec


def validate_taskbook(tasks: Iterable[ScenarioMiningTaskSpec]) -> list[ScenarioMiningTaskSpec]:
    rows = list(tasks)
    if len({task.task_id for task in rows}) != len(rows):
        raise ValueError("task ids must be unique")
    for task in rows:
        task.validate()
    hashes_by_split = {
        split: {task.geometry_hash for task in rows if task.geometry_split == split}
        for split in ("train", "validation", "test")
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if hashes_by_split[left] & hashes_by_split[right]:
            raise ValueError(f"{left} and {right} geometry hashes must be disjoint")
    families = {task.functional_scenario for task in rows}
    for family in families:
        family_splits = {
            split: {task.geometry_hash for task in rows
                    if task.functional_scenario == family and task.geometry_split == split}
            for split in ("train", "validation", "test")
        }
        if any(not family_splits[split] for split in ("train", "validation", "test")):
            raise ValueError(f"{family} must contain train, validation, and test geometries")
    return rows

def load_taskbook(path: str | Path) -> list[ScenarioMiningTaskSpec]:
    return validate_taskbook(ScenarioMiningTaskSpec.from_dict(row) for row in json.loads(Path(path).read_text(encoding="utf-8")))
