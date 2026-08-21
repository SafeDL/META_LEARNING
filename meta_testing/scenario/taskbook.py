"""Persist and validate task splits without putting SUT identities in models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .task_spec import MetaTestTaskSpec


def validate_taskbook(tasks: Iterable[MetaTestTaskSpec]) -> list[MetaTestTaskSpec]:
    rows = list(tasks)
    ids = [task.task_id for task in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique")
    for task in rows:
        task.validate()
    return rows


def save_taskbook(tasks: Iterable[MetaTestTaskSpec], path: str | Path) -> None:
    rows = validate_taskbook(tasks)
    Path(path).write_text(json.dumps([task.to_dict() for task in rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_taskbook(path: str | Path) -> list[MetaTestTaskSpec]:
    return validate_taskbook(MetaTestTaskSpec.from_dict(row) for row in json.loads(Path(path).read_text(encoding="utf-8")))
