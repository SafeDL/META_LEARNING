"""Disjoint, replayable per-task support/query case tables."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .io import content_hash, write_json
from .task_spec import LogicalScenarioTaskSpec


CASE_SPLITS = ("train_pool", "validation_support", "validation_query", "test_support", "test_query")


def build_casebook(task: LogicalScenarioTaskSpec, config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    counts = config["cases"]["per_task"]
    rng = np.random.default_rng(task.case_seed)
    result: dict[str, list[dict[str, Any]]] = {}
    used: set[int] = set()
    for group in CASE_SPLITS:
        entries = []
        for index in range(int(counts[group])):
            seed = int(rng.integers(1, 2**31 - 1))
            while seed in used:
                seed = int(rng.integers(1, 2**31 - 1))
            used.add(seed)
            entries.append({
                "case_id": f"{task.task_id}_{group}_{index:03d}", "case_seed": seed,
                "adversary_speed_mps": float(rng.uniform(10.0, 17.0)),
                "arrival_offset_m": float(rng.uniform(-8.0, 8.0)),
            })
        result[group] = entries
    validate_casebook(task, result)
    return result


def validate_casebook(task: LogicalScenarioTaskSpec, book: Mapping[str, list[Mapping[str, Any]]]) -> None:
    seen: set[str] = set()
    for group in CASE_SPLITS:
        for case in book.get(group, []):
            case_id = str(case.get("case_id", ""))
            if not case_id.startswith(f"{task.task_id}_{group}_"):
                raise ValueError(f"case has incorrect task/split namespace: {case_id}")
            if case_id in seen:
                raise ValueError(f"support/query leakage: duplicate case {case_id}")
            seen.add(case_id)


def save_casebook(task: LogicalScenarioTaskSpec, book: Mapping[str, list[dict[str, Any]]], output_root: str) -> str:
    validate_casebook(task, book)
    digest = content_hash(book)
    write_json(f"{output_root}/casebooks/{task.task_id}.json", {"task_id": task.task_id, "sha256": digest, "cases": book})
    return digest
