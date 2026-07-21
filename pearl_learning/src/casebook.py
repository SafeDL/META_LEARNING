"""Disjoint, replayable per-task support/query case tables."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .io import content_hash, write_json
from .task_spec import LogicalScenarioTaskSpec


CASE_SPLITS = ("train_pool", "validation_support", "validation_query", "test_support", "test_query")
CASEBOOK_SCHEMA = "logical_merge_casebook"


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
                "adversary_spawn_m": float(rng.uniform(*task.spawn_regions["adversary"])),
                "sut_spawn_m": float(rng.uniform(*task.spawn_regions["sut"])),
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
            for key in ("case_seed", "adversary_speed_mps", "adversary_spawn_m", "sut_spawn_m"):
                if key not in case:
                    raise ValueError(f"case {case_id} is missing {key}")
            seen.add(case_id)


def save_casebook(task: LogicalScenarioTaskSpec, book: Mapping[str, list[dict[str, Any]]], output_root: str) -> str:
    validate_casebook(task, book)
    digest = content_hash(book)
    write_json(f"{output_root}/casebooks/{task.task_id}.json", {"schema": CASEBOOK_SCHEMA, "task_id": task.task_id, "task_schema": task.schema, "sha256": digest, "cases": book})
    return digest


def load_casebook(task: LogicalScenarioTaskSpec, root: str) -> dict[str, list[dict[str, Any]]]:
    from pathlib import Path
    import json
    payload = json.loads((Path(root) / "casebooks" / f"{task.task_id}.json").read_text(encoding="utf-8"))
    if payload.get("schema") != CASEBOOK_SCHEMA or payload.get("task_schema") != task.schema or payload.get("task_id") != task.task_id:
        raise ValueError(f"casebook for {task.task_id} is incompatible with the current task")
    cases = payload.get("cases")
    if not isinstance(cases, Mapping):
        raise ValueError(f"casebook for {task.task_id} is malformed")
    book = {key: [dict(row) for row in value] for key, value in cases.items()}
    validate_casebook(task, book)
    if content_hash(book) != payload.get("sha256"):
        raise ValueError(f"casebook hash mismatch for {task.task_id}")
    return book


def validate_casebook_disjoint(casebooks: Mapping[str, Mapping[str, list[Mapping[str, Any]]]]) -> None:
    """Reject a seed or case-id shared by different task/split combinations."""
    case_ids: set[str] = set(); seeds: set[int] = set()
    for task_id, book in casebooks.items():
        for split in CASE_SPLITS:
            for case in book.get(split, []):
                if str(case["case_id"]) in case_ids or int(case["case_seed"]) in seeds:
                    raise ValueError(f"case leakage detected at {task_id}/{split}")
                case_ids.add(str(case["case_id"])); seeds.add(int(case["case_seed"]))
