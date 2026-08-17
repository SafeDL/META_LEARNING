"""Disjoint, replayable per-task support/query case tables."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .io import content_hash, write_json
from .task_spec import LogicalScenarioTaskSpec


CASE_SPLITS = ("train_pool", "validation_support", "validation_query", "test_support", "test_query")
LEGACY_CASEBOOK_SCHEMA = "logical_merge_casebook"
CASEBOOK_SCHEMA = "logical_merge_casebook_v2"
SUPPORTED_CASEBOOK_SCHEMAS = frozenset({LEGACY_CASEBOOK_SCHEMA, CASEBOOK_SCHEMA})


def physical_geometry_id(geometry_id: str) -> str:
    return str(geometry_id).split("__rule_", 1)[0]


def build_casebook(task: LogicalScenarioTaskSpec, config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    counts = config["cases"]["per_task"]
    match_variants = bool(config["cases"].get("match_rule_variant_initial_conditions", False))
    rng = np.random.default_rng(task.case_seed)
    value_rng = None
    if match_variants:
        value_seed = int(content_hash({
            "taskbook_seed": int(config.get("experiment", {}).get("taskbook_seed", 7301)),
            "physical_geometry_id": physical_geometry_id(task.geometry_id),
            "purpose": "matched_rule_variant_initial_conditions",
        })[:16], 16)
        value_rng = np.random.default_rng(value_seed)
    result: dict[str, list[dict[str, Any]]] = {}
    sampling = dict(config.get("case_sampling", {}))
    sut_range = sampling.get("sut_initial_speed_mps", (10.0, 14.0))
    adv_range = sampling.get("adversary_initial_speed_mps", (10.0, 17.0))
    used: set[int] = set()
    for group in CASE_SPLITS:
        entries = []
        for index in range(int(counts[group])):
            seed = int(rng.integers(1, 2**31 - 1))
            while seed in used:
                seed = int(rng.integers(1, 2**31 - 1))
            used.add(seed)
            sample_rng = value_rng or rng
            adversary_speed = float(sample_rng.uniform(*adv_range))
            entries.append({
                "case_id": f"{task.task_id}_{group}_{index:03d}", "case_seed": seed,
                "sut_initial_speed_mps": float(sample_rng.uniform(*sut_range)),
                "adversary_initial_speed_mps": adversary_speed,
                # Read compatibility for historical casebooks/scripts.
                "adversary_speed_mps": adversary_speed,
                "adversary_spawn_m": float(sample_rng.uniform(*task.spawn_regions["adversary"])),
                "sut_spawn_m": float(sample_rng.uniform(*task.spawn_regions["sut"])),
            })
        result[group] = entries
    validate_casebook(task, result)
    return result


def validate_casebook(
    task: LogicalScenarioTaskSpec,
    book: Mapping[str, list[Mapping[str, Any]]],
    *,
    schema: str = LEGACY_CASEBOOK_SCHEMA,
) -> None:
    seen: set[str] = set()
    for group in CASE_SPLITS:
        for case in book.get(group, []):
            case_id = str(case.get("case_id", ""))
            if not case_id.startswith(f"{task.task_id}_{group}_"):
                raise ValueError(f"case has incorrect task/split namespace: {case_id}")
            if case_id in seen:
                raise ValueError(f"support/query leakage: duplicate case {case_id}")
            for key in ("case_seed", "adversary_spawn_m", "sut_spawn_m"):
                if key not in case:
                    raise ValueError(f"case {case_id} is missing {key}")
            if "adversary_initial_speed_mps" not in case and "adversary_speed_mps" not in case:
                raise ValueError(f"case {case_id} is missing adversary initial speed")
            if schema == CASEBOOK_SCHEMA:
                required = {
                    "sut_initial_speed_mps", "target_initial_arrival_gap_s",
                    "actual_initial_arrival_gap_s", "initial_relative_speed_mps",
                    "adversary_initial_conflict_distance_m", "sut_initial_conflict_distance_m",
                    "difficulty_class", "calibration_hash",
                }
                missing = required - set(case)
                if missing:
                    raise ValueError(f"v2 case {case_id} misses provenance {sorted(missing)}")
                target = float(case["target_initial_arrival_gap_s"])
                actual = float(case["actual_initial_arrival_gap_s"])
                if not np.isfinite([target, actual]).all() or abs(target - actual) > 0.25:
                    raise ValueError(f"v2 case {case_id} does not realize its target arrival gap")
                if str(case["difficulty_class"]) not in {"heuristic_reachable", "harder"}:
                    raise ValueError(f"v2 case {case_id} has an unsupported difficulty class")
            seen.add(case_id)


def save_casebook(
    task: LogicalScenarioTaskSpec,
    book: Mapping[str, list[dict[str, Any]]],
    output_root: str,
    *,
    schema: str = LEGACY_CASEBOOK_SCHEMA,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    if schema not in SUPPORTED_CASEBOOK_SCHEMAS:
        raise ValueError(f"unsupported casebook schema: {schema!r}")
    validate_casebook(task, book, schema=schema)
    digest = content_hash(book)
    write_json(f"{output_root}/casebooks/{task.task_id}.json", {
        "schema": schema, "task_id": task.task_id, "task_schema": task.schema,
        "sha256": digest, "provenance": dict(provenance or {}), "cases": book,
    })
    return digest


def load_casebook(
    task: LogicalScenarioTaskSpec,
    root: str,
    *,
    required_schema: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    from pathlib import Path
    import json
    payload = json.loads((Path(root) / "casebooks" / f"{task.task_id}.json").read_text(encoding="utf-8"))
    schema = str(payload.get("schema", ""))
    if schema not in SUPPORTED_CASEBOOK_SCHEMAS or payload.get("task_schema") != task.schema or payload.get("task_id") != task.task_id:
        raise ValueError(f"casebook for {task.task_id} is incompatible with the current task")
    if required_schema is not None and schema != required_schema:
        raise ValueError(
            f"casebook for {task.task_id} uses {schema!r}; this run requires {required_schema!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, Mapping):
        raise ValueError(f"casebook for {task.task_id} is malformed")
    book = {key: [dict(row) for row in value] for key, value in cases.items()}
    validate_casebook(task, book, schema=schema)
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
