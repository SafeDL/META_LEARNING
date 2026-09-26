"""Import, deduplicate, filter, and incrementally persist measured episodes."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from method_chains.failure_memory_regression.schema_v2 import (
    EpisodeRecord, SCHEMA_VERSION, canonical_json, stable_hash,
)


ROOT = Path("results/method_chains/failure_memory_regression/memory_v2")
CORE = Path("results/method_chains/failure_memory_regression/standard_aligned/core")
LEGACY_CONTRACT = "highway-env;20Hz;functional-scripts;idm-reference"


def _bool(value: object) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _float(value: object) -> float | None:
    if value is None or str(value).strip() in {"", "None", "nan", "NaN"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _json(value: object, fallback):
    if isinstance(value, dict):
        return value
    if value is None or str(value).strip() == "":
        return fallback
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _source_build_fingerprint(build_id: str, contract: str) -> str:
    # Legacy source code is frozen at the repository baseline documented by the plan.
    return stable_hash({"build_id": build_id, "contract": contract,
                        "repository_commit": "6af8a502c90235063e593d2d36c8aede3aa027f2"})


def row_to_record(row: dict, source_file: str) -> EpisodeRecord:
    scenario = _json(row.get("scenario"), {})
    if not isinstance(scenario, dict):
        scenario = {}
    template = str(scenario.get("template_id") or row.get("template_id") or "unknown")
    build_id = str(row.get("build") or "unknown")
    contract = str(row.get("contract") or LEGACY_CONTRACT)
    seed_value = row.get("seed")
    try:
        seed = int(seed_value) if seed_value not in (None, "") else None
    except (TypeError, ValueError):
        seed = None
    event_times = _json(row.get("event_times"), {})
    lead_events = _json(row.get("lead_events"), [])
    collision = _bool(row.get("ego_collision"))
    completed = _bool(row.get("completed"))
    semantic_valid = _bool(row.get("semantic_valid"))
    inconclusive = semantic_valid is False or (completed is False and collision is not True)
    physical = {key: value for key, value in scenario.items()
                if key not in {"scenario_id", "scenario_fingerprint"}}
    scenario_fingerprint = stable_hash({"scenario": physical, "contract": contract})
    active_fields = {"initial_clearance_m", "lane_change_duration_s", "lead_deceleration_mps2",
                     "static_target_ttc_s"}
    fixed_context = {key: value for key, value in scenario.items()
                     if key not in active_fields and key not in {"scenario_id", "template_id"}}
    context_id = str(scenario.get("context_id") or (
        "legacy_exact:" + template + ":" + stable_hash({"fixed": fixed_context,
                                                         "contract": contract})[:12]))
    execution_id = "legacy-" + stable_hash({
        "build_id": build_id, "seed": seed, "scenario_fingerprint": scenario_fingerprint,
        "contract": contract,
    })[:24]
    collision_partner = row.get("collision_partner")
    signature = {
        "event_times": event_times if isinstance(event_times, dict) else {},
        "lead_phases": [item.get("phase") for item in lead_events if isinstance(item, dict)],
        "collision_partner_role": collision_partner or None,
        "source_signature_status": "partial_legacy" if not event_times else "observed_legacy",
    }
    return EpisodeRecord(
        execution_id=execution_id,
        scenario_id=str(row.get("scenario_id") or scenario.get("scenario_id") or "unknown"),
        build_id=build_id,
        build_fingerprint=_source_build_fingerprint(build_id, contract),
        scenario_fingerprint=scenario_fingerprint,
        template_id=template,
        context_id=context_id,
        scenario=scenario,
        simulator_seed=seed,
        execution_contract_version=contract,
        completed=completed,
        ego_collision=collision,
        inconclusive=inconclusive,
        collision_partner_role=str(collision_partner) if collision_partner else None,
        collision_time_s=_float(row.get("collision_time_s")),
        min_ttc=_float(row.get("min_ttc")),
        min_clearance=_float(row.get("min_clearance")),
        public_signature=signature,
        visibility="evaluator_only" if source_file == "target_response_bank.csv" else "historical",
        episode_cost=0,
        trajectory_path=None,
        source_file=source_file,
    )


def load_legacy_archive(core: Path = CORE) -> tuple[list[EpisodeRecord], dict]:
    records: list[EpisodeRecord] = []
    source_counts: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    for filename in ("reference_archive.csv", "target_response_bank.csv"):
        path = core / filename
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                record = row_to_record(row, filename)
                if record.execution_id in seen:
                    duplicate_ids.append(record.execution_id)
                    continue
                seen.add(record.execution_id)
                records.append(record)
                source_counts[filename] += 1
                for name in ("scenario", "seed", "completed", "ego_collision", "semantic_valid"):
                    if row.get(name) in (None, ""):
                        missing_fields[f"{filename}:{name}"] += 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_core": str(core.as_posix()),
        "source_files": dict(source_counts),
        "records_imported": len(records),
        "records_by_build": dict(Counter(record.build_id for record in records)),
        "records_by_template": dict(Counter(record.template_id for record in records)),
        "missing_fields": dict(missing_fields),
        "duplicate_execution_ids": duplicate_ids,
        "deduplicated_record_count": len(duplicate_ids),
        "unknown_values_preserved_as_null_or_partial_signature": True,
        "legacy_rows_relabelled_as_new_physical_data": False,
        "new_physical_episodes": 0,
        "archive_sha256": stable_hash([record.as_dict() for record in records]),
    }
    return records, report


def records_to_jsonl(path: Path, records: Iterable[EpisodeRecord | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            item = record.as_dict() if isinstance(record, EpisodeRecord) else record
            handle.write(canonical_json(item) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class SnapshotStore:
    """Append only observed records and keep reloadable snapshot hashes."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.snapshot_dir = root / "history_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.archive_path = root / "archive_v2.jsonl"

    def load_records(self) -> list[dict]:
        return read_jsonl(self.archive_path)

    def save_records(self, records: Iterable[EpisodeRecord | dict]) -> str:
        rows = [row.as_dict() if isinstance(row, EpisodeRecord) else row for row in records]
        rows.sort(key=lambda row: row["execution_id"])
        records_to_jsonl(self.archive_path, rows)
        digest = stable_hash(rows)
        payload = {"schema_version": SCHEMA_VERSION, "snapshot_hash": digest,
                   "record_count": len(rows), "records_file": "../archive_v2.jsonl"}
        (self.snapshot_dir / f"{digest}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.snapshot_dir / "latest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return digest

    def commit(self, new_records: Iterable[EpisodeRecord | dict]) -> tuple[str, str, list[str]]:
        before_rows = self.load_records()
        before_hash = stable_hash(before_rows)
        by_id = {row["execution_id"]: row for row in before_rows}
        inserted = []
        for record in new_records:
            row = record.as_dict() if isinstance(record, EpisodeRecord) else dict(record)
            row["visibility"] = "historical"
            if row["execution_id"] in by_id:
                continue
            by_id[row["execution_id"]] = row
            inserted.append(row["execution_id"])
        after_hash = self.save_records(by_id.values())
        return before_hash, after_hash, inserted


def build_visibility_view(records: list[dict], *, target_build_id: str,
                          target_family: str | None = None,
                          parent_build_id: str | None = None,
                          regression: bool = False,
                          include_parent_complete: bool = True) -> list[dict]:
    """Construct an allowed history without exposing evaluator-only outcomes."""
    output = []
    for row in records:
        if row.get("visibility") == "evaluator_only":
            continue
        if row.get("build_id") == target_build_id:
            continue
        if target_family and row.get("family") == target_family:
            continue
        if regression and row.get("build_id") != parent_build_id:
            # Regression experiments are explicitly conditioned on their parent's
            # complete records; other sources are omitted from this task view.
            continue
        if regression and include_parent_complete and not row.get("completed"):
            continue
        output.append(row)
    return output
