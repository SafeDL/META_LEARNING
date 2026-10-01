"""Paths and serialization for the historical IDM library A."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_ROOT = Path("results/method_chains/ras_frt_uq")
ROOT = RESULTS_ROOT / "historical_idm"
SOURCES = (
    "idm_source_short_headway", "idm_source_nominal",
    "idm_source_long_headway", "idm_source_limited_brake",
    "idm_source_strong_brake",
)
COUNT = 2048
# The frozen A tuning and first D development run used 100 queries.
# The expanded comparison on the same D defines its 200-query budget locally.
BUDGET = 100
REPEAT_SEEDS = (11, 23, 37, 53, 71)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")


def historical_manifest() -> list[dict]:
    return read_jsonl(ROOT / "history_manifest.jsonl")
