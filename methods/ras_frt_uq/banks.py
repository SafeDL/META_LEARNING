"""Read the five historical IDM response banks in A."""

from __future__ import annotations

from pathlib import Path

from highway_sim_env.s01_parameters import valid_label
from methods.ras_frt_uq.protocol import ROOT, read_jsonl


def bank_path(build: str) -> Path:
    return ROOT / "banks" / f"{build}.jsonl"


def labels(build: str, scenarios: list[dict]) -> tuple[list[int | None], list[dict]]:
    lookup = {row["scenario_id"]: row for row in read_jsonl(bank_path(build))}
    ordered = [lookup[scene["scenario_id"]] for scene in scenarios]
    return [valid_label(row) for row in ordered], ordered
