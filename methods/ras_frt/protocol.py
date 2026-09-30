"""Read-only validation for the frozen historical IDM library A."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from highway_sim_env.envs.fbrt_unified_env import EXECUTION_CONTRACT


ROOT = Path("results/method_chains/ras_frt/historical_idm")
FROZEN_RUNNER = ROOT / "source_snapshot" / "fbrt_unified_env.py"
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
CHECKPOINTS = (10, 30, 50, 100)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def manifest(split: str) -> list[dict]:
    if split != "history":
        raise ValueError("A is the historical IDM library; target candidates live in D")
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    if protocol["execution_contract"] != EXECUTION_CONTRACT or \
            protocol["runner_sha256"] != digest(FROZEN_RUNNER) or \
            protocol["controller_sha256"] != digest(
                Path("sut_algorithms/highway_env/idm_profiles.py")) or \
            protocol["scenario_schema_sha256"] != digest(
                Path("highway_sim_env/configs/scenario_parameter_space.yaml")):
        raise ValueError("historical simulation snapshot, controller, or scenario schema changed")
    if tuple(protocol["sources"]) != SOURCES or \
            set(protocol["builds"]) != set(SOURCES):
        raise ValueError("historical IDM source set changed")
    from sut_algorithms.highway_env.registry import build_spec_factory

    for build in SOURCES:
        if build_spec_factory(build).fingerprint != \
                protocol["builds"][build]["fingerprint"]:
            raise ValueError(f"historical IDM profile changed: {build}")
    path = ROOT / "history_manifest.jsonl"
    if digest(path) != protocol["history_manifest_sha256"]:
        raise ValueError("historical manifest changed")
    rows = read_jsonl(path)
    if len(rows) != COUNT or \
            len({row["scenario_id"] for row in rows}) != COUNT:
        raise ValueError("invalid historical manifest")
    return rows
