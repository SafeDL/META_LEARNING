"""Read and validate the five frozen IDM response banks in A."""

from __future__ import annotations

from pathlib import Path

from highway_sim_env.envs.fbrt_unified_env import EXECUTION_CONTRACT
from highway_sim_env.s01_parameters import valid_label
from highway_sim_env.build_spec import stable_hash
from methods.ras_frt.protocol import COUNT, ROOT, SOURCES, read_jsonl
from sut_algorithms.highway_env.registry import build_spec_factory


def bank_path(build: str) -> Path:
    if build not in SOURCES:
        raise ValueError(f"A only contains its five frozen IDM variants: {build}")
    return ROOT / "banks" / f"{build}.jsonl"


def verify_bank(build: str, scenarios: list[dict], *, complete: bool = True) -> list[dict]:
    path = bank_path(build)
    rows = read_jsonl(path)
    expected = {scene["scenario_id"]: scene for scene in scenarios}
    spec = build_spec_factory(build)
    seen = set()
    for row in rows:
        scene_id = row["scenario_id"]
        if scene_id not in expected or scene_id in seen or \
                row["build_id"] != build or \
                row["build_fingerprint"] != spec.fingerprint or \
                row["simulator_seed"] != expected[scene_id]["simulator_seed"] or \
                row["execution_contract_version"] != EXECUTION_CONTRACT or \
                row["scenario_fingerprint"] != stable_hash({
                    "scenario": expected[scene_id],
                    "execution_contract": EXECUTION_CONTRACT,
                }):
            raise ValueError(f"invalid A response: {build}/{scene_id}")
        seen.add(scene_id)
    if complete and (len(rows) != COUNT or seen != set(expected)):
        raise ValueError(f"incomplete A response: {build} ({len(rows)}/{COUNT})")
    return rows


def labels(build: str, scenarios: list[dict]) -> tuple[list[int | None], list[dict]]:
    rows = verify_bank(build, scenarios, complete=True)
    lookup = {row["scenario_id"]: row for row in rows}
    ordered = [lookup[scene["scenario_id"]] for scene in scenarios]
    return [valid_label(row) for row in ordered], ordered
