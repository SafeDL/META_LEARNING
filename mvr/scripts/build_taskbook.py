"""Build the reproducible multi-geometry taskbook from MetaDrive maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..map.metadrive_tokenizer import tokenize_road_network
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.interaction import InteractionCandidate
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.registry import load_adapters, load_geometry_catalog
from ..scenario.task_spec import ScenarioMiningTaskSpec


PROFILES = (
    ("idm_cautious", "train"),
    ("idm_defensive", "train"),
    ("idm_normal", "train"),
    ("idm_assertive", "train"),
    ("idm_fast_small_gap", "validation"),
    ("idm_late_response", "test"),
)


def _geometry_hash(geometry_id: str) -> str:
    geometry = load_geometry_catalog()[geometry_id]
    adapters = load_adapters()
    task = ScenarioMiningTaskSpec(
        task_id=f"hash-{geometry_id}",
        sut_ref="idm_cautious",
        functional_scenario=geometry.functional_scenario,
        geometry_id=geometry.geometry_id,
        geometry_hash="0" * 64,
        geometry_seed=geometry.seed,
        adapter_id=geometry.functional_scenario,
        interaction_schema_id="two_route_conflict_v1",
        sut_split="train",
        geometry_split=geometry.split,
    )
    env = adapters[geometry.functional_scenario].build_env(task, {})
    try:
        env.reset()
        tokens = tokenize_road_network(env.current_map.road_network)
        space = mvr_parameter_spaces()[f"{geometry.functional_scenario}_v1"]
        adapter = adapters[geometry.functional_scenario]
        for index in range(len(space.candidates)):
            action = NormalizedScenarioAction(
                index, (0.0,) * space.continuous_dim, space.options[0]
            )
            layout = adapter.resolve_layout(env, task, space.decode(action), space.candidates)
            InteractionCandidate.from_layout(env, layout)
        return tokens.map_hash
    finally:
        env.close()


def build_taskbook(output: str | Path) -> Path:
    tasks = []
    geometries = tuple(load_geometry_catalog().values())
    geometry_hashes = {
        geometry.geometry_id: _geometry_hash(geometry.geometry_id)
        for geometry in geometries
    }
    for geometry in geometries:
        geometry_hash = geometry_hashes[geometry.geometry_id]
        for sut_ref, sut_split in PROFILES:
            task = ScenarioMiningTaskSpec(
                task_id=f"{geometry.geometry_id}-{sut_ref.removeprefix('idm_')}",
                sut_ref=sut_ref,
                functional_scenario=geometry.functional_scenario,
                geometry_id=geometry.geometry_id,
                geometry_hash=geometry_hash,
                geometry_seed=geometry.seed,
                adapter_id=geometry.functional_scenario,
                interaction_schema_id="two_route_conflict_v1",
                sut_split=sut_split,
                geometry_split=geometry.split,
            )
            tasks.append(task.to_dict())
    hashes_by_split = {
        split: {row["geometry_hash"] for row in tasks if row["geometry_split"] == split}
        for split in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if not hashes_by_split[left] or hashes_by_split[left] & hashes_by_split[right]:
            raise RuntimeError(f"geometry {left}/{right} hashes must be non-empty and disjoint")
    expected_counts = {"train": 3, "validation": 1, "test": 1}
    for family in {geometry.functional_scenario for geometry in geometries}:
        for split, expected in expected_counts.items():
            actual = {
                row["geometry_hash"] for row in tasks
                if row["functional_scenario"] == family and row["geometry_split"] == split
            }
            if len(actual) != expected:
                raise RuntimeError(
                    f"{family} requires {expected} unique {split} geometries, got {len(actual)}"
                )
    path = Path(output)
    path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the taskbook from concrete MetaDrive geometries.")
    parser.add_argument("--output", default="mvr/configs/taskbook.json")
    args = parser.parse_args(argv)
    build_taskbook(args.output)


if __name__ == "__main__":
    main()
