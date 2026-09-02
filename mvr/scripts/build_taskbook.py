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
from ..scenario.task_spec import logical_parameter_names, ScenarioMiningTaskSpec
from ..scenario.taskbook import validate_taskbook


PROFILES = (
    ("idm_cautious", "train"),
    ("idm_defensive", "train"),
    ("idm_normal", "train"),
    ("idm_assertive", "train"),
    ("idm_fast_small_gap", "validation"),
    ("idm_late_response", "test"),
)

LOGICAL_DOMAINS = (
    ("interaction_core", "train", (-0.25, 0.25)),
    ("timing_shift", "validation", (-0.85, -0.35)),
    ("tight_gap", "test", (0.35, 0.85)),
)

CUTIN_LOGICAL_DOMAINS = (
    ("cutin_interaction_core", "train", {
        "cutin_gap_at_start_m": (-0.15, 0.15), "sut_initial_speed_mps": (-0.15, 0.15),
        "relative_speed_mps": (-0.15, 0.15), "cutin_start_progress": (-0.15, 0.15),
        "cutin_start_time_s": (-0.15, 0.15), "lane_change_length_m": (-0.15, 0.15),
    }),
    ("cutin_late_fast", "validation", {
        "cutin_gap_at_start_m": (-0.65, -0.25), "sut_initial_speed_mps": (0.25, 0.65),
        "relative_speed_mps": (0.25, 0.75), "cutin_start_progress": (-0.65, -0.25),
        "cutin_start_time_s": (0.25, 0.65), "lane_change_length_m": (-0.65, -0.25),
    }),
    ("cutin_tight_gap", "test", {
        "cutin_gap_at_start_m": (0.35, 0.85), "sut_initial_speed_mps": (-0.85, -0.35),
        "relative_speed_mps": (-0.85, -0.35), "cutin_start_progress": (0.35, 0.85),
        "cutin_start_time_s": (-0.85, -0.35), "lane_change_length_m": (0.35, 0.85),
    }),
)


def _logical_mask(family: str) -> tuple[bool, ...]:
    if family == "cutin":
        return (True, True, True, True, True, True)
    if family in {"merge", "roundabout"}:
        return (True, True, True, True, False)
    raise ValueError(f"unsupported scenario family: {family!r}")


def _logical_bounds(
    family: str, interval: tuple[float, float], mask: tuple[bool, ...]
) -> dict[str, tuple[float, float]]:
    return {
        name: interval if active else (-1.0, 1.0)
        for name, active in zip(logical_parameter_names(family), mask)
    }


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
        interaction_schema_id="two_route_conflict",
        sut_split="train",
        geometry_split=geometry.split,
        logical_domain_id="hash_probe",
        logical_domain_bounds=_logical_bounds(
            geometry.functional_scenario, (-1.0, 1.0),
            _logical_mask(geometry.functional_scenario),
        ),
        logical_parameter_mask=_logical_mask(geometry.functional_scenario),
        logical_split="train",
    )
    env = adapters[geometry.functional_scenario].build_env(task, {})
    try:
        env.reset()
        tokens = tokenize_road_network(env.current_map.road_network)
        space = mvr_parameter_spaces()[geometry.functional_scenario]
        adapter = adapters[geometry.functional_scenario]
        for index in range(len(space.candidates)):
            action = NormalizedScenarioAction(
                index, (0.0,) * space.continuous_dim
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
            domains = (
                CUTIN_LOGICAL_DOMAINS
                if geometry.functional_scenario == "cutin" else LOGICAL_DOMAINS
            )
            for domain_id, logical_split, interval in domains:
                mask = _logical_mask(geometry.functional_scenario)
                bounds = (
                    interval if geometry.functional_scenario == "cutin"
                    else _logical_bounds(geometry.functional_scenario, interval, mask)
                )
                task = ScenarioMiningTaskSpec(
                    task_id=f"{geometry.geometry_id}-{sut_ref.removeprefix('idm_')}-{domain_id}",
                    sut_ref=sut_ref,
                    functional_scenario=geometry.functional_scenario,
                    geometry_id=geometry.geometry_id,
                    geometry_hash=geometry_hash,
                    geometry_seed=geometry.seed,
                    adapter_id=geometry.functional_scenario,
                    interaction_schema_id="two_route_conflict",
                    sut_split=sut_split,
                    geometry_split=geometry.split,
                    logical_domain_id=domain_id,
                    logical_domain_bounds=bounds,
                    logical_parameter_mask=mask,
                    logical_split=logical_split,
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
    validate_taskbook(ScenarioMiningTaskSpec.from_dict(row) for row in tasks)
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
