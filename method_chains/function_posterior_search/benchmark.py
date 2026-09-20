"""Independent physical benchmark for function-aware failure search."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from diva_highway_env.data.response_bank import ResponseBank
from diva_highway_env.envs.cutin_env import (
    CutInScenario,
    EpisodeResult,
    run_cutin_episode,
)
from diva_highway_env.sut.idm_profiles import SUTProfile, get_profile
from method_chains.diva_function_conditioned_routing.benchmark import (
    ARCHETYPE_NAMES,
    MODE_BOUNDS,
    RELEASE_CODES,
    RELEASE_NAMES,
)


MODES = (*MODE_BOUNDS, "passing_cutin")
SCENARIOS_PER_MODE = 100
SCENARIO_COUNT = len(MODES) * SCENARIOS_PER_MODE
CONFIRMATION_SEEDS = (20261103, 20261117, 20261201, 20261213, 20261229)
REGIME_COUNTS = {"core": 40, "benign": 30, "boundary": 15, "outside": 15}
SOURCE_NAMES = tuple(f"Source-{name.removeprefix('Modular-')}" for name in RELEASE_NAMES)
TARGET_COVERAGES = ("exact", "interpolated", "unseen")
HETEROGENEITY_LEVELS = ("global", "partial", "full")


@dataclass(frozen=True)
class ReleaseSpec:
    """One executable controller release with a module for every function."""

    name: str
    coverage: str
    heterogeneity: str
    replicate: int
    modules: tuple[SUTProfile, ...]


def _profile_signature(profile: SUTProfile) -> tuple[object, ...]:
    return tuple(getattr(profile, field.name) for field in fields(profile) if field.name != "name")


def _blend_profile(name: str, weight: float) -> SUTProfile:
    """Interpolate two source IDM controllers without using target outcomes."""
    left, right = get_profile("SUT-B"), get_profile("SUT-C")
    values: dict[str, object] = {"name": name, "controller": "IDM"}
    for field in fields(SUTProfile):
        if field.name in values:
            continue
        a, b = float(getattr(left, field.name)), float(getattr(right, field.name))
        values[field.name] = (1.0 - weight) * a + weight * b
    return SUTProfile(**values)


def source_releases() -> tuple[ReleaseSpec, ...]:
    """Return six frozen source releases, balanced within every function."""
    profiles = tuple(get_profile(name) for name in ARCHETYPE_NAMES)
    passing_codes = (0, 1, 2, 0, 1, 2)
    releases = []
    for name, codes, passing in zip(SOURCE_NAMES, RELEASE_CODES, passing_codes, strict=True):
        releases.append(ReleaseSpec(
            name,
            "source",
            "mixed",
            0,
            tuple(profiles[int(code)] for code in (*codes.tolist(), passing)),
        ))
    return tuple(releases)


def _exact_modules(heterogeneity: str, replicate: int) -> tuple[SUTProfile, ...]:
    profiles = tuple(get_profile(name) for name in ARCHETYPE_NAMES)
    codes = {
        ("global", 1): (0, 0, 0, 0, 0, 0),
        ("global", 2): (2, 2, 2, 2, 2, 2),
        ("partial", 1): (0, 0, 0, 1, 1, 1),
        ("partial", 2): (2, 2, 1, 1, 1, 2),
        ("full", 1): (0, 1, 2, 1, 0, 2),
        ("full", 2): (2, 0, 1, 0, 2, 1),
    }[heterogeneity, replicate]
    return tuple(profiles[index] for index in codes)


def _interpolated_modules(heterogeneity: str, replicate: int) -> tuple[SUTProfile, ...]:
    weights = {
        ("global", 1): (0.25,) * 6,
        ("global", 2): (0.75,) * 6,
        ("partial", 1): (0.20, 0.20, 0.20, 0.80, 0.80, 0.80),
        ("partial", 2): (0.65, 0.65, 0.35, 0.35, 0.35, 0.65),
        ("full", 1): (0.15, 0.30, 0.45, 0.60, 0.75, 0.90),
        ("full", 2): (0.85, 0.70, 0.55, 0.40, 0.25, 0.10),
    }[heterogeneity, replicate]
    return tuple(
        _blend_profile(f"Interpolated-{heterogeneity}-{replicate}-{mode}", weight)
        for mode, weight in zip(MODES, weights, strict=True)
    )


def _unseen_modules(heterogeneity: str, replicate: int) -> tuple[SUTProfile, ...]:
    names = {
        ("global", 1): ("AV-Calibrated-IDM",) * 6,
        ("global", 2): ("SUT-E",) * 6,
        ("partial", 1): ("SUT-A",) * 3 + ("SUT-D",) * 3,
        ("partial", 2): ("SM-Limited-FVDM",) * 2 + ("AV-Predictive-Brake",) * 4,
        ("full", 1): (
            "SUT-A", "SUT-D", "SUT-E", "SM-Normal-IDM",
            "SM-Limited-FVDM", "AV-Predictive-Brake",
        ),
        ("full", 2): (
            "AV-Calibrated-IDM", "SM-Strong-FVDM", "SUT-D", "SUT-A",
            "AV-Predictive-Brake", "SUT-E",
        ),
    }[heterogeneity, replicate]
    return tuple(get_profile(name) for name in names)


def target_releases() -> tuple[ReleaseSpec, ...]:
    """Return the frozen 3 x 3 x 2 target factorial."""
    builders = {
        "exact": _exact_modules,
        "interpolated": _interpolated_modules,
        "unseen": _unseen_modules,
    }
    releases = []
    for coverage in TARGET_COVERAGES:
        for heterogeneity in HETEROGENEITY_LEVELS:
            for replicate in (1, 2):
                name = f"Target-{coverage}-{heterogeneity}-{replicate}"
                releases.append(ReleaseSpec(
                    name,
                    coverage,
                    heterogeneity,
                    replicate,
                    builders[coverage](heterogeneity, replicate),
                ))
    return tuple(releases)


def release_manifest() -> list[dict[str, object]]:
    """Describe controller modules without exposing simulation outcomes."""
    rows = []
    for release in (*source_releases(), *target_releases()):
        row: dict[str, object] = {
            "sut_name": release.name,
            "coverage": release.coverage,
            "heterogeneity": release.heterogeneity,
            "replicate": release.replicate,
        }
        row.update({
            mode: profile.name
            for mode, profile in zip(MODES, release.modules, strict=True)
        })
        rows.append(row)
    return rows


def physical_episode_count() -> int:
    """Return distinct physical executions per seed after exact-behaviour reuse."""
    releases = (*source_releases(), *target_releases())
    unique_modules = sum(
        len({_profile_signature(release.modules[index]) for release in releases})
        for index in range(len(MODES))
    )
    return unique_modules * SCENARIOS_PER_MODE


def _mode_bounds(mode: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if mode == "passing_cutin":
        return (5.0, 40.0), (-8.0, 2.0)
    return MODE_BOUNDS[mode]


def generate_confirmation_scenarios(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate six balanced modes and four outcome-independent regimes."""
    anchors, labels, controls, regimes = [], [], [], []
    regime_names = tuple(
        name for name, count in REGIME_COUNTS.items() for _ in range(count)
    )
    for offset, mode in enumerate(MODES):
        unit = qmc.Sobol(d=4, scramble=True, seed=seed + offset).random_base2(7)
        unit = unit[:SCENARIOS_PER_MODE]
        gap_bounds, speed_bounds = _mode_bounds(mode)
        gap_span = gap_bounds[1] - gap_bounds[0]
        speed_span = speed_bounds[1] - speed_bounds[0]
        mode_anchors = np.empty((SCENARIOS_PER_MODE, 2), dtype=float)
        for index, regime in enumerate(regime_names):
            gap_u, speed_u = unit[index, :2]
            if regime == "core":
                gap = gap_bounds[0] + gap_u * gap_span
                speed = speed_bounds[0] + speed_u * speed_span
            elif regime == "benign":
                gap = gap_bounds[1] + 5.0 + 20.0 * gap_u
                speed = max(0.0, speed_bounds[1]) + 1.0 + 4.0 * speed_u
            elif regime == "boundary":
                gap = (gap_bounds[0] + 0.08 * gap_span * gap_u
                       if speed_u < 0.5 else gap_bounds[1] - 0.08 * gap_span * gap_u)
                speed = (speed_bounds[0] + 0.08 * speed_span * speed_u
                         if gap_u < 0.5 else speed_bounds[1] - 0.08 * speed_span * speed_u)
            else:
                gap = (max(3.0, gap_bounds[0] - 0.15 * gap_span * gap_u)
                       if speed_u < 0.5 else gap_bounds[1] + 0.15 * gap_span * gap_u)
                speed = (speed_bounds[0] - 0.15 * speed_span * speed_u
                         if gap_u < 0.5 else speed_bounds[1] + 0.15 * speed_span * speed_u)
            mode_anchors[index] = gap, speed
        anchors.append(mode_anchors)
        labels.extend([mode] * SCENARIOS_PER_MODE)
        controls.append(unit[:, 2:4])
        regimes.extend(regime_names)
    return (
        np.vstack(anchors),
        np.asarray(labels, dtype="U32"),
        np.vstack(controls),
        np.asarray(regimes, dtype="U16"),
    )


def _execute(task: tuple[SUTProfile, CutInScenario, int]) -> EpisodeResult:
    profile, scenario, seed = task
    return run_cutin_episode(profile, scenario, seed)


def _empty_outputs(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "vulnerability": np.empty(shape, dtype=float),
        "collisions": np.empty(shape, dtype=bool),
        "near_misses": np.empty(shape, dtype=bool),
        "min_ttc": np.empty(shape, dtype=float),
        "min_distance": np.empty(shape, dtype=float),
        "completed": np.empty(shape, dtype=bool),
    }


def build_confirmation_bank(seed: int, workers: int = 12) -> ResponseBank:
    """Physically execute every distinct controller-module scenario pair."""
    anchors, modes, controls, _ = generate_confirmation_scenarios(seed)
    releases = (*source_releases(), *target_releases())
    scenarios = tuple(
        CutInScenario(float(anchor[0]), float(anchor[1]), str(mode),
                      float(control[0]), float(control[1]))
        for anchor, mode, control in zip(anchors, modes, controls, strict=True)
    )
    keys: list[tuple[tuple[object, ...], int]] = []
    tasks: list[tuple[SUTProfile, CutInScenario, int]] = []
    locations: dict[tuple[tuple[object, ...], int], int] = {}
    for release in releases:
        module_by_mode = dict(zip(MODES, release.modules, strict=True))
        for scenario_index, (mode, scenario) in enumerate(zip(modes, scenarios, strict=True)):
            profile = module_by_mode[str(mode)]
            key = (_profile_signature(profile), scenario_index)
            if key not in locations:
                locations[key] = len(tasks)
                keys.append(key)
                tasks.append((profile, scenario, seed + scenario_index))
    if workers == 1:
        results = list(map(_execute, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_execute, tasks, chunksize=16))
    outputs = _empty_outputs((len(releases), len(scenarios)))
    for release_index, release in enumerate(releases):
        module_by_mode = dict(zip(MODES, release.modules, strict=True))
        for scenario_index, mode in enumerate(modes):
            key = (_profile_signature(module_by_mode[str(mode)]), scenario_index)
            result = results[locations[key]]
            outputs["vulnerability"][release_index, scenario_index] = result.vulnerability
            outputs["collisions"][release_index, scenario_index] = result.collision
            outputs["near_misses"][release_index, scenario_index] = result.near_miss
            outputs["min_ttc"][release_index, scenario_index] = result.min_ttc
            outputs["min_distance"][release_index, scenario_index] = result.min_distance
            outputs["completed"][release_index, scenario_index] = result.completed
    return ResponseBank(
        anchors=anchors,
        sut_names=tuple(release.name for release in releases),
        modes=modes,
        scenario_controls=controls,
        **outputs,
    )


def load_or_build_bank(path: Path, seed: int, workers: int = 12) -> ResponseBank:
    """Resume completed physical banks while rejecting incompatible artifacts."""
    if path.exists():
        bank = ResponseBank.load(path)
        expected_names = tuple(
            release.name for release in (*source_releases(), *target_releases())
        )
        if bank.sut_names != expected_names or len(bank.anchors) != SCENARIO_COUNT:
            raise ValueError(f"incompatible confirmation bank: {path}")
        return bank
    bank = build_confirmation_bank(seed, workers)
    bank.save(path)
    return bank
