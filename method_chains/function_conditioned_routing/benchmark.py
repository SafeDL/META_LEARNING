"""Physical function-shift benchmark for function-conditioned transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from highway_env_benchmark.data.generate_anchor_bank import FUNCTIONAL_MODES
from highway_env_benchmark.data.response_bank import ResponseBank
from highway_env_benchmark.envs.cutin_env import CutInScenario, EpisodeResult, run_cutin_episode
from sut_algorithms.highway_env.idm_profiles import get_profile


ARCHETYPE_NAMES = ("SUT-B", "SUT-C", "SUT-F")
RELEASE_NAMES = (
    "Modular-Atlas",
    "Modular-Boreal",
    "Modular-Cascade",
    "Modular-Delta",
    "Modular-Estuary",
    "Modular-Fjord",
)

# Each column contains every archetype twice. No two releases share an entire
# row, while every held-out module has one historical release with the same
# physically executed controller behaviour for that function.
RELEASE_CODES = np.asarray(
    (
        (0, 0, 0, 1, 2),
        (0, 1, 2, 0, 1),
        (1, 2, 0, 1, 0),
        (1, 0, 1, 2, 2),
        (2, 1, 2, 2, 0),
        (2, 2, 1, 0, 1),
    ),
    dtype=int,
)

MODE_BOUNDS = {
    "fast_intrusion": ((8.0, 38.0), (-7.0, 1.0)),
    "cutin_braking": ((10.0, 40.0), (-6.0, 1.0)),
    "lead_braking": ((8.0, 40.0), (-5.0, 2.0)),
    "stop_and_go": ((8.0, 36.0), (-4.0, 2.0)),
    "slow_lead_following": ((5.0, 32.0), (-8.0, -1.0)),
}


@dataclass(frozen=True)
class FunctionalRelease:
    """One executable release with an explicit controller per function."""

    name: str
    modules: dict[str, str]


def functional_releases() -> tuple[FunctionalRelease, ...]:
    """Return the frozen module composition of all held-out releases."""
    releases = []
    for name, codes in zip(RELEASE_NAMES, RELEASE_CODES, strict=True):
        modules = {
            mode: ARCHETYPE_NAMES[int(code)]
            for mode, code in zip(FUNCTIONAL_MODES, codes, strict=True)
        }
        releases.append(FunctionalRelease(name, modules))
    return tuple(releases)


def generate_functional_scenarios(
    num_anchors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate balanced mode-specific physical parameters and schedules."""
    if num_anchors < len(FUNCTIONAL_MODES) or num_anchors % len(FUNCTIONAL_MODES):
        raise ValueError("num_anchors must be a positive multiple of the mode count")
    per_mode = num_anchors // len(FUNCTIONAL_MODES)
    anchors = []
    controls = []
    for offset, mode in enumerate(FUNCTIONAL_MODES):
        sample_power = int(np.ceil(np.log2(per_mode)))
        samples = qmc.Sobol(
            d=4,
            scramble=True,
            seed=seed + offset,
        ).random_base2(sample_power)[:per_mode]
        gap_bounds, speed_bounds = MODE_BOUNDS[mode]
        anchors.append(
            np.column_stack(
                (
                    gap_bounds[0] + samples[:, 0] * (gap_bounds[1] - gap_bounds[0]),
                    speed_bounds[0]
                    + samples[:, 1] * (speed_bounds[1] - speed_bounds[0]),
                )
            )
        )
        controls.append(samples[:, 2:4])
    return (
        np.vstack(anchors),
        np.repeat(np.asarray(FUNCTIONAL_MODES, dtype="U32"), per_mode),
        np.vstack(controls),
    )


def _empty_outputs(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "vulnerability": np.empty(shape, dtype=float),
        "collisions": np.empty(shape, dtype=bool),
        "near_misses": np.empty(shape, dtype=bool),
        "min_ttc": np.empty(shape, dtype=float),
        "min_distance": np.empty(shape, dtype=float),
        "completed": np.empty(shape, dtype=bool),
    }


def _store(outputs: dict[str, np.ndarray], row: int, column: int, result: EpisodeResult) -> None:
    outputs["vulnerability"][row, column] = result.vulnerability
    outputs["collisions"][row, column] = result.collision
    outputs["near_misses"][row, column] = result.near_miss
    outputs["min_ttc"][row, column] = result.min_ttc
    outputs["min_distance"][row, column] = result.min_distance
    outputs["completed"][row, column] = result.completed


def build_functional_release_bank(num_anchors: int, seed: int) -> ResponseBank:
    """Execute every distinct module-scenario pair in highway-env."""
    anchors, modes, controls = generate_functional_scenarios(num_anchors, seed)
    releases = functional_releases()
    outputs = _empty_outputs((len(releases), num_anchors))
    cache: dict[tuple[str, int], EpisodeResult] = {}
    for release_index, release in enumerate(releases):
        for scenario_index, (anchor, mode, control) in enumerate(
            zip(anchors, modes, controls, strict=True)
        ):
            profile_name = release.modules[str(mode)]
            key = (profile_name, scenario_index)
            if key not in cache:
                scenario = CutInScenario(
                    float(anchor[0]),
                    float(anchor[1]),
                    str(mode),
                    float(control[0]),
                    float(control[1]),
                )
                cache[key] = run_cutin_episode(
                    get_profile(profile_name),
                    scenario,
                    seed + scenario_index,
                )
            _store(outputs, release_index, scenario_index, cache[key])
    return ResponseBank(
        anchors=anchors,
        sut_names=RELEASE_NAMES,
        modes=modes,
        scenario_controls=controls,
        **outputs,
    )


def release_manifest() -> dict[str, object]:
    """Return the human-readable frozen design without simulation outcomes."""
    return {
        "release_modules": {
            release.name: release.modules for release in functional_releases()
        },
        "archetype_profiles": list(ARCHETYPE_NAMES),
        "mode_bounds": {
            mode: {
                "initial_gap_m": list(bounds[0]),
                "relative_speed_mps": list(bounds[1]),
                "timing": [0.0, 1.0],
                "intensity": [0.0, 1.0],
            }
            for mode, bounds in MODE_BOUNDS.items()
        },
        "physical_execution": (
            "Every distinct controller-module and scenario pair is executed in highway-env; "
            "no target response row is assembled after simulation."
        ),
    }
