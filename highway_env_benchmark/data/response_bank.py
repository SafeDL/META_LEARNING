"""Build and load the complete source-SUT by scenario response bank."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from highway_env_benchmark.data.generate_anchor_bank import generate_anchor_bank
from highway_env_benchmark.envs.cutin_env import CutInScenario, run_cutin_episode
from sut_algorithms.highway_env.idm_profiles import PROFILE_NAMES, SUTProfile, get_profile


@dataclass(frozen=True)
class ResponseBank:
    """Complete offline responses; LOSO code must reveal target entries selectively."""

    anchors: np.ndarray
    sut_names: tuple[str, ...]
    vulnerability: np.ndarray
    collisions: np.ndarray
    near_misses: np.ndarray
    min_ttc: np.ndarray
    min_distance: np.ndarray
    completed: np.ndarray
    modes: np.ndarray | None = None
    scenario_controls: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected = (len(self.sut_names), len(self.anchors))
        for name in (
            "vulnerability",
            "collisions",
            "near_misses",
            "min_ttc",
            "min_distance",
            "completed",
        ):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if (
            self.modes is not None
            and np.asarray(self.modes).shape != (len(self.anchors),)
        ):
            raise ValueError("modes must contain one interaction mode per anchor")
        if (
            self.scenario_controls is not None
            and np.asarray(self.scenario_controls).shape != (len(self.anchors), 2)
        ):
            raise ValueError("scenario_controls must contain timing and intensity")

    def index_of(self, sut_name: str) -> int:
        return self.sut_names.index(sut_name)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            anchors=self.anchors,
            sut_names=np.asarray(self.sut_names),
            vulnerability=self.vulnerability,
            collisions=self.collisions,
            near_misses=self.near_misses,
            min_ttc=self.min_ttc,
            min_distance=self.min_distance,
            completed=self.completed,
            modes=(
                np.asarray(self.modes, dtype="U32")
                if self.modes is not None
                else np.full(len(self.anchors), "fast_intrusion", dtype="U32")
            ),
            scenario_controls=(
                np.asarray(self.scenario_controls, dtype=float)
                if self.scenario_controls is not None
                else np.empty((0, 2), dtype=float)
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "ResponseBank":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                anchors=data["anchors"],
                sut_names=tuple(str(name) for name in data["sut_names"]),
                vulnerability=data["vulnerability"],
                collisions=data["collisions"].astype(bool),
                near_misses=data["near_misses"].astype(bool),
                min_ttc=data["min_ttc"],
                min_distance=data["min_distance"],
                completed=data["completed"].astype(bool),
                modes=(data["modes"].astype(str) if "modes" in data.files else None),
                scenario_controls=(
                    data["scenario_controls"].astype(float)
                    if "scenario_controls" in data.files
                    and data["scenario_controls"].shape == (len(data["anchors"]), 2)
                    else None
                ),
            )


def build_response_bank(
    anchors: np.ndarray,
    seed: int,
    modes: np.ndarray | None = None,
    profiles: tuple[SUTProfile, ...] | None = None,
) -> ResponseBank:
    """Run all six deterministic SUTs on exactly the same anchor bank."""
    selected_profiles = (
        tuple(get_profile(name) for name in PROFILE_NAMES)
        if profiles is None
        else tuple(profiles)
    )
    names = tuple(profile.name for profile in selected_profiles)
    if not names or len(set(names)) != len(names):
        raise ValueError("profiles must have unique names")
    if any(profile.controller not in {"IDM", "FVDM"} for profile in selected_profiles):
        raise ValueError("profiles must use a supported controller family")
    shape = (len(selected_profiles), len(anchors))
    vulnerability = np.empty(shape, dtype=float)
    collisions = np.empty(shape, dtype=bool)
    near_misses = np.empty(shape, dtype=bool)
    min_ttc = np.empty(shape, dtype=float)
    min_distance = np.empty(shape, dtype=float)
    completed = np.empty(shape, dtype=bool)
    interaction_modes = (
        np.full(len(anchors), "fast_intrusion", dtype="U32")
        if modes is None
        else np.asarray(modes, dtype="U32")
    )
    if interaction_modes.shape != (len(anchors),):
        raise ValueError("modes must contain one value per anchor")
    for sut_index, profile in enumerate(selected_profiles):
        for anchor_index, (initial_gap, relative_speed) in enumerate(anchors):
            result = run_cutin_episode(
                profile,
                CutInScenario(
                    float(initial_gap),
                    float(relative_speed),
                    interaction_modes[anchor_index],
                ),
                seed=seed + anchor_index,
            )
            vulnerability[sut_index, anchor_index] = result.vulnerability
            collisions[sut_index, anchor_index] = result.collision
            near_misses[sut_index, anchor_index] = result.near_miss
            min_ttc[sut_index, anchor_index] = result.min_ttc
            min_distance[sut_index, anchor_index] = result.min_distance
            completed[sut_index, anchor_index] = result.completed
    return ResponseBank(
        anchors=np.asarray(anchors, dtype=float),
        sut_names=names,
        vulnerability=vulnerability,
        collisions=collisions,
        near_misses=near_misses,
        min_ttc=min_ttc,
        min_distance=min_distance,
        completed=completed,
        modes=interaction_modes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/highway_replications/shared/response_bank.npz"),
    )
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    bank = build_response_bank(generate_anchor_bank(args.num_anchors, args.seed), args.seed)
    bank.save(args.output)
    print(f"Wrote {bank.vulnerability.size} highway-env episodes to {args.output}")


if __name__ == "__main__":
    main()
