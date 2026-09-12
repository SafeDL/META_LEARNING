"""Build and load the complete source-SUT by scenario response bank."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mvr.highway.data.generate_anchor_bank import generate_anchor_bank
from mvr.highway.envs.cutin_env import CutInScenario, run_cutin_episode
from mvr.highway.sut.idm_profiles import PROFILE_NAMES, get_profile


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
            )


def build_response_bank(anchors: np.ndarray, seed: int) -> ResponseBank:
    """Run all six deterministic SUTs on exactly the same anchor bank."""
    shape = (len(PROFILE_NAMES), len(anchors))
    vulnerability = np.empty(shape, dtype=float)
    collisions = np.empty(shape, dtype=bool)
    near_misses = np.empty(shape, dtype=bool)
    min_ttc = np.empty(shape, dtype=float)
    min_distance = np.empty(shape, dtype=float)
    completed = np.empty(shape, dtype=bool)
    for sut_index, sut_name in enumerate(PROFILE_NAMES):
        profile = get_profile(sut_name)
        for anchor_index, (initial_gap, relative_speed) in enumerate(anchors):
            result = run_cutin_episode(
                profile,
                CutInScenario(float(initial_gap), float(relative_speed)),
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
        sut_names=PROFILE_NAMES,
        vulnerability=vulnerability,
        collisions=collisions,
        near_misses=near_misses,
        min_ttc=min_ttc,
        min_distance=min_distance,
        completed=completed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/diva_highway/cutin_mvp/response_bank_highway.npz"),
    )
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    bank = build_response_bank(generate_anchor_bank(args.num_anchors, args.seed), args.seed)
    bank.save(args.output)
    print(f"Wrote {bank.vulnerability.size} highway-env episodes to {args.output}")


if __name__ == "__main__":
    main()
