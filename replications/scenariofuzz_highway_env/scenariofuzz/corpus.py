"""Topology-constrained local seed corpus and canonical scenario identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    initial_gap: float
    relative_speed: float
    mode: str

    @classmethod
    def create(cls, initial_gap: float, relative_speed: float, mode: str) -> "ScenarioSpec":
        canonical = {
            "initial_gap": round(float(initial_gap), 6),
            "relative_speed": round(float(relative_speed), 6),
            "mode": str(mode),
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return cls(f"scenario-{digest}", canonical["initial_gap"], canonical["relative_speed"], canonical["mode"])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LocalScenarioSeed:
    """A seed derived from the fixed straight two-lane RoadNetwork contract."""

    seed_id: str
    road_type: str
    road_network_id: str
    ego_lane: int
    npc_initial_lanes: tuple[int, ...]
    allowed_modes: tuple[str, ...]
    gap_bounds: tuple[float, float]
    relative_speed_bounds: tuple[float, float]
    road_length: float = 400.0
    lane_width: float = 4.0
    parent_seed_id: str | None = None
    sampling_rule: str = "uniform-within-frozen-contract"

    def bounds_hash(self) -> str:
        payload = {
            "gap_bounds": self.gap_bounds,
            "relative_speed_bounds": self.relative_speed_bounds,
            "allowed_modes": self.allowed_modes,
            "road_network_id": self.road_network_id,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def validate(self, spec: ScenarioSpec) -> tuple[bool, str | None]:
        if spec.mode not in self.allowed_modes:
            return False, "unsupported_mode"
        if not self.gap_bounds[0] <= spec.initial_gap <= self.gap_bounds[1]:
            return False, "gap_out_of_bounds"
        if not self.relative_speed_bounds[0] <= spec.relative_speed <= self.relative_speed_bounds[1]:
            return False, "relative_speed_out_of_bounds"
        if 25.0 + spec.relative_speed <= 0.0:
            return False, "nonpositive_npc_speed"
        lead_x = 60.0 + spec.initial_gap
        if spec.mode in {"fast_intrusion", "cutin_braking"}:
            lead_x -= spec.relative_speed
        if not 0.0 < lead_x < self.road_length:
            return False, "placement_out_of_road"
        return True, None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bounds_hash"] = self.bounds_hash()
        return data


def build_default_corpus(config: dict) -> list[LocalScenarioSeed]:
    bounds = config["bounds"]
    seed = LocalScenarioSeed(
        seed_id="straight-two-lane-001",
        road_type="straight_two_lane",
        road_network_id="highway-env:straight_road_network:2x400m",
        ego_lane=0,
        npc_initial_lanes=(0, 1),
        allowed_modes=tuple(config["mode_set"]),
        gap_bounds=tuple(float(x) for x in bounds["initial_gap"]),
        relative_speed_bounds=tuple(float(x) for x in bounds["relative_speed"]),
    )
    return [seed]


def save_corpus(path: Path, seeds: list[LocalScenarioSeed]) -> None:
    Path(path).write_text(
        json.dumps([seed.to_dict() for seed in seeds], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

