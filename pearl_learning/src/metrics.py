"""Strict, target-pair-aware safety outcomes."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping
import numpy as np


@dataclass
class EpisodeMetrics:
    task_id: str; case_id: str; episode_return: float = 0.0; episode_length: int = 0
    target_collision: bool = False; non_target_collision: bool = False; adversary_out_of_road: bool = False; sut_out_of_road: bool = False; wrong_route: bool = False; lane_marking_violation: bool = False
    min_ttc: float = float("inf"); min_distance: float = float("inf"); termination_reason: str = "running"; target_contact_method: str = "no_pairwise_contact"

    def update(self, reward: float, ttc: float, distance: float, events: Mapping[str, bool], contact_method: str) -> None:
        self.episode_return += float(reward); self.episode_length += 1
        self.min_ttc = min(self.min_ttc, float(ttc)); self.min_distance = min(self.min_distance, float(distance))
        for key in ("target_collision", "non_target_collision", "adversary_out_of_road", "sut_out_of_road", "wrong_route", "lane_marking_violation"):
            setattr(self, key, bool(getattr(self, key) or events.get(key, False)))
        if events.get("target_collision"):
            self.target_contact_method = contact_method

    def record(self, threshold: float) -> dict[str, object]:
        critical = self.target_collision or self.min_ttc <= threshold
        data = asdict(self)
        invalid = self.non_target_collision or self.adversary_out_of_road or self.sut_out_of_road or self.wrong_route
        data["critical"] = critical
        data["valid"] = not invalid
        data["invalid"] = invalid
        data["valid_critical_strict"] = critical and bool(data["valid"])
        return data


def summarize(records: list[Mapping[str, object]]) -> dict[str, float | int | None]:
    if not records: return {"episodes": 0}
    bool_rate = lambda key: float(np.mean([bool(row[key]) for row in records]))
    values = lambda key: np.asarray([float(row[key]) for row in records], dtype=float)
    first = next((index + 1 for index, row in enumerate(records) if row["valid_critical_strict"]), None)
    return {"episodes": len(records), "valid_critical_strict_rate": bool_rate("valid_critical_strict"), "target_collision_rate": bool_rate("target_collision"), "critical_rate": bool_rate("critical"), "invalid_rate": 1.0 - bool_rate("valid"), "median_min_ttc": float(np.median(values("min_ttc"))), "median_min_distance": float(np.median(values("min_distance"))), "episodes_to_first_valid_critical": first}
