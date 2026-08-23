"""Episode-level safety metrics; collision counts are deliberately Boolean."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import numpy as np


@dataclass
class EpisodeMetrics:
    case_id: str
    background_seed: int
    episode_return: float = 0.0
    episode_length: int = 0
    target_collision: bool = False
    any_vehicle_collision: bool = False
    non_target_collision: bool = False
    object_collision: bool = False
    adversary_out_of_road: bool = False
    sut_out_of_road: bool = False
    min_ttc: float = float("inf")
    min_distance: float = float("inf")
    action_l2_sum: float = 0.0
    action_delta_l2_sum: float = 0.0
    termination_reason: str = "unknown"
    invalid_before_critical: bool = False

    def update(self, reward: float, ttc: float, distance: float,
               action: np.ndarray, delta: np.ndarray,
               events: Mapping[str, bool]) -> None:
        self.episode_return += float(reward)
        self.episode_length += 1
        self.min_ttc = min(self.min_ttc, float(ttc))
        self.min_distance = min(self.min_distance, float(distance))
        self.action_l2_sum += float(np.dot(action, action))
        self.action_delta_l2_sum += float(np.dot(delta, delta))
        for key in ("target_collision", "any_vehicle_collision",
                    "non_target_collision", "object_collision",
                    "adversary_out_of_road", "sut_out_of_road"):
            setattr(self, key,
                    bool(getattr(self, key) or events.get(key, False)))
        self.invalid_before_critical |= bool(
            events.get("invalid_before_critical", False))

    def record(self, critical_threshold: float) -> dict:
        critical = self.target_collision or self.min_ttc <= critical_threshold
        valid = not self.invalid_before_critical
        data = asdict(self)
        data.update(critical=critical,
                    valid=valid,
                    valid_critical=critical and valid,
                    mean_action_l2=self.action_l2_sum /
                    max(self.episode_length, 1),
                    mean_action_delta_l2=self.action_delta_l2_sum /
                    max(self.episode_length, 1))
        data.pop("action_l2_sum")
        data.pop("action_delta_l2_sum")
        data.pop("invalid_before_critical")
        return data


def summarize(records: list[Mapping[str, object]]) -> dict[str, float | int]:
    if not records: return {"episodes": 0}
    rate = lambda k: float(np.mean([bool(r[k]) for r in records]))
    vals = lambda k: np.asarray([float(r[k]) for r in records], dtype=float)
    first = next((i + 1 for i, r in enumerate(records) if r["valid_critical"]),
                 None)
    return {
        "episodes": len(records),
        "valid_critical_rate": rate("valid_critical"),
        "target_collision_rate": rate("target_collision"),
        "critical_rate": rate("critical"),
        "invalid_rate": 1.0 - rate("valid"),
        "median_min_ttc": float(np.median(vals("min_ttc"))),
        "mean_min_ttc": float(np.mean(vals("min_ttc"))),
        "median_min_distance": float(np.median(vals("min_distance"))),
        "mean_episode_return": float(np.mean(vals("episode_return"))),
        "episodes_to_first_valid_critical": first
    }
