"""Strict, target-pair-aware safety outcomes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class EpisodeMetrics:
    task_id: str
    case_id: str
    episode_return: float = 0.0
    episode_length: int = 0
    target_collision: bool = False
    physical_critical_proximity: bool = False
    route_conflict_proximity: bool = False
    rule_satisfied_critical_proximity: bool = False
    non_target_collision: bool = False
    adversary_out_of_road: bool = False
    sut_out_of_road: bool = False
    wrong_route: bool = False
    lane_marking_violation: bool = False
    min_ttc: float = float("inf")
    min_distance: float = float("inf")
    termination_reason: str = "running"
    target_contact_method: str = "no_pairwise_contact"

    def update(
        self,
        reward: float,
        ttc: float,
        distance: float,
        events: Mapping[str, bool],
        contact_method: str,
    ) -> None:
        self.episode_return += float(reward)
        self.episode_length += 1
        self.min_ttc = min(self.min_ttc, float(ttc))
        self.min_distance = min(self.min_distance, float(distance))
        for key in EVENT_FIELDS:
            setattr(self, key, bool(getattr(self, key) or events.get(key, False)))
        if events.get("target_collision"):
            self.target_contact_method = contact_method

    def record(self, threshold: float) -> dict[str, object]:
        physical_critical = (
            self.target_collision or self.physical_critical_proximity
        )
        # A low-TTC encounter is a task success only when it realizes the
        # hidden interaction rule.  Otherwise one policy can score on both
        # members of an opposite-rule pair without identifying the task.
        critical = self.target_collision or self.rule_satisfied_critical_proximity
        data = asdict(self)
        invalid = self.non_target_collision or self.adversary_out_of_road or self.sut_out_of_road or self.wrong_route
        data["critical"] = critical
        data["physical_critical"] = physical_critical
        data["critical_rule_satisfied"] = critical
        data["valid"] = not invalid
        data["invalid"] = invalid
        data["valid_critical_strict"] = critical and bool(data["valid"])
        return data


EVENT_FIELDS = (
    "target_collision",
    "physical_critical_proximity",
    "route_conflict_proximity",
    "rule_satisfied_critical_proximity",
    "non_target_collision",
    "adversary_out_of_road",
    "sut_out_of_road",
    "wrong_route",
    "lane_marking_violation",
)
_DIVERSITY_FIELDS = ("adversary_speed_mps", "adversary_spawn_m", "sut_spawn_m")


def _valid_critical_initial_condition_diversity(
    records: list[Mapping[str, object]],
    case_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[int, float | None, float | None]:
    strict_ids = [str(row["case_id"]) for row in records if bool(row["valid_critical_strict"])]
    strict_set = set(strict_ids)
    if not strict_ids:
        return 0, None, None
    if case_metadata is None:
        return len(strict_set), None, None
    all_rows: list[list[float]] = []
    selected_rows: list[list[float]] = []
    for case_id, case in case_metadata.items():
        try:
            values = [float(case[field]) for field in _DIVERSITY_FIELDS]
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(values).all():
            continue
        all_rows.append(values)
        if str(case_id) in strict_set:
            selected_rows.append(values)
    coverage = float(len(selected_rows) / len(strict_set))
    if len(selected_rows) < 2 or not all_rows:
        return len(strict_set), None, coverage
    pool = np.asarray(all_rows, dtype=float)
    selected = np.asarray(selected_rows, dtype=float)
    scale = np.ptp(pool, axis=0)
    normalized = (selected - pool.min(axis=0)) / np.where(scale > 0.0, scale, 1.0)
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=-1)
    return len(strict_set), float(distances[np.triu_indices(len(normalized), k=1)].mean()), coverage


def summarize(
    records: list[Mapping[str, object]],
    *,
    case_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float | int | None]:
    if not records:
        return {"episodes": 0}

    bool_rate = lambda key: float(np.mean([bool(row[key]) for row in records]))
    values = lambda key: np.asarray([float(row[key]) for row in records], dtype=float)
    first = next((index + 1 for index, row in enumerate(records) if row["valid_critical_strict"]), None)
    episode_lengths = values("episode_length").astype(int)
    steps_to_first = None if first is None else int(episode_lengths[:first].sum())
    count, diversity, coverage = _valid_critical_initial_condition_diversity(records, case_metadata)
    return {
        "episodes": len(records),
        "mean_episode_return": float(np.mean(values("episode_return"))),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "query_environment_steps": int(episode_lengths.sum()),
        "valid_critical_strict_rate": bool_rate("valid_critical_strict"),
        "target_collision_rate": bool_rate("target_collision"),
        "critical_rate": bool_rate("critical"),
        "invalid_rate": 1.0 - bool_rate("valid"),
        "median_min_ttc": float(np.median(values("min_ttc"))),
        "median_min_distance": float(np.median(values("min_distance"))),
        "episodes_to_first_valid_critical": first,
        "environment_steps_to_first_valid_critical": steps_to_first,
        "valid_critical_case_count": count,
        "valid_critical_initial_condition_diversity": diversity,
        "valid_critical_case_metadata_coverage": coverage,
    }
