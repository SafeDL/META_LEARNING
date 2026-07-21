"""One reward definition for every logical task."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class RewardBreakdown:
    ttc: float
    proximity: float
    target_collision: float
    non_target_collision: float
    out_of_road: float
    wrong_route: float
    lane_marking_violation: float
    action_l2: float
    action_smoothness: float
    total: float
    def as_dict(self) -> dict[str, float]: return asdict(self)


def compute_reward(ttc: float, distance: float, action: np.ndarray, previous_action: np.ndarray, events: Mapping[str, bool], cfg: Mapping[str, float]) -> RewardBreakdown:
    dense = float(cfg["ttc_weight"]) * float(np.clip((float(cfg["ttc_dense_threshold"]) - ttc) / max(float(cfg["ttc_dense_threshold"]), 1e-6), 0.0, 1.0))
    proximity = float(cfg["proximity_weight"]) * float(np.exp(-max(0.0, distance) / 10.0))
    target = float(cfg["target_collision_bonus"]) if events.get("target_collision") else 0.0
    non_target = -float(cfg["non_target_collision_penalty"]) if events.get("non_target_collision") else 0.0
    out_of_road = -float(cfg["out_of_road_penalty"]) * sum(
        bool(events.get(key)) for key in ("adversary_out_of_road", "sut_out_of_road")
    )
    wrong_route = -float(cfg["wrong_route_penalty"]) if events.get("wrong_route") else 0.0
    lane_marking = -float(cfg.get("lane_marking_penalty", 0.0)) if events.get("lane_marking_violation") else 0.0
    a2 = -float(cfg["action_l2_weight"]) * float(np.dot(action, action))
    smooth = -float(cfg["action_smoothness_weight"]) * float(np.dot(action - previous_action, action - previous_action))
    total = dense + proximity + target + non_target + out_of_road + wrong_route + lane_marking + a2 + smooth
    return RewardBreakdown(dense, proximity, target, non_target, out_of_road, wrong_route, lane_marking, a2, smooth, total)
