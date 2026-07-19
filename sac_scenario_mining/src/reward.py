"""Componentized reward and TTC calculation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class RewardBreakdown:
    ttc_reward: float
    proximity_reward: float
    target_collision_bonus: float
    non_target_collision_penalty: float
    out_of_road_penalty: float
    reverse_penalty: float
    action_penalty: float
    action_smoothness_penalty: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_ttc(relative_position: np.ndarray, relative_velocity: np.ndarray,
                cap: float) -> float:
    p, v = np.asarray(relative_position,
                      dtype=float), np.asarray(relative_velocity, dtype=float)
    d = float(np.linalg.norm(p))
    if not np.isfinite(d) or not np.all(np.isfinite(v)): return float(cap)
    if d <= 1e-6: return 0.0
    closing = float(np.dot(p, v) / d)
    return float(min(cap, d / -closing)) if closing < 0.0 else float(cap)


def compute_reward(ttc: float, distance: float, action: np.ndarray,
                   previous_action: np.ndarray, events: Mapping[str, bool],
                   cfg: Mapping[str, float]) -> RewardBreakdown:
    ttc_dense = max(float(cfg["ttc_dense_threshold"]), 1e-6)
    ttc_r = float(cfg["ttc_weight"]) * np.clip(
        (ttc_dense - ttc) / ttc_dense, 0.0, 1.0)
    prox_r = float(cfg["proximity_weight"]) * np.exp(
        -max(0.0, distance) / max(float(cfg["proximity_scale"]), 1e-6))
    target = float(cfg["target_collision_bonus"]) if events.get(
        "target_collision", False) else 0.0
    other = -float(cfg["non_target_collision_penalty"]) if events.get(
        "non_target_collision", False) else 0.0
    road = -float(cfg["out_of_road_penalty"]) if events.get(
        "adversary_out_of_road", False) else 0.0
    reverse = -float(cfg["reverse_penalty"]) if events.get("wrong_way",
                                                           False) else 0.0
    act = -float(cfg["action_l2_weight"]) * float(np.dot(action, action))
    smooth = -float(cfg["action_smoothness_weight"]) * float(
        np.dot(action - previous_action, action - previous_action))
    return RewardBreakdown(
        ttc_r, prox_r, target, other, road, reverse, act, smooth,
        ttc_r + prox_r + target + other + road + reverse + act + smooth)
