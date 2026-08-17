"""One reward definition for every logical task."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping
import numpy as np


INVALID_EVENT_PENALTY_KEYS = (
    "non_target_collision_penalty",
    "out_of_road_penalty",
    "wrong_route_penalty",
)


def required_invalid_event_penalty(cfg: Mapping[str, float], horizon: int) -> float:
    """Bound all positive episode reward, including a coincident target bonus."""
    if int(horizon) < 1:
        raise ValueError("reward dominance requires a positive episode horizon")
    margin = float(cfg.get("invalid_penalty_margin", 1.0))
    if margin <= 0.0:
        raise ValueError("invalid_penalty_margin must be positive")
    per_step_positive = sum(max(0.0, float(cfg.get(key, 0.0))) for key in (
        "ttc_weight", "proximity_weight", "route_progress_weight", "priority_alignment_weight",
    ))
    return (
        max(
            max(0.0, float(cfg["target_collision_bonus"])),
            max(0.0, float(cfg.get("valid_critical_bonus", 0.0))),
        )
        + int(horizon) * per_step_positive
        + margin
    )


def validate_reward_contract(cfg: Mapping[str, float], horizon: int) -> float:
    required = required_invalid_event_penalty(cfg, horizon)
    insufficient = {
        key: float(cfg[key])
        for key in INVALID_EVENT_PENALTY_KEYS
        if float(cfg[key]) < required
    }
    if insufficient:
        raise ValueError(
            f"invalid-event penalties {insufficient} do not dominate the maximum positive "
            f"episode reward; each must be at least {required:.6g}"
        )
    return required


@dataclass(frozen=True)
class RewardBreakdown:
    ttc: float
    proximity: float
    route_progress: float
    priority_alignment: float
    route_deviation: float
    target_collision: float
    valid_critical: float
    non_target_collision: float
    out_of_road: float
    wrong_route: float
    lane_marking_violation: float
    action_l2: float
    action_smoothness: float
    total: float
    def as_dict(self) -> dict[str, float]: return asdict(self)


def compute_reward(
    ttc: float,
    distance: float,
    action: np.ndarray,
    previous_action: np.ndarray,
    events: Mapping[str, bool],
    cfg: Mapping[str, float],
    shaping: Mapping[str, float] | None = None,
) -> RewardBreakdown:
    shaping = shaping or {}
    dense = float(cfg["ttc_weight"]) * float(np.clip((float(cfg["ttc_dense_threshold"]) - ttc) / max(float(cfg["ttc_dense_threshold"]), 1e-6), 0.0, 1.0))
    proximity = float(cfg["proximity_weight"]) * float(np.exp(-max(0.0, distance) / 10.0))
    route_progress = float(cfg.get("route_progress_weight", 0.0)) * float(np.clip(shaping.get("route_progress", 0.0), -1.0, 1.0))
    priority_alignment = float(cfg.get("priority_alignment_weight", 0.0)) * float(np.clip(shaping.get("priority_alignment", 0.0), -1.0, 1.0))
    route_deviation = -float(cfg.get("route_deviation_weight", 0.0)) * float(max(0.0, shaping.get("route_deviation", 0.0)))
    legacy_near_miss = "valid_critical_near_miss" not in events and events.get("rule_satisfied_critical_proximity")
    target = float(cfg["target_collision_bonus"]) if (events.get("target_collision") or legacy_near_miss) else 0.0
    if "valid_critical_near_miss" in events:
        valid_critical = float(cfg.get("valid_critical_bonus", 0.0)) if events.get("valid_critical_near_miss") else 0.0
    else:
        # Historical metric/reward contract retained for non-v2 configs.
        valid_critical = 0.0
    non_target = -float(cfg["non_target_collision_penalty"]) if events.get("non_target_collision") else 0.0
    out_of_road = -float(cfg["out_of_road_penalty"]) * sum(
        bool(events.get(key)) for key in ("adversary_out_of_road", "sut_out_of_road")
    )
    wrong_route = -float(cfg["wrong_route_penalty"]) if events.get("wrong_route") else 0.0
    lane_marking = -float(cfg.get("lane_marking_penalty", 0.0)) if events.get("lane_marking_violation") else 0.0
    a2 = -float(cfg["action_l2_weight"]) * float(np.dot(action, action))
    smooth = -float(cfg["action_smoothness_weight"]) * float(np.dot(action - previous_action, action - previous_action))
    total = dense + proximity + route_progress + priority_alignment + route_deviation + target + valid_critical + non_target + out_of_road + wrong_route + lane_marking + a2 + smooth
    return RewardBreakdown(dense, proximity, route_progress, priority_alignment, route_deviation, target, valid_critical, non_target, out_of_road, wrong_route, lane_marking, a2, smooth, total)
