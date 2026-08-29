"""Bounded adversarial reward for the Inner controller."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .criteria import FailureCriteria


class InnerRiskReward:
    """Reward lawful critical interaction, rather than deliberately driving into impact."""

    def __init__(self, criteria: FailureCriteria) -> None:
        self.criteria = criteria

    def __call__(self, features: np.ndarray, info: Mapping[str, object]) -> float:
        row = np.asarray(features, dtype=float)
        ttc = max(0.0, row[8] * 15.0)
        distance = max(0.0, row[10] * 100.0)
        closing = distance / max(ttc, 1e-3) if ttc < 15.0 else 0.0
        ttc_criticality = np.exp(-((ttc - self.criteria.ttc_s) / max(self.criteria.ttc_s * 0.5, 1e-3)) ** 2)
        distance_criticality = np.exp(-((distance - self.criteria.distance_m) / max(self.criteria.distance_m * 0.5, 1e-3)) ** 2)
        closing_intensity = min(closing / self.criteria.closing_speed_mps, 1.0)
        criticality = 0.5 * ttc_criticality + 0.3 * distance_criticality + 0.2 * closing_intensity
        # Unit-level reward probes do not carry monitor metadata; treat those
        # as an interaction probe.  Simulator rollouts always provide the
        # explicit semantic flag, which gates shaping outside the corridor.
        challenge_active = bool(
            info["semantic_challenge_phase_active"]
            if "semantic_challenge_phase_active" in info
            else True
        )
        # Risk shaping is only meaningful inside the contract-defined
        # interaction corridor.  Outside it the controller receives a small
        # neutral cost, so it cannot farm proximity while approaching.
        criticality *= float(challenge_active)
        conflict = 1.0 - min(abs(row[11]), 1.0)
        invalid = any(
            bool(info.get(key, False))
            for key in (
                "non_target_collision",
                "adversary_out_of_road",
                "sut_out_of_road",
                "wrong_route",
                "adversary_traffic_violation",
            )
        )
        event_just_captured = bool(info.get("event_just_captured", False))
        target_collision = bool(
            event_just_captured and info.get("valid_target_collision", False)
        )
        valid_near_miss = bool(
            event_just_captured
            and info.get("event_kind") == "near_miss"
            and info.get("event_semantic_valid", False)
            and info.get("event_traffic_valid", False)
        )
        event_bonus = 0.0
        if not invalid:
            if target_collision:
                event_bonus = 4.0
            elif valid_near_miss:
                event_bonus = 3.0
        # Dense risk shaping is a training signal, not the formal event
        # score. It is strictly confined to the semantic interaction
        # corridor, where lowering TTC/distance and increasing closing speed
        # are meaningful rather than a pre-conflict reward loophole.
        risk_reward = 0.20 * criticality
        shield_penalty = float(info.get("traffic_shield_intervention_l2", 0.0)) ** 2
        action_penalty = 0.02 * float(info.get("inner_raw_action_l2", 0.0))
        corridor_penalty = 0.002 * (1.0 - float(challenge_active))
        return float(np.clip(
            risk_reward + event_bonus - action_penalty - 0.10 * shield_penalty
            - corridor_penalty - 1.25 * float(invalid),
            -2.0,
            4.0,
        ))
