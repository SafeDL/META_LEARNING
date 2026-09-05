"""Bounded adversarial reward for the Inner controller."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .criteria import FailureCriteria


class InnerRiskReward:
    """Reward valid target collisions first, with legal criticality as shaping."""

    def __init__(self, criteria: FailureCriteria) -> None:
        self.criteria = criteria
        self._previous_progress = 0.0
        self._previous_criticality = 0.0

    def reset(self) -> None:
        self._previous_progress = 0.0
        self._previous_criticality = 0.0

    def __call__(self, features: np.ndarray, info: Mapping[str, object]) -> float:
        row = np.asarray(features, dtype=float)
        ttc = max(0.0, row[8] * 15.0)
        distance = max(0.0, row[10] * 100.0)
        closing = distance / max(ttc, 1e-3) if ttc < 15.0 else 0.0
        # Accident search needs a monotone danger signal: moving from the
        # contract threshold toward contact must never reduce the shaping
        # reward.  A Gaussian around the threshold would peak at a near miss
        # and then fall again as the vehicles approach impact.
        ttc_criticality = np.exp(-ttc / max(self.criteria.ttc_s, 1e-3))
        distance_criticality = np.exp(
            -distance / max(self.criteria.distance_m, 1e-3)
        )
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
        # The reward remains continuous before the formal failure threshold;
        # those criteria still decide events and never become trainable rules.
        criticality *= float(challenge_active)
        invalid = any(
            bool(info.get(key, False))
            for key in (
                "non_target_collision",
                "adversary_out_of_road",
                "sut_out_of_road",
                "wrong_route",
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
            and info.get("event_execution_valid", False)
        )
        event_bonus = 0.0
        if not invalid:
            if target_collision:
                event_bonus = 12.0
            elif valid_near_miss:
                event_bonus = 6.0
        # Dense risk shaping is a training signal, not the formal event
        # score. It is strictly confined to the semantic interaction
        # corridor, where lowering TTC/distance and increasing closing speed
        # are meaningful rather than a pre-conflict reward loophole.
        risk_reward = 2.0 * (criticality - self._previous_criticality)
        self._previous_criticality = criticality
        tracking_penalty = 0.0
        progress = float(info.get("maneuver_reference_progress", 0.0))
        if progress > 0.0:
            lateral_error = abs(float(
                info.get("maneuver_reference_lateral_error_m", 0.0)
            ))
            heading_error = abs(float(
                info.get("maneuver_reference_heading_error_rad", 0.0)
            ))
            tracking_penalty = 0.03 * min(1.0, lateral_error / 3.5) + 0.01 * min(
                1.0, heading_error / (0.5 * np.pi)
            )
        # This is a bounded path-completion potential, not an absolute-speed
        # incentive.  It makes legal lateral completion preferable to simply
        # remaining in the source lane while risk is accumulated.
        progress_reward = 0.10 * max(0.0, progress - self._previous_progress)
        self._previous_progress = max(self._previous_progress, progress)
        shield_penalty = float(info.get("traffic_shield_intervention_l2", 0.0)) ** 2
        return float(np.clip(
            risk_reward + event_bonus + progress_reward - tracking_penalty
            - 0.10 * shield_penalty - 2.0 * float(invalid),
            -3.0,
            12.0,
        ))
