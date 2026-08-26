"""Bounded adversarial reward for the Inner controller."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .criteria import FailureCriteria


class InnerRiskReward:
    """Reward lawful critical interaction, rather than deliberately driving into impact."""

    def __init__(self, criteria: FailureCriteria) -> None:
        self.criteria = criteria

    def __call__(self, features: np.ndarray, info: Mapping[str, object], option: str, step: int, max_steps: int) -> float:
        row = np.asarray(features, dtype=float)
        ttc = max(0.0, row[8] * 15.0)
        distance = max(0.0, row[10] * 100.0)
        closing = distance / max(ttc, 1e-3) if ttc < 15.0 else 0.0
        ttc_criticality = np.exp(-((ttc - self.criteria.ttc_s) / max(self.criteria.ttc_s * 0.5, 1e-3)) ** 2)
        distance_criticality = np.exp(-((distance - self.criteria.distance_m) / max(self.criteria.distance_m * 0.5, 1e-3)) ** 2)
        closing_intensity = min(closing / self.criteria.closing_speed_mps, 1.0)
        criticality = 0.5 * ttc_criticality + 0.3 * distance_criticality + 0.2 * closing_intensity
        conflict = 1.0 - min(abs(row[11]), 1.0)
        if option == "yield_then_press":
            intent = 1.0 - conflict if step * 2 < max_steps else conflict
        elif option == "gap_close":
            intent = closing_intensity
        else:
            intent = conflict
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
        target_collision = bool(
            info.get("target_collision", info.get("crash_vehicle", False))
        )
        valid_near_miss = (
            0.0 < distance < self.criteria.distance_m
            and ttc < self.criteria.ttc_s
        )
        event_bonus = 0.0
        if not invalid:
            if target_collision:
                event_bonus = 1.5
            elif valid_near_miss:
                event_bonus = 1.0
        # Criticality and intent are only non-positive shaping here.  They
        # can guide SAC toward the conflict, but cannot be mistaken for a
        # successful attack.  Positive reward is reserved for a lawful
        # collision or a lawful thresholded near miss.
        shaping = -0.05 * (1.0 - criticality) - 0.01 * (1.0 - intent)
        return float(np.clip(shaping + event_bonus - 1.25 * float(invalid), -2.0, 2.0))
