"""Bounded adversarial reward for the Inner controller."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .criteria import FailureCriteria


class InnerRiskReward:
    """Reward risk creation while keeping invalid physics unattractive."""

    def __init__(self, criteria: FailureCriteria) -> None:
        self.criteria = criteria

    def __call__(self, features: np.ndarray, info: Mapping[str, object], option: str, step: int, max_steps: int) -> float:
        row = np.asarray(features, dtype=float)
        ttc = max(0.0, row[8] * 15.0)
        distance = max(0.0, row[10] * 100.0)
        closing = distance / max(ttc, 1e-3) if ttc < 15.0 else 0.0
        ttc_risk = 1.0 - min(ttc / self.criteria.ttc_s, 1.0)
        distance_risk = 1.0 - min(distance / self.criteria.distance_m, 1.0)
        closing_risk = min(closing / self.criteria.closing_speed_mps, 1.0)
        risk = 0.5 * ttc_risk + 0.3 * distance_risk + 0.2 * closing_risk
        conflict = 1.0 - min(abs(row[11]), 1.0)
        if option == "yield_then_press":
            intent = 1.0 - conflict if step * 2 < max_steps else conflict
        elif option == "gap_close":
            intent = closing_risk
        elif option == "route_block":
            intent = distance_risk
        else:
            intent = conflict
        invalid = any(bool(info.get(key, False)) for key in ("non_target_collision", "adversary_out_of_road", "sut_out_of_road", "wrong_route"))
        return float(np.clip(risk - float(invalid) + 0.1 * intent, -1.0, 1.1))
