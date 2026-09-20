"""Deterministic, identity-free Cut-in behavior template for DIVA."""
from __future__ import annotations

import numpy as np

from ..training.runner import InnerActionPhase


class DivaCutInBehavior:
    """Track a fixed path while retaining the red vehicle's reset speed."""

    speed_control_gain = 0.30
    maximum_longitudinal_action = 0.40

    def __init__(self) -> None:
        self._prescribed_speed_mps: float | None = None

    @property
    def prescribed_speed_mps(self) -> float | None:
        return self._prescribed_speed_mps

    def __call__(self, phase: InnerActionPhase) -> np.ndarray:
        if self._prescribed_speed_mps is None:
            self._prescribed_speed_mps = float(phase.adversary_speed_mps)
        target_speed = self._prescribed_speed_mps
        longitudinal = float(np.clip(
            self.speed_control_gain * (target_speed - float(phase.adversary_speed_mps)),
            -self.maximum_longitudinal_action,
            self.maximum_longitudinal_action,
        ))
        return np.asarray(
            (0.0, 0.0, 0.0, longitudinal), dtype=np.float32
        )
