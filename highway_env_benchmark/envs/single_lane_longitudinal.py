"""One-lane longitudinal interactions alongside unchanged two-lane cut-ins.

This is a separate scenario grammar. Existing CutInEnv/ExternalCutInEnv
behavior and all completed two-lane experiments remain unchanged.
"""

from __future__ import annotations

from highway_env_benchmark.envs.cutin_env import CutInEnv
from highway_env_benchmark.envs.external_cutin import ExternalCutInEnv


LONGITUDINAL_MODES = frozenset({
    CutInEnv.LEAD_BRAKING,
    CutInEnv.STOP_AND_GO,
    CutInEnv.SLOW_LEAD_FOLLOWING,
})


class _SingleLaneLongitudinalMixin:
    def _create_road(self) -> None:
        self.config["lanes_count"] = (
            1 if self.scenario.mode in LONGITUDINAL_MODES else 2
        )
        super()._create_road()


class SingleLaneLongitudinalEnv(_SingleLaneLongitudinalMixin, CutInEnv):
    """Profile-controlled ego: one lane only in longitudinal modes."""


class SingleLaneExternalCutInEnv(_SingleLaneLongitudinalMixin,
                                 ExternalCutInEnv):
    """External 20 Hz ego: one lane only in longitudinal modes."""
