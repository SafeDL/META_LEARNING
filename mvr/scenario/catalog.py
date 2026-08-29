"""Universal interaction-centric initial-condition search space."""
from __future__ import annotations

from .parameter_space import ParameterSpace


def mvr_parameter_spaces() -> dict[str, ParameterSpace]:
    def common(
        speed_limit_mps: float,
        initial_speed_limit_mps: float | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Keep sampled initial states inside the family traffic contract."""
        initial_speed_limit_mps = (
            speed_limit_mps
            if initial_speed_limit_mps is None
            else initial_speed_limit_mps
        )
        return {
            "adversary_distance_to_conflict_m": (0.5, 5.0),
            "sut_distance_to_conflict_m": (0.5, 5.0),
            "adversary_initial_speed_mps": (4.0, initial_speed_limit_mps),
            "sut_initial_speed_mps": (4.0, initial_speed_limit_mps),
            "maneuver_onset_progress": (0.2, 0.8),
        }

    cutin = {
        **common(20.0),
        "adversary_distance_to_conflict_m": (20.0, 80.0),
        "sut_distance_to_conflict_m": (20.0, 80.0),
    }
    return {
        "merge": ParameterSpace(
            "merge", ("main_conflict", "downstream_merge"), common(18.0)
        ),
        "cutin": ParameterSpace(
            "cutin",
            ("left_target_lane", "right_target_lane"),
            cutin,
        ),
        "roundabout": ParameterSpace(
            "roundabout",
            ("entry_0_exit_1", "entry_1_exit_2", "entry_2_exit_0"),
            common(12.0, initial_speed_limit_mps=6.5),
        ),
    }
