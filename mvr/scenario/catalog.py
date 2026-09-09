"""Universal interaction-centric initial-condition search space."""
from __future__ import annotations

import numpy as np

from ..physical_limits import CUTIN_LATERAL_ACCELERATION_LIMIT_MPS2
from .parameter_space import ParameterSpace

CUTIN_MIN_INITIAL_GAP_M = 2.0
CUTIN_MIN_INITIAL_SPEED_MPS = 7.0
CUTIN_MAX_INITIAL_SPEED_MPS = 13.0
CUTIN_REFERENCE_LATERAL_ACCELERATION_MPS2 = (
    CUTIN_LATERAL_ACCELERATION_LIMIT_MPS2
)
CUTIN_NOMINAL_LANE_SHIFT_M = 3.5
CUTIN_QUINTIC_MAX_SECOND_DERIVATIVE = 5.7735026919
CUTIN_PATH_LENGTH_SAFETY_FACTOR = 1.05
CUTIN_POST_MANEUVER_FOLLOW_THROUGH_M = 30.0


def minimum_cutin_path_length_m(red_speed_mps: float) -> float:
    """Minimum path length for a 3.5 m Cut-in below 0.6 g laterally."""
    return float(
        CUTIN_PATH_LENGTH_SAFETY_FACTOR
        * float(red_speed_mps)
        * np.sqrt(
            CUTIN_NOMINAL_LANE_SHIFT_M * CUTIN_QUINTIC_MAX_SECOND_DERIVATIVE
            / CUTIN_REFERENCE_LATERAL_ACCELERATION_MPS2
        )
    )


def valid_cutin_initial_state(
    ego_speed_mps: float,
    relative_speed_mps: float,
    initial_gap_m: float,
    path_length_m: float,
) -> bool:
    """Check the coupled physical reset constraints for a Cut-in design."""
    red_speed_mps = float(ego_speed_mps) + float(relative_speed_mps)
    return bool(
        CUTIN_MIN_INITIAL_SPEED_MPS <= float(ego_speed_mps) <= CUTIN_MAX_INITIAL_SPEED_MPS
        and red_speed_mps > 0.0
        and float(initial_gap_m) >= CUTIN_MIN_INITIAL_GAP_M
        and float(path_length_m) >= minimum_cutin_path_length_m(red_speed_mps)
    )


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
        # A Cut-in is specified in the two vehicles' shared longitudinal
        # frame.  It has no fixed conflict point: the route merely provides
        # a legal lane-change corridor.
        # The initial gap is the bumper-to-bumper separation at reset,
        # never a centre-to-centre proxy. The red speed is derived as ego
        # speed plus relative speed; it is only required to remain positive.
        "initial_gap_m": (7.0, 16.0),
        "ego_initial_speed_mps": (
            CUTIN_MIN_INITIAL_SPEED_MPS, CUTIN_MAX_INITIAL_SPEED_MPS,
        ),
        "relative_speed_mps": (-2.5, 2.5),
        "cutin_start_offset_m": (35.0, 600.0),
        "cutin_path_length_m": (30.0, 130.0),
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
