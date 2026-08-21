"""Frozen MVR family parameter spaces and held-out geometry-template labels."""
from __future__ import annotations

from .option import AdversarialOption
from .parameter_space import ParameterSpace


def mvr_parameter_spaces() -> dict[str, ParameterSpace]:
    options = tuple(AdversarialOption)
    common = {
        "adversary_spawn_m": (2.0, 40.0), "sut_spawn_m": (2.0, 40.0),
        "adversary_initial_speed_mps": (4.0, 20.0), "sut_initial_speed_mps": (4.0, 20.0),
    }
    return {
        "merge_v1": ParameterSpace("merge_v1", ("main_conflict", "downstream_merge"), common, options),
        "cutin_v1": ParameterSpace("cutin_v1", ("left_target_lane", "right_target_lane"), common, options),
        "roundabout_v1": ParameterSpace("roundabout_v1", ("entry_0_exit_1", "entry_1_exit_2", "entry_2_exit_0"), common, options),
    }
