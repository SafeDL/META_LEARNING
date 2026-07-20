"""Deterministic logical-case tables for the fixed ``on_ramp_merge`` task."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np


def _sample_case(rng: np.random.Generator, split: str, index: int,
                 cfg: Mapping[str, Any]) -> dict[str, Any]:
    spec = cfg["logical_scenario"]
    space = spec["parameter_space"]
    constraints = spec["constraints"]
    # The fixed SrS ramp reaches its conflict point after approximately 49 m;
    # the mainline approach ends there after 58.59 m.  Reject only cases that
    # cannot create a meaningful time-window interaction before the merge.
    for _ in range(1000):
        theta = {
            key: float(rng.uniform(*space[key]))
            for key in ("sut_speed_mps", "adversary_speed_mps", "longitudinal_gap_m",
                        "adversary_ramp_position_m", "background_density")
        }
        adv_remaining = 49.0 - theta["adversary_ramp_position_m"]
        sut_remaining = 58.59 - (
            10.0 + theta["adversary_ramp_position_m"] + theta["longitudinal_gap_m"])
        arrival_gap = abs(adv_remaining / theta["adversary_speed_mps"] -
                          sut_remaining / theta["sut_speed_mps"])
        # The ramp approaches the rightmost mainline lane with about 8.2 m of
        # lateral separation at reset, then physically merges into that lane.
        initial_distance = float(np.hypot(theta["longitudinal_gap_m"], 8.19))
        if (sut_remaining > 0.0 and initial_distance >= float(constraints["min_initial_distance_m"])
                and arrival_gap <= float(constraints["max_merge_arrival_time_gap_s"])):
            return {
                "case_id": f"{split}_{index:03d}",
                "background_seed": int(rng.integers(1, 2**31 - 1)),
                "theta": theta,
                "expected_initial_distance_m": initial_distance,
                "expected_merge_arrival_time_gap_s": arrival_gap,
            }
    raise RuntimeError(f"could not sample a feasible {split} case after 1000 attempts")


def build_case_table(cfg: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    """Build the explicit, non-overlapping case table for one named split."""
    try:
        set_cfg = cfg["logical_scenario"]["case_sets"][split]
    except KeyError as exc:
        raise ValueError(f"unknown case split: {split}") from exc
    rng = np.random.default_rng(int(set_cfg["case_seed"]))
    return [_sample_case(rng, split, index, cfg) for index in range(int(set_cfg["num_cases"]))]


def validate_case(case: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an externally supplied case before passing it to MetaDrive."""
    if not isinstance(case.get("case_id"), str) or not case["case_id"]:
        raise ValueError("case_id is required")
    if not isinstance(case.get("background_seed"), (int, np.integer)):
        raise ValueError("background_seed must be an integer")
    theta = case.get("theta")
    if not isinstance(theta, Mapping):
        raise ValueError("theta mapping is required")
    required = cfg["logical_scenario"]["parameter_space"]
    clean = deepcopy(dict(case))
    clean["theta"] = {}
    for key, limits in required.items():
        value = float(theta[key])
        low, high = map(float, limits)
        if not low <= value <= high:
            raise ValueError(f"theta.{key}={value} outside [{low}, {high}]")
        clean["theta"][key] = value
    clean["background_seed"] = int(case["background_seed"])
    return clean
