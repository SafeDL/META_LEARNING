"""S01 cut-in scenario coordinates and executable response labels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


CONFIG = Path(__file__).resolve().parent / "configs" / "scenario_parameter_space.yaml"
NAMES = ("initial_clearance_m", "lead_speed_mps", "lane_change_time_scale_s", "event_start_s")
ALIASES = {"lane_change_time_scale_s": "lane_change_duration_s"}


@dataclass(frozen=True)
class AlignedParameters:
    values: np.ndarray
    present: np.ndarray
    out_of_bounds: np.ndarray


@lru_cache(maxsize=1)
def bounds() -> dict[str, tuple[float, float]]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    item = next(row for row in config["scenarios"] if row["id"] == "S01")
    if tuple(item["axes"]) != NAMES or item["status"] != "executable":
        raise ValueError("S01 parameter schema differs from the frozen contract")
    return {name: tuple(map(float, item["axes"][name])) for name in NAMES}


def align_scenario(scenario: Mapping, *, historical: bool = False) -> AlignedParameters:
    if scenario.get("template_id") != "fbrt_cutin":
        raise ValueError("S01 accepts only fbrt_cutin")
    active = scenario.get("active_parameters")
    fixed = scenario.get("fixed_context")
    if active is not None and not isinstance(active, Mapping):
        raise ValueError("active_parameters must be a mapping")
    if fixed is not None and not isinstance(fixed, Mapping):
        raise ValueError("fixed_context must be a mapping")
    active, fixed = active or {}, fixed or {}
    if any(name in active and name in fixed for name in NAMES):
        raise ValueError("active axis also occurs in fixed_context")
    limits = bounds()
    values, present, outside = [], [], []
    for name in NAMES:
        keys = (name, ALIASES[name]) if historical and name in ALIASES else (name,)
        matches = [float(container[key]) for container in (active, fixed, scenario)
                   for key in keys if key in container and container[key] is not None]
        if matches and any(abs(value - matches[0]) > 1e-9 for value in matches[1:]):
            raise ValueError(f"conflicting values for {name}")
        if not matches and not historical:
            raise ValueError(f"missing required S01 axis: {name}")
        low, high = limits[name]
        u = (matches[0] - low) / (high - low) if matches else 0.0
        values.append(u)
        present.append(bool(matches))
        outside.append(bool(matches and (u < 0 or u > 1)))
    return AlignedParameters(np.asarray(values, dtype=np.float32),
                             np.asarray(present, dtype=np.bool_),
                             np.asarray(outside, dtype=np.bool_))


def coordinates(scenarios: list[dict]) -> np.ndarray:
    aligned = [align_scenario(item) for item in scenarios]
    return np.stack([item.values for item in aligned])


def valid_label(row: Mapping) -> int | None:
    if row.get("inconclusive", False):
        return None
    if row.get("ego_collision") is True:
        return 1
    if row.get("ego_collision") is False and row.get("completed") is True:
        return 0
    return None
