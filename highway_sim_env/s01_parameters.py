"""S01 cut-in scenario coordinates and executable response labels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


CONFIG = Path(__file__).resolve().parent / "configs" / "scenario_parameter_space.yaml"
NAMES = ("initial_clearance_m", "lead_speed_mps", "lane_change_time_scale_s", "event_start_s")
@lru_cache(maxsize=1)
def bounds() -> dict[str, tuple[float, float]]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    item = next(row for row in config["scenarios"] if row["id"] == "S01")
    if tuple(item["axes"]) != NAMES or item["status"] != "executable":
        raise ValueError("S01 parameter schema differs from the frozen contract")
    return {name: tuple(map(float, item["axes"][name])) for name in NAMES}


def coordinates(scenarios: list[dict]) -> np.ndarray:
    """Normalize the four active axes in the A and D scene schema."""
    limits = bounds()
    return np.asarray([
        [(float(scene["active_parameters"][name]) - limits[name][0]) /
         (limits[name][1] - limits[name][0]) for name in NAMES]
        for scene in scenarios
    ], dtype=np.float32)


def valid_label(row: Mapping) -> int | None:
    if row["inconclusive"]:
        return None
    if row["ego_collision"] is True:
        return 1
    if row["ego_collision"] is False and row["completed"] is True:
        return 0
    return None
