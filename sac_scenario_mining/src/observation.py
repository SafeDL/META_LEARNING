"""Fixed, versioned 38-dimensional observation schema for Stage 1."""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

OBSERVATION_SCHEMA = "stage1_obs_v1"
OBS_FIELDS = (
    "adv_vx",
    "adv_vy",
    "adv_heading",
    "adv_yaw_rate",
    "adv_spawn_dx",
    "adv_spawn_dy",
    "sut_dx",
    "sut_dy",
    "sut_dvx",
    "sut_dvy",
    "sut_heading_rel",
    "sut_distance",
    "near1_dx",
    "near1_dy",
    "near1_dvx",
    "near1_dvy",
    "near1_heading_rel",
    "near1_distance",
    "near2_dx",
    "near2_dy",
    "near2_dvx",
    "near2_dvy",
    "near2_heading_rel",
    "near2_distance",
    "near3_dx",
    "near3_dy",
    "near3_dvx",
    "near3_dvy",
    "near3_heading_rel",
    "near3_distance",
    "route_dir_0",
    "route_dir_1",
    "route_dir_2",
    "route_dir_3",
    "lane_width",
    "curvature",
    "num_lanes",
    "merge_flag",
)
assert len(OBS_FIELDS) == 38


def clipped(value: float, scale: float) -> float:
    return float(np.clip(float(value) / max(float(scale), 1e-6), -1.0, 1.0))


def vehicle_features(adv: Mapping[str, float], other: Mapping[str, float],
                     norm: Mapping[str, float]) -> list[float]:
    dx, dy = float(other["x"] - adv["x"]), float(other["y"] - adv["y"])
    distance = float(np.hypot(dx, dy))
    return [
        clipped(dx, norm["relative_distance"]),
        clipped(dy, norm["lateral_distance"]),
        clipped(other["vx"] - adv["vx"], norm["velocity"]),
        clipped(other["vy"] - adv["vy"], norm["lateral_velocity"]),
        clipped(other["heading"] - adv["heading"], norm["heading"]),
        clipped(distance, norm["relative_distance"])
    ]


def build_observation(adv: Mapping[str, float], sut: Mapping[str, float],
                      nearby: Sequence[Mapping[str, float]],
                      spawn: tuple[float, float], road: Mapping[str, float],
                      norm: Mapping[str, float]) -> np.ndarray:
    values = [
        clipped(adv["vx"], norm["velocity"]),
        clipped(adv["vy"], norm["lateral_velocity"]),
        clipped(adv["heading"], norm["heading"]),
        clipped(adv.get("yaw_rate", 0.0), norm["yaw_rate"]),
        clipped(adv["x"] - spawn[0], norm["longitudinal_distance"]),
        clipped(adv["y"] - spawn[1], norm["lateral_distance"])
    ]
    values += vehicle_features(adv, sut, norm)
    for vehicle in list(nearby)[:3]:
        values += vehicle_features(adv, vehicle, norm)
    values += [0.0] * (18 - 6 * min(len(nearby), 3))
    values += [clipped(road.get(f"route_dir_{i}", 0.0), 1.0) for i in range(4)]
    values += [
        clipped(road.get("lane_width", 0.0), norm["lane_width"]),
        clipped(road.get("curvature", 0.0), norm["curvature"]),
        clipped(road.get("num_lanes", 0.0), norm["num_lanes"]), 1.0
    ]
    obs = np.asarray(values, dtype=np.float32)
    if obs.shape != (38, ):
        raise ValueError(
            f"{OBSERVATION_SCHEMA} expected 38 fields, got {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("observation contains NaN or Inf")
    return np.clip(obs, -1.0, 1.0).astype(np.float32)
