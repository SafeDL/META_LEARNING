"""The label-free, 37 dimensional logical_merge_obs contract."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

OBSERVATION_SCHEMA = "logical_merge_obs"
OBS_FIELDS = (
    "adv_signed_distance_to_conflict", "adv_route_speed", "adv_route_acceleration", "adv_lateral_offset", "adv_heading_error", "adv_time_to_conflict", "adv_route_progress", "adv_on_route_flag",
    "sut_signed_distance_to_conflict", "sut_route_speed", "sut_route_acceleration", "sut_lateral_offset", "sut_heading_error", "sut_time_to_conflict", "sut_route_progress", "sut_on_route_flag",
    "arrival_time_difference", "euclidean_distance", "relative_route_speed", "closing_speed", "ttc", "conflict_angle", "adversary_priority", "sut_priority",
    "num_incoming_branches", "num_outgoing_branches", "adversary_lane_count", "sut_lane_count", "merge_length", "adversary_route_remaining", "sut_route_remaining", "conflict_radius", "adversary_route_curvature", "sut_route_curvature", "adversary_speed_limit", "sut_speed_limit", "num_conflict_zones",
)
OBSERVATION_DIM = 37
assert len(OBS_FIELDS) == OBSERVATION_DIM


def _clip(value: float, scale: float) -> float:
    return float(np.clip(float(value) / max(float(scale), 1e-6), -1.0, 1.0))


def _vehicle_state(vehicle: Any) -> dict[str, float]:
    pos, velocity = np.asarray(vehicle.position, dtype=float), np.asarray(vehicle.velocity, dtype=float)
    return {"x": float(pos[0]), "y": float(pos[1]), "vx": float(velocity[0]), "vy": float(velocity[1]),
            "speed": float(np.linalg.norm(velocity)), "heading": float(getattr(vehicle, "heading_theta", 0.0)),
            "acceleration": float(np.linalg.norm(np.asarray(getattr(vehicle, "acceleration", [0.0, 0.0]), dtype=float)))}


def _route_terms(state: Mapping[str, float], frame: Mapping[str, Any], vehicle: Any, norm: Mapping[str, float]) -> tuple[list[float], float]:
    origin = np.asarray(frame["origin"], dtype=float)
    delta = np.array([state["x"], state["y"]]) - origin
    distance = float(np.linalg.norm(delta))
    lane = getattr(vehicle, "lane", None)
    length = float(getattr(lane, "length", 1.0))
    longitudinal = float(lane.local_coordinates(np.asarray(vehicle.position))[0]) if lane is not None else 0.0
    remaining = max(0.0, length - longitudinal)
    time = min(float(norm["time_s"]), distance / max(state["speed"], 0.1))
    heading_error = float(np.arctan2(delta[1], delta[0]) - state["heading"])
    return [
        _clip(distance, norm["distance_m"]), _clip(state["speed"], norm["speed_mps"]), _clip(state["acceleration"], norm["acceleration_mps2"]),
        _clip(float(lane.local_coordinates(np.asarray(vehicle.position))[1]) if lane is not None else 0.0, 5.0), _clip(heading_error, norm["angle_rad"]),
        _clip(time, norm["time_s"]), _clip(longitudinal / max(length, 1.0), 1.0), 1.0 if bool(getattr(vehicle, "on_lane", True)) else 0.0,
    ], remaining


def _ttc(relative_position: np.ndarray, relative_velocity: np.ndarray, cap: float) -> float:
    distance = float(np.linalg.norm(relative_position))
    if distance <= 1e-6:
        return 0.0
    closing = float(np.dot(relative_position, relative_velocity) / distance)
    return min(cap, distance / -closing) if closing < 0.0 else cap


def build_observation(adversary: Any, sut: Any, frame: Mapping[str, Any], topology: Mapping[str, float], config: Mapping[str, Any]) -> np.ndarray:
    norm = config["normalization"]
    adv, target = _vehicle_state(adversary), _vehicle_state(sut)
    adv_terms, adv_remaining = _route_terms(adv, frame, adversary, norm)
    sut_terms, sut_remaining = _route_terms(target, frame, sut, norm)
    rel_p = np.array([target["x"] - adv["x"], target["y"] - adv["y"]])
    rel_v = np.array([target["vx"] - adv["vx"], target["vy"] - adv["vy"]])
    adv_time, sut_time = adv_terms[5] * norm["time_s"], sut_terms[5] * norm["time_s"]
    interaction = [
        _clip(adv_time - sut_time, norm["time_s"]), _clip(float(np.linalg.norm(rel_p)), norm["distance_m"]),
        _clip(target["speed"] - adv["speed"], norm["speed_mps"]), _clip(float(np.dot(rel_p, rel_v) / max(np.linalg.norm(rel_p), 1e-6)), norm["speed_mps"]),
        _clip(_ttc(rel_p, rel_v, 5.0), norm["time_s"]), _clip(target["heading"] - adv["heading"], norm["angle_rad"]), 0.0, 1.0,
    ]
    lane_count = topology["lane_count"]
    descriptor = [
        _clip(topology["num_incoming_branches"], norm["lane_count"]), _clip(topology["num_outgoing_branches"], norm["lane_count"]), _clip(lane_count, norm["lane_count"]), _clip(lane_count, norm["lane_count"]),
        _clip(topology["merge_length_m"], norm["distance_m"]), _clip(adv_remaining, norm["distance_m"]), _clip(sut_remaining, norm["distance_m"]), _clip(topology["conflict_radius_m"], norm["distance_m"]),
        _clip(topology["route_curvature"], norm["curvature"]), _clip(topology["route_curvature"], norm["curvature"]), _clip(topology["speed_limit_mps"], norm["speed_mps"]), _clip(topology["speed_limit_mps"], norm["speed_mps"]), _clip(topology["num_conflict_zones"], 2.0),
    ]
    obs = np.asarray(adv_terms + sut_terms + interaction + descriptor, dtype=np.float32)
    if obs.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(obs)):
        raise ValueError(f"{OBSERVATION_SCHEMA} contract violated: {obs.shape}")
    return np.clip(obs, -1.0, 1.0)
