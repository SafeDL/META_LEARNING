"""Versioned route-correct observation contracts for logical merge tasks."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .routes import RoutePolyline


OBSERVATION_SCHEMA = "logical_merge_obs"
OBS_FIELDS = (
    "adv_signed_distance_to_conflict", "adv_route_speed", "adv_route_acceleration", "adv_lateral_offset", "adv_heading_error", "adv_time_to_conflict", "adv_route_progress", "adv_on_route_flag",
    "sut_signed_distance_to_conflict", "sut_route_speed", "sut_route_acceleration", "sut_lateral_offset", "sut_heading_error", "sut_time_to_conflict", "sut_route_progress", "sut_on_route_flag",
    "arrival_time_difference", "euclidean_distance", "relative_route_speed", "closing_speed", "ttc", "conflict_angle", "adversary_priority", "sut_priority",
    "num_incoming_branches", "num_outgoing_branches", "adversary_lane_count", "sut_lane_count", "merge_length", "adversary_route_remaining", "sut_route_remaining", "conflict_radius", "adversary_route_curvature", "sut_route_curvature", "adversary_speed_limit", "sut_speed_limit", "num_conflict_zones",
)
OBSERVATION_DIM = 37
assert len(OBS_FIELDS) == OBSERVATION_DIM

DYNAMIC_OBSERVATION_SCHEMA = "logical_merge_dynamic_obs_v1"
DYNAMIC_OBS_FIELDS = OBS_FIELDS[:22] + (
    "adversary_route_remaining",
    "sut_route_remaining",
)
DYNAMIC_OBSERVATION_DIM = 24
assert len(DYNAMIC_OBS_FIELDS) == DYNAMIC_OBSERVATION_DIM

OBSERVATION_SCHEMAS = {
    OBSERVATION_SCHEMA: OBS_FIELDS,
    DYNAMIC_OBSERVATION_SCHEMA: DYNAMIC_OBS_FIELDS,
}


def observation_fields(schema: str) -> tuple[str, ...]:
    try:
        return OBSERVATION_SCHEMAS[str(schema)]
    except KeyError as error:
        raise ValueError(f"unsupported observation schema: {schema!r}") from error


def observation_dim(schema: str) -> int:
    return len(observation_fields(schema))


def _clip(value: float, scale: float) -> float:
    return float(np.clip(float(value) / max(float(scale), 1e-6), -1.0, 1.0))


def _vehicle_state(vehicle: Any) -> dict[str, Any]:
    position = np.asarray(vehicle.position, dtype=float)
    velocity = np.asarray(vehicle.velocity, dtype=float)
    acceleration = np.asarray(getattr(vehicle, "acceleration", [0.0, 0.0]), dtype=float)
    return {"position": position, "velocity": velocity, "acceleration": acceleration, "heading": float(getattr(vehicle, "heading_theta", 0.0))}


def _route_terms(state: Mapping[str, Any], vehicle: Any, route: RoutePolyline, conflict_s_m: float, norm: Mapping[str, float]) -> tuple[list[float], dict[str, float]]:
    lane_width = float(getattr(getattr(vehicle, "lane", None), "width", 3.8))
    projection = route.projection(state["position"], state["heading"], lane_width)
    distance = float(conflict_s_m - projection.s_m)
    route_speed = float(np.dot(state["velocity"], projection.tangent))
    route_acceleration = float(np.dot(state["acceleration"], projection.tangent))
    time = distance / route_speed if route_speed > 1e-6 and distance > 0.0 else float(norm["time_s"])
    remaining = max(0.0, route.length_m - projection.s_m)
    values = [
        _clip(distance, norm["distance_m"]), _clip(route_speed, norm["speed_mps"]), _clip(route_acceleration, norm["acceleration_mps2"]),
        _clip(projection.lateral_m, 5.0), _clip(projection.heading_error, norm["angle_rad"]), _clip(time, norm["time_s"]),
        _clip(projection.s_m / max(route.length_m, 1e-6), 1.0), 1.0 if projection.on_route else 0.0,
    ]
    return values, {"distance": distance, "speed": route_speed, "time": min(float(norm["time_s"]), time), "remaining": remaining, "tangent": projection.tangent}


def _ttc(relative_position: np.ndarray, relative_velocity: np.ndarray, cap: float) -> float:
    distance = float(np.linalg.norm(relative_position))
    if distance <= 1e-6:
        return 0.0
    closing = float(np.dot(relative_position, relative_velocity) / distance)
    return min(cap, distance / -closing) if closing < 0.0 else cap


def build_observation(adversary: Any, sut: Any, frame: Mapping[str, Any], topology: Mapping[str, float], config: Mapping[str, Any]) -> np.ndarray:
    norm = config["normalization"]
    adv_state, sut_state = _vehicle_state(adversary), _vehicle_state(sut)
    adv_terms, adv = _route_terms(adv_state, adversary, frame["adversary_route"], float(frame["adversary_conflict_s_m"]), norm)
    sut_terms, target = _route_terms(sut_state, sut, frame["sut_route"], float(frame["sut_conflict_s_m"]), norm)
    relative_position = sut_state["position"] - adv_state["position"]
    relative_velocity = sut_state["velocity"] - adv_state["velocity"]
    tangent_dot = float(np.clip(np.dot(adv["tangent"], target["tangent"]), -1.0, 1.0))
    interaction = [
        _clip(adv["time"] - target["time"], norm["time_s"]), _clip(float(np.linalg.norm(relative_position)), norm["distance_m"]),
        _clip(target["speed"] - adv["speed"], norm["speed_mps"]), _clip(float(np.dot(relative_position, relative_velocity) / max(np.linalg.norm(relative_position), 1e-6)), norm["speed_mps"]),
        _clip(_ttc(relative_position, relative_velocity, float(config["reward"]["ttc_cap"])), norm["time_s"]), _clip(float(np.arccos(tangent_dot)), norm["angle_rad"]),
        0.0 if bool(frame["priority_spec"]["sut_has_priority"]) else 1.0, 1.0 if bool(frame["priority_spec"]["sut_has_priority"]) else 0.0,
    ]
    descriptor = [
        _clip(topology["num_incoming_branches"], norm["lane_count"]), _clip(topology["num_outgoing_branches"], norm["lane_count"]),
        _clip(topology["adversary_lane_count"], norm["lane_count"]), _clip(topology["sut_lane_count"], norm["lane_count"]),
        _clip(topology["merge_length_m"], norm["distance_m"]), _clip(adv["remaining"], norm["distance_m"]), _clip(target["remaining"], norm["distance_m"]),
        _clip(topology["conflict_radius_m"], norm["distance_m"]), _clip(topology["adversary_route_curvature"], norm["curvature"]), _clip(topology["sut_route_curvature"], norm["curvature"]),
        _clip(topology["adversary_speed_limit_mps"], norm["speed_mps"]), _clip(topology["sut_speed_limit_mps"], norm["speed_mps"]), _clip(topology["num_conflict_zones"], 2.0),
    ]
    if bool(config.get("ablation", {}).get("no_topology", False)):
        descriptor = [0.0] * len(descriptor)
    full = np.asarray(adv_terms + sut_terms + interaction + descriptor, dtype=np.float32)
    schema = str(config["environment"]["observation_schema"])
    fields = observation_fields(schema)
    if schema == OBSERVATION_SCHEMA:
        observation = full
    else:
        indexes = [OBS_FIELDS.index(name) for name in fields]
        observation = full[indexes]
    if observation.shape != (len(fields),) or not np.all(np.isfinite(observation)):
        raise ValueError(f"{schema} contract violated: {observation.shape}")
    return np.clip(observation, -1.0, 1.0)
