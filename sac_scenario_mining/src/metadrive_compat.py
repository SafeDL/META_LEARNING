"""The sole MetaDrive-version-dependent access layer used by Stage 1.

Stage 1 uses MetaDrive's built-in ``SrS`` map template, whose road-block
sequence is fixed. This module passes a scenario seed and traffic settings to
MetaDrive; it does not generate roads or traffic itself. The seed
deterministically instantiates traffic and initial interaction for that fixed
road layout; it does not vary the ``SrS`` topology. The default agent is the
adversary and the nearest IDM traffic vehicle selected at reset is the SUT.
Actions use MetaDrive's native interface:
``[steering, throttle_or_brake]``.
"""
from __future__ import annotations
from typing import Any
import numpy as np


class MetaDriveUnavailableError(RuntimeError):
    pass


def import_metadrive() -> tuple[Any, Any]:
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        from metadrive.manager.traffic_manager import TrafficMode
        return MetaDriveEnv, TrafficMode
    except ImportError as exc:
        raise MetaDriveUnavailableError(
            "MetaDrive is required for Stage 1. Activate the metadrive conda environment."
        ) from exc


def make_merge_env(config: dict[str, Any], seed: int) -> Any:
    """Instantiate fixed ``SrS`` roads and seed-controlled traffic for one scenario."""
    MetaDriveEnv, TrafficMode = import_metadrive()
    env_cfg = {
        "map": "SrS",
        "start_seed": int(seed),
        "num_scenarios": 1,
        "horizon": int(config["horizon"]),
        "traffic_density": float(config["traffic_density"]),
        "traffic_mode": TrafficMode.Basic,
        # Keep each template instantiation repeatable for its scenario seed.
        "random_traffic": False,
        "use_render": bool(config.get("use_render", False)),
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "out_of_road_done": False,
        "vehicle_config": {
            "enable_reverse": False
        }
    }
    return MetaDriveEnv(env_cfg)


def reset_env(env: Any, seed: int) -> tuple[Any, dict]:
    try:
        return env.reset(seed=int(seed))
    except TypeError:
        return env.reset(force_seed=int(seed))


def adversary(env: Any) -> Any:
    vehicle = getattr(env, "agent", None)
    if vehicle is None: vehicle = env.agents.get("default_agent")
    if vehicle is None:
        raise RuntimeError(
            "MetaDrive default agent (adversary) unavailable after reset")
    return vehicle


def traffic_vehicles(env: Any) -> list[Any]:
    manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    vehicles = getattr(manager, "traffic_vehicles", None)
    return list(vehicles) if vehicles is not None else []


def vehicle_id(vehicle: Any) -> str:
    return str(getattr(vehicle, "id", getattr(vehicle, "name", "unknown")))


def state(vehicle: Any) -> dict[str, float]:
    pos, vel = np.asarray(vehicle.position,
                          dtype=float), np.asarray(vehicle.velocity,
                                                   dtype=float)
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "vx": float(vel[0]),
        "vy": float(vel[1]),
        "heading": float(getattr(vehicle, "heading_theta", 0.0)),
        "yaw_rate": float(getattr(vehicle, "angular_velocity", 0.0))
    }


def out_of_road(env: Any, vehicle: Any) -> bool:
    checker = getattr(env, "_is_out_of_road", None)
    return bool(checker(vehicle)) if checker else bool(
        getattr(vehicle, "out_of_road", False))


def target_contact(adv: Any, sut: Any) -> bool:
    """Identify a collision between the fixed adversary/SUT pair.

    MetaDrive 0.4.3 ``contact_results`` is only a set of *type names* (for
    example ``"VEHICLE"``), not object IDs.  It therefore cannot identify a
    specific traffic vehicle.  The physics callback sets ``crash_vehicle`` on
    each vehicle in a pairwise collision during the same simulation step, so
    both flags are the reliable version-compatible identity signal here.
    """
    return bool(
        getattr(adv, "crash_vehicle", False)
        and getattr(sut, "crash_vehicle", False))


def route_features(vehicle: Any) -> dict[str, float]:
    nav = getattr(vehicle, "navigation", None)
    lane = getattr(vehicle, "lane", None)
    width = float(getattr(lane, "width", 0.0)) if lane is not None else 0.0
    lanes = float(
        getattr(getattr(vehicle, "lane_index",
                        (None, None, 0)), "__getitem__", lambda _: 0)(2) +
        1) if getattr(vehicle, "lane_index", None) else 0.0
    return {
        "lane_width": width,
        "num_lanes": lanes,
        "curvature": 0.0,
        "route_dir_0": 1.0,
        "route_dir_1": 0.0,
        "route_dir_2": 0.0,
        "route_dir_3": 0.0
    }
