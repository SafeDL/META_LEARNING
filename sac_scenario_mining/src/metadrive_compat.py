"""The sole MetaDrive-version-dependent layer for ``on_ramp_merge``.

The road is MetaDrive's fixed ``SrS`` template.  A case explicitly places the
SAC-controlled adversary on the in-ramp and an IDM-controlled SUT on the
mainline before the merge.  Background vehicles, when requested by a case,
are separate IDM vehicles and never determine either role.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# These lane IDs are the forward in-ramp and mainline approach of ``SrS`` in
# MetaDrive 0.4.3.  Validate them against the created map at every reset.
RAMP_LANE = ("2r1_0_", "2r1_1_", 0)
MAINLINE_LANE = ("1S0_0_", "2r0_0_", 2)
BACKGROUND_LANE = ("1S0_0_", "2r0_0_", 1)
SUT_COLOR = (0.12, 0.43, 0.95)
ADVERSARY_COLOR = (0.92, 0.16, 0.14)


class MetaDriveUnavailableError(RuntimeError):
    pass


def import_metadrive() -> tuple[Any, Any, Any, Any]:
    try:
        from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle
        from metadrive.envs.metadrive_env import MetaDriveEnv
        from metadrive.manager.traffic_manager import TrafficMode
        from metadrive.policy.idm_policy import IDMPolicy
        return MetaDriveEnv, TrafficMode, TrafficDefaultVehicle, IDMPolicy
    except ImportError as exc:
        raise MetaDriveUnavailableError(
            "MetaDrive is required. Activate the metadrive conda environment."
        ) from exc


def make_merge_env(config: Mapping[str, Any], case: Mapping[str, Any]) -> Any:
    """Create an empty fixed-template environment with the ramp adversary."""
    MetaDriveEnv, TrafficMode, _, _ = import_metadrive()
    theta = case["theta"]
    dual_view = bool(config.get("dual_view", False))
    env_cfg = {
        "map": str(config["metadrive_map"]),
        "start_seed": int(config.get("engine_seed", 0)),
        "num_scenarios": 1,
        "horizon": int(config["horizon"]),
        # Roles and optional background traffic are created explicitly below.
        "traffic_density": 0.0,
        "traffic_mode": TrafficMode.Basic,
        "random_traffic": False,
        # ``dual_view`` uses an off-screen main camera; OpenCV presents it
        # beside an off-screen top-down frame in ``visualize.py``.
        "use_render": bool(config.get("use_render", False)) and not dual_view,
        "image_observation": dual_view,
        "window_size": (640, 360),
        "interface_panel": [],
        "sensors": {"main_camera": ()} if dual_view else {},
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "out_of_road_done": False,
        "on_continuous_line_done": False,
        "agent_configs": {
            "default_agent": {
                "spawn_lane_index": RAMP_LANE,
                "spawn_longitude": float(theta["adversary_ramp_position_m"]),
                "spawn_lateral": 0.0,
                "use_special_color": False,
            }
        },
        "vehicle_config": {
            "enable_reverse": False,
            "image_source": "main_camera" if dual_view else "lidar",
        },
    }
    return MetaDriveEnv(env_cfg)


def configure_adversary_spawn(env: Any, case: Mapping[str, Any]) -> None:
    """Update only the fixed-template adversary spawn before a new reset."""
    env.engine.global_config["agent_configs"]["default_agent"].update({
        "spawn_lane_index": RAMP_LANE,
        "spawn_longitude": float(case["theta"]["adversary_ramp_position_m"]),
        "spawn_lateral": 0.0,
    })


def reset_env(env: Any, seed: int) -> tuple[Any, dict]:
    try:
        return env.reset(seed=int(seed))
    except TypeError:
        return env.reset(force_seed=int(seed))


def adversary(env: Any) -> Any:
    vehicle = getattr(env, "agent", None)
    if vehicle is None:
        vehicle = env.agents.get("default_agent")
    if vehicle is None:
        raise RuntimeError("MetaDrive default agent (adversary) unavailable after reset")
    return vehicle


def _lane(env: Any, index: tuple[str, str, int]) -> Any:
    try:
        return env.current_map.road_network.get_lane(index)
    except Exception as exc:
        raise RuntimeError(f"SrS lane is unavailable: {index!r}") from exc


def _spawn_idm(env: Any, lane_index: tuple[str, str, int], longitudinal: float,
               speed_mps: float, policy_seed: int) -> Any:
    _, _, TrafficDefaultVehicle, IDMPolicy = import_metadrive()
    manager = env.engine.traffic_manager
    vehicle = manager.spawn_object(
        TrafficDefaultVehicle,
        vehicle_config={
            "spawn_lane_index": lane_index,
            "spawn_longitude": float(longitudinal),
            "spawn_lateral": 0.0,
            "enable_reverse": False,
        },
    )
    manager.add_policy(vehicle.id, IDMPolicy, vehicle, int(policy_seed))
    # MetaDrive exposes ``traffic_vehicles`` as a copy.  The manager must own
    # the vehicle in its internal list so that it invokes the IDM policy each
    # physics step.
    manager._traffic_vehicles.append(vehicle)
    vehicle.set_velocity(np.asarray(vehicle.heading, dtype=float) * float(speed_mps))
    # IDM's internal units are km/h while the logical case uses m/s.
    env.engine.get_policy(vehicle.id).target_speed = float(speed_mps) * 3.6
    return vehicle


def establish_case_roles(env: Any, case: Mapping[str, Any]) -> tuple[Any, Any]:
    """Place and initialize the fixed mainline SUT and optional background."""
    theta = case["theta"]
    adv = adversary(env)
    ramp_lane, main_lane = _lane(env, RAMP_LANE), _lane(env, MAINLINE_LANE)
    _lane(env, BACKGROUND_LANE)
    adv_long = float(theta["adversary_ramp_position_m"])
    adv_x = float(ramp_lane.position(adv_long, 0.0)[0])
    main_long = adv_x + float(theta["longitudinal_gap_m"]) - float(main_lane.position(0.0, 0.0)[0])
    if not 0.0 <= main_long <= main_lane.length:
        raise ValueError(f"case {case['case_id']} places the SUT outside its mainline lane")
    adv.set_velocity(np.asarray(adv.heading, dtype=float) * float(theta["adversary_speed_mps"]))
    set_role_appearance(adv, ADVERSARY_COLOR, force_paint=True)
    sut = _spawn_idm(env, MAINLINE_LANE, main_long, float(theta["sut_speed_mps"]), int(case["background_seed"]) + 1)
    set_role_appearance(sut, SUT_COLOR)

    rng = np.random.default_rng(int(case["background_seed"]))
    if rng.random() < min(1.0, float(theta["background_density"]) / 0.08):
        background_lane = _lane(env, BACKGROUND_LANE)
        # One independently seeded car in the adjacent lane creates motion
        # without obscuring the two semantic roles at the merge conflict zone.
        bg_long = float(rng.uniform(4.0, max(5.0, background_lane.length - 8.0)))
        bg_speed = float(rng.uniform(8.0, 16.0))
        _spawn_idm(env, BACKGROUND_LANE, bg_long, bg_speed,
                   int(case["background_seed"]) + 2)
    return adv, sut


def traffic_vehicles(env: Any) -> list[Any]:
    manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    vehicles = getattr(manager, "traffic_vehicles", None)
    return list(vehicles) if vehicles is not None else []


def vehicle_id(vehicle: Any) -> str:
    return str(getattr(vehicle, "id", getattr(vehicle, "name", "unknown")))


def state(vehicle: Any) -> dict[str, float]:
    pos, vel = np.asarray(vehicle.position, dtype=float), np.asarray(vehicle.velocity, dtype=float)
    return {
        "x": float(pos[0]), "y": float(pos[1]), "vx": float(vel[0]), "vy": float(vel[1]),
        "heading": float(getattr(vehicle, "heading_theta", 0.0)),
        "yaw_rate": float(getattr(vehicle, "angular_velocity", 0.0)),
    }


def out_of_road(env: Any, vehicle: Any) -> bool:
    checker = getattr(env, "_is_out_of_road", None)
    return bool(checker(vehicle)) if checker else bool(getattr(vehicle, "out_of_road", False))


def target_contact(adv: Any, sut: Any) -> bool:
    """Pairwise contact signal supported by MetaDrive 0.4.3."""
    return bool(getattr(adv, "crash_vehicle", False) and getattr(sut, "crash_vehicle", False))


def has_expected_roles(env: Any, adv: Any, sut: Any) -> bool:
    return tuple(adv.lane_index) == RAMP_LANE and tuple(sut.lane_index) == MAINLINE_LANE


def track_vehicle(env: Any, vehicle: Any) -> None:
    """Make MetaDrive's interactive chase camera follow ``vehicle``.

    This intentionally lives in the compatibility layer: the public scenario
    environment exposes roles, while MetaDrive owns the Panda3D camera API.
    """
    camera = getattr(getattr(env, "engine", None), "main_camera", None)
    if camera is None:
        raise RuntimeError("MetaDrive interactive camera is unavailable; enable use_render")
    camera.track(vehicle)


def camera_frame(env: Any) -> np.ndarray:
    """Return the current main-camera RGB frame for off-screen dual-view use."""
    try:
        camera = env.engine.get_sensor("main_camera")
    except (AttributeError, ValueError) as exc:
        raise RuntimeError("main-camera output is unavailable; enable dual_view") from exc
    # MetaDrive 0.4.3's off-screen main camera returns BGR despite the public
    # image-observation convention.  Normalize here so callers always get RGB.
    return np.asarray(camera.perceive(to_float=False))[..., ::-1].copy()


def set_role_appearance(vehicle: Any, color: tuple[float, float, float], force_paint: bool = False) -> None:
    """Set a role's semantic colour and, when needed, its visible car paint."""
    vehicle._panda_color = tuple(float(channel) for channel in color)
    if not getattr(vehicle, "render", False):
        return
    try:
        from panda3d.core import LVecBase4, Material

        material = Material()
        coefficient = float(getattr(vehicle, "MATERIAL_COLOR_COEFF", 1.0))
        material.setBaseColor(LVecBase4(*(min(1.0, channel * coefficient) for channel in color), 1.0))
        material.setMetallic(float(getattr(vehicle, "MATERIAL_METAL_COEFF", 0.0)))
        material.setSpecular(getattr(vehicle, "MATERIAL_SPECULAR_COLOR", (0.0, 0.0, 0.0, 1.0)))
        material.setRoughness(float(getattr(vehicle, "MATERIAL_ROUGHNESS", 0.5)))
        material.setShininess(float(getattr(vehicle, "MATERIAL_SHININESS", 0.0)))
        material.setTwoside(False)
        vehicle.origin.setMaterial(material, True)
        if force_paint:
            # The adversary's baked yellow texture otherwise dominates its
            # material in MetaDrive's off-screen renderer.  A high-priority
            # tint produces a stable red car without touching its dynamics.
            vehicle.origin.setColor(LVecBase4(1.0, 0.08, 0.06, 1.0), 1000)
    except ImportError:
        pass


def route_features(vehicle: Any) -> dict[str, float]:
    lane = getattr(vehicle, "lane", None)
    width = float(getattr(lane, "width", 0.0)) if lane is not None else 0.0
    lanes = float(getattr(getattr(vehicle, "lane_index", (None, None, 0)), "__getitem__", lambda _: 0)(2) + 1)
    return {
        "lane_width": width, "num_lanes": lanes, "curvature": 0.0,
        "route_dir_0": 1.0, "route_dir_1": 0.0, "route_dir_2": 0.0, "route_dir_3": 0.0,
    }
