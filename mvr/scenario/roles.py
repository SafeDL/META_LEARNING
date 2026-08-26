"""Explicit adversary/SUT spawning extracted from the legacy IDM coupling."""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from ..sut.base import ControllerProfile, SUTAdapter


def spawn_sut(
    env: Any,
    *,
    lane_index: tuple[Any, Any, int],
    longitudinal_m: float,
    speed_mps: float,
    destination: Any,
    route: Sequence[tuple[Any, Any, int]],
    adapter: SUTAdapter,
    profile: ControllerProfile,
    seed: int,
) -> Any:
    from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle

    lane = env.current_map.road_network.get_lane(lane_index)
    if not 0.0 <= float(longitudinal_m) <= float(lane.length):
        raise ValueError(f"SUT spawn {longitudinal_m:.2f} m is outside lane {lane_index!r}")
    manager = env.engine.traffic_manager
    vehicle = manager.spawn_object(TrafficDefaultVehicle, vehicle_config={
        "spawn_lane_index": lane_index, "spawn_longitude": float(longitudinal_m), "spawn_lateral": 0.0,
        "destination": destination, "enable_reverse": False,
    })
    manager._traffic_vehicles.append(vehicle)
    adapter.attach(env, vehicle, profile, int(seed), route)
    vehicle.set_velocity(np.asarray(vehicle.heading, dtype=float) * float(speed_mps))
    return vehicle
