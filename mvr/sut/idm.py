"""IDM implementation behind the new black-box SUT boundary."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np
from metadrive.component.road_network import Road
from metadrive.policy.idm_policy import IDMPolicy

from .base import ControllerProfile


class RouteBoundIDMPolicy(IDMPolicy):
    """Bind MetaDrive IDM to the exact route resolved by a scenario adapter."""

    def __init__(
        self,
        control_object: Any,
        random_seed: int,
        route: Sequence[tuple[Any, Any, int]],
    ) -> None:
        super().__init__(control_object, random_seed)
        lane_sequence = tuple(route)
        if not lane_sequence:
            raise ValueError("IDM route must not be empty")
        if any(
            current[1] != following[0]
            for current, following in zip(lane_sequence, lane_sequence[1:])
        ):
            raise ValueError("IDM route lanes must be contiguous")
        navigation = self.control_object.navigation
        graph = navigation.map.road_network.graph
        checkpoints = [lane_sequence[0][0], *(lane[1] for lane in lane_sequence)]
        if any(
            end not in graph.get(start, {})
            for start, end in zip(checkpoints, checkpoints[1:])
        ):
            raise ValueError("IDM route is absent from the runtime road network")
        navigation.checkpoints = checkpoints
        navigation._target_checkpoints_index = [0, 1]
        navigation.current_ref_lanes = graph[checkpoints[0]][checkpoints[1]]
        navigation.next_ref_lanes = (
            graph[checkpoints[1]][checkpoints[2]] if len(checkpoints) > 2 else None
        )
        navigation.current_road = Road(checkpoints[0], checkpoints[1])
        navigation.next_road = (
            Road(checkpoints[1], checkpoints[2]) if len(checkpoints) > 2 else None
        )
        navigation.final_road = Road(checkpoints[-2], checkpoints[-1])
        navigation.final_lane = graph[checkpoints[-2]][checkpoints[-1]][lane_sequence[-1][2]]
        navigation.total_length = sum(
            float(graph[start][end][lane[2]].length)
            for start, end, lane in zip(checkpoints, checkpoints[1:], lane_sequence)
        )
        navigation.travelled_length = 0.0
        navigation._last_long_in_ref_lane = 0.0


class IDMSUTAdapter:
    name = "idm"

    def reset(self, env: Any, task: Any, config: Mapping[str, Any], seed: int) -> None:
        del env, task, config, seed

    def attach(
        self,
        env: Any,
        vehicle: Any,
        profile: ControllerProfile,
        seed: int,
        route: Sequence[tuple[Any, Any, int]],
    ) -> Any:
        env.engine.traffic_manager.add_policy(
            vehicle.id,
            RouteBoundIDMPolicy,
            vehicle,
            int(seed),
            tuple(route),
        )
        policy = env.engine.get_policy(vehicle.id)
        target_speed_kmh = float(profile.target_speed_mps) * 3.6
        # IDM restores target_speed from NORMAL_SPEED while lane following.
        policy.NORMAL_SPEED = target_speed_kmh
        policy.target_speed = target_speed_kmh
        # The route is part of the executable SUT contract.  Letting IDM
        # choose an unrelated lane change makes the SUT leave that contract
        # and produces the visible lateral oscillation in replays.  A route
        # change must be supplied by the scenario adapter, not invented by
        # the black-box SUT controller.
        policy.enable_lane_change = False
        policy.DISTANCE_WANTED = float(profile.distance_wanted_m)
        policy.TIME_WANTED = float(profile.time_headway_s)
        policy.ACC_FACTOR = float(profile.acceleration_factor)
        policy.DEACC_FACTOR = float(profile.deceleration_factor)
        return policy

    def observe(self, env: Any, vehicle: Any) -> Mapping[str, Any]:
        return {
            "position": np.asarray(vehicle.position, dtype=np.float32).copy(),
            "velocity": np.asarray(vehicle.velocity, dtype=np.float32).copy(),
            "speed_mps": float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6,
            "episode_step": int(getattr(env, "episode_step", 0)),
        }

    def step(self, observation: Mapping[str, Any]) -> Any:
        # MetaDrive invokes the attached traffic policy.
        return observation

    def metadata(self, profile: ControllerProfile) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "profile": profile.profile_id,
            "model_input_fields": (),
            "profile_is_model_input": False,
        }
