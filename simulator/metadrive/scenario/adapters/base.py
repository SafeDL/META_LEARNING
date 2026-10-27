"""Headless MetaDrive family adapter base with reset/config provenance."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..layout import LaneIndex, ScenarioLayout
from ..registry import load_geometry_catalog
from ..route_geometry import RoutePolyline
from ..task_spec import ScenarioMiningTaskSpec


class MetaDriveFamilyAdapter:
    family = ""
    SPEED_LIMITS_MPS = {"cutin": 20.0, "merge": 18.0, "roundabout": 12.0}
    SUT_NOMINAL_SPEEDS_MPS = {"cutin": 8.3, "merge": 8.3, "roundabout": 4.5}

    def env_config(
        self,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        layout: ScenarioLayout | None = None,
    ) -> dict[str, Any]:
        result = {
            **self.geometry_config(task),
            "traffic_density": 0.0,
            "random_traffic": False,
            "random_spawn_lane_index": False,
            "use_render": False,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "out_of_road_done": False,
        }
        if layout is not None:
            result["agent_configs"] = {"default_agent": self.adversary_agent_config(layout, config)}
        return result

    def build_env(
        self,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        layout: ScenarioLayout | None = None,
        environment_overrides: Mapping[str, Any] | None = None,
    ) -> Any:
        from metadrive.envs.metadrive_env import MetaDriveEnv

        if task.functional_scenario != self.family or task.adapter_id != self.family:
            raise ValueError(f"{self.family} adapter cannot execute {task.functional_scenario}")
        runtime_config = self.env_config(task, config, layout)
        if environment_overrides:
            runtime_config.update(environment_overrides)
        return MetaDriveEnv(runtime_config)

    @staticmethod
    def geometry_config(task: ScenarioMiningTaskSpec) -> dict[str, Any]:
        geometry = load_geometry_catalog()[task.geometry_id]
        if geometry.functional_scenario != task.functional_scenario:
            raise ValueError("task functional scenario and geometry do not match")
        if geometry.seed != task.geometry_seed:
            raise ValueError("task geometry seed does not match the geometry catalog")
        return geometry.env_overrides()

    @staticmethod
    def adversary_agent_config(
        layout: ScenarioLayout, config: Mapping[str, float | str]
    ) -> dict[str, Any]:
        return {
            "spawn_lane_index": config.get("adversary_spawn_lane", layout.adversary_lane),
            "spawn_longitude": float(config["adversary_spawn_m"]),
            "spawn_lateral": 0.0,
            "spawn_velocity": [float(config["adversary_initial_speed_mps"]), 0.0],
            "spawn_velocity_car_frame": True,
            "destination": layout.adversary_destination,
        }

    def reset(
        self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], seed: int
    ) -> tuple[Any, Mapping[str, Any]]:
        # MetaDrive's procedural maps use ``start_seed``/``num_scenarios`` at
        # construction time; passing an arbitrary reset seed is invalid when
        # only one scenario is configured.
        del seed
        observation, info = env.reset()
        info = dict(info)
        info["mvr_task_id"] = task.task_id
        info["mvr_config"] = dict(config)
        setattr(env, "_mvr_initial_config", dict(config))
        setattr(env, "_mvr_observation", observation)
        return observation, info

    def validate_runtime(
        self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str]
    ) -> None:
        if getattr(env, "_mvr_initial_config", None) != dict(config):
            raise RuntimeError("outer configuration was not recorded by simulator reset")
        if task.functional_scenario != self.family:
            raise RuntimeError("runtime scenario family mismatch")

    @staticmethod
    def spawn_from_conflict_distance(
        route: RoutePolyline,
        conflict_xy: tuple[float, float],
        distance_to_conflict_m: float,
    ) -> float:
        spawn = route.conflict_s(conflict_xy) - float(distance_to_conflict_m)
        if not 0.0 <= spawn <= route.length_m:
            raise ValueError("distance to conflict places a vehicle outside its route")
        return spawn

    @staticmethod
    def _lane_rows(road_network: Any, minimum_length: float) -> list[tuple[LaneIndex, Any]]:
        rows: list[tuple[LaneIndex, Any]] = []
        for start, ends in road_network.graph.items():
            for end, lanes in ends.items():
                for number, lane in enumerate(lanes):
                    if float(lane.length) > float(minimum_length):
                        rows.append(((start, end, number), lane))
        return sorted(rows, key=lambda row: tuple(map(str, row[0])))

    @staticmethod
    def _route_from(road_network: Any, lane_index: LaneIndex) -> tuple[tuple[LaneIndex, ...], Any]:
        """Follow the geometrically continuous successor of one concrete lane."""
        start, end, number = lane_index
        successors = road_network.graph.get(end, {})
        if not successors:
            return (lane_index,), end
        current_lane = road_network.get_lane(lane_index)
        current_heading = float(current_lane.heading_theta_at(float(current_lane.length)))
        current_end = np.asarray(
            current_lane.position(float(current_lane.length), 0.0), dtype=float
        )

        def continuity(row: tuple[Any, Any]) -> tuple[float, float, str]:
            next_end, next_lanes = row
            next_lane = next_lanes[min(int(number), len(next_lanes) - 1)]
            next_heading = float(next_lane.heading_theta_at(0.0))
            alignment = float(np.cos(next_heading - current_heading))
            gap = float(
                np.linalg.norm(np.asarray(next_lane.position(0.0, 0.0), dtype=float) - current_end)
            )
            return alignment, -gap, str(next_end)

        next_end, next_lanes = max(successors.items(), key=continuity)
        next_index = (end, next_end, min(int(number), len(next_lanes) - 1))
        return (lane_index, next_index), next_end
