"""Explicit SrS on-ramp adapter."""
from __future__ import annotations
from typing import Any, Mapping

from .base import MetaDriveAdapterBase
from ..routes import lane_index
from ..task_spec import LogicalScenarioTaskSpec


class OnRampMergeAdapter(MetaDriveAdapterBase):
    def build_env(self, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
        MetaDriveEnv, TrafficMode, _, _ = self._imports()
        adversary_lane = lane_index(task.adversary_route["initial_lane"])
        spawn = self._case_spawn(case, "adversary", task.spawn_regions["adversary"])
        return MetaDriveEnv({
            "map": task.map_config["map"], "start_seed": int(task.map_config.get("start_seed", 0)), "num_scenarios": 1,
            "horizon": int(config["environment"]["horizon"]), "traffic_density": 0.0, "traffic_mode": TrafficMode.Basic,
            "random_traffic": False, "random_spawn_lane_index": False, "use_render": bool(config["environment"].get("use_render", False)),
            "crash_vehicle_done": False, "crash_object_done": False, "out_of_road_done": False,
            "on_continuous_line_done": False,
            "agent_configs": {"default_agent": {"spawn_lane_index": adversary_lane, "spawn_longitude": spawn, "spawn_lateral": 0.0, "enable_reverse": False}},
        })
