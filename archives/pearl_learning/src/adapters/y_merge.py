"""Independent Y-merge adapter backed by a two-arm real PGMap Merge block."""
from __future__ import annotations
from typing import Any, Mapping

from .base import MetaDriveAdapterBase
from ..maps import y_merge_env_class
from ..routes import lane_index
from ..task_spec import LogicalScenarioTaskSpec


class YMergeAdapter(MetaDriveAdapterBase):
    def build_env(self, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
        Env = y_merge_env_class()
        recipe = task.map_config
        spawn = self._case_spawn(case, "adversary", task.spawn_regions["adversary"])
        return Env({
            "start_seed": int(case["case_seed"]), "num_scenarios": 1, "horizon": int(config["environment"]["horizon"]),
            "traffic_density": 0.0, "random_traffic": False, "random_spawn_lane_index": False, "use_render": bool(config["environment"].get("use_render", False)),
            "crash_vehicle_done": False, "crash_object_done": False, "out_of_road_done": False, "on_continuous_line_done": False,
            "agent_configs": {"default_agent": {"spawn_lane_index": lane_index(task.adversary_route["initial_lane"]), "spawn_longitude": spawn, "spawn_lateral": 0.0, "enable_reverse": False}},
            "map_config": {"exit_length": 60, "bottle_lane_num": int(recipe["bottle_lane_num"]), "neck_lane_num": int(recipe["neck_lane_num"]), "neck_length": float(recipe["merge_length_m"]), "lane_num": int(recipe["bottle_lane_num"])},
        })
