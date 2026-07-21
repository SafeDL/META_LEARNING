from __future__ import annotations
from typing import Any, Mapping
from .base import MetaDriveAdapterBase
from ..maps import bottleneck_env_class
from ..task_spec import LogicalScenarioTaskSpec


class BottleneckMergeAdapter(MetaDriveAdapterBase):
    def build_env(self, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
        Env = bottleneck_env_class()
        recipe = task.map_config
        return Env({"start_seed": int(case["case_seed"]), "num_scenarios": 1, "horizon": int(config["environment"]["horizon"]),
            "traffic_density": 0.0, "random_traffic": False, "use_render": bool(config["environment"].get("use_render", False)),
            "crash_vehicle_done": False, "crash_object_done": False, "out_of_road_done": False, "on_continuous_line_done": False,
            "map_config": {"exit_length": 60, "bottle_lane_num": int(recipe["bottle_lane_num"]), "neck_lane_num": int(recipe["neck_lane_num"]), "neck_length": float(recipe["merge_length_m"]), "lane_num": int(recipe["bottle_lane_num"])}})
