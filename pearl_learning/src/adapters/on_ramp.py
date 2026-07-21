from __future__ import annotations
from typing import Any, Mapping
from .base import MetaDriveAdapterBase
from ..task_spec import LogicalScenarioTaskSpec


class OnRampMergeAdapter(MetaDriveAdapterBase):
    def build_env(self, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
        MetaDriveEnv, TrafficMode, _, _ = self._imports()
        return MetaDriveEnv({"map": task.map_config["map"], "start_seed": int(case["case_seed"]), "num_scenarios": 1,
            "horizon": int(config["environment"]["horizon"]), "traffic_density": 0.0, "traffic_mode": TrafficMode.Basic,
            "random_traffic": False, "use_render": bool(config["environment"].get("use_render", False)),
            "crash_vehicle_done": False, "crash_object_done": False, "out_of_road_done": False, "on_continuous_line_done": False})
