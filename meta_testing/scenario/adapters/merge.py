from __future__ import annotations

from typing import Any, Mapping

from .base import MetaDriveFamilyAdapter
from ..task_spec import MetaTestTaskSpec


class MergeScenarioAdapter(MetaDriveFamilyAdapter):
    family = "merge"

    def env_config(self, task: MetaTestTaskSpec, config: Mapping[str, float |str]) -> dict[str, Any]:
        return {
            "map": "SrS", "start_seed": int(task.seed), "num_scenarios": 1, "horizon": 180,
            "traffic_density": 0.0, "random_traffic": False, "random_spawn_lane_index": False,
            "use_render": False, "crash_vehicle_done": False, "crash_object_done": False,
            "out_of_road_done": False, "on_continuous_line_done": False,
        }
