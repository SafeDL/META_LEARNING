from __future__ import annotations

from typing import Any, Mapping

from .base import MetaDriveFamilyAdapter
from ..layout import ScenarioLayout
from ..task_spec import MetaTestTaskSpec


class RoundaboutScenarioAdapter(MetaDriveFamilyAdapter):
    """MetaDrive's native ``Roundabout`` PG block (`O`) scenario fixture."""
    family = "roundabout"

    def env_config(self, task: MetaTestTaskSpec, config: Mapping[str, float | str], layout: ScenarioLayout | None = None) -> dict[str, Any]:
        result = {
            "map": "O", "start_seed": int(task.seed),
            "num_scenarios": 1, "horizon": 240, "traffic_density": 0.0, "random_traffic": False,
            "random_spawn_lane_index": False, "use_render": False, "crash_vehicle_done": False,
            "crash_object_done": False, "out_of_road_done": False,
        }
        if layout is not None:
            result["agent_configs"] = {"default_agent": self.adversary_agent_config(layout, config)}
        return result
