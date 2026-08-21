from __future__ import annotations

from typing import Any, Mapping

from .base import MetaDriveFamilyAdapter
from ..task_spec import MetaTestTaskSpec


class CutInScenarioAdapter(MetaDriveFamilyAdapter):
    """Procedural multi-lane straight-road Cut-in fixture.

    The outer action selects the target lane candidate and initial gap; role
    spawning is performed by the hierarchical runner after this deterministic
    map reset.
    """
    family = "cutin"

    def env_config(self, task: MetaTestTaskSpec, config: Mapping[str, float | str]) -> dict[str, Any]:
        return {
            "map": "S",
            "start_seed": int(task.seed), "num_scenarios": 1, "horizon": 180, "traffic_density": 0.0,
            "random_traffic": False, "random_spawn_lane_index": False, "use_render": False,
            "crash_vehicle_done": False, "crash_object_done": False, "out_of_road_done": False,
        }
