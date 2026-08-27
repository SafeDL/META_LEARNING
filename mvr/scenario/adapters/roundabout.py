from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from .base import MetaDriveFamilyAdapter
from ..layout import LaneIndex, ScenarioLayout, TrafficBehaviorContract
from ..task_spec import ScenarioMiningTaskSpec


class RoundaboutScenarioAdapter(MetaDriveFamilyAdapter):
    """MetaDrive's native ``Roundabout`` PG block (`O`) scenario fixture."""
    family = "roundabout"

    @staticmethod
    def _candidate_entry_exit(candidate: str) -> tuple[int, int]:
        match = re.fullmatch(r"entry_([0-2])_exit_([0-2])", candidate)
        if match is None:
            raise ValueError(f"invalid roundabout candidate {candidate!r}")
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _nodes_for_route(entry: int, exit_: int) -> tuple[str, ...]:
        if entry == 0:
            nodes = [">>>", "1O0_0_", "1O0_1_"]
        else:
            previous = entry - 1
            nodes = [
                f"-1O{previous}_3_",
                f"-1O{previous}_2_",
                f"1O{entry}_0_",
                f"1O{entry}_1_",
            ]
        current = entry
        while current != exit_:
            current = (current + 1) % 4
            nodes.extend((f"1O{current}_0_", f"1O{current}_1_"))
        nodes.extend((f"1O{exit_}_2_", f"1O{exit_}_3_"))
        return tuple(nodes)

    @classmethod
    def _route(
        cls,
        road_network: Any,
        entry: int,
        exit_: int,
        lane_number: int,
    ) -> tuple[LaneIndex, ...]:
        nodes = cls._nodes_for_route(entry, exit_)
        route: list[LaneIndex] = []
        for start, end in zip(nodes, nodes[1:]):
            lanes = road_network.graph.get(start, {}).get(end)
            if lanes is None or lane_number >= len(lanes):
                raise RuntimeError(
                    f"roundabout route {entry}->{exit_} is absent from the generated map"
                )
            route.append((start, end, lane_number))
        return tuple(route)

    def resolve_layout(
        self,
        env: Any,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        candidates: tuple[str, ...],
    ) -> ScenarioLayout:
        del task, candidates
        candidate = str(config["route_or_conflict_candidate"])
        sut_entry, sut_exit = self._candidate_entry_exit(candidate)
        adversary_entry, adversary_exit = sut_exit, (sut_exit + 1) % 3
        road_network = env.current_map.road_network
        sut_route = self._route(road_network, sut_entry, sut_exit, lane_number=0)
        adversary_route = self._route(road_network, adversary_entry, adversary_exit, lane_number=0)
        shared = [lane_index for lane_index in adversary_route if lane_index in sut_route]
        if not shared:
            raise RuntimeError(f"roundabout candidate {candidate!r} has no shared conflict lane")
        conflict_lane = road_network.get_lane(shared[0])
        conflict = np.asarray(conflict_lane.position(0.0, 0.0), dtype=float)
        return ScenarioLayout(
            candidate=candidate,
            conflict_zone_id=f"roundabout:{candidate}:{adversary_entry}->{adversary_exit}",
            adversary_lane=adversary_route[0],
            sut_lane=sut_route[0],
            adversary_destination=adversary_route[-1][1],
            sut_destination=sut_route[-1][1],
            adversary_route=adversary_route,
            sut_route=sut_route,
            conflict_xy=(float(conflict[0]), float(conflict[1])),
            traffic_contract=TrafficBehaviorContract(
                self.SPEED_LIMITS_MPS[self.family],
                self.SUT_NOMINAL_SPEEDS_MPS[self.family],
                (0,),
                0,
                adversary_intent="roundabout_conflict_entry",
                sut_role="lane_stable_roundabout_route",
                min_completion_steps=600,
            ),
        )

    def env_config(self, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], layout: ScenarioLayout | None = None) -> dict[str, Any]:
        result = {
            **self.geometry_config(task),
            "traffic_density": 0.0, "random_traffic": False,
            "random_spawn_lane_index": False, "use_render": False, "crash_vehicle_done": False,
            "crash_object_done": False, "out_of_road_done": False,
        }
        if layout is not None:
            result["agent_configs"] = {"default_agent": self.adversary_agent_config(layout, config)}
        return result
