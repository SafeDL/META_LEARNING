from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from .base import MetaDriveFamilyAdapter
from ..layout import LaneIndex, ScenarioLayout, TrafficBehaviorContract
from ..task_spec import ScenarioMiningTaskSpec


class MergeScenarioAdapter(MetaDriveFamilyAdapter):
    """Merge fixture: the adversary enters from a branch into the SUT mainline.

    Candidate labels select a conflict reference on the same verified merge,
    never which vehicle is assigned the ramp.  The role assignment is thus
    stable throughout outer sampling, rollout, reward, and visualization.
    """

    family = "merge"

    def _branch_mainline_layout(
        self,
        road_network: Any,
        rank: int,
    ) -> tuple[LaneIndex, LaneIndex, LaneIndex]:
        """Return ramp, mainline, and their common downstream lane.

        MetaDrive's ``r`` block represents the physical ramp as a one-lane
        incoming road joining a multi-lane through road.  This topological
        distinction is more reliable than lexical road ordering or a small
        heading difference at a smooth merge.
        """
        incoming: dict[Any, list[tuple[LaneIndex, Any]]] = defaultdict(list)
        for index, lane in self._lane_rows(road_network, 8.0):
            incoming[index[1]].append((index, lane))
        layouts: list[tuple[LaneIndex, LaneIndex, LaneIndex]] = []
        for junction, members in incoming.items():
            distinct = [row for row in members if row[0][0] != junction]
            if len({index[0] for index, _ in distinct}) < 2:
                continue
            for downstream, lanes in road_network.graph.get(junction, {}).items():
                for lane_number, common_lane in enumerate(lanes):
                    common_index = (junction, downstream, lane_number)
                    del common_lane
                    by_source: dict[Any, list[LaneIndex]] = defaultdict(list)
                    for index, _ in distinct:
                        if int(index[2]) == lane_number:
                            by_source[index[0]].append(index)
                    if len(by_source) < 2:
                        continue
                    source_widths = {
                        source: sum(
                            1 for index, _ in distinct if index[0] == source
                        )
                        for source in by_source
                    }
                    ramp_source = min(source_widths, key=lambda source: (source_widths[source], str(source)))
                    main_source = max(source_widths, key=lambda source: (source_widths[source], str(source)))
                    if source_widths[ramp_source] != 1 or source_widths[main_source] < 2:
                        continue
                    layouts.append((
                        by_source[ramp_source][0],
                        by_source[main_source][0],
                        common_index,
                    ))
        if not layouts:
            raise RuntimeError(
                "merge geometry has no one-lane branch joining a multi-lane mainline"
            )
        layouts.sort(key=lambda row: tuple(map(str, row)))
        ramp, mainline, common = layouts[rank % len(layouts)]
        return ramp, mainline, common

    def resolve_layout(
        self,
        env: Any,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        candidates: tuple[str, ...],
    ) -> ScenarioLayout:
        del task
        candidate = str(config["route_or_conflict_candidate"])
        if candidate not in candidates:
            raise ValueError(f"invalid merge candidate {candidate!r}")
        road_network = env.current_map.road_network
        ramp, mainline, common = self._branch_mainline_layout(
            road_network, list(candidates).index(candidate)
        )
        adversary_route = (ramp, common)
        sut_route = (mainline, common)
        common_lane = road_network.get_lane(common)
        if candidate == "main_conflict":
            conflict_s = 0.0
        else:
            conflict_s = min(0.45 * float(common_lane.length), 24.0)
        conflict = np.asarray(common_lane.position(conflict_s, 0.0), dtype=float)
        return ScenarioLayout(
            candidate=candidate,
            conflict_zone_id=f"merge:{candidate}:{ramp[0]}->{common[1]}",
            adversary_lane=ramp,
            sut_lane=mainline,
            adversary_destination=common[1],
            sut_destination=common[1],
            adversary_route=adversary_route,
            sut_route=sut_route,
            conflict_xy=(float(conflict[0]), float(conflict[1])),
            traffic_contract=TrafficBehaviorContract(
                self.SPEED_LIMITS_MPS[self.family],
                self.SUT_NOMINAL_SPEEDS_MPS[self.family],
                tuple(sorted({lane[2] for lane in adversary_route})),
                ramp[2],
                adversary_intent="merge_from_branch",
                sut_role="lane_stable_mainline",
                min_completion_steps=180,
            ),
        )

    def env_config(self, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], layout: ScenarioLayout | None = None) -> dict[str, Any]:
        result = {
            **self.geometry_config(task),
            "traffic_density": 0.0, "random_traffic": False, "random_spawn_lane_index": False,
            "use_render": False, "crash_vehicle_done": False, "crash_object_done": False,
            "out_of_road_done": False, "on_continuous_line_done": False,
        }
        if layout is not None:
            result["agent_configs"] = {"default_agent": self.adversary_agent_config(layout, config)}
        return result
