"""Explicit legal cut-in layouts on a shared multi-lane corridor."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import MetaDriveFamilyAdapter
from ..layout import LaneIndex, ScenarioLayout, TrafficBehaviorContract
from ..task_spec import ScenarioMiningTaskSpec


class CutInScenarioAdapter(MetaDriveFamilyAdapter):
    """Three-lane cut-in fixture with a legal adjacent-lane merge window."""

    family = "cutin"

    @staticmethod
    def _corridor(road_network: Any) -> tuple[LaneIndex, ...]:
        """Choose one long, unbranched three-lane direction of travel."""
        candidates: list[tuple[float, tuple[LaneIndex, ...]]] = []
        for start, ends in road_network.graph.items():
            for end, lanes in ends.items():
                if len(lanes) < 3 or float(lanes[0].length) < 40.0:
                    continue
                route: list[LaneIndex] = [(start, end, 1)]
                node = end
                while node in road_network.graph:
                    successors = [
                        (next_node, next_lanes)
                        for next_node, next_lanes in road_network.graph[node].items()
                        if len(next_lanes) >= 3
                    ]
                    if len(successors) != 1:
                        break
                    next_node, _ = successors[0]
                    route.append((node, next_node, 1))
                    node = next_node
                length = sum(
                    float(road_network.get_lane(lane_index).length)
                    for lane_index in route
                )
                if length >= 200.0:
                    candidates.append((length, tuple(route)))
        if not candidates:
            raise RuntimeError(
                "cut-in geometry has no continuous three-lane corridor of at least 200 m"
            )
        return max(candidates, key=lambda row: (row[0], tuple(map(str, row[1]))))[1]

    @staticmethod
    def _boundary_is_broken(lane: Any, source_lane: int, target_lane: int) -> bool:
        boundary_index = 1 if target_lane > source_lane else 0
        line_types = tuple(str(value) for value in getattr(lane, "line_types", ()))
        return len(line_types) == 2 and "BROKEN" in line_types[boundary_index]

    def _legal_merge_windows(
        self,
        road_network: Any,
        corridor: tuple[LaneIndex, ...],
        source_lane: int,
        target_lane: int,
    ) -> tuple[tuple[float, float], ...]:
        """Return every continuous dashed boundary interval in route metres."""
        windows: list[tuple[float, float]] = []
        offset = 0.0
        start: float | None = None
        for base in corridor:
            lane = road_network.get_lane((base[0], base[1], source_lane))
            length = float(lane.length)
            if self._boundary_is_broken(lane, source_lane, target_lane):
                if start is None:
                    start = offset
            elif start is not None:
                windows.append((start, offset))
                start = None
            offset += length
        if start is not None:
            windows.append((start, offset))
        return tuple(windows)

    def resolve_layout(
        self,
        env: Any,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        candidates: tuple[str, ...],
    ) -> ScenarioLayout:
        del task, candidates
        candidate = str(config["route_or_conflict_candidate"])
        try:
            source_lane, target_lane = {
                "left_target_lane": (0, 1),
                "right_target_lane": (2, 1),
            }[candidate]
        except KeyError as error:
            raise ValueError(f"invalid cut-in candidate {candidate!r}") from error
        road_network = env.current_map.road_network
        corridor = self._corridor(road_network)
        # The direct SAC steering action, rather than native navigation, is
        # the only mechanism that may cross from source to target lane.
        adversary_route = tuple((start, end, source_lane) for start, end, _ in corridor)
        sut_route = tuple((start, end, target_lane) for start, end, _ in corridor)
        windows = self._legal_merge_windows(
            road_network, corridor, source_lane, target_lane
        )
        valid_windows = [window for window in windows if window[1] - window[0] >= 60.0]
        if not valid_windows:
            raise RuntimeError(
                "cut-in geometry has no continuous 60 m dashed lane-change corridor"
            )
        merge_start, merge_end = max(valid_windows, key=lambda row: row[1] - row[0])
        cumulative = 0.0
        target_index = corridor[0]
        for lane_index in corridor:
            length = float(road_network.get_lane(lane_index).length)
            if cumulative <= 0.5 * (merge_start + merge_end) <= cumulative + length:
                target_index = lane_index
                break
            cumulative += length
        target_runtime_lane = road_network.get_lane(
            (target_index[0], target_index[1], target_lane)
        )
        source_runtime_lane = road_network.get_lane(
            (target_index[0], target_index[1], source_lane)
        )
        if not self._boundary_is_broken(source_runtime_lane, source_lane, target_lane):
            raise RuntimeError("cut-in source/target boundary is not a broken lane line")
        conflict_longitudinal = 0.5 * (merge_start + merge_end) - cumulative
        conflict = np.asarray(
            target_runtime_lane.position(conflict_longitudinal, 0.0), dtype=float
        )
        return ScenarioLayout(
            candidate=candidate,
            conflict_zone_id=f"cutin:{candidate}:merge_window",
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
                (source_lane, target_lane),
                source_lane,
                target_lane,
                (merge_start, merge_end),
                "broken",
                adversary_intent="cut_in_to_sut_lane",
                sut_role="lane_stable_main_corridor",
                min_completion_steps=300,
            ),
        )

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
            result["agent_configs"] = {
                "default_agent": {
                    **self.adversary_agent_config(layout, config),
                    # MetaDrive applies this force at all four wheels. These
                    # values bound full direct action at +3 / -6 m/s^2.
                    "max_engine_force": 825.0,
            # MetaDrive's ``max_brake_force`` is applied as a wheel brake
            # torque.  33.0 calibrates a full negative low-level action to
            # the Cut-in contract's 6 m/s² deceleration bound.
            "max_brake_force": 33.0,
                }
            }
        return result
