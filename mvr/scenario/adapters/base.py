"""Headless MetaDrive family adapter base with reset/config provenance."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from ..layout import LaneIndex, ScenarioLayout, TrafficBehaviorContract
from ..registry import load_geometry_catalog
from ..route_geometry import RoutePolyline
from ..task_spec import ScenarioMiningTaskSpec


class MetaDriveFamilyAdapter:
    family = ""
    SPEED_LIMITS_MPS = {"cutin": 20.0, "merge": 18.0, "roundabout": 12.0}
    SUT_NOMINAL_SPEEDS_MPS = {"cutin": 8.3, "merge": 8.3, "roundabout": 4.5}

    def env_config(
        self,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        layout: ScenarioLayout | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def build_env(
        self,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        layout: ScenarioLayout | None = None,
        environment_overrides: Mapping[str, Any] | None = None,
    ) -> Any:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        if task.functional_scenario != self.family or task.adapter_id != self.family:
            raise ValueError(f"{self.family} adapter cannot execute {task.functional_scenario}")
        runtime_config = self.env_config(task, config, layout)
        if environment_overrides:
            runtime_config.update(environment_overrides)
        return MetaDriveEnv(runtime_config)

    @staticmethod
    def geometry_config(task: ScenarioMiningTaskSpec) -> dict[str, Any]:
        geometry = load_geometry_catalog()[task.geometry_id]
        if geometry.functional_scenario != task.functional_scenario:
            raise ValueError("task functional scenario and geometry do not match")
        if geometry.seed != task.geometry_seed:
            raise ValueError("task geometry seed does not match the geometry catalog")
        return geometry.env_overrides()

    @staticmethod
    def adversary_agent_config(layout: ScenarioLayout, config: Mapping[str, float | str]) -> dict[str, Any]:
        return {
            "spawn_lane_index": layout.adversary_lane,
            "spawn_longitude": float(config["adversary_spawn_m"]),
            "spawn_lateral": 0.0,
            "spawn_velocity": [float(config["adversary_initial_speed_mps"]), 0.0],
            "spawn_velocity_car_frame": True,
            "destination": layout.adversary_destination,
        }

    def reset(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], seed: int) -> tuple[Any, Mapping[str, Any]]:
        # MetaDrive's procedural maps use ``start_seed``/``num_scenarios`` at
        # construction time; passing an arbitrary reset seed is invalid when
        # only one scenario is configured.
        del seed
        observation, info = env.reset()
        info = dict(info)
        info["mvr_task_id"] = task.task_id
        info["mvr_config"] = dict(config)
        setattr(env, "_mvr_initial_config", dict(config))
        setattr(env, "_mvr_observation", observation)
        return observation, info

    def validate_runtime(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str]) -> None:
        if getattr(env, "_mvr_initial_config", None) != dict(config):
            raise RuntimeError("outer configuration was not recorded by simulator reset")
        if task.functional_scenario != self.family:
            raise RuntimeError("runtime scenario family mismatch")

    @staticmethod
    def spawn_from_conflict_distance(
        route: RoutePolyline,
        conflict_xy: tuple[float, float],
        distance_to_conflict_m: float,
    ) -> float:
        spawn = route.conflict_s(conflict_xy) - float(distance_to_conflict_m)
        if not 0.0 <= spawn <= route.length_m:
            raise ValueError("distance to conflict places a vehicle outside its route")
        return spawn

    @staticmethod
    def _lane_rows(road_network: Any, minimum_length: float) -> list[tuple[LaneIndex, Any]]:
        rows: list[tuple[LaneIndex, Any]] = []
        for start, ends in road_network.graph.items():
            for end, lanes in ends.items():
                for number, lane in enumerate(lanes):
                    if float(lane.length) > float(minimum_length):
                        rows.append(((start, end, number), lane))
        return sorted(rows, key=lambda row: tuple(map(str, row[0])))

    @staticmethod
    def _midpoint(lane: Any) -> tuple[float, float]:
        point = np.asarray(lane.position(0.5 * float(lane.length), 0.0), dtype=float)
        return float(point[0]), float(point[1])

    @staticmethod
    def _route_from(road_network: Any, lane_index: LaneIndex) -> tuple[tuple[LaneIndex, ...], Any]:
        """Follow the geometrically continuous successor of one concrete lane."""
        start, end, number = lane_index
        successors = road_network.graph.get(end, {})
        if not successors:
            return (lane_index,), end
        current_lane = road_network.get_lane(lane_index)
        current_heading = float(current_lane.heading_theta_at(float(current_lane.length)))
        current_end = np.asarray(current_lane.position(float(current_lane.length), 0.0), dtype=float)

        def continuity(row: tuple[Any, Any]) -> tuple[float, float, str]:
            next_end, next_lanes = row
            next_lane = next_lanes[min(int(number), len(next_lanes) - 1)]
            next_heading = float(next_lane.heading_theta_at(0.0))
            alignment = float(np.cos(next_heading - current_heading))
            gap = float(np.linalg.norm(
                np.asarray(next_lane.position(0.0, 0.0), dtype=float) - current_end
            ))
            return alignment, -gap, str(next_end)

        next_end, next_lanes = max(successors.items(), key=continuity)
        next_index = (end, next_end, min(int(number), len(next_lanes) - 1))
        return (lane_index, next_index), next_end

    def resolve_layout(
        self,
        env: Any,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        candidates: tuple[str, ...],
    ) -> ScenarioLayout:
        """Resolve candidate strings against the generated map, never lane order alone."""
        road_network = env.current_map.road_network
        rows = self._lane_rows(road_network, 2.0)
        if len(rows) < 2:
            raise RuntimeError("generated map has no two lanes long enough for the selected outer spawn")
        candidate = str(config["route_or_conflict_candidate"])
        rank = list(candidates).index(candidate)

        # Cut-in candidates are explicitly left/right members of one road.
        grouped: dict[tuple[Any, Any], list[tuple[LaneIndex, Any]]] = defaultdict(list)
        for index, lane in rows:
            grouped[index[:2]].append((index, lane))
        multi_lane_roads = [members for _, members in sorted(grouped.items(), key=lambda row: tuple(map(str, row[0]))) if len(members) >= 2]

        if self.family == "cutin" and multi_lane_roads:
            members = multi_lane_roads[rank % len(multi_lane_roads)]
            adversary_index, adversary_lane = members[0 if candidate.startswith("left") else -1]
            sut_index, sut_lane = members[-1 if candidate.startswith("left") else 0]
        else:
            adversary_index, adversary_lane = rows[(2 * rank) % len(rows)]
            sut_index, sut_lane = rows[(2 * rank + 1) % len(rows)]

        adversary_route, adversary_destination = self._route_from(road_network, adversary_index)
        sut_route, sut_destination = self._route_from(road_network, sut_index)
        shared = [lane_index for lane_index in adversary_route if lane_index in sut_route]
        if shared:
            conflict = np.asarray(road_network.get_lane(shared[0]).position(0.0, 0.0), dtype=float)
        elif self.family == "merge" and adversary_index[1] == sut_index[1]:
            # The incoming lanes meet at this node, even when their natural
            # outgoing branches differ.  Using the branch midpoint here
            # creates a fictitious conflict and forces the SUT to turn away
            # from its physically continuous route.
            conflict = np.asarray(
                adversary_lane.position(float(adversary_lane.length), 0.0), dtype=float
            )
        else:
            conflict = 0.5 * (np.asarray(self._midpoint(adversary_lane)) + np.asarray(self._midpoint(sut_lane)))
        return ScenarioLayout(
            candidate=candidate,
            conflict_zone_id=f"{self.family}:{adversary_index!s}|{sut_index!s}",
            adversary_lane=adversary_index,
            sut_lane=sut_index,
            adversary_destination=adversary_destination,
            sut_destination=sut_destination,
            adversary_route=adversary_route,
            sut_route=sut_route,
            conflict_xy=(float(conflict[0]), float(conflict[1])),
            traffic_contract=TrafficBehaviorContract(
                self.SPEED_LIMITS_MPS[self.family],
                self.SUT_NOMINAL_SPEEDS_MPS[self.family],
                tuple(sorted({lane[2] for lane in adversary_route})),
                adversary_route[0][2],
                adversary_intent="route_follow",
                sut_role="route_following",
                min_completion_steps=240,
            ),
        )
