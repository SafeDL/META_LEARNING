"""Headless MetaDrive family adapter base with reset/config provenance."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from ..layout import LaneIndex, ScenarioLayout
from ..task_spec import MetaTestTaskSpec


class MetaDriveFamilyAdapter:
    family = ""

    def env_config(
        self,
        task: MetaTestTaskSpec,
        config: Mapping[str, float | str],
        layout: ScenarioLayout | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def build_env(self, task: MetaTestTaskSpec, config: Mapping[str, float | str], layout: ScenarioLayout | None = None) -> Any:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        if task.scenario_family != self.family:
            raise ValueError(f"{self.family} adapter cannot execute {task.scenario_family}")
        return MetaDriveEnv(self.env_config(task, config, layout))

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

    def reset(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str], seed: int) -> tuple[Any, Mapping[str, Any]]:
        # MetaDrive's procedural maps use ``start_seed``/``num_scenarios`` at
        # construction time; passing an arbitrary reset seed is invalid when
        # only one scenario is configured.
        del seed
        observation, info = env.reset()
        info = dict(info)
        info["meta_testing_task_id"] = task.task_id
        info["meta_testing_config"] = dict(config)
        setattr(env, "_meta_testing_initial_config", dict(config))
        setattr(env, "_meta_testing_observation", observation)
        return observation, info

    def validate_runtime(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str]) -> None:
        if getattr(env, "_meta_testing_initial_config", None) != dict(config):
            raise RuntimeError("outer configuration was not recorded by simulator reset")
        if task.scenario_family != self.family:
            raise RuntimeError("runtime scenario family mismatch")

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
        """Take one concrete successor road so a candidate changes navigation, too."""
        start, end, number = lane_index
        successors = road_network.graph.get(end, {})
        if not successors:
            return (lane_index,), end
        next_end, next_lanes = sorted(successors.items(), key=lambda row: str(row[0]))[0]
        next_index = (end, next_end, min(int(number), len(next_lanes) - 1))
        return (lane_index, next_index), next_end

    def resolve_layout(
        self,
        env: Any,
        task: MetaTestTaskSpec,
        config: Mapping[str, float | str],
        candidates: tuple[str, ...],
    ) -> ScenarioLayout:
        """Resolve candidate strings against the generated map, never lane order alone."""
        road_network = env.current_map.road_network
        required = max(float(config["adversary_spawn_m"]), float(config["sut_spawn_m"]))
        rows = self._lane_rows(road_network, required)
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
        elif self.family == "merge":
            incoming: dict[Any, list[tuple[LaneIndex, Any]]] = defaultdict(list)
            for index, lane in rows:
                incoming[index[1]].append((index, lane))
            merge_groups = [members for members in incoming.values() if len({index[0] for index, _ in members}) >= 2]
            if merge_groups:
                members = sorted(merge_groups[rank % len(merge_groups)], key=lambda row: tuple(map(str, row[0])))
                adversary_index, adversary_lane = members[0]
                sut_index, sut_lane = members[-1]
            else:
                adversary_index, adversary_lane = rows[(2 * rank) % len(rows)]
                sut_index, sut_lane = rows[(2 * rank + 1) % len(rows)]
        else:
            adversary_index, adversary_lane = rows[(2 * rank) % len(rows)]
            sut_index, sut_lane = rows[(2 * rank + 1) % len(rows)]

        midpoint = 0.5 * (np.asarray(self._midpoint(adversary_lane)) + np.asarray(self._midpoint(sut_lane)))
        adversary_route, adversary_destination = self._route_from(road_network, adversary_index)
        sut_route, sut_destination = self._route_from(road_network, sut_index)
        return ScenarioLayout(
            candidate=candidate,
            adversary_lane=adversary_index,
            sut_lane=sut_index,
            adversary_destination=adversary_destination,
            sut_destination=sut_destination,
            adversary_route=adversary_route,
            sut_route=sut_route,
            conflict_xy=(float(midpoint[0]), float(midpoint[1])),
        )
