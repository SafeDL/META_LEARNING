"""Adapter protocol and shared MetaDrive role utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol
import numpy as np

from ..task_spec import LogicalScenarioTaskSpec


class LogicalScenarioAdapter(Protocol):
    def build_env(self, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> Any: ...
    def establish_roles(self, env: Any, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, Any]: ...
    def conflict_frame(self, env: Any, task: LogicalScenarioTaskSpec, adversary: Any, sut: Any) -> dict[str, Any]: ...
    def topology_features(self, env: Any, task: LogicalScenarioTaskSpec) -> dict[str, float]: ...
    def target_contact(self, env: Any, adversary: Any, sut: Any) -> tuple[bool, str]: ...
    def validate_episode_roles(self, env: Any, adversary: Any, sut: Any) -> None: ...


class MetaDriveAdapterBase(ABC):
    """All version-sensitive MetaDrive access is isolated here."""

    def _imports(self) -> tuple[Any, Any, Any, Any]:
        from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle
        from metadrive.envs.metadrive_env import MetaDriveEnv
        from metadrive.manager.traffic_manager import TrafficMode
        from metadrive.policy.idm_policy import IDMPolicy
        return MetaDriveEnv, TrafficMode, TrafficDefaultVehicle, IDMPolicy

    @abstractmethod
    def build_env(self, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    @staticmethod
    def _lanes(env: Any) -> list[tuple[tuple[Any, Any, int], Any]]:
        lanes = []
        for start, ends in env.current_map.road_network.graph.items():
            for end, group in ends.items():
                for index, lane in enumerate(group):
                    lanes.append(((start, end, index), lane))
        if not lanes:
            raise RuntimeError("topology audit found an empty lane graph")
        return lanes

    def _select_sut_lane(self, env: Any, adversary: Any) -> tuple[tuple[Any, Any, int], Any, float]:
        """Select another incoming route by geometry, never by a guessed lane ID."""
        adv_pos = np.asarray(adversary.position, dtype=float)
        candidates = []
        graph = env.current_map.road_network.graph
        for lane_index, lane in self._lanes(env):
            if tuple(lane_index) == tuple(adversary.lane_index):
                continue
            # IDM navigation needs at least one continuation after its spawn
            # road; terminal lanes can render but cannot host an IDM role.
            if lane_index[1] not in graph or not graph[lane_index[1]]:
                continue
            samples = np.linspace(0.0, float(lane.length), 25)
            positions = [np.asarray(lane.position(float(x), 0.0), dtype=float) for x in samples]
            distances = [float(np.linalg.norm(p - adv_pos)) for p in positions]
            eligible = [(distance, float(x)) for distance, x in zip(distances, samples) if distance >= 12.0]
            if eligible:
                distance, longitudinal = min(eligible, key=lambda pair: abs(pair[0] - 18.0))
                candidates.append((abs(distance - 18.0), lane_index, lane, longitudinal))
        if not candidates:
            raise RuntimeError("could not locate a separate SUT route at a valid separation")
        _, index, lane, longitudinal = min(candidates, key=lambda item: item[0])
        return index, lane, longitudinal

    def establish_roles(self, env: Any, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, Any]:
        _, _, TrafficDefaultVehicle, IDMPolicy = self._imports()
        adversary = getattr(env, "agent", None) or env.agents.get("default_agent")
        if adversary is None:
            raise RuntimeError("MetaDrive did not create the adversary agent")
        lane_index, lane, longitudinal = self._select_sut_lane(env, adversary)
        manager = env.engine.traffic_manager
        spawn_longitudinal = float(np.clip(longitudinal + float(case["arrival_offset_m"]), 2.0, max(2.0, float(lane.length) - 2.0)))
        sut = manager.spawn_object(TrafficDefaultVehicle, vehicle_config={
            "spawn_lane_index": lane_index, "spawn_longitude": spawn_longitudinal,
            "spawn_lateral": 0.0, "enable_reverse": False,
        })
        manager.add_policy(sut.id, IDMPolicy, sut, int(case["case_seed"]) + 1)
        manager._traffic_vehicles.append(sut)
        adversary.set_velocity(np.asarray(adversary.heading, dtype=float) * float(case["adversary_speed_mps"]))
        sut_speed_mps = float(config["sut"]["target_speed_mps"])
        sut.set_velocity(np.asarray(sut.heading, dtype=float) * sut_speed_mps)
        policy = env.engine.get_policy(sut.id)
        if hasattr(policy, "target_speed"):
            policy.target_speed = sut_speed_mps * 3.6
        if hasattr(policy, "enable_lane_change"):
            policy.enable_lane_change = bool(config["sut"]["enable_lane_change"])
        self.validate_episode_roles(env, adversary, sut)
        return adversary, sut

    @staticmethod
    def _navigation_lanes(vehicle: Any) -> list[Any]:
        """Return the roles' actual current/next route lanes, without labels."""
        navigation = getattr(vehicle, "navigation", None)
        candidates = [getattr(vehicle, "lane", None)]
        for name in ("current_ref_lanes", "next_ref_lanes"):
            candidates.extend(list(getattr(navigation, name, None) or []))
        lanes: list[Any] = []
        for lane in candidates:
            if lane is not None and all(lane is not known for known in lanes):
                lanes.append(lane)
        if not lanes:
            raise RuntimeError("could not recover a role route for conflict-frame construction")
        return lanes

    def conflict_frame(self, env: Any, task: LogicalScenarioTaskSpec, adversary: Any, sut: Any) -> dict[str, Any]:
        """Locate the closest geometric encounter on the two actual role routes.

        The old endpoint-average heuristic could move as unrelated map branches
        were added.  This construction only uses the lanes selected by the
        adversary and SUT navigations and finds their closest sampled encounter.
        """
        best: tuple[float, np.ndarray, float] | None = None
        for adv_lane in self._navigation_lanes(adversary):
            adv_s = np.linspace(0.0, float(adv_lane.length), 64)
            adv_points = np.asarray([adv_lane.position(float(s), 0.0) for s in adv_s], dtype=float)
            for sut_lane in self._navigation_lanes(sut):
                sut_s = np.linspace(0.0, float(sut_lane.length), 64)
                sut_points = np.asarray([sut_lane.position(float(s), 0.0) for s in sut_s], dtype=float)
                distances = np.linalg.norm(adv_points[:, None, :] - sut_points[None, :, :], axis=-1)
                row, col = np.unravel_index(int(np.argmin(distances)), distances.shape)
                distance = float(distances[row, col])
                origin = (adv_points[row] + sut_points[col]) / 2.0
                headings = np.asarray([adv_lane.heading_theta_at(float(adv_s[row])), sut_lane.heading_theta_at(float(sut_s[col]))])
                heading = float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))
                if best is None or distance < best[0]:
                    best = distance, origin, heading
        if best is None:
            raise RuntimeError("could not locate a role-route encounter for conflict frame")
        _, origin, heading = best
        return {"origin": origin, "heading": heading, "radius_m": float(task.conflict_spec["conflict_radius_m"])}

    def topology_features(self, env: Any, task: LogicalScenarioTaskSpec) -> dict[str, float]:
        lanes = self._lanes(env)
        graph = env.current_map.road_network.graph
        return {
            "num_incoming_branches": float(len(graph)),
            "num_outgoing_branches": float(sum(len(v) for v in graph.values())),
            "lane_count": float(max(len(v) for ends in graph.values() for v in ends.values())),
            "merge_length_m": float(task.conflict_spec["merge_length_m"]),
            "conflict_radius_m": float(task.conflict_spec["conflict_radius_m"]),
            "route_curvature": 0.0,
            "speed_limit_mps": 30.0,
            "num_conflict_zones": 1.0,
            "lane_graph_edges": float(len(lanes)),
        }

    def target_contact(self, env: Any, adversary: Any, sut: Any) -> tuple[bool, str]:
        # Prefer engine collision state; pair it with a geometric fallback so a
        # third-party collision cannot be credited as a target collision.
        both_crashed = bool(getattr(adversary, "crash_vehicle", False) and getattr(sut, "crash_vehicle", False))
        distance = float(np.linalg.norm(np.asarray(adversary.position) - np.asarray(sut.position)))
        threshold = 0.55 * (float(getattr(adversary, "LENGTH", 5.0)) + float(getattr(sut, "LENGTH", 5.0)))
        if both_crashed and distance <= threshold:
            return True, "crash_flags_plus_bbox"
        if both_crashed:
            return False, "crash_flags_rejected_by_bbox"
        return False, "no_pairwise_contact"

    def validate_episode_roles(self, env: Any, adversary: Any, sut: Any) -> None:
        if adversary is sut or str(adversary.id) == str(sut.id):
            raise RuntimeError("adversary and SUT must be distinct objects")
        distance = float(np.linalg.norm(np.asarray(adversary.position) - np.asarray(sut.position)))
        if distance < 10.0:
            raise RuntimeError(f"roles spawned too close: {distance:.2f} m")
        if tuple(adversary.lane_index) == tuple(sut.lane_index):
            raise RuntimeError("role routes are not distinct at reset")
