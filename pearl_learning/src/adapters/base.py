"""Explicit-role MetaDrive adapters and pairwise collision utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol
import math
import numpy as np

from ..io import content_hash
from ..routes import RoutePolyline, lane_index
from ..task_spec import LogicalScenarioTaskSpec


class LogicalScenarioAdapter(Protocol):
    def build_env(self, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> Any: ...
    def establish_roles(self, env: Any, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, Any]: ...
    def conflict_frame(self, env: Any, task: LogicalScenarioTaskSpec, adversary: Any, sut: Any) -> dict[str, Any]: ...
    def topology_features(self, env: Any, task: LogicalScenarioTaskSpec) -> dict[str, float]: ...
    def target_contact(self, env: Any, adversary: Any, sut: Any) -> tuple[bool, str]: ...
    def route_status(self, env: Any, vehicle: Any, role: str, previous_s_m: float | None) -> tuple[float, bool, bool]: ...
    def map_hash(self, env: Any) -> str: ...
    def validate_episode_roles(self, env: Any, adversary: Any, sut: Any) -> None: ...


def _corners(vehicle: Any) -> np.ndarray:
    length, width = float(getattr(vehicle, "LENGTH", 5.0)), float(getattr(vehicle, "WIDTH", 2.0))
    heading = float(getattr(vehicle, "heading_theta", 0.0))
    forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
    lateral = np.asarray([-forward[1], forward[0]], dtype=float)
    centre = np.asarray(vehicle.position, dtype=float)
    return np.asarray([centre + sx * length / 2.0 * forward + sy * width / 2.0 * lateral for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def _obb_intersects(first: Any, second: Any) -> bool:
    a, b = _corners(first), _corners(second)
    for polygon in (a, b):
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            axis = np.asarray([-(end - start)[1], (end - start)[0]], dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            axis /= norm
            a_min, a_max = np.dot(a, axis).min(), np.dot(a, axis).max()
            b_min, b_max = np.dot(b, axis).min(), np.dot(b, axis).max()
            if a_max < b_min or b_max < a_min:
                return False
    return True


def _route_obb_intersects(
    first_center: np.ndarray,
    first_forward: np.ndarray,
    first_length: float,
    first_width: float,
    second_center: np.ndarray,
    second_forward: np.ndarray,
    second_length: float,
    second_width: float,
) -> bool:
    """SAT overlap for virtual vehicles placed on two route centrelines."""
    first_forward = first_forward / max(float(np.linalg.norm(first_forward)), 1e-12)
    second_forward = second_forward / max(float(np.linalg.norm(second_forward)), 1e-12)
    first_lateral = np.asarray([-first_forward[1], first_forward[0]])
    second_lateral = np.asarray([-second_forward[1], second_forward[0]])
    delta = np.asarray(second_center, dtype=float) - np.asarray(first_center, dtype=float)
    for axis in (first_forward, first_lateral, second_forward, second_lateral):
        first_radius = 0.5 * first_length * abs(float(np.dot(axis, first_forward))) + 0.5 * first_width * abs(float(np.dot(axis, first_lateral)))
        second_radius = 0.5 * second_length * abs(float(np.dot(axis, second_forward))) + 0.5 * second_width * abs(float(np.dot(axis, second_lateral)))
        if abs(float(np.dot(delta, axis))) > first_radius + second_radius:
            return False
    return True


class MetaDriveAdapterBase(ABC):
    """Version-sensitive MetaDrive handling shared by explicit route adapters."""

    def __init__(self) -> None:
        self._routes: dict[str, RoutePolyline] = {}
        self._frame: dict[str, Any] | None = None

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
    def _lane_graph_payload(env: Any) -> list[dict[str, Any]]:
        rows = []
        for start, ends in env.current_map.road_network.graph.items():
            for end, lanes in ends.items():
                for index, lane in enumerate(lanes):
                    samples = np.linspace(0.0, float(lane.length), 9)
                    rows.append({
                        "lane_index": [str(start), str(end), int(index)],
                        "points": [np.asarray(lane.position(float(s), 0.0), dtype=float).round(6).tolist() for s in samples],
                    })
        return sorted(rows, key=lambda row: tuple(row["lane_index"]))

    def map_hash(self, env: Any) -> str:
        return content_hash(self._lane_graph_payload(env))

    @staticmethod
    def _vehicle(env: Any) -> Any:
        vehicle = getattr(env, "agent", None) or getattr(env, "agents", {}).get("default_agent")
        if vehicle is None:
            raise RuntimeError("MetaDrive did not create the adversary agent")
        return vehicle

    def _spawn_idm(self, env: Any, lane: tuple[Any, Any, int], longitudinal: float, speed_mps: float, seed: int) -> Any:
        _, _, TrafficDefaultVehicle, IDMPolicy = self._imports()
        target_lane = env.current_map.road_network.get_lane(lane)
        if not 0.0 <= longitudinal <= float(target_lane.length):
            raise ValueError(f"SUT spawn {longitudinal:.2f} m is outside explicit lane {lane!r}")
        manager = env.engine.traffic_manager
        vehicle = manager.spawn_object(TrafficDefaultVehicle, vehicle_config={
            "spawn_lane_index": lane, "spawn_longitude": float(longitudinal), "spawn_lateral": 0.0,
            "enable_reverse": False,
        })
        manager.add_policy(vehicle.id, IDMPolicy, vehicle, int(seed))
        manager._traffic_vehicles.append(vehicle)
        return vehicle

    @staticmethod
    def _case_spawn(case: Mapping[str, Any], role: str, region: list[float]) -> float:
        key = f"{role}_spawn_m"
        value = float(case.get(key, sum(region) / 2.0))
        if not float(region[0]) <= value <= float(region[1]):
            raise ValueError(f"case {key}={value} violates frozen task spawn region {region}")
        return value

    def establish_roles(self, env: Any, task: LogicalScenarioTaskSpec, case: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, Any]:
        adversary = self._vehicle(env)
        expected_adv = lane_index(task.adversary_route["initial_lane"])
        expected_sut = lane_index(task.sut_route["initial_lane"])
        if tuple(adversary.lane_index) != expected_adv:
            raise RuntimeError(f"adversary spawned on {adversary.lane_index!r}, expected explicit route lane {expected_adv!r}")
        sut_spawn = self._case_spawn(case, "sut", task.spawn_regions["sut"])
        target_speed = float(config["sut"]["target_speed_mps"])
        sut = self._spawn_idm(env, expected_sut, sut_spawn, target_speed, int(case["case_seed"]) + 1)
        adversary_speed = float(case.get("adversary_initial_speed_mps", case["adversary_speed_mps"]))
        sut_initial_speed = float(case.get("sut_initial_speed_mps", target_speed))
        adversary.set_velocity(np.asarray(adversary.heading, dtype=float) * adversary_speed)
        sut.set_velocity(np.asarray(sut.heading, dtype=float) * sut_initial_speed)
        policy = env.engine.get_policy(sut.id)
        if hasattr(policy, "target_speed"):
            policy.target_speed = target_speed * 3.6
        if hasattr(policy, "enable_lane_change"):
            policy.enable_lane_change = bool(config["sut"]["enable_lane_change"])
        self._routes = {
            "adversary": RoutePolyline.from_env(env, task.adversary_route),
            "sut": RoutePolyline.from_env(env, task.sut_route),
        }
        self.validate_episode_roles(env, adversary, sut)
        return adversary, sut

    def conflict_frame(self, env: Any, task: LogicalScenarioTaskSpec, adversary: Any, sut: Any) -> dict[str, Any]:
        adv, target = self._routes["adversary"], self._routes["sut"]
        delta = adv.points[:, None, :] - target.points[None, :, :]
        distances = np.linalg.norm(delta, axis=-1)
        # A conservative centreline envelope uses half the sum of vehicle
        # lengths. On shallow-angle merges, longitudinal extent—not width—is
        # what makes physical contact possible before lane centres coincide.
        adv_length = float(getattr(adversary, "LENGTH", 5.0))
        sut_length = float(getattr(sut, "LENGTH", 5.0))
        # Conflict-zone geometry follows the route corridor, not an idealized
        # perfectly centred chassis. Pair distance remains a separate strict
        # condition, so this does not turn corridor proximity into near-miss.
        adv_width = max(
            float(getattr(adversary, "WIDTH", 2.0)),
            float(getattr(getattr(adversary, "lane", None), "width", 3.5)),
        )
        sut_width = max(
            float(getattr(sut, "WIDTH", 2.0)),
            float(getattr(getattr(sut, "lane", None), "width", 3.5)),
        )
        clearance = 0.5 * (adv_length + sut_length)
        centre_candidates = np.argwhere(distances <= clearance)
        obb_candidates: list[tuple[int, int]] = []
        for row_value, col_value in centre_candidates:
            row_index, col_index = int(row_value), int(col_value)
            adv_tangent = adv.tangent_at_s(float(adv.arc_lengths_m[row_index]))
            sut_tangent = target.tangent_at_s(float(target.arc_lengths_m[col_index]))
            adv_lane_index = int(np.clip(
                np.searchsorted(adv.lane_end_s_m, adv.arc_lengths_m[row_index], side="right"),
                0, len(adv.lane_indices) - 1,
            ))
            sut_lane_index = int(np.clip(
                np.searchsorted(target.lane_end_s_m, target.arc_lengths_m[col_index], side="right"),
                0, len(target.lane_indices) - 1,
            ))
            distinct_parallel_lanes = bool(
                adv.lane_indices[adv_lane_index] != target.lane_indices[sut_lane_index]
                and abs(float(adv_tangent[0] * sut_tangent[1] - adv_tangent[1] * sut_tangent[0])) <= 1e-3
                and float(np.dot(adv_tangent, sut_tangent)) > 0.0
            )
            if distinct_parallel_lanes:
                continue
            if _route_obb_intersects(
                adv.points[row_index], adv_tangent,
                adv_length, adv_width,
                target.points[col_index], sut_tangent,
                sut_length, sut_width,
            ):
                obb_candidates.append((row_index, col_index))
        candidates = np.asarray(obb_candidates, dtype=int).reshape((-1, 2))
        if len(candidates) == 0:
            row, col = np.unravel_index(int(np.argmin(distances)), distances.shape)
            distance = float(distances[row, col])
            if distance > float(task.conflict_spec["max_route_distance_m"]):
                raise RuntimeError(
                    f"task {task.task_id} has no true route encounter within "
                    f"{task.conflict_spec['max_route_distance_m']} m (closest={distance:.3f} m)"
                )
        else:
            # First geometrically collision-capable cross-section, rather than
            # the later fully-shared lane origin. This makes one conflict-point
            # metric comparable for gradual bottlenecks and abrupt lane drops.
            progress = np.maximum(
                adv.arc_lengths_m[candidates[:, 0]] / max(adv.length_m, 1e-6),
                target.arc_lengths_m[candidates[:, 1]] / max(target.length_m, 1e-6),
            )
            separation = distances[candidates[:, 0], candidates[:, 1]]
            selected = int(np.lexsort((separation, progress))[0])
            row, col = map(int, candidates[selected])
        adv_s = float(adv.arc_lengths_m[row])
        sut_s = float(target.arc_lengths_m[col])
        origin = (adv.points[row] + target.points[col]) / 2.0
        adv_tangent, sut_tangent = adv.tangent_at_s(adv_s), target.tangent_at_s(sut_s)
        heading = float(math.atan2((adv_tangent + sut_tangent)[1], (adv_tangent + sut_tangent)[0]))
        frame = {
            "origin": origin,
            "heading": heading,
            "radius_m": float(task.conflict_spec["conflict_radius_m"]),
            "adversary_conflict_s_m": adv_s,
            "sut_conflict_s_m": sut_s,
            "adversary_route": adv,
            "sut_route": target,
        }
        self._frame = frame
        return frame

    @staticmethod
    def _curvature(route: RoutePolyline) -> float:
        directions = np.diff(route.points, axis=0)
        headings = np.unwrap(np.arctan2(directions[:, 1], directions[:, 0]))
        lengths = np.linalg.norm(directions, axis=1)
        if len(headings) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(headings)) / np.maximum(lengths[1:], 1e-6)))

    def topology_features(self, env: Any, task: LogicalScenarioTaskSpec) -> dict[str, float]:
        graph = env.current_map.road_network.graph
        lanes = self._lane_graph_payload(env)
        adv, sut = self._routes["adversary"], self._routes["sut"]
        frame = self._frame or {}
        return {
            "num_incoming_branches": float(len(graph)),
            "num_outgoing_branches": float(sum(len(v) for v in graph.values())),
            "adversary_lane_count": float(len(adv.lane_indices)),
            "sut_lane_count": float(len(sut.lane_indices)),
            "merge_length_m": float(max(0.0, frame.get("adversary_conflict_s_m", 0.0))),
            "conflict_radius_m": float(task.conflict_spec["conflict_radius_m"]),
            "adversary_route_curvature": self._curvature(adv),
            "sut_route_curvature": self._curvature(sut),
            "adversary_speed_limit_mps": 30.0,
            "sut_speed_limit_mps": 30.0,
            "num_conflict_zones": 1.0,
            "lane_graph_edges": float(len(lanes)),
        }

    def target_contact(self, env: Any, adversary: Any, sut: Any) -> tuple[bool, str]:
        try:
            world = env.engine.physics_world.dynamic_world
            result = world.contactTestPair(adversary.chassis.node(), sut.chassis.node())
            if result.getNumContacts() > 0:
                return True, "physics_contact_pair"
        except (AttributeError, TypeError):
            pass
        if _obb_intersects(adversary, sut):
            return True, "obb_overlap"
        distance = float(np.linalg.norm(np.asarray(adversary.position) - np.asarray(sut.position)))
        threshold = 0.55 * (float(getattr(adversary, "LENGTH", 5.0)) + float(getattr(sut, "LENGTH", 5.0)))
        if bool(getattr(adversary, "crash_vehicle", False) and getattr(sut, "crash_vehicle", False)) and distance <= threshold:
            return False, "center_distance_low_confidence"
        return False, "no_pairwise_contact"

    def route_status(self, env: Any, vehicle: Any, role: str, previous_s_m: float | None) -> tuple[float, bool, bool]:
        route = self._routes[role]
        projection = route.projection(vehicle.position, float(getattr(vehicle, "heading_theta", 0.0)), float(getattr(vehicle.lane, "width", 3.8)))
        current = tuple(getattr(vehicle, "lane_index", (None, None, -1)))
        route_membership = current in route.lane_indices
        graph = env.current_map.road_network.graph
        # A temporary connector is valid only when it leads into an explicitly
        # planned route segment.  This decision is made from road topology, not
        # from task metadata or a nearby-lane heuristic.
        planned_nodes = {node for index in route.lane_indices for node in index[:2]}
        # Bottleneck maps may choose either of two parallel merge connectors;
        # both are legitimate when the connector's next graph edge enters the
        # next frozen route node. The old direct-node-only test falsely marked
        # this one-hop connector as a wrong route.
        outgoing_from_destination = set(graph.get(current[1], {}))
        forward_connected = current[1] in planned_nodes or bool(outgoing_from_destination & planned_nodes)
        backwards = abs(projection.heading_error) > math.pi / 2.0
        regressed = previous_s_m is not None and projection.s_m < previous_s_m - 1.0
        # A frozen route ends at the experiment's ODD boundary.  MetaDrive may
        # either keep the final lane index while setting ``on_road=False`` or
        # switch immediately to an unplanned successor lane.  Both are normal
        # completion, not an invalid route departure.  The check is deliberately
        # narrow: the vehicle centre must be within its final half-length, on
        # the route corridor, and aligned forward.  Mid-route branches remain
        # invalid regardless of graph connectivity.
        completion_margin = max(1.0, 0.5 * float(getattr(vehicle, "LENGTH", 5.0)))
        completed_route = bool(
            projection.s_m >= route.length_m - completion_margin
            and projection.on_route
            and not backwards
        )
        geometric_lane_change = bool(
            route.in_lane_change(projection.s_m)
            and projection.on_route
            and not backwards
            and not regressed
        )
        wrong_route = bool(
            ((not route_membership and not forward_connected) and not completed_route and not geometric_lane_change)
            or backwards
            or regressed
        )
        return projection.s_m, wrong_route, completed_route

    def validate_episode_roles(self, env: Any, adversary: Any, sut: Any) -> None:
        if adversary is sut or str(adversary.id) == str(sut.id):
            raise RuntimeError("adversary and SUT must be distinct objects")
        if tuple(adversary.lane_index) == tuple(sut.lane_index):
            raise RuntimeError("roles must begin on their explicit distinct route lanes")
        if _obb_intersects(adversary, sut):
            raise RuntimeError("explicit roles overlap at reset")
