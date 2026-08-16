"""Route-polyline construction and route-relative kinematics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import math
import re
import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def lane_index(value: Iterable[Any]) -> tuple[Any, Any, int]:
    values = tuple(value)
    if len(values) != 3:
        raise ValueError("lane index must contain origin, destination, and lane number")
    # Normalize route-spec branch aliases to runtime road-graph node names.
    def node(item: Any) -> Any:
        match = re.fullmatch(r"([0-9]+[A-Za-z][0-9]+)-(\d+)-", str(item))
        return f"{match.group(1)}_{match.group(2)}_" if match else item
    return node(values[0]), node(values[1]), int(values[2])


@dataclass(frozen=True)
class RouteProjection:
    s_m: float
    lateral_m: float
    tangent: np.ndarray
    heading_error: float
    on_route: bool


@dataclass(frozen=True)
class RoutePolyline:
    """Sampled lane-centre polyline with monotonic arc-length coordinates."""

    lane_indices: tuple[tuple[Any, Any, int], ...]
    points: np.ndarray
    arc_lengths_m: np.ndarray
    lane_end_s_m: tuple[float, ...]
    lane_change_intervals_m: tuple[tuple[float, float], ...] = ()

    @property
    def length_m(self) -> float:
        return float(self.arc_lengths_m[-1])

    @classmethod
    def from_env(cls, env: Any, route: Mapping[str, Any], samples_per_lane: int = 48) -> "RoutePolyline":
        indices = tuple(lane_index(item) for item in route["lane_sequence"])
        graph = env.current_map.road_network
        points: list[np.ndarray] = []
        ends: list[float] = []
        lane_changes: list[tuple[float, float]] = []
        distance = 0.0
        for index in indices:
            lane = graph.get_lane(index)
            samples = np.linspace(0.0, float(lane.length), max(2, int(samples_per_lane)))
            lane_points = [np.asarray(lane.position(float(s), 0.0), dtype=float) for s in samples]
            lane_change_start: float | None = None
            if points:
                # A logical route may require a lane change between adjacent
                # lane centres.  Concatenating those centres directly creates
                # a lateral teleport segment (and a near-90-degree tangent)
                # that corrupts progress, heading error, and route rewards.
                # Blend the offset smoothly across the next lane instead: the
                # resulting reference begins at the previous centreline and
                # converges to the declared target lane with zero endpoint
                # slope distortion (cubic smoothstep).
                offset = np.asarray(points[-1], dtype=float) - lane_points[0]
                if float(np.linalg.norm(offset)) > 1e-6:
                    lane_change_start = distance
                    fractions = samples / max(float(lane.length), 1e-6)
                    smooth = fractions * fractions * (3.0 - 2.0 * fractions)
                    lane_points = [
                        point + (1.0 - float(weight)) * offset
                        for point, weight in zip(lane_points, smooth)
                    ]
                lane_points = lane_points[1:]
            for point in lane_points:
                if points:
                    distance += float(np.linalg.norm(point - points[-1]))
                points.append(point)
            ends.append(distance)
            if lane_change_start is not None:
                lane_changes.append((lane_change_start, distance))
        if len(points) < 2 or distance <= 0.0:
            raise RuntimeError(f"route {route.get('route_id')} has no usable geometry")
        values = np.asarray(points, dtype=float)
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))
        return cls(indices, values, arc, tuple(ends), tuple(lane_changes))

    def projection(self, position: Any, heading: float, lane_width_m: float = 3.8) -> RouteProjection:
        point = np.asarray(position, dtype=float)
        starts, ends = self.points[:-1], self.points[1:]
        segments = ends - starts
        lengths2 = np.einsum("ij,ij->i", segments, segments)
        factors = np.clip(np.einsum("ij,ij->i", point - starts, segments) / np.maximum(lengths2, 1e-12), 0.0, 1.0)
        closest = starts + segments * factors[:, None]
        distances2 = np.einsum("ij,ij->i", point - closest, point - closest)
        index = int(np.argmin(distances2))
        tangent = segments[index] / max(float(np.linalg.norm(segments[index])), 1e-12)
        normal = np.asarray([-tangent[1], tangent[0]])
        s = float(self.arc_lengths_m[index] + factors[index] * np.linalg.norm(segments[index]))
        lateral = float(np.dot(point - closest[index], normal))
        route_heading = float(math.atan2(tangent[1], tangent[0]))
        return RouteProjection(s, lateral, tangent, wrap_to_pi(float(heading) - route_heading), abs(lateral) <= 0.5 * lane_width_m)

    def in_lane_change(self, s_m: float) -> bool:
        return any(start <= float(s_m) <= end for start, end in self.lane_change_intervals_m)

    def conflict_s(self, point: Any) -> float:
        return self.projection(point, 0.0).s_m

    def tangent_at_s(self, s_m: float) -> np.ndarray:
        index = int(np.clip(np.searchsorted(self.arc_lengths_m, float(s_m), side="right") - 1, 0, len(self.points) - 2))
        delta = self.points[index + 1] - self.points[index]
        return delta / max(float(np.linalg.norm(delta)), 1e-12)


def route_hash_payload(route: Mapping[str, Any]) -> dict[str, Any]:
    return {"route_id": route["route_id"], "lane_sequence": [list(lane_index(x)) for x in route["lane_sequence"]]}
