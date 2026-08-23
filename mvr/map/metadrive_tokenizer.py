"""Canonicalize a MetaDrive road network into fixed-length lane polylines."""
from __future__ import annotations

from typing import Any
import numpy as np

from ..provenance import content_hash
from .relations import lane_relations
from .schema import MapPolyline, MapTokens


def _resample_lane(lane: Any, points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = float(lane.length)
    samples = np.linspace(0.0, length, points, dtype=np.float32)
    xy = np.asarray([lane.position(float(s), 0.0) for s in samples], dtype=np.float32)
    tangent = np.gradient(xy, axis=0)
    headings = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0])).astype(np.float32)
    distances = np.maximum(np.linalg.norm(np.gradient(xy, axis=0), axis=1), 1e-6)
    curvature = (np.gradient(headings) / distances).astype(np.float32)
    return xy, headings, curvature


def tokenize_road_network(road_network: Any, *, points_per_polyline: int = 16) -> MapTokens:
    if points_per_polyline < 2:
        raise ValueError("points_per_polyline must be at least two")
    rows: list[tuple[tuple[Any, Any, int], Any]] = []
    for start, ends in road_network.graph.items():
        for end, lanes in ends.items():
            rows.extend(((start, end, index), lane) for index, lane in enumerate(lanes))
    rows.sort(key=lambda row: tuple(map(str, row[0])))
    lane_indices = [index for index, _ in rows]
    polylines = []
    payload = []
    for lane_index, lane in rows:
        xy, headings, curvature = _resample_lane(lane, points_per_polyline)
        payload.append({"lane_index": list(map(str, lane_index)), "points": xy.round(6).tolist()})
        polylines.append(MapPolyline(
            polyline_id="|".join(map(str, lane_index)), polyline_type=type(lane).__name__, points_xy=xy,
            headings=headings, curvature=curvature, lane_width=float(getattr(lane, "width", 3.5)),
            speed_limit=float(getattr(lane, "speed_limit", 0.0)), attributes={"lane_index": lane_index},
        ))
    return MapTokens(content_hash(payload), tuple(polylines), lane_relations(lane_indices))
