from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MapPolyline:
    polyline_id: str
    polyline_type: str
    points_xy: np.ndarray
    headings: np.ndarray
    curvature: np.ndarray
    lane_width: float
    speed_limit: float
    attributes: Mapping[str, Any]

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xy, dtype=np.float32)
        if (
            not self.polyline_id
            or points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
        ):
            raise ValueError("polyline requires at least two finite XY points")
        if np.asarray(self.headings).shape != (len(points),) or np.asarray(
            self.curvature
        ).shape != (len(points),):
            raise ValueError("heading and curvature must align with points")


@dataclass(frozen=True)
class MapTokens:
    map_hash: str
    polylines: tuple[MapPolyline, ...]
    relations: Mapping[str, tuple[tuple[int, int], ...]]

    def __post_init__(self) -> None:
        if len(self.map_hash) != 64:
            raise ValueError("MapTokens requires a SHA-256 map hash")
        count = len(self.polylines)
        for kind, edges in self.relations.items():
            if not kind:
                raise ValueError("relation type cannot be empty")
            if any(not (0 <= src < count and 0 <= dst < count) for src, dst in edges):
                raise ValueError(f"relation {kind!r} references an unknown polyline")
