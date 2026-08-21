from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
import torch


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
        if not self.polyline_id or points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("polyline requires at least two finite XY points")
        if np.asarray(self.headings).shape != (len(points),) or np.asarray(self.curvature).shape != (len(points),):
            raise ValueError("heading and curvature must align with points")

    def local_features(self) -> np.ndarray:
        points = np.asarray(self.points_xy, dtype=np.float32)
        centre = points.mean(axis=0)
        heading = float(np.asarray(self.headings, dtype=np.float32)[len(points) // 2])
        c, s = np.cos(heading), np.sin(heading)
        rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
        local = (points - centre) @ rotation.T
        return np.column_stack((local, np.cos(self.headings - heading), np.sin(self.headings - heading), self.curvature,
                                np.full(len(points), self.lane_width), np.full(len(points), self.speed_limit))).astype(np.float32)


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

    def tensorize(self, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.polylines:
            raise ValueError("cannot encode an empty map")
        lengths = {len(polyline.points_xy) for polyline in self.polylines}
        if len(lengths) != 1:
            raise ValueError("tokens must be resampled to a shared point count")
        features = torch.as_tensor(np.stack([polyline.local_features() for polyline in self.polylines]), device=device)
        centres = torch.as_tensor(np.stack([np.asarray(polyline.points_xy).mean(axis=0) for polyline in self.polylines]), device=device)
        headings = torch.as_tensor([polyline.headings[len(polyline.headings) // 2] for polyline in self.polylines], device=device)
        return features.float(), centres.float(), headings.float()
