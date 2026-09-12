"""Geometry-only interaction candidates used by the universal Outer policy."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .layout import LaneIndex, ScenarioLayout
from .route_geometry import RoutePolyline


@dataclass(frozen=True)
class InteractionCandidate:
    candidate_id: str
    sut_route: tuple[LaneIndex, ...]
    adversary_route: tuple[LaneIndex, ...]
    conflict_xy: tuple[float, float]
    crossing_angle_rad: float
    sut_distance_available_m: float
    adversary_distance_available_m: float
    sut_route_curvature: float
    adversary_route_curvature: float
    conflict_zone_id: str
    sut_distance_min_m: float = 0.0
    adversary_distance_min_m: float = 0.0

    def features(self) -> np.ndarray:
        """SE(2)-invariant descriptors; labels and family names are excluded."""
        # Keep descriptor magnitudes comparable before the learned projection.
        # Distances are bounded by the scenario parameter space and curvature is
        # an angle-like quantity in radians.
        values = np.asarray((
            math.sin(self.crossing_angle_rad), math.cos(self.crossing_angle_rad),
            self.sut_distance_available_m / 100.0,
            self.adversary_distance_available_m / 100.0,
            self.sut_route_curvature / math.pi,
            self.adversary_route_curvature / math.pi,
        ), dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

    @classmethod
    def from_layout(cls, env: Any, layout: ScenarioLayout) -> "InteractionCandidate":
        adversary = RoutePolyline.from_env(
            env, {"route_id": "adversary", "lane_sequence": layout.adversary_route}
        )
        sut = RoutePolyline.from_env(env, {"route_id": "sut", "lane_sequence": layout.sut_route})
        return cls.from_routes(layout, adversary, sut)

    @classmethod
    def from_routes(
        cls,
        layout: ScenarioLayout,
        adversary: RoutePolyline,
        sut: RoutePolyline,
    ) -> "InteractionCandidate":
        adversary_s = adversary.conflict_s(layout.conflict_xy)
        sut_s = sut.conflict_s(layout.conflict_xy)
        adv_tangent = adversary.tangent_at_s(adversary_s)
        sut_tangent = sut.tangent_at_s(sut_s)
        angle = math.atan2(
            float(adv_tangent[0] * sut_tangent[1] - adv_tangent[1] * sut_tangent[0]),
            float(np.dot(adv_tangent, sut_tangent)),
        )
        def curvature(route: RoutePolyline, s_m: float) -> float:
            index = int(np.clip(np.searchsorted(route.arc_lengths_m, s_m), 0, len(route.points) - 2))
            first = route.tangent_at_s(route.arc_lengths_m[max(0, index - 1)])
            second = route.tangent_at_s(route.arc_lengths_m[min(len(route.points) - 2, index + 1)])
            return float(np.arctan2(first[0] * second[1] - first[1] * second[0], np.dot(first, second)))
        return cls(
            layout.candidate, layout.sut_route, layout.adversary_route, layout.conflict_xy,
            angle, sut_s, adversary_s, curvature(sut, sut_s), curvature(adversary, adversary_s),
            layout.conflict_zone_id,
            max(0.0, sut_s - sut.lane_end_s_m[0]),
            max(0.0, adversary_s - adversary.lane_end_s_m[0]),
        )
