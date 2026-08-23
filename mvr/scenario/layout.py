"""Runtime-resolved physical layout for one outer scenario candidate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LaneIndex = tuple[Any, Any, int]


@dataclass(frozen=True)
class ScenarioLayout:
    """Lane/route binding resolved from the *actual* generated road network.

    Candidate labels are intentionally task-level strings.  This object is the
    auditable bridge from that label to concrete MetaDrive lane indices and
    destinations; it is never a learned model input.
    """

    candidate: str
    conflict_zone_id: str
    adversary_lane: LaneIndex
    sut_lane: LaneIndex
    adversary_destination: Any
    sut_destination: Any
    adversary_route: tuple[LaneIndex, ...]
    sut_route: tuple[LaneIndex, ...]
    conflict_xy: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.candidate or not self.conflict_zone_id or not self.adversary_route or not self.sut_route:
            raise ValueError("scenario layout requires a candidate and non-empty routes")
        if self.adversary_route[0] != self.adversary_lane or self.sut_route[0] != self.sut_lane:
            raise ValueError("each runtime route must begin on its spawn lane")
        if not np.isfinite(np.asarray(self.conflict_xy, dtype=float)).all():
            raise ValueError("scenario layout conflict reference must be finite")
