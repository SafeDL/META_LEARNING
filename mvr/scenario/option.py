from __future__ import annotations

from enum import Enum


class AdversarialOption(str, Enum):
    APPROACH_CONFLICT = "approach_conflict"
    YIELD_THEN_PRESS = "yield_then_press"
    GAP_CLOSE = "gap_close"
    ROUTE_BLOCK = "route_block"
