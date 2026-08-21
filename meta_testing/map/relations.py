from __future__ import annotations

from collections import defaultdict
from typing import Iterable


RELATION_TYPES = ("successor", "predecessor", "left", "right", "merge", "split", "conflict", "crossing", "route_membership")


def lane_relations(lane_indices: Iterable[tuple[object, object, int]]) -> dict[str, tuple[tuple[int, int], ...]]:
    """Infer stable typed road-graph relations from MetaDrive lane indices."""
    indices = list(lane_indices)
    relations: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for source, (start, end, lane_number) in enumerate(indices):
        for target, (next_start, next_end, next_lane_number) in enumerate(indices):
            if source == target:
                continue
            if end == next_start:
                relations["successor"].add((source, target))
                relations["predecessor"].add((target, source))
            if start == next_start and end == next_end:
                if next_lane_number == lane_number - 1:
                    relations["left"].add((source, target))
                if next_lane_number == lane_number + 1:
                    relations["right"].add((source, target))
    return {kind: tuple(sorted(edges)) for kind, edges in relations.items()}
