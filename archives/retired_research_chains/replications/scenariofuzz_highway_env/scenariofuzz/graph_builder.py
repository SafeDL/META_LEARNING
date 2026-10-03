"""Leakage-safe pre-execution graph construction and explicit line graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .corpus import LocalScenarioSeed, ScenarioSpec


NODE_FEATURE_NAMES = (
    "relative_x", "relative_y", "type_waypoint", "type_ego", "type_npc",
    "type_goal", "lane_0", "lane_1", "planned_speed",
)
EDGE_FEATURE_NAMES = (
    "distance", "path_lane", "path_ego", "path_npc", "straight", "lane_change",
)
GLOBAL_FEATURE_NAMES = (
    "mode_fast_intrusion", "mode_cutin_braking", "mode_lead_braking",
    "cutin_start", "cutin_duration", "brake_start", "brake_duration", "brake_deceleration",
)


@dataclass(frozen=True)
class ScenarioGraph:
    node_features: np.ndarray
    node_adjacency: np.ndarray
    edge_features: np.ndarray
    line_adjacency: np.ndarray
    edge_index: np.ndarray
    global_features: np.ndarray
    node_labels: tuple[str, ...]
    edge_labels: tuple[str, ...]

    def validate(self) -> None:
        n, e = len(self.node_features), len(self.edge_features)
        if self.node_adjacency.shape != (n, n):
            raise ValueError("node adjacency does not match node entities")
        if self.line_adjacency.shape != (e, e):
            raise ValueError("line adjacency does not match edge entities")
        if self.edge_index.shape != (2, e):
            raise ValueError("edge_index must identify every edge entity")
        if np.any(self.edge_index < 0) or np.any(self.edge_index >= n):
            raise ValueError("edge_index contains an invalid node index")


def _schedule(mode: str) -> tuple[float, float, float, float, float]:
    if mode == "fast_intrusion":
        return 1.0, 0.45, 0.0, 0.0, 0.0
    if mode == "cutin_braking":
        return 1.0, 1.5, 2.5, 1.0, 4.5
    if mode == "lead_braking":
        return 0.0, 0.0, 1.0, 1.0, 6.5
    raise ValueError(f"Unsupported graph mode: {mode}")


def build_graph(seed: LocalScenarioSeed, spec: ScenarioSpec) -> ScenarioGraph:
    """Build a graph from topology, planned routes and configured inputs only."""
    valid, reason = seed.validate(spec)
    if not valid:
        raise ValueError(f"Cannot graph invalid scenario: {reason}")
    lane_x = (20.0, 100.0, 180.0, 260.0)
    positions: list[tuple[float, float, str, int, float]] = []
    labels: list[str] = []
    for lane in (0, 1):
        for j, x in enumerate(lane_x):
            positions.append((x, lane * seed.lane_width, "waypoint", lane, 35.0))
            labels.append(f"lane{lane}_wp{j}")
    lead_x = 60.0 + spec.initial_gap
    lead_lane = 0
    if spec.mode in {"fast_intrusion", "cutin_braking"}:
        lead_x -= spec.relative_speed
        lead_lane = 1
    positions.extend([
        (60.0, 0.0, "ego", 0, 25.0),
        (280.0, 0.0, "goal", 0, 25.0),
        (lead_x, lead_lane * seed.lane_width, "npc", lead_lane, 25.0 + spec.relative_speed),
        (min(lead_x + 170.0, 360.0), 0.0, "goal", 0, 25.0 + spec.relative_speed),
    ])
    labels.extend(("ego_start", "ego_goal", "npc_start", "npc_goal"))
    type_index = {"waypoint": 0, "ego": 1, "npc": 2, "goal": 3}
    node = np.zeros((len(positions), len(NODE_FEATURE_NAMES)), dtype=np.float32)
    for i, (x, y, kind, lane, speed) in enumerate(positions):
        node[i, 0] = (x - 60.0) / 220.0
        node[i, 1] = y / seed.lane_width
        node[i, 2 + type_index[kind]] = 1.0
        node[i, 6 + lane] = 1.0
        node[i, 8] = speed / 35.0

    edges: list[tuple[int, int, str, str]] = []
    for lane in (0, 1):
        base = lane * len(lane_x)
        for j in range(len(lane_x) - 1):
            edges.append((base + j, base + j + 1, "lane", "straight"))
            edges.append((base + j + 1, base + j, "lane", "straight"))
    edges.append((8, 9, "ego", "straight"))
    direction = "lane_change" if lead_lane == 1 else "straight"
    edges.append((10, 11, "npc", direction))
    edge_index = np.asarray([(a, b) for a, b, _, _ in edges], dtype=np.int64).T
    edge = np.zeros((len(edges), len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    edge_labels: list[str] = []
    path_offset = {"lane": 1, "ego": 2, "npc": 3}
    for i, (a, b, kind, direction) in enumerate(edges):
        pa = np.asarray(positions[a][:2]); pb = np.asarray(positions[b][:2])
        edge[i, 0] = float(np.linalg.norm(pb - pa) / 250.0)
        edge[i, path_offset[kind]] = 1.0
        edge[i, 4 if direction == "straight" else 5] = 1.0
        edge_labels.append(f"{labels[a]}->{labels[b]}:{kind}")
    node_adj = np.eye(len(node), dtype=np.float32)
    for a, b, _, _ in edges:
        node_adj[a, b] = node_adj[b, a] = 1.0
    line_adj = np.eye(len(edge), dtype=np.float32)
    for i, (a, b, _, _) in enumerate(edges):
        for j, (c, d, _, _) in enumerate(edges):
            if i != j and ({a, b} & {c, d} or b == c):
                line_adj[i, j] = 1.0
    modes = ["fast_intrusion", "cutin_braking", "lead_braking"]
    global_features = np.zeros(len(GLOBAL_FEATURE_NAMES), dtype=np.float32)
    global_features[modes.index(spec.mode)] = 1.0
    schedule = _schedule(spec.mode)
    global_features[3:] = np.asarray(schedule, dtype=np.float32) / np.asarray([2, 2, 3, 2, 10])
    graph = ScenarioGraph(node, node_adj, edge, line_adj, edge_index, global_features, tuple(labels), tuple(edge_labels))
    graph.validate()
    return graph


def stack_graphs(graphs: list[ScenarioGraph], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    if not graphs:
        raise ValueError("at least one graph is required")
    return {
        "nodes": torch.as_tensor(np.stack([g.node_features for g in graphs]), device=device),
        "node_adj": torch.as_tensor(np.stack([g.node_adjacency for g in graphs]), device=device),
        "edges": torch.as_tensor(np.stack([g.edge_features for g in graphs]), device=device),
        "line_adj": torch.as_tensor(np.stack([g.line_adjacency for g in graphs]), device=device),
        "global_features": torch.as_tensor(np.stack([g.global_features for g in graphs]), device=device),
    }

