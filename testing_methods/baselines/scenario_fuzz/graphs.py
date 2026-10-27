"""Waypoint/path graphs derived from public HighwayEnv scenario parameters only."""

import math

import torch
from torch_geometric.data import Data

from experiments.scenarios import validate
from simulator.highway.interaction_env import GUARD_FRONT_GAP_M, GUARD_REAR_GAP_M


class ScenarioGraph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_graph_index":
            return self.edge_index.shape[1]
        return super().__inc__(key, value, *args, **kwargs)


def graph(scenario):
    validate(scenario)
    nodes, edges, attributes = [], [], []

    def node(x, y, kind=0, appearance=0, *, time_s=0, speed=0):
        nodes.append([float(x), float(y), 0.0, kind, 0, appearance, time_s / 10, speed / 40])
        return len(nodes) - 1

    def edge(a, b, kind=0):
        x, y = nodes[b][0] - nodes[a][0], nodes[b][1] - nodes[a][1]
        direction = 1 if y < -0.1 else 2 if y > 0.1 else 0
        edges.append((a, b))
        attributes.append([math.hypot(x, y), kind, direction])

    # The fixed map is a three-lane straight road. Node coordinates are ego-relative.
    lanes = (-4, 0) if scenario.kind == "lane_change_rear" else (-4, 0, 4)
    road = {(x, y): node(x, y) for x in (0, 100, 200, 400) for y in lanes}
    for (x, y), source in road.items():
        if x < 400:
            following = {0: 100, 100: 200, 200: 400}[x]
            edge(source, road[following, y])
        for neighbor in (y - 4, y + 4):
            if (x, neighbor) in road:
                edge(source, road[x, neighbor])

    def connect(index):
        nearest = min(
            road.values(),
            key=lambda j: (
                (nodes[j][0] - nodes[index][0]) ** 2 + (nodes[j][1] - nodes[index][1]) ** 2
            ),
        )
        edge(index, nearest)
        edge(nearest, index)

    p = scenario.parameters
    ego_start = node(0, 0, 1, 1, speed=p["ego_speed_mps"])
    ego_end = node(p["ego_speed_mps"] * 10, 0, 2, 1, time_s=10, speed=p["ego_speed_mps"])
    edge(ego_start, ego_end, 1)
    connect(ego_start)
    connect(ego_end)

    def background_path(x0, y0, speed):
        path = [node(x0 + speed * t, y0, 3, 1, time_s=t, speed=speed) for t in (0, 10)]
        edge(*path, 2)
        for index in path:
            connect(index)

    if scenario.kind == "front_brake":
        speed = p["ego_speed_mps"] - p["closing_speed_mps"]
        ramp, deceleration, event = p["brake_ramp_s"], p["deceleration_mps2"], p["event_time_s"]
        after_ramp = speed - deceleration * ramp / 2
        stop_time = event + ramp + after_ramp / deceleration
        path = []
        for t in sorted({0, event, event + ramp, min(10, stop_time), 10}):
            braking = max(0, t - event)
            first = min(braking, ramp)
            later = min(max(0, braking - ramp), after_ramp / deceleration)
            distance = speed * min(t, event) + speed * first - deceleration * first**3 / (6 * ramp)
            distance += after_ramp * later - deceleration * later**2 / 2
            velocity = max(0, speed - deceleration * first**2 / (2 * ramp) - deceleration * later)
            path.append(node(p["gap_m"] + 5 + distance, 0, 3, 1, time_s=t, speed=velocity))
        for start, end in zip(path, path[1:]):
            edge(start, end, 2)
        for index in path:
            connect(index)
    elif scenario.kind == "lane_change_rear":
        background_path(p["lead_gap_m"] + 5, 0, p["lead_speed_mps"])
        background_path(-p["rear_gap_m"] - 5, -4, p["rear_speed_mps"])
        background_path(p["adjacent_front_gap_m"] + 5, -4, p["ego_speed_mps"])
    elif scenario.kind in ("cut_in", "cut_in_interaction"):
        initial_y = -4 if scenario.categories["side"] == "left" else 4
        speed = p["ego_speed_mps"] - p["closing_speed_mps"]
        final_y = 0
    else:
        initial_y, final_y, speed = 0, -4, p["lead_speed_mps"]
    if scenario.kind in ("cut_in", "cut_out", "cut_in_interaction"):
        times = (0, p["event_time_s"], p["event_time_s"] + p["lateral_duration_s"], 10)
        path = [
            node(
                p["gap_m"] + 5 + speed * t,
                initial_y if i < 2 else final_y,
                3,
                1,
                time_s=t,
                speed=speed,
            )
            for i, t in enumerate(times)
        ]
        for start, end in zip(path, path[1:]):
            edge(start, end, 2)
        for index in path:
            connect(index)
    if (
        scenario.kind == "cut_in_interaction"
        and scenario.categories["adjacent_space"] == "restricted"
    ):
        opposite_y = 4 if scenario.categories["side"] == "left" else -4
        background_path(GUARD_FRONT_GAP_M + 5, opposite_y, p["ego_speed_mps"])
        background_path(-GUARD_REAR_GAP_M - 5, opposite_y, p["ego_speed_mps"])
    if scenario.kind == "cut_out":
        connect(node(p["obstacle_distance_m"], 0, 4, 2))
    # Edge features are themselves graph entities. Their adjacency is the line graph.
    line_edges = [
        (i, j)
        for i, (_, end) in enumerate(edges)
        for j, (start, _) in enumerate(edges)
        if end == start
    ]
    return ScenarioGraph(
        x=torch.tensor(nodes, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
        edge_attr=torch.tensor(attributes, dtype=torch.float32),
        edge_graph_index=torch.tensor(line_edges, dtype=torch.long).T.contiguous(),
        weather_attr=torch.zeros(1, 8),
    )
