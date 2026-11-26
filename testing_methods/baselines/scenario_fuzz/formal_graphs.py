"""Public road and scheduled actor paths for the four registered integer scenes.

Coordinates are relative to the ego spawn. Paths describe the public event,
without running a controller or reading an execution record.
"""

import math

import torch

from experiments.scenarios import validate

from .graphs import ScenarioGraph


def graph(scenario):
    validate(scenario)
    p, kind = scenario.parameters, scenario.kind
    duration = 14 if kind == "car_following_braking_onset" else 10
    side = -4 if scenario.categories.get("side") == "left" else 4
    lanes = {
        "car_following_braking_onset": (0,),
        "cut_in_onset": (-4, 0, 4),
        "cut_out_rear_gap": (-4, 0),
        "side_encroachment": tuple(sorted((0, side))),
    }[kind]
    nodes, edges, attributes = [], [], []

    def node(x, y, actor=0, *, time=0, speed=0):
        nodes.append([float(x), float(y), 0, actor, 0, int(actor != 0), time / 10, speed / 40])
        return len(nodes) - 1

    def edge(start, end, actor=0):
        dx = nodes[end][0] - nodes[start][0]
        dy = nodes[end][1] - nodes[start][1]
        edges.append((start, end))
        attributes.append([math.hypot(dx, dy), actor, 1 if dy < -0.1 else 2 if dy > 0.1 else 0])

    road = {(x, y): node(x, y) for x in (0, 100, 200, 400) for y in lanes}
    for (x, y), index in road.items():
        if x < 400:
            edge(index, road[{0: 100, 100: 200, 200: 400}[x], y])
        for neighbor in (y - 4, y + 4):
            if (x, neighbor) in road:
                edge(index, road[x, neighbor])

    def connect(index):
        nearest = min(road.values(), key=lambda j: sum(
            (nodes[j][i] - nodes[index][i]) ** 2 for i in (0, 1)
        ))
        edge(index, nearest)
        edge(nearest, index)

    def path(points, actor=3):
        indices = [node(x, y, actor, time=t, speed=v) for t, x, y, v in points]
        for start, end in zip(indices, indices[1:]):
            edge(start, end, 1 if actor in (1, 2) else 2)
        for index in indices:
            connect(index)

    speed = p["ego_speed_mps"]
    start = node(0, 0, 1, speed=speed)
    end = node(speed * duration, 0, 2, time=duration, speed=speed)
    edge(start, end, 1)
    connect(start)
    connect(end)

    def cruise(x, y, velocity):
        path([(0, x, y, velocity), (duration, x + velocity * duration, y, velocity)])

    def lateral(x, y0, y1, velocity):
        shift = p["lateral_duration_s"]
        points = []
        for t in sorted({0, shift / 2, shift, duration}):
            ratio = min(1, t / shift)
            y = y0 + (y1 - y0) * (10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5)
            points.append((t, x + velocity * t, y, velocity))
        path(points)

    if kind == "car_following_braking_onset":
        lead = speed - p["closing_speed_mps"]
        deceleration = p["deceleration_mps2"]
        stop = lead / deceleration
        path([(t, p["gap_m"] + 5 + lead * min(t, stop)
               - deceleration * min(t, stop)**2 / 2, 0, max(0, lead - deceleration * t))
              for t in sorted({0, stop, duration})])
    elif kind == "cut_in_onset":
        lateral(p["gap_m"] + 5, side, 0, speed - p["closing_speed_mps"])
        cruise(15, -side, speed)
        cruise(-15, -side, speed)
    elif kind == "cut_out_rear_gap":
        lateral(16.5, 0, -4, speed)
        cruise(-p["rear_gap_m"], -4, speed)
        connect(node(p["obstacle_distance_m"], 0, 4))
    else:
        lateral(p["longitudinal_offset_m"], side, 0, speed + p["relative_speed_mps"])
    line_edges = [(i, j) for i, (_, end) in enumerate(edges)
                  for j, (start, _) in enumerate(edges) if end == start]
    return ScenarioGraph(
        x=torch.tensor(nodes, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
        edge_attr=torch.tensor(attributes, dtype=torch.float32),
        edge_graph_index=torch.tensor(line_edges, dtype=torch.long).T.contiguous(),
        weather_attr=torch.zeros(1, 8),
    )
