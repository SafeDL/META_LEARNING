"""UCT with tree expansion/backpropagation and deterministic beam planning."""

import math
from dataclasses import dataclass, field

import numpy as np

from .control import Control, lane_steering
from .planning import advance_visible, overlap_margin


def actions(world):
    current = min(
        range(len(world.lane_centers)), key=lambda i: abs(world.lane_centers[i] - world.ego.y)
    )
    return [
        (a, lane)
        for lane in range(max(0, current - 1), min(len(world.lane_centers), current + 2))
        for a in (-8.0, -3.0, 0.0, 2.0)
    ]


def transition(world, action, desired_speed):
    acceleration, lane = action
    state = world
    margin = math.inf
    terminal = False
    for _ in range(3):
        state = advance_visible(state, acceleration, lane, duration_s=0.2)
        margin = min(margin, overlap_margin(state.ego, state.vehicles))
        terminal = (
            margin <= 0
            or not min(state.lane_centers) - 0.5 <= state.ego.y <= max(state.lane_centers) + 0.5
        )
        if terminal:
            break
    reward = (state.ego.x - world.ego.x) / 15
    reward -= 0.04 * (state.ego.speed - desired_speed) ** 2
    reward -= 0.003 * acceleration**2
    for actor in state.vehicles:
        if actor.x > state.ego.x and abs(actor.y - state.ego.y) < 2.2:
            gap = actor.x - state.ego.x - (actor.length + state.ego.length) / 2
            reward -= 0.04 * max(0, 2 + 1.4 * state.ego.speed - gap) ** 2
    if terminal:
        reward -= 1000
    # Penalize approaching objects before a geometric collision occurs.
    if margin < 5:
        reward -= 0.5 * max(0, 5 - margin)
    return state, float(reward), terminal


@dataclass
class Node:
    world: object
    depth: int
    parent: object = None
    action: tuple | None = None
    reward: float = 0.0
    terminal: bool = False
    children: list = field(default_factory=list)
    untried: list = field(default_factory=list)
    visits: int = 0
    total: float = 0.0


class SearchController:
    def __init__(self, route, parameters, *, seed=1709):
        if route not in ("uct", "beam"):
            raise ValueError(route)
        self.route = route
        self.parameters = dict(parameters)
        self.generator = np.random.default_rng(seed)
        self.last_diagnostics = {}

    def _rollout_action(self, world):
        ego = world.ego
        lane = min(range(len(world.lane_centers)), key=lambda i: abs(world.lane_centers[i] - ego.y))
        desired = self.parameters["desired_speed_mps"]
        front = [v for v in world.vehicles if abs(v.y - ego.y) < 2.2 and v.x > ego.x]
        if front:
            lead = min(front, key=lambda v: v.x)
            gap = lead.x - ego.x - (lead.length + ego.length) / 2
            desired = min(desired, max(0, lead.speed + 0.6 * (gap - 2 - 1.4 * ego.speed)))
        acceleration = min(
            (-8.0, -3.0, 0.0, 2.0), key=lambda a: abs(a - np.clip(desired - ego.speed, -8, 2))
        )
        return acceleration, lane

    def _uct(self, world):
        p = self.parameters
        horizon = int(p["horizon_steps"])
        root = Node(world, 0, untried=actions(world))
        for _ in range(int(p["iterations"])):
            node = root
            path = [root]
            rewards = []
            while not node.terminal and node.depth < horizon:
                if node.untried:
                    action = node.untried.pop(int(self.generator.integers(len(node.untried))))
                    state, reward, terminal = transition(node.world, action, p["desired_speed_mps"])
                    child = Node(
                        state,
                        node.depth + 1,
                        parent=node,
                        action=action,
                        reward=reward,
                        terminal=terminal,
                        untried=actions(state),
                    )
                    node.children.append(child)
                    node = child
                    path.append(node)
                    rewards.append(reward)
                    break
                # The exploitation value excludes rewards already earned above this node.
                node = max(
                    node.children,
                    key=lambda c: (
                        c.total / c.visits
                        + p["exploration"] * math.sqrt(math.log(node.visits) / c.visits),
                        c.action,
                    ),
                )
                path.append(node)
                rewards.append(node.reward)
            state, depth, terminal = node.world, node.depth, node.terminal
            rollout = 0.0
            discount = 1.0
            while not terminal and depth < horizon:
                action = self._rollout_action(state)
                state, reward, terminal = transition(state, action, p["desired_speed_mps"])
                rollout += discount * reward
                discount *= 0.98
                depth += 1
            # Each edge reward belongs to the child value used by the parent's UCT.
            value = rollout
            for position in range(len(path) - 1, -1, -1):
                if position:
                    value = rewards[position - 1] + 0.98 * value
                path[position].visits += 1
                path[position].total += value
        best = max(root.children, key=lambda c: (c.visits, c.total / c.visits, c.action))
        stack = [root]
        maximum_depth = 0
        count = 0
        while stack:
            node = stack.pop()
            maximum_depth = max(maximum_depth, node.depth)
            count += 1
            stack.extend(node.children)
        self.last_diagnostics = {
            "route": "uct",
            "nodes": count,
            "maximum_tree_depth": maximum_depth,
            "root_visits": root.visits,
            "root_children": [
                {"action": c.action, "visits": c.visits, "mean_return": c.total / c.visits}
                for c in root.children
            ],
        }
        return best.action

    def _beam(self, world):
        p = self.parameters
        beam = [(0.0, (), world, False)]
        expanded = []
        for depth in range(int(p["horizon_steps"])):
            candidates = []
            for total, sequence, state, terminal in beam:
                if terminal:
                    candidates.append((total, sequence, state, True))
                    continue
                for action in actions(state):
                    next_state, reward, done = transition(state, action, p["desired_speed_mps"])
                    candidates.append(
                        (total + 0.98**depth * reward, sequence + (action,), next_state, done)
                    )
            expanded.append(len(candidates))
            candidates.sort(key=lambda row: (-row[0], row[1]))
            beam = candidates[: int(p["beam_width"])]
        self.last_diagnostics = {
            "route": "beam",
            "expanded_per_depth": expanded,
            "beam_width": int(p["beam_width"]),
            "best_sequence": beam[0][1],
        }
        return beam[0][1][0]

    def act(self, world):
        action = self._uct(world) if self.route == "uct" else self._beam(world)
        acceleration, lane = action
        return Control(acceleration, lane_steering(world.ego, world.lane_centers[lane]))
