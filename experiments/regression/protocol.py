"""Coordinates, stopping rules and reset selection for a policy reference chain."""

from math import prod

import numpy as np

from experiments.formal_scenarios import FREE_BOUNDS
from experiments.scenarios import DOMAINS, validate
from experiments.schema import Scenario

COLLISION_TARGET = 100
QUERY_LIMIT = 500
INITIAL_COUNT = 20
TRAINING_STEPS = 131072
DQN_SEED = 6119
DQN_WARMUP = 2048
DQN_DECAY = 26214


def numeric_coordinate(scene):
    return tuple(scene.parameters[name] for name in FREE_BOUNDS[scene.kind])


def sample_block(kind, count, seed, used, label):
    bounds = FREE_BOUNDS[kind]
    sizes = [high - low + 1 for low, high in bounds.values()]
    generator = np.random.default_rng(seed)
    coordinates = []
    if count > prod(sizes) - len(used):
        raise ValueError("Insufficient unused physical coordinates")
    while len(coordinates) < count:
        index = int(generator.integers(prod(sizes)))
        values = []
        for (low, _), size in zip(bounds.values(), sizes):
            index, digit = divmod(index, size)
            values.append(low + digit)
        point = tuple(values)
        if point not in used:
            used.add(point)
            coordinates.append(point)
    categories = {
        name: np.random.default_rng(seed).permutation(
            np.tile(options, (count + len(options) - 1) // len(options))[:count]
        )
        for name, options in DOMAINS[kind].categories.items()
    }
    scenes = [Scenario(
        f"regression:{kind}:{label}:{i:04d}", kind, dict(zip(bounds, point)),
        {name: str(values[i]) for name, values in categories.items()},
    ) for i, point in enumerate(coordinates)]
    for scene in scenes:
        validate(scene)
    return scenes


def repair_subset(scenes, limit=COLLISION_TARGET):
    """Deterministic farthest points; category disagreement adds unit distance."""
    unique = {scene.coordinate(): scene for scene in scenes}
    ordered = sorted(unique.values(), key=lambda s: (
        numeric_coordinate(s), tuple(sorted(s.categories.items()))))
    if len(ordered) <= limit:
        return ordered
    bounds = FREE_BOUNDS[ordered[0].kind]
    x = np.array([[(s.parameters[n] - low) / (high - low)
                   for n, (low, high) in bounds.items()] for s in ordered])
    categories = [tuple(sorted(s.categories.items())) for s in ordered]
    selected = [0]
    distance = np.full(len(ordered), np.inf)
    while len(selected) < limit:
        last = selected[-1]
        updated = ((x - x[last]) ** 2).sum(1)
        updated += np.array([c != categories[last] for c in categories])
        distance = np.minimum(distance, updated)
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return [ordered[i] for i in selected]


def stop_reason(collisions, queries):
    if collisions >= COLLISION_TARGET:
        return "collision_target_reached"
    if queries >= QUERY_LIMIT:
        return "query_limit_reached"
    return None
