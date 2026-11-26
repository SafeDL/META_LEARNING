"""SPECTRE's five public fitness components and epsilon-IBEA for a paid fixed suite."""

import hashlib

import numpy as np
from scipy.spatial.distance import cdist

from testing_methods.baselines.detour.method import parameter_features


DEMAND_RULES = {
    "car_following_braking_onset": [
        ["ego_speed_mps", ">=", 27], ["closing_speed_mps", ">=", 6],
        ["gap_m", "<=", 36], ["deceleration_mps2", ">=", 6]],
    "cut_in_onset": [
        ["ego_speed_mps", ">=", 27], ["closing_speed_mps", ">=", 6],
        ["gap_m", "<=", 25], ["lateral_duration_s", "<=", 2]],
    "cut_out_rear_gap": [
        ["ego_speed_mps", ">=", 29], ["obstacle_distance_m", "<=", 130],
        ["rear_gap_m", "<=", 40]],
    "side_encroachment": [
        ["ego_speed_mps", ">=", 27], ["longitudinal_offset_m", "abs<=", 4],
        ["lateral_duration_s", "<=", 2]],
}

SETTINGS = {
    "upstream": "83613463676a0c3fa629cb1399541830499d48bf",
    "optimizer": "epsilon-IBEA", "population": 100, "offspring": 100,
    "evaluations_per_search": 30000, "kappa": 1.0, "crossover_probability": 0.9,
    "distribution_index": 20, "mutation": "1 / block length", "seed": 11,
    "block_size": 200, "neighbors": 5, "distance": "unit-range Euclidean",
    "collision": "expected native collision count",
    "proximity": "clipped negative RSS margin; potential interval (0.3, 0.9]",
    "demand": DEMAND_RULES, "high_demand": "strict majority of applicable rules",
    "diversity": "within-subset mean pairwise unit-range L1 / dimension",
    "position": "equal shares of collision, proximity, demand count and mean suite distance",
    "integer_mutation_rounding": "nearest, ties to even",
    "front_choice": "collision count, position score, potential count, demand count, diversity",
    "within_block": "descending unrounded positional utility, then suite index",
}


def demand_count(scene):
    count = 0
    for name, direction, threshold in DEMAND_RULES[scene.kind]:
        value = scene.parameters[name]
        if direction == "abs<=":
            value = abs(value)
        count += (value >= threshold if direction == ">=" else value <= threshold)
    return count


def attributes(candidates, history, observed, x):
    """Estimate candidate properties once using only admitted and prefix responses."""
    samples = [(s, y, r) for a in history
               for s, y, r in zip(a.scenarios, a.collision, a.continuous_margin, strict=True)]
    samples.extend((candidates[i], f.collision, f.safety.rss_margin) for i, f in observed)
    train = np.stack([parameter_features(s) for s, _, _ in samples])
    collision = np.array([y for _, y, _ in samples])
    deficit = np.clip([-r for _, _, r in samples], 0, 1)
    labels = np.column_stack((collision, deficit, (deficit > .3) & (deficit <= .9)))
    distances = cdist(x, train)
    # Include all observations tied at the k-th distance, including duplicate historical coordinates.
    k = min(SETTINGS["neighbors"], len(train))
    radius = np.partition(distances, k - 1, axis=1)[:, k - 1, None]
    weights = np.where(distances <= radius + 1e-12, 1 / np.maximum(distances, 1e-12), 0)
    exact = distances == 0
    weights = np.where(exact.any(axis=1, keepdims=True), exact, weights)
    probability, proximity, potential = ((weights / weights.sum(axis=1, keepdims=True)) @ labels).T
    diversity = cdist(x, x, "cityblock").sum(axis=1) / (max(1, len(x) - 1) * x.shape[1])
    demand = np.array([demand_count(s) for s in candidates])
    high_demand = 2 * demand > len(DEMAND_RULES[candidates[0].kind])
    priority = (probability + proximity + demand + diversity) / 4
    return np.column_stack((probability, potential, high_demand, priority))


def fitness(population, values, features):
    """Indices in each row are already ordered by descending positional utility."""
    length = population.shape[1]
    selected = values[population]
    weights = (length + 1 - np.arange(length)) / length
    # Within each sorted dimension x[i] occurs i times as the upper endpoint
    # and L-1-i times as the lower endpoint of an unordered pair.
    ordered = np.sort(features[population], axis=1)
    coefficients = 2 * np.arange(length) - length + 1
    total_distance = np.einsum("pld,l->p", ordered, coefficients)
    mean_distance = total_distance / (max(1, length * (length - 1) // 2) * features.shape[1])
    return np.column_stack((selected[:, :, :3].sum(axis=1),
                            np.round(mean_distance, 3),
                            np.round((selected[:, :, 3] * weights).mean(axis=1), 3)))


def survive(scores, count):
    # impact[i,j] is the additive-epsilon fitness penalty competitor j imposes on i.
    epsilon = (scores[:, None, :] - scores[None, :, :]).max(axis=2)
    impact = np.exp(-epsilon / SETTINGS["kappa"])
    np.fill_diagonal(impact, 0)
    penalty = impact.sum(axis=1)
    active = np.ones(len(scores), dtype=bool)
    for _ in range(len(scores) - count):
        worst = np.argmax(np.where(active, penalty, -np.inf))
        active[worst] = False
        penalty -= impact[:, worst]
    keep = np.flatnonzero(active)
    return keep, impact[np.ix_(keep, keep)].sum(axis=1)


def offspring(population, penalty, size, rng):
    """Bounded integer SBX and polynomial mutation, followed by no-repeat repair."""
    n, length = population.shape
    pairs = rng.integers(n, size=(n, 2))
    a, b = pairs.T
    parents = population[np.where(penalty[a] <= penalty[b], a, b)]
    left, right = parents[::2], parents[1::2]
    lo, hi = np.minimum(left, right), np.maximum(left, right)
    gap = np.maximum(hi - lo, 1)
    eta = SETTINGS["distribution_index"] + 1
    u = rng.random(lo.shape)

    def spread(distance):
        alpha = 2 - (1 + 2 * distance / gap) ** -float(eta)
        return np.where(u <= 1 / alpha, (u * alpha) ** (1 / eta),
                        (1 / (2 - u * alpha)) ** (1 / eta))

    cross = ((rng.random((n // 2, 1)) < SETTINGS["crossover_probability"])
             & (rng.random(lo.shape) <= .5) & (hi > lo))
    low = (lo + hi - spread(lo) * gap) * .5
    high = (lo + hi + spread(size - 1 - hi) * gap) * .5
    swap = rng.random(lo.shape) < .5
    children = parents.copy()
    children[::2] = np.where(cross, np.where(swap, low, high), left)
    children[1::2] = np.where(cross, np.where(swap, high, low), right)
    rows, cols = np.nonzero(rng.random(children.shape) < 1 / length)
    y = children[rows, cols] / (size - 1)
    u = rng.random(len(y))
    down = (2 * u + (1 - 2 * u) * (1 - y) ** eta).clip(min=0) ** (1 / eta) - 1
    up = 1 - (2 * (1 - u) + (2 * u - 1) * y ** eta).clip(min=0) ** (1 / eta)
    children[rows, cols] = np.rint(np.clip((y + np.where(u < .5, down, up)) * (size - 1), 0, size - 1))
    for row in children:
        present, first = np.unique(row, return_index=True)
        if len(present) < length:
            duplicates = np.ones(length, dtype=bool)
            duplicates[first] = False
            available = np.ones(size, dtype=bool)
            available[present] = False
            row[duplicates] = rng.choice(np.flatnonzero(available), duplicates.sum(), replace=False)
    return children


def select_block(values, features, length, rng):
    """The local candidate IDs are in descending priority, breaking ties by suite index."""
    size = len(values)
    if size <= length:
        return np.arange(size)
    count = SETTINGS["population"]
    population = np.array([rng.choice(size, length, replace=False) for _ in range(count)])
    # Include the exact maximum of the fifth objective as a feasible starting solution.
    population[0] = np.arange(length)
    population.sort(axis=1)
    scores = fitness(population, values, features)
    _, penalty = survive(scores, count)
    for _ in range(SETTINGS["evaluations_per_search"] // count - 1):
        children = offspring(population, penalty, size, rng)
        children.sort(axis=1)
        scores = np.vstack((scores, fitness(children, values, features)))
        keep, penalty = survive(scores, count)
        population = np.vstack((population, children))[keep]
        scores = scores[keep]
    # Lexicographic maximization with positive direction on all five objectives
    # selects a non-dominated survivor. Equal scores retain a deterministic order.
    winner = np.lexsort(tuple(-scores[:, j] for j in (3, 2, 1, 4, 0)))[0]
    return population[winner]


def prioritize(candidates, history, observed):
    features = np.stack([parameter_features(s) for s in candidates])
    values = attributes(candidates, history, observed, features)
    remaining = np.ones(len(values), dtype=bool)
    remaining[[i for i, _ in observed]] = False
    indices = np.flatnonzero(remaining)
    indices = indices[np.argsort(-values[indices, 3], kind="stable")]
    rng = np.random.default_rng(SETTINGS["seed"])
    order = []
    while len(indices):
        block = select_block(values[indices], features[indices], SETTINGS["block_size"], rng)
        order.extend(indices[block].tolist())
        indices = np.delete(indices, block)
    return order


def order_digest(order):
    return hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()
