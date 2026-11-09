"""Evaluator statistics for fixed candidate libraries, using measured collision IDs."""

import math

import numpy as np

from .scenarios import DOMAINS, validate

METRIC_CONTRACT = "candidate-id-and-normalized-parameter-distance"


def parameter_distances(scenes, others):
    """Gower distance: each original physical or categorical attribute has equal weight."""
    if not scenes or not others:
        raise ValueError("Parameter distances require nonempty scene lists")
    kind = scenes[0].kind
    if any(scene.kind != kind for scene in (*scenes, *others)):
        raise ValueError("Parameter distances require one functional scenario")
    domain = DOMAINS[kind]
    distances = np.zeros((len(scenes), len(others)), dtype=float)
    for name, (lower, upper) in domain.bounds.items():
        left = np.array([s.parameters[name] for s in scenes])
        right = np.array([s.parameters[name] for s in others])
        distances += np.abs(left[:, None] - right) / (upper - lower)
    for name in domain.categories:
        left = np.array([s.categories[name] for s in scenes])
        right = np.array([s.categories[name] for s in others])
        distances += left[:, None] != right
    return distances / (len(domain.bounds) + len(domain.categories))


def discovery_distances(scenes):
    """Distances for every ordered discovery prefix; no full candidate distance matrix."""
    pairs, neighbors = [], []
    nearest = np.empty(0)
    total = 0.0
    for i, scene in enumerate(scenes):
        if i == 0:
            nearest = np.array([np.inf])
            pairs.append(None)
            neighbors.append(None)
            continue
        distances = parameter_distances([scene], scenes[:i])[0]
        total += float(distances.sum())
        nearest = np.append(np.minimum(nearest, distances), distances.min())
        pairs.append(2 * total / (i * (i + 1)))
        neighbors.append(float(nearest.mean()))
    return {"pair_distance": pairs, "nearest_neighbor_distance": neighbors}


def summarize(candidates, labels, queries, trajectories, *, initial_count, query_ledger):
    """Only the evaluator receives the exhaustive labels and measured execution traces."""
    if not candidates or len(labels) != len(candidates) or any(type(y) is not bool for y in labels):
        raise ValueError("Metrics require one measured Boolean for every candidate")
    identities = [s.scenario_id for s in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("Candidate IDs must be unique")
    if len({s.kind for s in candidates}) != 1:
        raise ValueError("Metrics require one functional scenario per task")
    for scene in candidates:
        validate(scene)
    if not queries or not len(queries) == len(trajectories) == len(query_ledger):
        raise ValueError("Each paid query requires a matching trajectory and ledger entry")
    if not 0 <= initial_count <= len(queries):
        raise ValueError("Initialization must fit the paid query budget")

    discovered, found, counts = [], set(), []
    first = None
    brier, nll = [], []
    timings = {name: [] for name in ("execution", "selection", "adaptation", "rss", "replay")}
    negative_rss = 0
    for position, (query, trace, record) in enumerate(zip(queries, trajectories, query_ledger), 1):
        index = query["index"]
        if type(index) is not int or not 0 <= index < len(candidates):
            raise ValueError("Query index is outside the candidate library")
        feedback = query["feedback"]
        if (
            feedback["scenario_id"] != identities[index]
            or type(feedback["collision"]) is not bool
            or feedback["collision"] != labels[index]
        ):
            raise ValueError("Paid feedback differs from the complete candidate truth")
        if not trace or any(bool(frame["collision"]) for frame in trace) != labels[index]:
            raise ValueError("Collision label differs from the actual execution trajectory")
        if (
            record["status"] != "complete"
            or record["index"] != index
            or record["feedback"] != feedback
        ):
            raise ValueError("Query differs from its paid execution ledger")
        if labels[index] and identities[index] not in found:
            first = position if first is None else first
            found.add(identities[index])
            discovered.append(index)
        counts.append(len(found))
        probability = query.get("prequery_probability")
        if probability is not None:
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Query-before-feedback probability must be in [0, 1]")
            brier.append((probability - int(labels[index])) ** 2)
            probability = np.clip(probability, 1e-12, 1 - 1e-12)
            nll.append(-float(np.log(probability if labels[index] else 1 - probability)))
        margin = feedback["safety"].get("rss_margin")
        negative_rss += int(margin is not None and margin < 0)
        timings["execution"].append(feedback["execution_s"])
        timings["selection"].append(query.get("selection_s", 0.0))
        timings["adaptation"].append(query.get("adaptation_s", 0.0))
        timings["rss"].append(record.get("rss_compute_s", 0.0))
        timings["replay"].append(record["wall_s"])

    distance_prefixes = discovery_distances([candidates[i] for i in discovered])
    distance_curves = {
        name: [values[n - 1] if n else None for n in counts]
        for name, values in distance_prefixes.items()
    }
    failures = sum(labels)
    recall = [n / failures if failures else None for n in counts]
    initial_failures = counts[initial_count - 1] if initial_count else 0
    cumulative = {name: np.cumsum(values).tolist() for name, values in timings.items()}
    equivalent = np.sum(
        [cumulative[name] for name in ("execution", "selection", "adaptation", "rss")], axis=0
    ).tolist()
    return {
        "metric_contract": METRIC_CONTRACT,
        "scenario": candidates[0].kind,
        "candidate_count": len(candidates),
        "queries": len(queries),
        "budget_fraction": len(queries) / len(candidates),
        "initial_count": initial_count,
        "initial_collisions": initial_failures,
        "unique_collisions": len(found),
        "collision_executions": sum(q["feedback"]["collision"] for q in queries),
        "discovered_ids": [identities[i] for i in discovered],
        "discovery_curve": counts,
        "discovery_auc": float(np.mean(counts)),
        "post_initial_collisions": len(found) - initial_failures,
        "post_initial_curve": [
            n - initial_failures if t >= initial_count else None for t, n in enumerate(counts, 1)
        ],
        "pool_unique_collisions": failures,
        "pool_failure_rate": failures / len(candidates),
        "collision_recall": recall[-1],
        "recall_curve": recall,
        "recall_auc": float(np.mean(recall)) if failures else None,
        "pair_distance": distance_curves["pair_distance"][-1],
        "nearest_neighbor_distance": distance_curves["nearest_neighbor_distance"][-1],
        "distance_curves": distance_curves,
        "distances_by_discovery_count": distance_prefixes,
        "first_collision_query": first,
        "restricted_first_collision_queries": first if first is not None else len(queries),
        "no_collision_within_budget": first is None,
        "queries_to_recall": {
            str(level): next(
                (i for i, r in enumerate(recall, 1) if r is not None and r >= level), None
            )
            for level in (0.25, 0.5, 0.75, 1.0)
        },
        "collision_nll": float(np.mean(nll)) if nll else None,
        "brier": float(np.mean(brier)) if brier else None,
        "prediction_count": len(brier),
        "negative_rss_executions": negative_rss,
        "equivalent_execution_s": sum(timings["execution"]),
        "selection_s": sum(timings["selection"]),
        "adaptation_s": sum(timings["adaptation"]),
        "rss_compute_s": sum(timings["rss"]),
        "replay_wall_s": sum(timings["replay"]),
        "cumulative_costs_s": cumulative,
        "equivalent_testing_curve_s": equivalent,
    }


def equal_count_distances(summaries):
    """Compare the first k discoveries, retaining tasks without a comparable pair."""
    if not summaries or len({s["scenario"] for s in summaries}) != 1:
        raise ValueError("Equal-count comparison requires methods from the same scenario")
    common = min(s["unique_collisions"] for s in summaries)
    return {
        "common_maximum": common,
        "counts": list(range(2, common + 1)),
        "methods": [
            {name: values[1:common] for name, values in s["distances_by_discovery_count"].items()}
            for s in summaries
        ],
    }


def aggregate(summaries):
    """Equal-task means and separately labelled pooled recall; never pool coordinates."""
    if not summaries:
        raise ValueError("Aggregation requires completed tasks")
    means = {}
    for name in (
        "unique_collisions",
        "post_initial_collisions",
        "discovery_auc",
        "collision_recall",
        "recall_auc",
        "pair_distance",
        "nearest_neighbor_distance",
        "collision_nll",
        "brier",
    ):
        values = [s[name] for s in summaries if s[name] is not None]
        means[name] = {"mean": float(np.mean(values)) if values else None, "tasks": len(values)}
    total = sum(s["pool_unique_collisions"] for s in summaries)
    return {
        "tasks": len(summaries),
        "zero_failure_tasks": sum(s["pool_unique_collisions"] == 0 for s in summaries),
        "distance_undefined_tasks": sum(s["unique_collisions"] < 2 for s in summaries),
        "equal_task_means": means,
        "pooled_recall": sum(s["unique_collisions"] for s in summaries) / total if total else None,
    }
