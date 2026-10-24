"""Fixed-budget collision discovery with binary conditional lookahead."""

import time

import numpy as np
import torch

from benchmark.schema import QueryFeedback
from methods.common.classification import NODES, WEIGHTS, site_derivatives


def acquisition_values(model, available, horizon, *, device=None):
    available = np.asarray(available, dtype=int)
    if not 1 <= horizon <= len(available):
        raise ValueError("Lookahead must fit the remaining pool and budget")
    q = model.probabilities()
    if horizon == 1:
        return q[available]
    if device is None:
        device = "cuda" if len(available) >= 128 and torch.cuda.is_available() else "cpu"
    if device != "cpu":
        return batched_acquisition(model, available, horizon, device=device)
    values = []
    for index in available:
        future = available[available != index]
        gains = []
        for label in (False, True):
            p = model.conditional_probabilities(index, label, future)
            gains.append(np.partition(p, len(p) - horizon + 1)[-horizon + 1:].sum())
        values.append(q[index] + (1 - q[index]) * gains[0] + q[index] * gains[1])
    return np.asarray(values)


def batched_acquisition(model, available, horizon, *, device):
    """Enumerate every candidate and both labels, without pruning the decision set."""
    mean = torch.as_tensor(model.mean[:, available], dtype=torch.float64, device=device)
    covariance = torch.as_tensor(model.covariance[:, available][:, :, available],
                                 dtype=torch.float64, device=device)
    variance = covariance.diagonal(dim1=1, dim2=2)
    cross = covariance.permute(2, 0, 1)
    q = torch.as_tensor(model.probabilities(available), dtype=torch.float64, device=device)
    weights = torch.as_tensor(model.log_weights_at(available).copy(), dtype=torch.float64, device=device)
    influence = torch.as_tensor(model.reliability_influence(available, available),
                                dtype=torch.float64, device=device)
    values = q.clone()
    positions = torch.arange(len(available), device=device)
    for label in (False, True):
        logp, first, second = site_derivatives(
            model.mean[:, available],
            model.covariance.diagonal(axis1=1, axis2=2)[:, available], label)
        gradient = torch.as_tensor(first.T, dtype=torch.float64, device=device)
        curvature = torch.as_tensor(second.T, dtype=torch.float64, device=device)
        conditional_mean = mean[None] + gradient[:, :, None] * cross
        conditional_sd = (2 * (variance[None] + curvature[:, :, None] * cross.square()).clamp_min(0)).sqrt()
        probability = torch.zeros_like(conditional_mean)
        for node, weight in zip(NODES, WEIGHTS):
            probability += float(weight) * torch.sigmoid(conditional_mean + float(node) * conditional_sd)
        if model.fixed_weights:
            conditional_weights = weights.exp()[None]
        else:
            likelihood = torch.as_tensor(logp.T, dtype=torch.float64, device=device)
            conditional_weights = torch.softmax(
                weights[None] + likelihood[:, :, None] * influence[:, None, :], dim=1)
        conditional = (probability * conditional_weights).sum(1)
        conditional[positions, positions] = -torch.inf
        gains = conditional.topk(horizon - 1, dim=1).values.sum(1)
        values += (q if label else 1 - q) * gains
    return values.cpu().numpy()


def choose(model, available, scenario_ids, horizon):
    available = np.asarray(available, dtype=int)
    values = acquisition_values(model, available, horizon)
    best = min(np.flatnonzero(values >= values.max() - 1e-12),
               key=lambda i: scenario_ids[available[i]])
    return int(available[best]), float(values[best])


def select(model, candidates, oracle, *, initial_indices, budget=128,
           horizon=5, greedy=False):
    size = len(candidates)
    ids = [s.scenario_id for s in candidates]
    initial = list(initial_indices)
    if (len(set(ids)) != size or len(set(initial)) != len(initial)
            or not 0 < len(initial) <= budget <= size
            or any(type(i) is not int or not 0 <= i < size for i in initial)
            or horizon < 1):
        raise ValueError("Invalid candidates, common initialization or budget")
    if model.observed or model.mean.shape[1] != size:
        raise ValueError("A task needs a fresh posterior matching its public candidates")
    remaining = set(range(size))
    queries = []
    for step in range(budget):
        started = time.perf_counter()
        h = 1 if greedy else min(horizon, budget - step)
        if step < len(initial):
            index, value = initial[step], None
            reason = "common_initialization"
        else:
            index, value = choose(model, sorted(remaining), ids, h)
            reason = "collision_greedy" if greedy else "binary_lookahead"
        p = float(model.probabilities([index])[0])
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if not isinstance(feedback, QueryFeedback) or feedback.scenario_id != ids[index]:
            raise ValueError("Query returned feedback for a different scenario")
        started = time.perf_counter()
        evidence = model.observe(index, feedback.collision)
        adaptation_s = time.perf_counter() - started
        remaining.remove(index)
        queries.append({"query": step + 1, "index": index, "feedback": feedback.to_dict(),
                        "prequery_probability": p,
                        "source_weights": np.exp(model.log_weights_at([index])[:, 0]).tolist(),
                        "source_names": list(model.component_names),
                        "log_evidence": evidence, "selection_s": selection_s,
                        "adaptation_s": adaptation_s, "horizon": h,
                        "acquisition": value, "reason": reason})
    return {"queries": queries, "selected_indices": [q["index"] for q in queries],
            "actual_queries": len(queries), "maximum_queries": budget,
            "feedback_used": "paid collision labels; auxiliary measurements available",
            "termination": "budget_exhausted"}
