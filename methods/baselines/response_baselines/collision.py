"""Matched fixed-pool collision baselines using only paid target feedback."""

import copy
import time

import numpy as np
import torch

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.baselines.pooled_ft.method import fine_tune
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.common.response import HistoricalPredictor, source_weights
from methods.gcrt.posterior import CollisionPosterior


METHODS = ("random", "space_filling", "history_ranking", "nearest_residual",
           "pooled_ft", "gp_collision", "fsbo")


def physical_covariance(x, lengthscale=0.3, scale=1.0):
    distance = np.linalg.norm(x[:, None] - x[None], axis=-1) / lengthscale
    r = np.sqrt(5) * distance
    return scale**2 * (1 + r + r**2 / 3) * np.exp(-r) + 1e-6 * np.eye(len(x))


def run(candidates, oracle, *, method, initial_indices, budget=128, seed=11,
        history=(), pretrained=None, adaptation_config=None, adaptation_steps=None):
    if method not in METHODS:
        raise ValueError(f"Unknown collision baseline: {method}")
    size = len(candidates)
    initial = list(initial_indices)
    if (len(set(initial)) != len(initial) or not 0 < len(initial) <= budget <= size
            or any(type(i) is not int or not 0 <= i < size for i in initial)):
        raise ValueError("Initialization must be unique and included in the total budget")
    if len({c.scenario_id for c in candidates}) != size:
        raise ValueError("Candidate identifiers must be unique")
    torch.set_num_threads(1)
    x = features(candidates)
    generator = np.random.default_rng(seed)
    random_order = generator.permutation(size).tolist()
    model, base_model, settings = None, None, {}
    if method in ("history_ranking", "nearest_residual"):
        historical = HistoricalPredictor(history)
        probabilities = historical.predict(x)
        if method == "history_ranking":
            # Rank by the most dangerous historical source; ties use the fixed ID.
            static_probability = probabilities.max(0)
        else:
            means = historical.latent_means(x)
            covariance = np.broadcast_to(physical_covariance(x), (len(history), size, size))
            model = CollisionPosterior(means, covariance, source_weights(history),
                                       component_names=[a.source_id for a in history])
    elif method in ("pooled_ft", "fsbo"):
        if pretrained is None or adaptation_config is None:
            raise ValueError("Neural baselines receive validated source weights and adaptation settings")
        base_model, settings = pretrained, {"config": adaptation_config}
        model = copy.deepcopy(base_model)
    elif method == "gp_collision":
        base_model = ClassificationKernel(x.shape[1])
        model = copy.deepcopy(base_model)
    remaining = set(range(size))
    selected, labels, queries = [], [], []
    for step in range(budget):
        started = time.perf_counter()
        if method in ("random", "space_filling"):
            q = None
        elif method == "history_ranking":
            q = static_probability
        elif method == "nearest_residual":
            q = model.probabilities()
        else:
            with torch.no_grad():
                q = (model(x).sigmoid().numpy() if method == "pooled_ft" else
                     collision_probabilities(model, x[selected], labels, x).numpy())
        available = sorted(remaining)
        if step < len(initial):
            index = initial[step]
        elif method == "random":
            index = next(i for i in random_order if i in remaining)
        elif method == "space_filling":
            distances = ((x[available, None] - x[selected][None])**2).sum(-1).min(-1)
            best = np.flatnonzero(distances >= distances.max() - 1e-12)
            index = min((available[i] for i in best), key=lambda i: candidates[i].scenario_id)
        else:
            maximum = max(q[i] for i in available)
            index = min((i for i in available if q[i] >= maximum - 1e-12),
                        key=lambda i: candidates[i].scenario_id)
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if not isinstance(feedback, QueryFeedback) or feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("Baseline feedback does not match the paid query")
        selected.append(index)
        labels.append(feedback.collision)
        remaining.remove(index)
        started = time.perf_counter()
        if method == "nearest_residual":
            model.observe(index, feedback.collision)
            if step + 1 == len(initial):
                best = int(np.argmax(model.weights))
                model.mean = model.mean[best:best + 1].copy()
                model.covariance = model.covariance[best:best + 1].copy()
                model.component_names = [model.component_names[best]]
                model.log_weight = np.zeros(1)
                model.fixed_weights = True
        elif method in ("pooled_ft", "fsbo", "gp_collision"):
            # Every round starts from the source initialization and all paid feedback.
            model = copy.deepcopy(base_model)
            config = settings.get("config", {})
            steps = adaptation_steps if adaptation_steps is not None else config.get("adaptation_steps", 10)
            if method == "pooled_ft":
                fine_tune(model, x[selected], labels, steps=steps,
                          learning_rate=config.get("adaptation_rate", 0.001))
            else:
                fit_target(model, x[selected], labels, steps=steps,
                           learning_rate=config.get("adaptation_rate", 0.01))
        adaptation_s = time.perf_counter() - started
        row = {"query": step + 1, "index": index, "feedback": feedback.to_dict(),
               "prequery_probability": float(q[index]) if q is not None else None,
               "selection_s": selection_s, "adaptation_s": adaptation_s,
               "reason": "common_initialization" if step < len(initial) else method}
        if method == "nearest_residual":
            row.update(source_names=list(model.component_names), source_weights=model.weights.tolist())
        queries.append(row)
    return {"method": method, "queries": queries, "selected_indices": selected,
            "actual_queries": len(queries), "maximum_queries": budget,
            "termination": "budget_exhausted", "feedback_used": "paid collision labels"}
