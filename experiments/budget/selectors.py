"""E02 selectors: existing methods, active Pooled GP and fixed initial ranking."""

import copy
import time

import numpy as np
import torch

from testing_methods.baselines.response_baselines.pooled_gp import predict_pooled
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel

from experiments.baselines import BASELINES
from experiments.costs import measure_resources, synchronized_time
from experiments.rss import measurement_contract
from experiments.scenarios import features


def greedy_index(candidates, probability, remaining):
    maximum = max(probability[i] for i in remaining)
    return min((i for i in remaining if probability[i] >= maximum - 1e-12),
               key=lambda i: candidates[i].scenario_id)


def static_order(candidates, probability, initial, budget):
    order = list(initial)
    remaining = set(range(len(candidates))) - set(order)
    while len(order) < budget:
        index = greedy_index(candidates, probability, remaining)
        order.append(index)
        remaining.remove(index)
    return order


def pooled_select(candidates, oracle, *, pretrained, history, initial_indices,
                  budget, fit_settings, snapshot_counts, device="cuda"):
    base = copy.deepcopy(pretrained).double().to(device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    x = features(candidates)
    selected, labels, margins, queries, snapshots = [], [], [], [], []
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = synchronized_time(device)
        # Initialization is prescribed: fit the same GP but predict only its next point.
        subset = [initial_indices[step]] if step < len(initial_indices) and step not in snapshot_counts else None
        probability, parameters, diagnostic = predict_pooled(
            base, np.concatenate((history["x"], x[selected])),
            np.concatenate((history["collision"], labels)),
            np.concatenate((history["rss_margin"], margins)),
            x if subset is None else x[subset], batch_size=128,
            inference_backend="grouped", fit_settings=fit_settings,
        )
        probability = np.asarray(probability)
        if not np.isfinite(probability).all():
            raise FloatingPointError("Nonfinite Pooled GP acquisition")
        fit_s = synchronized_time(device) - started
        started = synchronized_time(device)
        index = initial_indices[step] if step < len(initial_indices) else greedy_index(candidates, probability, remaining)
        if step in snapshot_counts:
            snapshots.append({"paid_queries": step, "observed_indices": list(selected),
                              "probabilities": probability.tolist(),
                              "target_parameters": parameters, "conditional_mode": diagnostic})
        selection_s = synchronized_time(device) - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial_indices) else "target")
        if (feedback.scenario_id != candidates[index].scenario_id
                or feedback.rss_contract != measurement_contract()
                or feedback.safety.rss_margin is None):
            raise ValueError("Pooled GP needs matching paid collision and RSS feedback")
        selected.append(index)
        remaining.remove(index)
        labels.append(feedback.collision)
        margins.append(feedback.safety.rss_margin)
        queries.append({"query": step + 1, "index": index, "feedback": feedback.to_dict(),
                        "prequery_probability": float(probability[index if subset is None else 0]),
                        "conditioning_queries": step, "target_parameters": parameters,
                        "conditional_mode": diagnostic, "adaptation_s": fit_s,
                        "selection_s": selection_s,
                        "reason": "common_initialization" if step < len(initial_indices) else "greedy_pooled"})
    return {"method": "pooled_history", "queries": queries, "selected_indices": selected,
            "actual_queries": budget, "maximum_queries": budget, "termination": "budget_exhausted",
            "historical_count": len(history["collision"]), "fit_settings": fit_settings.to_dict(),
            "prediction_snapshots": snapshots, "inference_backend": "grouped",
            "feedback_used": "joint", "conditional_inference_device": device}


@measure_resources
def run(candidates, oracle, *, method, initial_indices, budget, device="cpu", **inputs):
    torch.set_num_threads(1)
    if method in ("full", "target_only"):
        options = {} if method == "full" else {"raw_kernel_type": SharedPrivateKernel}
        result = select(candidates, oracle, initial_indices=initial_indices, budget=budget,
                        device=device, **inputs, **options)
    elif method == "pooled_history":
        result = pooled_select(candidates, oracle, initial_indices=initial_indices,
                               budget=budget, device=device, **inputs)
    elif method == "static_prior":
        started = time.perf_counter()
        order = static_order(candidates, inputs["probabilities"], initial_indices, budget)
        sorting_s = time.perf_counter() - started
        queries = []
        for step, index in enumerate(order):
            feedback = oracle.query(index, purpose="initial" if step < len(initial_indices) else "target")
            queries.append({"query": step + 1, "index": index, "feedback": feedback.to_dict(),
                            "prequery_probability": float(inputs["probabilities"][index]),
                            "selection_s": sorting_s if step == 0 else 0.0,
                            "adaptation_s": 0.0, "conditioning_queries": 0,
                            "reason": "common_initialization" if step < len(initial_indices) else "fixed_k0"})
        result = {"method": method, "queries": queries, "selected_indices": order,
                  "actual_queries": budget, "maximum_queries": budget, "termination": "budget_exhausted",
                  "fixed_order": order, "sorting_s": sorting_s, "target_feedback_updates": False}
    else:
        result = BASELINES[method].run(candidates, oracle, initial_indices=initial_indices,
                                      budget=budget, **inputs)
    return result
