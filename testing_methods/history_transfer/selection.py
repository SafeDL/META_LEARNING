"""Matched H-by-F controls with global reliability and a shared historical teacher."""

import copy
import time

import numpy as np
import torch

from experiments.scenarios import features
from testing_methods.common.gp_classifier import ClassificationKernel, fit_target
from testing_methods.selection import validate_initial_indices

from .inference import component_probabilities, residual_prior

MODES = ("H0F0", "H1F0", "H0F1", "H1F1")


def aggregate_history(sources):
    """A single Gaussian teacher preserving the mixture's latent mean and total covariance."""
    weights = np.asarray(sources["weights"], dtype=float)
    means, covariances = np.asarray(sources["means"]), np.asarray(sources["covariances"])
    mean = weights @ means
    centered = means - mean
    covariance = np.einsum("k,kij->ij", weights, covariances)
    covariance += np.einsum("k,ki,kj->ij", weights, centered, centered)
    return {
        "names": ["aggregate_history_teacher"],
        "weights": np.ones(1),
        "means": mean[None],
        "covariances": covariance[None],
    }


def select(
    candidates,
    oracle,
    *,
    pretrained,
    sources,
    mode,
    initial_indices,
    budget=128,
    device=None,
    feature_fn=features,
):
    if mode not in MODES:
        raise ValueError("Unknown H-by-F cell")
    initial = list(initial_indices)
    validate_initial_indices(len(candidates), initial, budget)
    if not sources or (mode.startswith("H0") and sources["names"] != ["aggregate_history_teacher"]):
        raise ValueError(
            "H0 receives only the aggregate teacher; H1 receives explicit source priors"
        )
    weights = np.asarray(sources["weights"])
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Historical prior weights must form a positive probability distribution")
    torch.set_num_threads(1)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = copy.deepcopy(pretrained).to(device).double()
    model = copy.deepcopy(base)
    x = feature_fn(candidates)
    physical_base = ClassificationKernel(x.shape[1]).to(device)
    physical = copy.deepcopy(physical_base)
    names = [*sources["names"], "independent_target", "physical_target"]
    historical = {
        key: torch.as_tensor(sources[key], dtype=torch.float64, device=device)
        for key in ("weights", "means", "covariances")
    }
    prior = torch.cat((0.75 * historical["weights"], base.mean.new_tensor([0.25])))
    scores = prior.log()
    reference = float(base.mean.detach())
    feedback_enabled = mode.endswith("F1")
    selected, labels, queries = [], [], []
    remaining = set(range(len(candidates)))
    cached = None
    for step in range(budget):
        started = time.perf_counter()
        if cached is None or feedback_enabled:
            paid, used_labels = (selected, labels) if feedback_enabled else ([], [])
            with torch.no_grad():
                means, covariances = residual_prior(
                    model.mean,
                    model.covariance(x),
                    historical["means"],
                    historical["covariances"],
                    reference,
                )
                learned, diagnostics = component_probabilities(
                    means.cpu(), covariances.cpu(), paid, used_labels
                )
                raw, physical_info = component_probabilities(
                    physical.mean.expand(1, len(x)).cpu(),
                    physical.covariance(x)[None].cpu(),
                    paid,
                    used_labels,
                )
                group_weights = scores.cpu().softmax(0)
                components = torch.cat((learned, raw))
                effective = torch.cat((0.9 * group_weights, group_weights.new_tensor([0.1])))
                probability = (effective[:, None] * components).sum(0).numpy()
            cached = probability, components, effective, diagnostics, physical_info[0]
        probability, components, effective, diagnostics, physical_info = cached
        if step < len(initial):
            index = initial[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min(
                (i for i in remaining if probability[i] >= maximum - 1e-12),
                key=lambda i: candidates[i].scenario_id,
            )
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("H-by-F feedback differs from the paid candidate")
        before = effective.tolist()
        selected.append(index)
        remaining.remove(index)
        started = time.perf_counter()
        likelihood = None
        if feedback_enabled:
            labels.append(feedback.collision)
            p = components[:-1, index].to(device)
            log_likelihood = (p if feedback.collision else 1 - p).clamp_min(1e-15).log()
            scores = torch.log_softmax(scores + log_likelihood, dim=0)
            likelihood = log_likelihood.cpu().tolist()
            model = copy.deepcopy(base)
            fit_target(model, x[selected], labels, steps=10, learning_rate=0.003)
            physical = copy.deepcopy(physical_base)
            fit_target(physical, x[selected], labels, steps=10, learning_rate=0.01)
            if model.mean.is_cuda:
                torch.cuda.synchronize(model.mean.device)
        after = torch.cat((0.9 * scores.cpu().softmax(0), scores.new_tensor([0.1]).cpu())).tolist()
        queries.append(
            {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                "prequery_probability": float(probability[index]),
                "prequery_component_probabilities": components[:, index].tolist(),
                "prequery_component_weights": before,
                "component_weights_after_query": after,
                "prequery_log_likelihood": likelihood,
                "conditional_modes": diagnostics,
                "physical_mode": physical_info,
                "conditioning_queries": step if feedback_enabled else 0,
                "selection_s": selection_s,
                "adaptation_s": time.perf_counter() - started,
                "reason": "common_initialization" if step < len(initial) else mode,
            }
        )
    return {
        "method": mode,
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "component_names": names,
        "explicit_source_components": mode.startswith("H1"),
        "target_feedback_updates": feedback_enabled,
        "global_reliability": True,
        "group_prior_weights": prior.cpu().tolist(),
        "physical_mass": 0.1,
        "beta": 1.0,
        "feedback_used": "paid collision Boolean" if feedback_enabled else "paid indices only",
        "conditional_inference_device": "cpu",
        "kernel_and_adaptation_device": str(device),
        "online_feature_precision": "float64",
    }
