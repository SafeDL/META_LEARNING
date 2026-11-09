"""Budgeted collision discovery with paid joint feedback and frozen embeddings."""

import copy

import numpy as np
import torch

from experiments.costs import measure_resources, synchronized_time
from experiments.rss import measurement_contract
from experiments.scenarios import features
from testing_methods.selection import validate_initial_indices

from .inference import (
    TARGET_PARAMETERS,
    adapted_model,
    initial_weights,
    mixture_weights,
    predict_components,
    source_tensors,
    update_weights,
)
from .model import JointResponseKernel
from .reliability import local_weights

MODES = ("H0F0", "H1F0", "H0F1", "H1F1")


def aggregate_history(sources):
    """Represent total covariance as point blocks plus between-source low rank."""
    weights = np.asarray(sources["weights"], dtype=float)
    means = np.asarray(sources["means"])
    mean = np.einsum("s,sni->ni", weights, means)
    low_rank = ((means - mean) * np.sqrt(weights)[:, None, None]).transpose(1, 2, 0)
    if "low_rank" in sources:
        previous = sources["low_rank"] * np.sqrt(weights)[:, None, None, None]
        low_rank = np.concatenate(
            (low_rank, previous.transpose(1, 2, 0, 3).reshape(*mean.shape, -1)), axis=-1
        )
    return {
        **sources,
        "names": ["aggregate_history_teacher"],
        "weights": np.ones(1),
        "means": mean[None],
        "blocks": np.einsum("s,snij->nij", weights, sources["blocks"])[None],
        "low_rank": low_rank[None],
    }


@measure_resources
def select(
    candidates,
    oracle,
    *,
    pretrained=None,
    sources=None,
    mode="H1F1",
    initial_indices,
    budget=200,
    device="cpu",
    feature_fn=features,
    batch_size=128,
    feedback_mode="joint",
    update_source_weights=True,
    snapshot_counts=(0, 20, 50, 200),
    query_order=None,
    local_reliability=False,
):
    if mode not in MODES or feedback_mode not in ("joint", "collision"):
        raise ValueError("Unknown source control or feedback mode")
    raw_only = pretrained is None
    if raw_only and sources is not None:
        raise ValueError("Raw target-only GP cannot use historical sources")
    if local_reliability and (
        raw_only or sources is None or mode != "H1F1" or not update_source_weights
    ):
        raise ValueError("Local reliability requires separate sources and paid feedback updates")
    if not raw_only and not isinstance(pretrained, JointResponseKernel):
        raise TypeError("Joint discovery requires a joint response target kernel")
    initial = list(initial_indices)
    validate_initial_indices(len(candidates), initial, budget)
    if query_order is not None:
        query_order = list(query_order)
        validate_initial_indices(len(candidates), query_order, budget)
        if len(query_order) != budget or query_order[: len(initial)] != initial:
            raise ValueError("Replay order must cover the budget and preserve initialization")
    if sources is not None:
        weights = np.asarray(sources["weights"])
        if np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("Source weights must form a positive probability distribution")
        if mode.startswith("H0") and sources["names"] != ["aggregate_history_teacher"]:
            raise ValueError("H0 uses the aggregate historical teacher")
    torch.set_num_threads(1)
    x = feature_fn(candidates)
    base = (
        (JointResponseKernel(x.shape[1], deep=False) if raw_only else copy.deepcopy(pretrained))
        .to(device)
        .double()
        .eval()
    )
    physical_base = (
        None if raw_only else JointResponseKernel(base.dimension, deep=False).double().to(device)
    )
    for model in (base,) if raw_only else (base, physical_base):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    encoded_target = base.features(x).detach()
    encoded_physical = None if raw_only else physical_base.features(x).detach()
    model, physical = base, physical_base
    tensors = source_tensors(sources, base)
    reference = base.point_mean(1)[0].detach().clone()
    prior = initial_weights(sources, reference)
    scores = prior.log()
    names = (
        ["raw_target"]
        if raw_only
        else [*(sources["names"] if sources else []), "independent_target", "physical_target"]
    )
    selected, labels, margins, queries, snapshots = [], [], [], [], []
    remaining = set(range(len(candidates)))
    enabled = mode.endswith("F1")
    cached = None
    reliability_observations = []
    reliability_bandwidth = base.log_lengthscale.exp().clamp_min(0.01).detach()

    def weights_at(indices=None):
        if not local_reliability:
            return mixture_weights(scores, raw_only=raw_only)
        return local_weights(
            scores,
            encoded_target if indices is None else encoded_target[indices],
            encoded_target[selected],
            reliability_observations,
            reliability_bandwidth,
            batch_size=batch_size,
        )

    def snapshot(count, probability, weights):
        return {
            "paid_queries": count,
            "observed_indices": list(selected),
            "probabilities": probability.tolist(),
            "component_weights": weights.cpu().tolist(),
            "target_parameters": {
                name: float(getattr(model, name).detach()) for name in TARGET_PARAMETERS
            },
            "physical_parameters": None
            if physical is None
            else {name: float(getattr(physical, name).detach()) for name in TARGET_PARAMETERS},
        }

    for step in range(budget):
        started = synchronized_time(device)
        # A prescribed sequence only needs the next point, except at report checkpoints.
        prediction_indices = (
            [query_order[step]]
            if query_order is not None and step not in snapshot_counts and enabled
            else None
        )
        if cached is None or enabled:
            with torch.no_grad():
                components, diagnostics = predict_components(
                    model,
                    physical,
                    x,
                    tensors,
                    reference,
                    selected if enabled else [],
                    labels if enabled else [],
                    margins if enabled and feedback_mode == "joint" else None,
                    batch_size=batch_size,
                    indices=prediction_indices,
                    encoded_target=encoded_target,
                    encoded_physical=encoded_physical,
                )
                effective = weights_at(prediction_indices)
                weighted = effective if local_reliability else effective[:, None]
                probability = (weighted * components).sum(0).cpu().numpy()
            cached = components, diagnostics, effective, probability
        components, diagnostics, effective, probability = cached
        if step in snapshot_counts:
            snapshots.append(snapshot(step, probability, effective))
        if query_order is not None:
            index = query_order[step]
        elif step < len(initial):
            index = initial[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min(
                (i for i in remaining if probability[i] >= maximum - 1e-12),
                key=lambda i: candidates[i].scenario_id,
            )
        predicted_index = index if prediction_indices is None else 0
        selection_s = synchronized_time(device) - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("Feedback differs from the paid candidate")
        if feedback_mode == "joint" and (
            feedback.safety.rss_margin is None or feedback.rss_contract != measurement_contract()
        ):
            raise ValueError(
                "Joint discovery requires measured paid RSS with the registered contract"
            )
        selected.append(index)
        remaining.remove(index)
        started = synchronized_time(device)
        likelihood = None
        if enabled:
            labels.append(feedback.collision)
            margins.append(feedback.safety.rss_margin)
            if not raw_only and update_source_weights:
                scores, likelihood = update_weights(
                    scores, components[:-1, predicted_index], feedback.collision
                )
                if local_reliability:
                    reliability_observations.append(likelihood.detach())
            used_margins = margins if feedback_mode == "joint" else None
            model = adapted_model(
                base,
                x[selected],
                labels,
                used_margins,
                learning_rate=0.01 if raw_only else 0.003,
                encoded=encoded_target[selected],
            )
            if not raw_only:
                physical = adapted_model(
                    physical_base,
                    x[selected],
                    labels,
                    used_margins,
                    learning_rate=0.01,
                    encoded=encoded_physical[selected],
                )
        after = mixture_weights(scores, raw_only=raw_only)
        queries.append(
            {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                "prequery_probability": float(probability[predicted_index]),
                "prequery_component_probabilities": components[:, predicted_index].cpu().tolist(),
                "prequery_component_weights": (
                    effective[:, predicted_index] if local_reliability else effective
                )
                .cpu()
                .tolist(),
                "component_weights_after_query": after.cpu().tolist(),
                "prequery_log_likelihood": None
                if likelihood is None
                else likelihood.cpu().tolist(),
                "conditional_modes": diagnostics if raw_only else diagnostics[:-1],
                "physical_mode": None if raw_only else diagnostics[-1],
                "conditioning_queries": step if enabled else 0,
                "selection_s": selection_s,
                "adaptation_s": synchronized_time(device) - started,
                "reason": "common_initialization" if step < len(initial) else mode,
            }
        )
    diagnostic_prediction_s = 0.0
    if budget in snapshot_counts:
        started = synchronized_time(device)
        with torch.no_grad():
            components, _ = predict_components(
                model,
                physical,
                x,
                tensors,
                reference,
                selected if enabled else [],
                labels if enabled else [],
                margins if enabled and feedback_mode == "joint" else None,
                batch_size=batch_size,
                encoded_target=encoded_target,
                encoded_physical=encoded_physical,
            )
            effective = weights_at()
            weighted = effective if local_reliability else effective[:, None]
            probability = (weighted * components).sum(0).cpu().numpy()
        snapshots.append(snapshot(budget, probability, effective))
        diagnostic_prediction_s = synchronized_time(device) - started
    return {
        "method": mode,
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "component_names": names,
        "explicit_source_components": sources is not None and mode.startswith("H1"),
        "target_feedback_updates": enabled,
        "global_reliability": not raw_only and enabled and update_source_weights,
        "local_reliability": local_reliability,
        "source_weight_updates": not raw_only and enabled and update_source_weights,
        "group_prior_weights": prior.cpu().tolist(),
        "physical_mass": 1.0 if raw_only else 0.1,
        "representation": "normalized_parameters" if raw_only else "historical_embedding",
        "beta": 1.0,
        "feedback_used": feedback_mode if enabled else "paid indices only",
        "rss_contract": measurement_contract(),
        "covariance_mode": "point_joint_blocks",
        "batch_size": batch_size,
        "online_feature_precision": "float64",
        "conditional_inference_device": str(device),
        "kernel_and_adaptation_device": str(device),
        "prediction_snapshots": snapshots,
        "diagnostic_prediction_s": diagnostic_prediction_s,
    }
