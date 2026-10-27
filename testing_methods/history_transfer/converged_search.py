"""Collision-only source transfer with converged conditional posteriors."""

import copy
import time

import torch

from experiments.scenarios import features
from experiments.schema import QueryFeedback
from testing_methods.common.gp_classifier import ClassificationKernel, fit_target
from testing_methods.selection import greedy_index, validate_initial_indices

from .converged_response import component_probabilities
from .residual_scale import residual_prior, update_reliability

MODES = ("converged_local", "converged_global", "converged_target", "converged_dual_target")


def learned_predictions(model, auxiliary, x, paid, labels, sources, reference, beta):
    if sources is not None:
        means, covariances = residual_prior(
            model.mean,
            model.covariance(x),
            sources["means"],
            sources["covariances"],
            reference,
            beta,
        )
    else:
        models = [auxiliary, model] if auxiliary is not None else [model]
        means = torch.stack([m.mean.expand(len(x)) for m in models])
        covariances = torch.stack([m.covariance(x) for m in models])
    probability, information = component_probabilities(means.cpu(), covariances.cpu(), paid, labels)
    return probability.to(model.mean.device), information


def select(
    candidates,
    oracle,
    *,
    pretrained,
    sources,
    mode,
    beta,
    initial_indices,
    auxiliary=None,
    budget=128,
    device=None,
    feature_fn=features,
):
    torch.set_num_threads(1)
    if mode not in MODES:
        raise ValueError("Unknown converged-source comparison")
    validate_initial_indices(len(candidates), initial_indices, budget)
    target_only = mode in ("converged_target", "converged_dual_target")
    if (sources is None) != target_only or (auxiliary is not None) != (
        mode == "converged_dual_target"
    ):
        raise ValueError(
            "Only registered target controls can omit sources or receive the second teacher"
        )
    if beta < 0:
        raise ValueError("Residual amplitude must be nonnegative")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = copy.deepcopy(pretrained).to(device)
    other_base = copy.deepcopy(auxiliary).to(device) if auxiliary is not None else None
    model, other = copy.deepcopy(base), copy.deepcopy(other_base)
    x = feature_fn(candidates)
    public_x = torch.as_tensor(x, dtype=torch.float64, device=device)
    physical_base = ClassificationKernel(x.shape[1]).to(device)
    physical = copy.deepcopy(physical_base)
    if sources is not None:
        names = [*sources["names"], "target_fsbo"]
        sources = {
            k: torch.as_tensor(v, dtype=torch.float64, device=device)
            for k, v in sources.items()
            if k != "names"
        }
        prior = torch.cat((0.75 * sources["weights"], public_x.new_tensor([0.25])))
    elif other is not None:
        names, prior = ["target_joint", "target_fsbo"], public_x.new_tensor([0.75, 0.25])
    else:
        names, prior = ["target_fsbo"], public_x.new_ones(1)
    bandwidth = 0.3 if mode in ("converged_local", "converged_dual_target") else None
    scores = prior.log()[:, None].expand(-1, len(x)).clone()
    reference = float(base.mean.detach())
    selected, labels, queries = [], [], []
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = time.perf_counter()
        with torch.no_grad():
            learned, mode_info = learned_predictions(
                model, other, x, selected, labels, sources, reference, beta
            )
            raw, raw_info = component_probabilities(
                physical.mean.expand(1, len(x)).cpu(),
                physical.covariance(x)[None].cpu(),
                selected,
                labels,
            )
            raw = raw.to(model.mean.device)
            weights = scores.softmax(0)
            learned_pooled = (weights * learned).sum(0)
            probability = (0.9 * learned_pooled + 0.1 * raw[0]).cpu().numpy()
        if step < len(initial_indices):
            index = initial_indices[step]
        else:
            index = greedy_index(probability, remaining, candidates)
        selection_s = time.perf_counter() - started
        feedback = oracle.query(
            index, purpose="initial" if step < len(initial_indices) else "target"
        )
        if (
            not isinstance(feedback, QueryFeedback)
            or feedback.scenario_id != candidates[index].scenario_id
        ):
            raise ValueError("Converged source search received another paid scenario")
        with torch.no_grad():
            likelihood = (
                (learned[:, index] if feedback.collision else 1 - learned[:, index])
                .clamp_min(1e-15)
                .log()
            )
            scores = update_reliability(
                scores, learned, index, feedback.collision, public_x, bandwidth
            )
            row = {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                "prequery_probability": float(probability[index]),
                "prequery_component_probabilities": torch.cat(
                    (learned[:, index], raw[0, index : index + 1])
                )
                .cpu()
                .tolist(),
                "prequery_component_weights": torch.cat(
                    (0.9 * weights[:, index], prior.new_tensor([0.1]))
                )
                .cpu()
                .tolist(),
                "prequery_group_probability": float(learned_pooled[index]),
                "prequery_group_weights": weights[:, index].cpu().tolist(),
                "prequery_group_log_likelihood": likelihood.cpu().tolist(),
                "group_weights_after_query": scores[:, index].exp().cpu().tolist(),
                "prequery_physical_probability": float(raw[0, index]),
                "conditional_modes": mode_info,
                "physical_mode": raw_info[0],
                "selection_s": selection_s,
                "reason": "common_initialization" if step < len(initial_indices) else mode,
            }
        selected.append(index)
        labels.append(feedback.collision)
        remaining.remove(index)
        started = time.perf_counter()
        model = copy.deepcopy(base)
        fit_target(model, x[selected], labels, steps=10, learning_rate=0.003)
        if model.mean.is_cuda:
            torch.cuda.synchronize(model.mean.device)
        row["fsbo_adaptation_s"] = time.perf_counter() - started
        row["joint_adaptation_s"] = 0.0
        if other_base is not None:
            started = time.perf_counter()
            other = copy.deepcopy(other_base)
            fit_target(other, x[selected], labels, steps=10, learning_rate=0.003)
            if other.mean.is_cuda:
                torch.cuda.synchronize(other.mean.device)
            row["joint_adaptation_s"] = time.perf_counter() - started
        started = time.perf_counter()
        physical = copy.deepcopy(physical_base)
        fit_target(physical, x[selected], labels, steps=10, learning_rate=0.01)
        if physical.mean.is_cuda:
            torch.cuda.synchronize(physical.mean.device)
        row["physical_adaptation_s"] = time.perf_counter() - started
        row["adaptation_s"] = sum(
            row[k] for k in ("fsbo_adaptation_s", "joint_adaptation_s", "physical_adaptation_s")
        )
        queries.append(row)
    return {
        "method": mode,
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "component_names": [*names, "target_physical"],
        "group_prior_weights": prior.cpu().tolist(),
        "physical_mass": 0.1,
        "beta": beta if sources is not None else None,
        "reliability_bandwidth": bandwidth,
        "reference_target_mean": reference,
        "feedback_used": "paid collision labels only",
        "uses_source_posteriors": sources is not None,
        "conditional_inference_device": "cpu",
        "kernel_and_adaptation_device": str(device),
        "conditional_mode_solver": "whitened damped Newton; gradient at most 1e-10",
    }
