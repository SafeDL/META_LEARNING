"""RSS historical posteriors and collision-only target conditioning."""

import copy

import numpy as np
import torch

from testing_methods.common.classification import torch_moments
from testing_methods.common.gp_classifier import ClassificationKernel, fit_target
from testing_methods.common.gp_laplace import laplace, posterior

from .model import MARGIN_NOISE


def component_probabilities(means, covariances, paid, labels):
    if (
        means.ndim != 2
        or covariances.shape != (*means.shape, means.shape[-1])
        or len(paid) != len(labels)
        or len(set(paid)) != len(paid)
        or any(not 0 <= i < means.shape[1] for i in paid)
    ):
        raise ValueError(
            "Conditional component means, covariances and unique paid labels must align"
        )
    probabilities, diagnostics = [], []
    for mean, covariance in zip(means, covariances):
        if len(paid):
            state = laplace(mean[paid], covariance[paid][:, paid], labels)
            cross = covariance[:, paid]
            predicted_mean = mean + cross @ state.alpha
            solved = torch.linalg.solve_triangular(
                state.factor, state.root[:, None] * cross.T, upper=False
            )
            variance = covariance.diagonal() - solved.square().sum(0)
            info = {
                "iterations": state.iterations,
                "gradient_max": state.gradient_max,
                "backtracks": state.backtracks,
            }
        else:
            predicted_mean, variance = mean, covariance.diagonal()
            info = {"iterations": 0, "gradient_max": 0.0, "backtracks": 0}
        probabilities.append(torch_moments(predicted_mean, variance.clamp_min(1e-10))[0])
        diagnostics.append(info)
    return torch.stack(probabilities), diagnostics


def residual_prior(
    target_mean, target_covariance, source_means, source_covariances, reference_mean
):
    count = target_covariance.shape[-1]
    means = source_means + (target_mean - reference_mean)
    covariances = source_covariances + target_covariance[None]
    return (
        torch.cat((means, target_mean.expand(1, count))),
        torch.cat((covariances, target_covariance[None])),
    )


def historical_prior(model, archive, query_x):
    """Condition on historical RSS margins, then historical collision labels."""
    if archive.role != "history":
        raise ValueError(
            "Only completed historical systems can supply an episode's response context"
        )
    count = len(archive.x)
    x = np.concatenate((archive.x, query_x))
    size = len(x)
    mean, covariance = model.joint_mean(size), model.joint_covariance(x)
    indices = torch.arange(size, size + count, device=mean.device)
    cross = covariance[:, indices]
    factor = torch.linalg.cholesky(
        covariance[indices][:, indices]
        + MARGIN_NOISE**2 * torch.eye(count, dtype=mean.dtype, device=mean.device)
    )
    values = torch.as_tensor(archive.continuous_margin, dtype=mean.dtype, device=mean.device)
    mean = mean + cross @ torch.cholesky_solve((values - mean[indices])[:, None], factor)[:, 0]
    solved = torch.linalg.solve_triangular(factor, cross.T, upper=False)
    covariance = covariance - solved.T @ solved
    mean, covariance, _ = posterior(
        mean[:size], covariance[:size, :size], list(range(count)), archive.collision
    )
    return mean[count:], covariance[count:, count:]


def adaptation_states(pretrained, x, labels):
    """Target/physical updates are independent of the historical encoder and reusable."""
    device = pretrained.mean.device
    physical_base = ClassificationKernel(x.shape[1]).to(device)
    states = []
    for count in range(len(labels) + 1):
        target = copy.deepcopy(pretrained).double()
        physical = copy.deepcopy(physical_base)
        if count:
            fit_target(target, x[:count], labels[:count], steps=10, learning_rate=0.003)
            fit_target(physical, x[:count], labels[:count], steps=10, learning_rate=0.01)
        with torch.no_grad():
            target_covariance, physical_covariance = target.covariance(x), physical.covariance(x)
            target_probability, _ = component_probabilities(
                target.mean.expand(1, len(x)),
                target_covariance[None],
                list(range(count)),
                labels[:count],
            )
            physical_probability, _ = component_probabilities(
                physical.mean.expand(1, len(x)),
                physical_covariance[None],
                list(range(count)),
                labels[:count],
            )
            states.append(
                {
                    "target_mean": target.mean.detach().cpu().clone(),
                    "target_covariance": target_covariance.cpu(),
                    "independent_probability": target_probability[0].cpu(),
                    "physical_probability": physical_probability[0].cpu(),
                }
            )
    return states


def query_probabilities(
    source_means,
    source_covariances,
    source_weights,
    states,
    labels,
    support_counts,
    query_count,
):
    """Follow the actual prequery reliability sequence and condition on paid Booleans."""
    if len(states) != len(labels) + 1 or max(support_counts) > len(labels):
        raise ValueError("Every requested support size needs its exact prefix adaptation state")
    source_covariances = (
        source_covariances
        * torch.eye(source_means.shape[-1], dtype=source_means.dtype, device=source_means.device)[
            None
        ]
    )
    prior = torch.cat((0.75 * source_weights, source_weights.new_tensor([0.25])))
    scores = prior.log()
    reference = states[0]["target_mean"].to(source_means)
    output = {}
    for count, state in enumerate(states):
        mean = source_means + state["target_mean"].to(source_means) - reference
        covariance = source_covariances + state["target_covariance"].to(source_means)[None]
        historical, _ = component_probabilities(
            mean, covariance, list(range(count)), labels[:count]
        )
        components = torch.cat(
            (historical, state["independent_probability"].to(source_means)[None])
        )
        if count in support_counts:
            mixed = 0.9 * (scores.softmax(0)[:, None] * components).sum(0)
            mixed = mixed + 0.1 * state["physical_probability"].to(source_means)
            output[count] = mixed[-query_count:]
        if count < len(labels):
            likelihood = components[:, count] if labels[count] else 1 - components[:, count]
            scores = torch.log_softmax(scores + likelihood.clamp_min(1e-15).log(), dim=0)
    return output
