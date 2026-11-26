"""Joint pooled inference with sufficient statistics for exactly repeated inputs.

Every original response remains in the likelihood. One latent pair is represented
per distinct coordinate; Gaussian RSS replicates and Bernoulli counts are grouped.
"""

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from testing_methods.common.classification import torch_moments
from testing_methods.common.gp_laplace import MODE_ITERATIONS, MODE_TOLERANCE, LaplaceState
from testing_methods.history_transfer.inference import TARGET_PARAMETERS, JointPosterior
from testing_methods.history_transfer.model import MARGIN_NOISE


@dataclass(frozen=True)
class GroupedResponses:
    x: np.ndarray
    counts: np.ndarray
    successes: np.ndarray
    rss_mean: np.ndarray
    rss_squared_residuals: np.ndarray


@dataclass(frozen=True)
class CountedLaplaceState(LaplaceState):
    precision_steps: int


def group_responses(x, labels, margins):
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    margins = np.asarray(margins, dtype=np.float64)
    if (
        x.ndim != 2
        or not len(x)
        or labels.shape != (len(x),)
        or not np.isin(labels, (0, 1)).all()
        or margins.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(margins).all()
    ):
        raise ValueError(
            "Grouped responses require finite coordinates and aligned binary/RSS responses"
        )
    unique, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    successes = np.bincount(inverse, weights=labels, minlength=len(unique))
    means = np.bincount(inverse, weights=margins, minlength=len(unique)) / counts
    residuals = np.bincount(inverse, weights=(margins - means[inverse]) ** 2, minlength=len(unique))
    return GroupedResponses(unique, counts, successes, means, residuals)


def binomial_laplace(mean, covariance, successes, counts):
    """Solve the replicated Bernoulli likelihood on unique latent coordinates."""
    successes = torch.as_tensor(successes, dtype=mean.dtype, device=mean.device)
    counts = torch.as_tensor(counts, dtype=mean.dtype, device=mean.device)
    eye = torch.eye(len(mean), dtype=mean.dtype, device=mean.device)
    covariance = covariance + 1e-5 * eye
    prior_factor = torch.linalg.cholesky(covariance)
    whitened = torch.zeros_like(mean)
    iterations, backtracks, precision_steps = 0, 0, 0

    def negative_log_likelihood(latent):
        # Labels remain ordered observations: no binomial combinatorial factor.
        return (counts * F.softplus(latent) - successes * latent).sum()

    for _ in range(MODE_ITERATIONS):
        latent = mean + prior_factor @ whitened
        p = latent.sigmoid()
        weight = (counts * p * (1 - p)).clamp_min(1e-12)
        gradient = whitened + prior_factor.T @ (counts * p - successes)
        if float(gradient.detach().abs().max()) <= MODE_TOLERANCE:
            break
        hessian = eye + prior_factor.T @ (weight[:, None] * prior_factor)
        direction = -torch.cholesky_solve(gradient[:, None], torch.linalg.cholesky(hessian))[:, 0]
        slope = (gradient * direction).sum()
        objective = 0.5 * whitened.square().sum() + negative_log_likelihood(latent)
        precision_limit = max(
            1e-12,
            16 * torch.finfo(mean.dtype).eps * max(1.0, abs(float(objective.detach()))),
        )
        # Summed evidence can hide a valid Newton decrease at floating-point
        # precision. A <= .1 latent step bounds logistic Hessian variation by
        # exp(.1); keep refining the score instead of declaring convergence.
        near_mode = (
            float(-slope.detach()) <= precision_limit
            and float((prior_factor @ direction).detach().abs().max()) <= 0.1
        )
        if near_mode:
            whitened = whitened + direction
            precision_steps += 1
        else:
            rate = 1.0
            for _ in range(30):
                candidate = whitened + rate * direction
                value = 0.5 * candidate.square().sum() + negative_log_likelihood(
                    mean + prior_factor @ candidate
                )
                if float(value.detach()) <= float((objective + 1e-4 * rate * slope).detach()):
                    whitened = candidate
                    break
                rate *= 0.5
                backtracks += 1
            else:
                raise FloatingPointError("Grouped Bernoulli mode line search failed")
        iterations += 1
    latent = mean + prior_factor @ whitened
    p = latent.sigmoid()
    gradient_max = float(
        (whitened + prior_factor.T @ (counts * p - successes)).detach().abs().max()
    )
    if not math.isfinite(gradient_max) or gradient_max > MODE_TOLERANCE:
        raise FloatingPointError(
            f"Grouped Bernoulli mode did not converge: gradient {gradient_max:.3g}"
        )
    root = (counts * p * (1 - p)).clamp_min(1e-12).sqrt()
    factor = torch.linalg.cholesky(eye + root[:, None] * covariance * root[None, :])
    alpha = torch.linalg.solve_triangular(prior_factor.T, whitened[:, None], upper=True)[:, 0]
    nll = (
        0.5 * whitened.square().sum()
        + negative_log_likelihood(latent)
        + factor.diagonal().log().sum()
    )
    return CountedLaplaceState(
        nll, alpha, root, factor, iterations, gradient_max, backtracks, precision_steps
    )


class GroupedJointPosterior(JointPosterior):
    """Keep Gaussian replicate constants and counted collision likelihoods."""

    def __init__(self, mean, covariance, grouped, *, margin_noise=MARGIN_NOISE):
        self.count = len(mean)
        counts = torch.as_tensor(grouped.counts, dtype=mean.dtype, device=mean.device)
        values = torch.as_tensor(grouped.rss_mean, dtype=mean.dtype, device=mean.device)
        residuals = torch.as_tensor(
            grouped.rss_squared_residuals, dtype=mean.dtype, device=mean.device
        )
        variance = margin_noise**2
        rr = covariance[1::2, 1::2]
        self.r_factor = torch.linalg.cholesky(rr + torch.diag(variance / counts))
        difference = values - mean[:, 1]
        self.r_alpha = torch.cholesky_solve(difference[:, None], self.r_factor)[:, 0]
        self.r_to_f = torch.cholesky_solve(covariance[1::2, ::2], self.r_factor)
        collision_mean = mean[:, 0] + covariance[::2, 1::2] @ self.r_alpha
        collision_covariance = covariance[::2, ::2] - covariance[::2, 1::2] @ self.r_to_f
        replicate_constant = (
            0.5
            * (
                (counts - 1) * math.log(2 * math.pi * variance)
                + counts.log()
                + residuals / variance
            ).sum()
        )
        self.r_nll = (
            0.5 * (difference * self.r_alpha).sum()
            + self.r_factor.diagonal().log().sum()
            + 0.5 * self.count * math.log(2 * math.pi)
            + replicate_constant
        )
        self.state = binomial_laplace(
            collision_mean, collision_covariance, grouped.successes, counts
        )
        self.info = {
            "iterations": self.state.iterations,
            "gradient_max": self.state.gradient_max,
            "backtracks": self.state.backtracks,
            "precision_steps": self.state.precision_steps,
            "original_records": int(grouped.counts.sum()),
            "unique_coordinates": self.count,
            "classification_jitter": "1e-5 per unique latent coordinate",
        }


def predict_grouped(
    base,
    train_x,
    labels,
    margins,
    query_x,
    *,
    batch_size=128,
    steps=10,
    learning_rate=0.003,
    trainable=TARGET_PARAMETERS,
):
    grouped = group_responses(train_x, labels, margins)
    model = copy.deepcopy(base).double()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if set(trainable) != {n for n, _ in parameters}:
        raise ValueError("Unknown pooled GP trainable parameter")
    encoded = model.features(grouped.x).detach()
    optimizer = torch.optim.Adam([p for _, p in parameters], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = GroupedJointPosterior(
            model.point_mean(len(grouped.x)), model.kernel_from_features(encoded), grouped
        )
        loss = state.nll / int(grouped.counts.sum())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite grouped pooled GP evidence")
        loss.backward()
        for group in model.adaptation_groups():
            torch.nn.utils.clip_grad_norm_(group, 5)
        optimizer.step()
    with torch.no_grad():
        state = GroupedJointPosterior(
            model.point_mean(len(grouped.x)), model.kernel_from_features(encoded), grouped
        )
        query = model.features(query_x)
        probability = []
        for start in range(0, len(query), batch_size):
            z = query[start : start + batch_size]
            mean, blocks = state.predict(
                model.point_mean(len(z)),
                model.kernel_from_features(z, encoded),
                model.point_covariance(len(z)),
            )
            probability.extend(
                torch_moments(mean[:, 0], blocks[:, 0, 0].clamp_min(1e-10))[0].cpu().tolist()
            )
    return probability, {n: float(p.detach()) for n, p in parameters}, state.info
