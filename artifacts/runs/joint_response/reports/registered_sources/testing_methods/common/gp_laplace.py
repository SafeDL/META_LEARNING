"""Converged Bernoulli/Laplace inference for a finite Gaussian prior."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

MODE_TOLERANCE = 1e-10
MODE_ITERATIONS = 100


@dataclass(frozen=True)
class LaplaceState:
    nll: torch.Tensor
    alpha: torch.Tensor
    root: torch.Tensor
    factor: torch.Tensor
    iterations: int
    gradient_max: float
    backtracks: int


def laplace(mean, covariance, labels):
    """Find the unique mode in whitened coordinates using damped Newton steps."""
    labels = torch.as_tensor(labels, dtype=mean.dtype, device=mean.device)
    eye = torch.eye(len(labels), dtype=mean.dtype, device=mean.device)
    covariance = covariance + 1e-5 * eye
    prior_factor = torch.linalg.cholesky(covariance)
    whitened = torch.zeros_like(mean)
    backtracks = 0
    iterations = 0
    for _ in range(MODE_ITERATIONS):
        latent = mean + prior_factor @ whitened
        p = latent.sigmoid()
        weight = (p * (1 - p)).clamp_min(1e-12)
        gradient = whitened + prior_factor.T @ (p - labels)
        if float(gradient.detach().abs().max()) <= MODE_TOLERANCE:
            break
        hessian = eye + prior_factor.T @ (weight[:, None] * prior_factor)
        factor = torch.linalg.cholesky(hessian)
        direction = -torch.cholesky_solve(gradient[:, None], factor)[:, 0]
        slope = (gradient * direction).sum()
        # Near the mode, objective differences reach floating-point precision.
        if float(-slope.detach()) < 1e-12:
            whitened = whitened + direction
        else:
            objective = 0.5 * whitened.square().sum() + F.binary_cross_entropy_with_logits(
                latent, labels, reduction="sum"
            )
            rate = 1.0
            for _ in range(30):
                candidate = whitened + rate * direction
                f = mean + prior_factor @ candidate
                value = 0.5 * candidate.square().sum() + F.binary_cross_entropy_with_logits(
                    f, labels, reduction="sum"
                )
                if float(value.detach()) <= float((objective + 1e-4 * rate * slope).detach()):
                    whitened = candidate
                    break
                rate *= 0.5
                backtracks += 1
            else:
                raise FloatingPointError(
                    "Bernoulli mode line search did not find a decreasing step"
                )
        iterations += 1
    latent = mean + prior_factor @ whitened
    p = latent.sigmoid()
    gradient_max = float((whitened + prior_factor.T @ (p - labels)).detach().abs().max())
    if gradient_max > MODE_TOLERANCE:
        raise FloatingPointError(f"Bernoulli mode did not converge: gradient {gradient_max:.3g}")
    root = (p * (1 - p)).clamp_min(1e-12).sqrt()
    factor = torch.linalg.cholesky(eye + root[:, None] * covariance * root[None, :])
    alpha = torch.linalg.solve_triangular(prior_factor.T, whitened[:, None], upper=True)[:, 0]
    nll = (
        0.5 * whitened.square().sum()
        + F.binary_cross_entropy_with_logits(latent, labels, reduction="sum")
        + factor.diagonal().log().sum()
    )
    return LaplaceState(nll, alpha, root, factor, iterations, gradient_max, backtracks)


def posterior(mean, covariance, observed, labels):
    if not len(observed):
        return mean, covariance, {"iterations": 0, "gradient_max": 0.0, "backtracks": 0}
    indices = torch.as_tensor(observed, dtype=torch.long, device=mean.device)
    cross = covariance[indices]
    state = laplace(mean[indices], cross[:, indices], labels)
    result_mean = mean + cross.T @ state.alpha
    solved = torch.linalg.solve_triangular(state.factor, state.root[:, None] * cross, upper=False)
    result_covariance = covariance - solved.T @ solved
    return (
        result_mean,
        0.5 * (result_covariance + result_covariance.T),
        {
            "iterations": state.iterations,
            "gradient_max": state.gradient_max,
            "backtracks": state.backtracks,
        },
    )
