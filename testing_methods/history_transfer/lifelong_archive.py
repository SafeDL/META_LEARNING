"""Likelihood weighting and prior anchoring for completed paid task archives."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rss import measurement_contract
from testing_methods.common.gp_laplace import (
    MODE_ITERATIONS, MODE_TOLERANCE, PRIOR_JITTER, LaplaceState,
)
from testing_methods.common.response import archive_fingerprints, shared_domain

from .inference import JointPosterior, historical_prior
from .model import MARGIN_NOISE

INITIAL_COUNT = 20
MINIMUM_WEIGHT = 0.15
MINIMUM_PROBABILITY = 0.05


def archive_weights(collision, prequery_probability, *, initial_count=INITIAL_COUNT):
    labels = np.asarray(collision)
    probability = np.asarray(prequery_probability, dtype=np.float64)
    if (
        labels.dtype != bool or labels.ndim != 1 or not 0 < initial_count <= len(labels)
        or probability.shape != labels.shape or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("Archive probabilities must align with a completed paid collision sequence")
    rate = (labels[:initial_count].sum() + 1) / (initial_count + 2)
    weights = np.ones(len(labels), dtype=np.float64)
    weights[initial_count:] = np.clip(
        rate / np.maximum(probability[initial_count:], MINIMUM_PROBABILITY), MINIMUM_WEIGHT, 1,
    )
    return weights, float(np.log(rate / (1 - rate)))


def effective_size(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("ESS requires positive finite observation weights")
    return float(weights.sum() ** 2 / np.square(weights).sum())


def source_weights(weight_vectors):
    sizes = np.array([min(effective_size(item), 500) for item in weight_vectors])
    if not len(sizes):
        raise ValueError("At least one historical source is required")
    return sizes / sizes.sum()


def weighted_laplace(mean, covariance, labels, weights):
    labels = torch.as_tensor(labels, dtype=mean.dtype, device=mean.device)
    eye = torch.eye(len(labels), dtype=mean.dtype, device=mean.device)
    covariance = covariance + PRIOR_JITTER * eye
    prior_factor = torch.linalg.cholesky(covariance)
    whitened = torch.zeros_like(mean)
    backtracks = 0

    def objective(latent, state):
        return 0.5 * state.square().sum() + F.binary_cross_entropy_with_logits(
            latent, labels, weight=weights, reduction="sum",
        )

    for iteration in range(MODE_ITERATIONS + 1):
        latent = mean + prior_factor @ whitened
        p = latent.sigmoid()
        gradient = whitened + prior_factor.T @ (weights * (p - labels))
        gradient_max = float(gradient.detach().abs().max())
        curvature = weights * (p * (1 - p)).clamp_min(1e-12)
        if gradient_max <= MODE_TOLERANCE:
            break
        if iteration == MODE_ITERATIONS:
            raise FloatingPointError(f"Weighted Bernoulli mode did not converge: {gradient_max:.3g}")
        hessian = eye + prior_factor.T @ (curvature[:, None] * prior_factor)
        step = -torch.cholesky_solve(gradient[:, None], torch.linalg.cholesky(hessian))[:, 0]
        slope = gradient @ step
        if float(-slope.detach()) < 1e-12:
            whitened = whitened + step
            continue
        current = objective(latent, whitened)
        rate = 1.0
        for _ in range(30):
            new = whitened + rate * step
            value = objective(mean + prior_factor @ new, new)
            if float(value.detach()) <= float((current + 1e-4 * rate * slope).detach()):
                whitened = new
                break
            rate *= 0.5
            backtracks += 1
        else:
            raise FloatingPointError("Weighted Bernoulli mode line search failed")
    root = curvature.sqrt()
    factor = torch.linalg.cholesky(eye + root[:, None] * covariance * root[None, :])
    alpha = torch.linalg.solve_triangular(prior_factor.T, whitened[:, None], upper=True)[:, 0]
    nll = objective(latent, whitened) + factor.diagonal().log().sum()
    return LaplaceState(nll, alpha, root, factor, iteration, gradient_max, backtracks)


class WeightedJointPosterior(JointPosterior):
    """Condition measured RSS and collisions with a shared observation weight."""

    def __init__(self, mean, covariance, labels, margins, weights):
        self.count = len(mean)
        weights = torch.as_tensor(weights, dtype=mean.dtype, device=mean.device)
        values = torch.as_tensor(margins, dtype=mean.dtype, device=mean.device)
        if (
            weights.shape != (self.count,) or values.shape != (self.count,)
            or not torch.isfinite(weights).all() or not torch.isfinite(values).all()
            or not ((weights > 0) & (weights <= 1)).all()
        ):
            raise ValueError("Weighted posterior requires aligned finite RSS and likelihood weights")
        rr = covariance[1::2, 1::2]
        rf = covariance[1::2, ::2]
        self.r_factor = torch.linalg.cholesky(rr + torch.diag(MARGIN_NOISE**2 / weights))
        difference = values - mean[:, 1]
        self.r_alpha = torch.cholesky_solve(difference[:, None], self.r_factor)[:, 0]
        self.r_to_f = torch.cholesky_solve(rf, self.r_factor)
        collision_mean = mean[:, 0] + rf.T @ self.r_alpha
        collision_covariance = covariance[::2, ::2] - rf.T @ self.r_to_f
        self.r_nll = (
            0.5 * (difference @ self.r_alpha) + self.r_factor.diagonal().log().sum()
            + 0.5 * self.count * math.log(2 * math.pi)
        )
        self.state = weighted_laplace(collision_mean, collision_covariance, labels, weights)
        self.info = {
            "iterations": self.state.iterations, "gradient_max": self.state.gradient_max,
            "backtracks": self.state.backtracks,
        }


def point_mean(model, count, collision_anchor):
    mean = model.point_mean(count)
    if collision_anchor is not None:
        return torch.stack((torch.full_like(mean[:, 0], collision_anchor), mean[:, 1]), dim=1)
    return mean


def weighted_prior(model, archive, x, weights, collision_anchor, *, encoded_query=None, batch_size=128):
    if archive.role != "target" or not archive.provenance.get("completed_task"):
        raise ValueError("Only completed paid target archives can use a propensity weight")
    z_history = model.features(archive.x)
    z_query = model.features(x) if encoded_query is None else encoded_query
    state = WeightedJointPosterior(
        point_mean(model, len(z_history), collision_anchor), model.kernel_from_features(z_history),
        archive.collision, archive.continuous_margin, weights,
    )
    means, blocks = [], []
    for z in z_query.split(batch_size):
        mean, block = state.predict(
            point_mean(model, len(z), collision_anchor), model.kernel_from_features(z, z_history),
            model.point_covariance(len(z)),
        )
        means.append(mean)
        blocks.append(block)
    return torch.cat(means), torch.cat(blocks)


def source_snapshot(model, archives, candidates, weighting, *, batch_size=128):
    names = [archive.source_id for archive in archives]
    if not names or len(names) != len(set(names)) or set(weighting) - set(names):
        raise ValueError("The selected archive set must identify each source once")
    domain = shared_domain(archives)
    from experiments.scenarios import features

    x = features(candidates, domain=domain)
    kind = candidates[0].kind
    target_coordinates = {scene.coordinate() for scene in candidates}
    means, blocks, weights = [], [], []
    with torch.no_grad():
        encoded = model.features(x)
        for archive in archives:
            if archive.scenarios[0].kind != kind or any(
                scene.coordinate() in target_coordinates for scene in archive.scenarios
            ):
                raise ValueError("Historical sources must match the scenario and exclude target coordinates")
            item = weighting.get(archive.source_id)
            if item is None:
                mean, block = historical_prior(
                    model, archive, x, encoded_query=encoded, batch_size=batch_size,
                )
                weights.append(np.ones(len(archive.scenarios)))
            else:
                weight, anchor = item
                mean, block = weighted_prior(
                    model, archive, x, weight, anchor, encoded_query=encoded, batch_size=batch_size,
                )
                weights.append(weight)
            means.append(mean.cpu().numpy())
            blocks.append(block.cpu().numpy())
    return {
        "names": names, "means": np.stack(means), "blocks": np.stack(blocks),
        "weights": source_weights(weights),
        "specification": {
            "scenario": kind, "parameter_definition": None if domain is None else domain.to_dict(),
            "archives": archive_fingerprints(archives), "rss_contract": measurement_contract(),
            "weighting": {
                name: {"weights": weight.tolist(), "collision_anchor": anchor}
                for name, (weight, anchor) in weighting.items()
            },
        },
    }
