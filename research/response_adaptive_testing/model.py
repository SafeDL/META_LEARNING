"""An empirical joint response prior, conditioned only on target risk."""
import math

import numpy as np
from scipy.special import logit, ndtri
import torch

from methods.history_guided_testing.kernel import matern52

from .config import RISK_EPSILON


def risk_coordinate(value, transform):
    value = np.asarray(value)
    if transform == "raw":
        return value
    return logit(np.clip(value, RISK_EPSILON, 1 - RISK_EPSILON))


def positive_expectation(mean, deviation, boundary):
    z = (mean - boundary) / deviation
    return ((mean - boundary) * torch.special.ndtr(z) +
            deviation * torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi))


def clipped_expectation(mean, variance):
    deviation = variance.clamp(min=1e-14).sqrt()
    return (positive_expectation(mean, deviation, 0) -
            positive_expectation(mean, deviation, 1))


def logistic_expectation(mean, variance):
    nodes, masses = np.polynomial.hermite.hermgauss(32)
    nodes = mean.new_tensor(nodes * math.sqrt(2))
    masses = mean.new_tensor(masses / math.sqrt(math.pi))
    values = mean[..., None] + variance.clamp(min=0).sqrt()[..., None] * nodes
    return (values.sigmoid() * masses).sum(-1)


def joint_prior(x,
                risks,
                collisions,
                options,
                device="cuda",
                full_collision=False):
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    risk = torch.as_tensor(risk_coordinate(risks, options["transform"]),
                           dtype=torch.float64,
                           device=device)
    collision_transform = options.get("collision_transform", "raw")
    collision_values = (ndtri(
        np.clip(collisions, RISK_EPSILON, 1 -
                RISK_EPSILON)) if collision_transform == "probit" else
                        risk_coordinate(collisions, collision_transform))
    collision = torch.as_tensor(collision_values,
                                dtype=torch.float64,
                                device=device)
    mean_r, mean_c = risk.mean(1), collision.mean(1)
    normalizer = math.sqrt(risk.shape[1] - 1)
    basis_r = (risk - mean_r[:, None]) / normalizer
    basis_c = (collision - mean_c[:, None]) / normalizer
    family = x[:, 4:5] == x[:, 4:5].T
    source_r = basis_r @ basis_r.T * family
    source_cr = basis_c @ basis_r.T * family
    response_distance = torch.cdist(risk, risk) / math.sqrt(risk.shape[1])
    local = (0.9 * matern52(response_distance, options["length"]) +
             0.1 * matern52(torch.cdist(x[:, :4], x[:, :4]), 0.2)) * family
    strength, scale_r, scale_c = (options[name]
                                  for name in ("source_scale", "risk_scale",
                                               "collision_scale"))
    rr = strength * source_r + scale_r**2 * local
    residual_mode = options.get("residual_mode", "correlated")
    if residual_mode in ("sensitivity", "calibrated_sensitivity",
                         "learned_sensitivity"):
        if residual_mode in ("sensitivity", "learned_sensitivity"):
            sensitivity = (basis_r * basis_c).sum(1) / (
                basis_r.square().sum(1) +
                options.get("sensitivity_regularization", options["noise"]))
            if residual_mode == "learned_sensitivity":
                features = torch.stack(
                    (torch.ones_like(mean_r),
                     mean_r, basis_r.square().sum(1).sqrt(), mean_c,
                     basis_c.square().sum(1).sqrt(), sensitivity), 1)
                loading = torch.as_tensor(options["loading"],
                                          dtype=risk.dtype,
                                          device=device)
                sensitivity = (features * loading[x[:, 4].long()]).sum(1)
                if options.get("positive_loading", False):
                    sensitivity = torch.nn.functional.softplus(sensitivity)
                offsets = torch.as_tensor(options["collision_offset"],
                                          dtype=risk.dtype,
                                          device=device)
                mean_c = mean_c + offsets[x[:, 4].long()]
        else:
            slopes = risk.new_tensor(options["historical_slopes"])
            sensitivity = slopes[x[:, 4].long()]
            if options["transform"] == "logit":
                physical_risk = risk.new_tensor(risks).mean(1)
                sensitivity = sensitivity * physical_risk * (1 - physical_risk)
        cr = strength * source_cr + sensitivity[:, None] * scale_r**2 * local
        cc = (strength * basis_c.square().sum(1) +
              sensitivity.square() * scale_r**2 + scale_c**2)
    else:
        cr = (strength * source_cr +
              options["correlation"] * scale_r * scale_c * local)
        cc = strength * basis_c.square().sum(1) + scale_c**2
    if full_collision:
        source_cc = basis_c @ basis_c.T * family
        cc = strength * source_cc + scale_c**2 * local
        if residual_mode in ("sensitivity", "calibrated_sensitivity",
                             "learned_sensitivity"):
            cc += sensitivity[:,
                              None] * sensitivity[None, :] * scale_r**2 * local
    if options.get("predictive_mean") == "empirical_probability":
        prior_probability = risk.new_tensor(collisions).mean(1).clamp(
            RISK_EPSILON, 1 - RISK_EPSILON)
        prior_variance = cc.diag() if full_collision else cc
        mean_c = (
            1 + prior_variance).sqrt() * torch.special.ndtri(prior_probability)
        if residual_mode == "learned_sensitivity":
            mean_c = mean_c + offsets[x[:, 4].long()]
    # Separate coefficients per scenario family; no source probabilities.
    coefficients = torch.cat(
        (basis_r * (x[:, 4:5] == 0), basis_r * (x[:, 4:5] == 1)), dim=1).T
    coefficient_r = strength**0.5 * coefficients
    return mean_r, mean_c, rr, cr, cc, coefficient_r
