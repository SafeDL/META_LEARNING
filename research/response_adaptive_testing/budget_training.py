"""Risk-conditioned response moments and budget-weighted offline ranking."""
import math

import numpy as np
from scipy.special import ndtri
import torch
from torch.nn import functional as F

from .config import BUDGET, RISK_EPSILON
from .model import joint_prior, risk_coordinate


def response_parts(x, risks, collisions, options, device="cuda"):
    mr, _, rr, _, _, ur = joint_prior(x, risks, collisions, options, device)
    risk = mr.new_tensor(risk_coordinate(risks, "logit"))
    collision = mr.new_tensor(
        ndtri(np.clip(collisions, RISK_EPSILON, 1 - RISK_EPSILON)))
    family = torch.as_tensor(x[:, 4], dtype=torch.long, device=device)
    normalizer = math.sqrt(risks.shape[1] - 1)
    br = (risk - risk.mean(1)[:, None]) / normalizer
    bc = (collision - collision.mean(1)[:, None]) / normalizer
    sensitivity = (br * bc).sum(1) / (br.square().sum(1) +
                                      options["sensitivity_regularization"])
    features = torch.stack(
        (torch.ones_like(mr), mr, br.square().sum(1).sqrt(), collision.mean(1),
         bc.square().sum(1).sqrt(), sensitivity), 1)
    uc = options["source_scale"]**0.5 * torch.cat(
        (bc * (family[:, None] == 0), bc * (family[:, None] == 1)), 1).T
    return {
        "mean_r": mr,
        "rr": rr,
        "ur": ur,
        "uc": uc,
        "collision_values": collision,
        "family": family,
        "features": features
    }


def condition_parts(parts, context, observed_risk, options):
    mr, rr, ur, uc = (parts[key] for key in ("mean_r", "rr", "ur", "uc"))
    covariance = rr[context][:, context] + options["noise"] * torch.eye(
        len(context), dtype=mr.dtype, device=mr.device)
    source_cross = uc.T @ ur[:, context]
    local_cross = rr[:, context] - ur.T @ ur[:, context]
    residual = mr.new_tensor(risk_coordinate(observed_risk,
                                             "logit")) - mr[context]
    update = torch.linalg.solve(covariance, residual)
    source_solution = torch.linalg.solve(covariance, source_cross.T).T
    local_solution = torch.linalg.solve(covariance, local_cross.T).T
    return {
        "source_mean":
        source_cross @ update,
        "local_mean":
        local_cross @ update,
        "source_variance":
        uc.square().sum(0) - (source_cross * source_solution).sum(1),
        "local_variance":
        options["risk_scale"]**2 - (local_cross * local_solution).sum(1),
        "source_local_covariance":
        -(source_cross * local_solution).sum(1)
    }


def conditional_scores(episode, mean_logits, loading, options):
    family = episode["family"]
    weights = mean_logits[family].softmax(1)
    mean = (episode["collision_values"] * weights).sum(1)
    mean += mean.new_tensor(options["collision_offset"])[family]
    a = F.softplus((episode["features"] * loading[family]).sum(1))
    mean += episode["source_mean"] + a * episode["local_mean"]
    variance = (episode["source_variance"] +
                a.square() * episode["local_variance"] +
                2 * a * episode["source_local_covariance"] +
                options["collision_scale"]**2).clamp(min=0)
    return mean / (1 + variance).sqrt()


def position_gains(size, remaining_budget, budget=BUDGET, device="cpu"):
    position = torch.arange(1, size + 1, dtype=torch.float64, device=device)
    return torch.where(position <= remaining_budget,
                       0.5 + 0.5 * (remaining_budget - position + 1) / budget,
                       torch.zeros_like(position))


def ranking_loss(score, collision, remaining_budget, weighted):
    log_positive = torch.special.log_ndtr(score)
    log_negative = torch.special.log_ndtr(-score)
    log_odds = log_positive - log_negative
    positive, negative = log_odds[collision], log_odds[~collision]
    calibration = -torch.where(collision, log_positive, log_negative).mean()
    if not len(positive) or not len(negative):
        return 0.1 * calibration
    pairwise = F.softplus(negative[None, :] - positive[:, None])
    if weighted:
        order = torch.argsort(score.detach(), descending=True, stable=True)
        gains = position_gains(len(score),
                               remaining_budget,
                               device=score.device)
        current = torch.empty_like(gains)
        current[order] = gains
        weight = (current[collision, None] - current[None, ~collision]).abs()
        ranking = (pairwise * weight).sum() / weight.sum().clamp(min=1e-15)
    else:
        ranking = pairwise.mean()
    return ranking + 0.1 * calibration
