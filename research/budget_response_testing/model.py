"""A compact risk-only joint response model for budget-trained testing."""
import math

import numpy as np
from scipy.special import logit, ndtri
import torch
from torch.nn import functional as F

from methods.history_guided_testing.kernel import matern52

RISK_EPSILON = 1e-4


def risk_coordinate(value):
    return logit(np.clip(np.asarray(value), RISK_EPSILON, 1 - RISK_EPSILON))


def joint_prior(x, risks, collisions, state, device="cuda"):
    options = state["options"]
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    risk = torch.as_tensor(risk_coordinate(risks),
                           dtype=torch.float64,
                           device=device)
    collision = torch.as_tensor(ndtri(
        np.clip(collisions, RISK_EPSILON, 1 - RISK_EPSILON)),
                                dtype=torch.float64,
                                device=device)
    family = x[:, 4].long()
    same_family = family[:, None] == family[None, :]
    mean_r = risk.mean(1)
    mean_logits = risk.new_tensor(state["mean_logits"])
    mean_c = (collision * mean_logits[family].softmax(1)).sum(1)
    mean_c += risk.new_tensor(options["collision_offset"])[family]
    normalizer = math.sqrt(risk.shape[1] - 1)
    br = (risk - mean_r[:, None]) / normalizer
    bc = (collision - collision.mean(1)[:, None]) / normalizer
    sensitivity = (br * bc).sum(1) / (br.square().sum(1) +
                                      options["sensitivity_regularization"])
    features = torch.stack(
        (torch.ones_like(mean_r), mean_r, br.square().sum(1).sqrt(),
         collision.mean(1), bc.square().sum(1).sqrt(), sensitivity), 1)
    loading = risk.new_tensor(options["loading"])
    a = F.softplus((features * loading[family]).sum(1))
    response_distance = torch.cdist(risk, risk) / math.sqrt(risk.shape[1])
    local = (0.9 * matern52(response_distance, options["length"]) + 0.1 *
             matern52(torch.cdist(x[:, :4], x[:, :4]), 0.2)) * same_family
    strength = options["source_scale"]
    local_risk = options["risk_scale"]**2 * local
    rr = strength * (br @ br.T) * same_family + local_risk
    cr = strength * (bc @ br.T) * same_family + a[:, None] * local_risk
    cc = (strength * bc.square().sum(1) +
          a.square() * options["risk_scale"]**2 +
          options["collision_scale"]**2)
    source_r = strength**0.5 * torch.cat(
        (br * (family[:, None] == 0), br * (family[:, None] == 1)), 1).T
    return mean_r, mean_c, rr, cr, cc, source_r
