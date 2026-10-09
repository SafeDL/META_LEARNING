"""Learn which historical risk contrasts transfer to failure contrasts."""
import math

import numpy as np
from scipy.special import ndtri
import torch

from .config import RISK_EPSILON
from .model import joint_prior


def collision_sources(x, collisions, options, projection, source_indices):
    values = torch.as_tensor(ndtri(
        np.clip(collisions, RISK_EPSILON, 1 - RISK_EPSILON)),
                             dtype=projection.dtype,
                             device=projection.device)
    family = torch.as_tensor(x[:, 4],
                             dtype=torch.long,
                             device=projection.device)
    basis = (values - values.mean(1)[:, None]) / math.sqrt(values.shape[1] - 1)
    matrices = projection[family][:, source_indices][:, :, source_indices]
    projected = torch.einsum("ni,nij->nj", basis, matrices)
    scale = math.sqrt(options["source_scale"])
    original = scale * torch.cat(
        (basis * (family[:, None] == 0), basis * (family[:, None] == 1)), 1).T
    changed = scale * torch.cat(
        (projected * (family[:, None] == 0), projected *
         (family[:, None] == 1)), 1).T
    return values, family, original, changed


def projected_prior(x,
                    risks,
                    collisions,
                    options,
                    projection,
                    mean_logits,
                    source_indices,
                    full_collision=False):
    mr, _, rr, cr, cc, ur = joint_prior(x,
                                        risks,
                                        collisions,
                                        options,
                                        projection.device,
                                        full_collision=full_collision)
    values, family, original, changed = collision_sources(
        x, collisions, options, projection, source_indices)
    cr = cr + (changed - original).T @ ur
    if full_collision:
        cc = cc - original.T @ original + changed.T @ changed
    else:
        cc = cc - original.square().sum(0) + changed.square().sum(0)
    weights = mean_logits[family][:, source_indices].softmax(1)
    mc = (values * weights).sum(1) + values.new_tensor(
        options["collision_offset"])[family]
    return mr, mc, rr, cr, cc, ur


def apply_projection(session, x, prediction, state):
    projection = session.mean_r.new_tensor(state["projection"])
    mean_logits = session.mean_r.new_tensor(state["mean_logits"])
    source_indices = list(range(prediction[1].shape[1]))
    values, family, original, changed = collision_sources(
        x, prediction[1], session.options, projection, source_indices)
    session.cr += (changed - original).T @ session.coefficient_r
    session.cc += changed.square().sum(0) - original.square().sum(0)
    weights = mean_logits[family][:, source_indices].softmax(1)
    session.mean_c = (values * weights).sum(1) + values.new_tensor(
        session.options["collision_offset"])[family]
