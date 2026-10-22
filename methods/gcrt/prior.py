"""Construct the same component prior for training, selection and ablations."""

import numpy as np
import torch

from methods.common.response import HistoricalPredictor, source_weights

from .posterior import CollisionPosterior


def component_prior(encoder, x, contexts, *, response=True, target=True,
                    target_weight=0.25):
    if not 0 < target_weight < 1:
        raise ValueError("Target prior mass must be strictly between zero and one")
    means = HistoricalPredictor(contexts).latent_means(np.asarray(x))
    covariance = encoder.covariance(x, contexts, response=response)
    covariances = covariance[None].expand(len(contexts), -1, -1)
    weights = source_weights(contexts)
    if target:
        means = np.concatenate((means, np.zeros((1, len(x)))), axis=0)
        covariances = torch.cat((covariances, encoder.physical_kernel(x).double()[None]))
        weights = np.r_[weights * (1 - target_weight), target_weight]
    means = torch.as_tensor(means, device=covariance.device, dtype=torch.float64)
    eye = torch.eye(len(x), device=covariance.device, dtype=torch.float64)
    return means, covariances + 1e-6 * eye, torch.as_tensor(
        weights, device=covariance.device, dtype=torch.float64)


def posterior(encoder, x, contexts, *, response=True, target=True,
              fixed_weights=False, target_weight=0.25):
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Online selection requires a frozen evaluation-mode encoder")
    with torch.no_grad():
        mean, covariance, weight = component_prior(
            encoder, x, contexts, response=response, target=target,
            target_weight=target_weight)
    return CollisionPosterior(mean.cpu().numpy(), covariance.cpu().numpy(),
                              weight.cpu().numpy(), fixed_weights=fixed_weights,
                              component_names=[a.source_id for a in contexts] + (["target_only"] if target else []))
