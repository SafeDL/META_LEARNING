"""Source-specific kernels and spatial predictive-likelihood reliability."""

import numpy as np
from scipy.special import logsumexp
import torch

from methods.common.classification import site_derivatives
from methods.common.response import HistoricalPredictor, source_weights

from .posterior import CollisionPosterior


class LocalCollisionPosterior(CollisionPosterior):
    def __init__(self, means, covariances, weights, *, x, bandwidth, component_names):
        super().__init__(means, covariances, weights, component_names=component_names)
        self.x = np.asarray(x, dtype=float).copy()
        if self.x.shape[0] != self.mean.shape[1] or not np.isfinite(self.x).all():
            raise ValueError("Reliability features must match the public candidate pool")
        if bandwidth <= 0:
            raise ValueError("Reliability bandwidth must be positive")
        self.bandwidth = bandwidth
        self.log_prior = self.log_weight.copy()
        self.predictive_evidence = []
        self.evidence_indices = []
        self.local_log_weight = np.broadcast_to(
            self.log_prior[:, None], self.mean.shape).copy()

    def log_weights_at(self, indices):
        weights = self.local_log_weight[:, indices]
        return weights - logsumexp(weights, axis=0)

    def reliability_influence(self, queries, candidates):
        distance = ((self.x[queries, None] - self.x[None, candidates]) ** 2).mean(-1)
        return np.exp(-distance / (2 * self.bandwidth ** 2))

    def observe(self, index, collision):
        likelihood = site_derivatives(
            self.mean[:, index], self.covariance[:, index, index], collision)[0]
        evidence = float(logsumexp(self.log_weights_at([index])[:, 0] + likelihood))
        super().observe(index, collision)
        self.predictive_evidence.append(likelihood.copy())
        self.evidence_indices.append(int(index))
        self.local_log_weight += likelihood[:, None] * self.reliability_influence(
            [index], np.arange(len(self.x)))
        return evidence


def discovery_posterior(encoder, x, contexts, *, independent, bandwidth=None,
                        target_weight=0.25):
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Pilot inference must use the same frozen historical encoder")
    if not 0 < target_weight < 1:
        raise ValueError("Target prior mass must be strictly between zero and one")
    with torch.no_grad():
        physical = encoder.physical_kernel(x).double()
        if independent:
            encoded = encoder.encode(x, contexts).double()
            covariance = physical[None] + encoder.log_response_scale.exp().double().square() * (
                encoded @ encoded.transpose(-1, -2))
        else:
            covariance = encoder.covariance(x, contexts)[None].expand(len(contexts), -1, -1)
        covariance = torch.cat((covariance, physical[None]))
        covariance = covariance + 1e-6 * torch.eye(len(x), device=physical.device)
    means = np.concatenate((HistoricalPredictor(contexts).latent_means(np.asarray(x)),
                            np.zeros((1, len(x)))))
    weights = np.r_[source_weights(contexts) * (1 - target_weight), target_weight]
    names = [a.source_id for a in contexts] + ["target_only"]
    if bandwidth is None:
        return CollisionPosterior(means, covariance.cpu().numpy(), weights, component_names=names)
    return LocalCollisionPosterior(means, covariance.cpu().numpy(), weights,
                                   x=x, bandwidth=bandwidth, component_names=names)
