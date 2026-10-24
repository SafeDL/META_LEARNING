"""Reweight stored prequery evidence using the currently available feature geometry."""

import torch


def squared_distance(encoded, scale=1.):
    if encoded.ndim != 2 or not encoded.shape[1] or scale <= 0:
        raise ValueError('Reliability distance needs a feature matrix and a positive frozen scale')
    return (encoded[:, None]-encoded[None]).square().mean(-1)/(scale**2)


def log_weights(prior, prequery_log_likelihoods, paid, distance, bandwidth):
    if (prior.ndim != 1 or prequery_log_likelihoods.shape != (len(prior), len(paid))
            or distance.ndim != 2 or distance.shape[0] != distance.shape[1]
            or len(set(paid)) != len(paid) or any(not 0 <= i < len(distance) for i in paid)
            or bandwidth is not None and bandwidth <= 0):
        raise ValueError('Reliability uses exactly one stored prequery likelihood per source and paid query')
    influence = (torch.ones_like(distance[paid]) if bandwidth is None else
                 (-distance[paid]/(2*bandwidth**2)).exp())
    score = prior.log()[:, None]+prequery_log_likelihoods@influence
    return score-torch.logsumexp(score, dim=0, keepdim=True)
