"""Censored conditional collision labels for recorded interaction stages."""

import copy
import math

import numpy as np
import torch

from methods.common.gp_classifier import fit_target, laplace
from methods.common.response import source_weights


def phase_labels(events, collision_time, episode_end, *, horizon=10.):
    onset, complete = events['measured_background_onset_s'], events['background_complete_s']
    if collision_time is not None:
        phase = 0 if onset is None or collision_time < onset else (1 if complete is None or collision_time < complete else 2)
        return tuple(0 if k < phase else 1 if k == phase else None for k in range(3))
    return (0 if onset is not None and episode_end >= onset else None,
            0 if complete is not None and episode_end >= complete else None,
            0 if complete is not None and episode_end >= horizon else None)


def phase_support(paid, observations, phase, *, zero_fill=False):
    values = observations[paid, phase]
    observed = np.ones(len(values), dtype=bool) if zero_fill else np.isfinite(values)
    return np.asarray(paid)[observed].tolist(), np.nan_to_num(values[observed], nan=0.).astype(bool)


def total_collision_probability(hazards):
    return 1-(1-hazards).prod(dim=0)


def phase_bases(pretrained, history, observations):
    if any(archive.role != 'history' for archive in history):
        raise ValueError('Stage initialization must use historical sources only')
    weights = source_weights(history)
    bases = []
    for phase in range(3):
        count = sum(w*np.isfinite(observations[a.source_id][:, phase]).sum() for w, a in zip(weights, history))
        collisions = sum(w*np.nansum(observations[a.source_id][:, phase]) for w, a in zip(weights, history))
        probability = (collisions+.5)/(count+1)
        base = copy.deepcopy(pretrained)
        with torch.no_grad():
            base.mean.fill_(math.log(probability/(1-probability)))
        bases.append(base)
    return bases


def latent_posterior(model, support, labels, queries):
    _, alpha, root, factor = laplace(model, support, labels)
    cross = model.covariance(support, queries)
    mean = model.mean+cross.T @ alpha
    solved = torch.linalg.solve_triangular(factor, root[:, None]*cross, upper=False)
    covariance = model.covariance(queries)-solved.T @ solved
    return mean, (covariance+covariance.T)/2


def source_phase_priors(bases, history, observations, x, *, zero_fill=False):
    priors, supports = [], []
    for phase, base in enumerate(bases):
        means, covariances, sizes = [], [], []
        for archive in history:
            if archive.role != 'history':
                raise ValueError('Stage posterior construction cannot use a target archive')
            indices, labels = phase_support(list(range(len(archive.scenarios))), observations[archive.source_id],
                                             phase, zero_fill=zero_fill)
            model = copy.deepcopy(base)
            fit_target(model, archive.x[indices], labels, steps=10, learning_rate=.003)
            with torch.no_grad():
                mean, covariance = latent_posterior(model, archive.x[indices], labels, x)
            means.append(mean)
            covariances.append(covariance)
            sizes.append({'source_id': archive.source_id, 'observed': len(indices), 'collisions': int(labels.sum())})
        priors.append((torch.stack(means), torch.stack(covariances)))
        supports.append(sizes)
    return priors, supports
