"""Fixed-kernel virtual Bernoulli updates using the real converged solver."""

import numpy as np
from scipy.special import logsumexp
import torch

from .converged_response import component_probabilities


def mix_virtual(components, before, scores, influence, index, label):
    """Use pre-observation component likelihoods; preserve the fixed physical mass."""
    likelihood = np.log(np.maximum(before[:-1, index] if label else 1 - before[:-1, index], 1e-15))
    weight = scores + likelihood[:, None] * influence[None]
    weight = np.exp(weight - logsumexp(weight, axis=0))
    return .9 * (weight * components[:-1]).sum(0) + .1 * components[-1]


def virtual_components(means, covariances, paid, labels, index, label):
    if index in paid or type(label) is not bool:
        raise ValueError('A virtual observation must be a new coordinate and a Boolean')
    with torch.no_grad():
        probability, diagnostics = component_probabilities(means, covariances, [*paid, index], [*labels, label])
    return probability.numpy(), diagnostics


def acquisitions(probability, conditional, available, horizons):
    """Conditional axes are candidate, virtual label and complete public pool."""
    available = np.asarray(available, dtype=int)
    if conditional.shape != (len(available), 2, len(probability)):
        raise ValueError('Virtual probabilities must include every candidate and both labels')
    result = {}
    for horizon in horizons:
        if not 1 <= horizon <= len(available):
            raise ValueError('The batch must fit the remaining candidate pool')
        values = np.asarray(probability)[available].copy()
        if horizon > 1:
            for position, index in enumerate(available):
                future = available[available != index]
                for label in (False, True):
                    p = conditional[position, int(label), future]
                    gain = np.partition(p, len(p) - horizon + 1)[-horizon + 1:].sum()
                    values[position] += (probability[index] if label else 1 - probability[index]) * gain
        result[horizon] = values
    return result


def choose(values, available, scenario_ids):
    maximum = max(values)
    positions = np.flatnonzero(np.asarray(values) >= maximum - 1e-12)
    return int(min((available[i] for i in positions), key=lambda i: scenario_ids[i]))
