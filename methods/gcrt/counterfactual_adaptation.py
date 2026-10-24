"""Separate posterior conditioning from refitting after a virtual collision label."""

import copy
import time

import numpy as np
import torch

from methods.common.gp_classifier import collision_probabilities, fit_target
from .residual_scale import update_reliability
from .residual_search import learned_probabilities


MODES = ('fixed', 'refit_target', 'refit_both')


def virtual_update(pretrained, physical_pretrained, current, physical_current, x, paid, labels,
                   sources, reference_mean, beta, log_weights, prequery_components, index, label):
    if index in paid or type(label) is not bool:
        raise ValueError('A virtual branch needs an unqueried candidate and a Boolean label')
    extended, feedback = [*paid, index], [*labels, label]
    public_x = torch.as_tensor(x, dtype=log_weights.dtype, device=log_weights.device)
    updated_weights = update_reliability(log_weights, prequery_components, index, label, public_x, .3)
    if current.mean.is_cuda:
        torch.cuda.synchronize(current.mean.device)
    started = time.perf_counter()
    adapted = copy.deepcopy(pretrained)
    fit_target(adapted, x[extended], feedback, steps=10, learning_rate=.003)
    if adapted.mean.is_cuda:
        torch.cuda.synchronize(adapted.mean.device)
    target_s = time.perf_counter()-started
    started = time.perf_counter()
    physical = copy.deepcopy(physical_pretrained)
    fit_target(physical, x[extended], feedback, steps=10, learning_rate=.01)
    if physical.mean.is_cuda:
        torch.cuda.synchronize(physical.mean.device)
    physical_s = time.perf_counter()-started
    started = time.perf_counter()
    with torch.no_grad():
        fixed_components = learned_probabilities(current, None, x, extended, feedback, sources, reference_mean, beta)
        refitted_components = learned_probabilities(adapted, None, x, extended, feedback, sources, reference_mean, beta)
        fixed_physical = collision_probabilities(physical_current, x[extended], feedback, x)
        refitted_physical = collision_probabilities(physical, x[extended], feedback, x)
        weights = updated_weights.softmax(0)
        fixed = (weights*fixed_components).sum(0)
        refitted = (weights*refitted_components).sum(0)
        probabilities = {'fixed': .9*fixed+.1*fixed_physical,
                         'refit_target': .9*refitted+.1*fixed_physical,
                         'refit_both': .9*refitted+.1*refitted_physical}
    if current.mean.is_cuda:
        torch.cuda.synchronize(current.mean.device)
    return {'probabilities': probabilities, 'target': adapted, 'physical': physical,
        'log_weights': updated_weights, 'components': {'fixed': fixed_components, 'refit': refitted_components},
        'physical_probabilities': {'fixed': fixed_physical, 'refit': refitted_physical},
        'cost': {'target_refit_s': target_s, 'physical_refit_s': physical_s,
                 'conditional_prediction_s': time.perf_counter()-started}}


def batch_utility(prequery_probability, index, branches, available, horizon):
    if not 1 <= horizon <= len(available) or index not in available:
        raise ValueError('The horizon includes the current query and must fit the available candidates')
    future = [i for i in available if i != index]
    probability = float(prequery_probability[index])
    gains = [float(np.sort(np.asarray(branches[y])[future])[-(horizon-1):].sum())
             if horizon > 1 else 0. for y in (False, True)]
    return probability+(1-probability)*gains[0]+probability*gains[1]
