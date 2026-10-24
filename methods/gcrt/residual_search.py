"""Collision-only search with historically learned residual amplitude."""

import copy
import time

import torch

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from .conditional_adaptation import component_probabilities, conditional_prior
from .residual_scale import residual_prior, update_reliability


MODES = ('residual_local', 'residual_global', 'residual_unit_local', 'residual_unit_global',
         'residual_target', 'residual_dual_target')


def learned_probabilities(model, auxiliary, x, paid, labels, sources, reference_mean, beta):
    if sources is not None:
        mean, covariance = residual_prior(model.mean, model.covariance(x), sources['means'],
                                           sources['covariances'], reference_mean, beta)
        return component_probabilities(mean, covariance, paid, labels)
    mean, covariance = conditional_prior(model, x)
    probability = component_probabilities(mean, covariance, paid, labels)
    if auxiliary is not None:
        mean, covariance = conditional_prior(auxiliary, x)
        probability = torch.cat((component_probabilities(mean, covariance, paid, labels), probability))
    return probability


def select(candidates, oracle, *, pretrained, sources, mode, beta, initial_indices, auxiliary=None,
           budget=128, device=None, feature_fn=features):
    torch.set_num_threads(1)
    if mode not in MODES:
        raise ValueError('Unknown residual-search comparison')
    if (not 0 < len(initial_indices) <= budget <= len(candidates)
            or len(set(initial_indices)) != len(initial_indices)
            or any(type(i) is not int or not 0 <= i < len(candidates) for i in initial_indices)):
        raise ValueError('Residual-search initialization must fit the paid budget')
    target_only = mode in ('residual_target', 'residual_dual_target')
    if (sources is None) != target_only or (auxiliary is not None) != (mode == 'residual_dual_target'):
        raise ValueError('Only registered target controls can omit sources or receive the second teacher')
    effective_beta = 1. if mode.startswith('residual_unit_') else beta
    if effective_beta < 0:
        raise ValueError('Residual amplitude must be nonnegative')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    base = copy.deepcopy(pretrained).to(device)
    auxiliary_base = copy.deepcopy(auxiliary).to(device) if auxiliary is not None else None
    model = copy.deepcopy(base)
    other_model = copy.deepcopy(auxiliary_base) if auxiliary_base is not None else None
    x = feature_fn(candidates)
    physical_base = ClassificationKernel(x.shape[1]).to(device)
    physical = copy.deepcopy(physical_base)
    public_x = torch.as_tensor(x, dtype=torch.float64, device=device)
    if sources is not None:
        names = [*sources['names'], 'target_fsbo']
        sources = {k: torch.as_tensor(v, dtype=torch.float64, device=device)
                   for k, v in sources.items() if k != 'names'}
        prior = torch.cat((.75*sources['weights'], torch.tensor([.25], dtype=torch.float64, device=device)))
    elif other_model is not None:
        names, prior = ['target_joint', 'target_fsbo'], torch.tensor([.75, .25], dtype=torch.float64, device=device)
    else:
        names, prior = ['target_fsbo'], torch.ones(1, dtype=torch.float64, device=device)
    bandwidth = .3 if mode in ('residual_local', 'residual_unit_local', 'residual_dual_target') else None
    log_weights = prior.log()[:, None].expand(-1, len(x)).clone()
    reference_mean = float(base.mean.detach())
    selected, labels, queries = [], [], []
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = time.perf_counter()
        with torch.no_grad():
            learned_p = learned_probabilities(model, other_model, x, selected, labels, sources,
                                               reference_mean, effective_beta)
            weights = log_weights.softmax(0)
            learned_pooled = (weights*learned_p).sum(0)
            raw_p = collision_probabilities(physical, x[selected], labels, x)
            probability = (.9*learned_pooled+.1*raw_p).cpu().numpy()
        if step < len(initial_indices):
            index = initial_indices[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min((i for i in remaining if probability[i] >= maximum-1e-12),
                        key=lambda i: candidates[i].scenario_id)
        selection_s = time.perf_counter()-started
        feedback = oracle.query(index, purpose='initial' if step < len(initial_indices) else 'target')
        if not isinstance(feedback, QueryFeedback) or feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError('Residual search received a different paid scenario')
        with torch.no_grad():
            likelihood = (learned_p[:, index] if feedback.collision else 1-learned_p[:, index]).clamp_min(1e-15).log()
            log_weights = update_reliability(log_weights, learned_p, index, feedback.collision, public_x, bandwidth)
            row = {'query': step+1, 'index': index, 'feedback': feedback.to_dict(),
                'prequery_probability': float(probability[index]),
                'prequery_component_probabilities': torch.cat((learned_p[:, index], raw_p[index:index+1])).cpu().tolist(),
                'prequery_component_weights': torch.cat((.9*weights[:, index], prior.new_tensor([.1]))).cpu().tolist(),
                'prequery_group_probability': float(learned_pooled[index]),
                'prequery_group_weights': weights[:, index].cpu().tolist(),
                'prequery_group_log_likelihood': likelihood.cpu().tolist(),
                'group_weights_after_query': log_weights[:, index].exp().cpu().tolist(),
                'prequery_physical_probability': float(raw_p[index]), 'selection_s': selection_s,
                'reason': 'common_initialization' if step < len(initial_indices) else mode}
        selected.append(index)
        labels.append(feedback.collision)
        remaining.remove(index)
        del learned_p, weights, learned_pooled, raw_p
        started = time.perf_counter()
        model = copy.deepcopy(base)
        fit_target(model, x[selected], labels, steps=10, learning_rate=.003)
        if model.mean.is_cuda:
            torch.cuda.synchronize()
        row['fsbo_adaptation_s'] = time.perf_counter()-started
        row['joint_adaptation_s'] = 0.
        if auxiliary_base is not None:
            started = time.perf_counter()
            other_model = copy.deepcopy(auxiliary_base)
            fit_target(other_model, x[selected], labels, steps=10, learning_rate=.003)
            if other_model.mean.is_cuda:
                torch.cuda.synchronize()
            row['joint_adaptation_s'] = time.perf_counter()-started
        started = time.perf_counter()
        physical = copy.deepcopy(physical_base)
        fit_target(physical, x[selected], labels, steps=10, learning_rate=.01)
        if physical.mean.is_cuda:
            torch.cuda.synchronize()
        row['physical_adaptation_s'] = time.perf_counter()-started
        row['adaptation_s'] = sum(row[k] for k in ('fsbo_adaptation_s', 'joint_adaptation_s', 'physical_adaptation_s'))
        queries.append(row)
    return {'method': mode, 'queries': queries, 'selected_indices': selected, 'actual_queries': len(queries),
        'maximum_queries': budget, 'termination': 'budget_exhausted',
        'component_names': [*names, 'target_physical'], 'group_prior_weights': prior.cpu().tolist(),
        'physical_mass': .1, 'beta': effective_beta if sources is not None else None,
        'reliability_bandwidth': bandwidth, 'reference_target_mean': reference_mean,
        'feedback_used': 'paid collision labels only', 'uses_source_posteriors': sources is not None}
