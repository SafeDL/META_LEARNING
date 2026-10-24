"""Prequential source reliability with a separately adapted target GP."""

import copy
import time

import numpy as np
from scipy.special import logsumexp
import torch

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target


def reliability_update(log_weight, x, index, probability, collision, bandwidth):
    if type(collision) is not bool:
        raise TypeError('Reliability updates use actual Boolean collision feedback')
    likelihood = np.log(np.maximum(probability if collision else 1-probability,1e-15))
    influence = (np.ones(len(x)) if bandwidth is None else
                 np.exp(-np.square(x-x[index]).mean(1)/(2*bandwidth**2)))
    updated = log_weight + likelihood[:,None]*influence
    return updated-logsumexp(updated,axis=0),likelihood


def select(candidates, oracle, *, initial_indices, pretrained, source_model,
           bandwidth=.3, raw_target=False, budget=128):
    torch.set_num_threads(1)
    if (not 0 < len(initial_indices) <= budget <= len(candidates)
            or len(set(initial_indices)) != len(initial_indices)
            or any(type(i) is not int or not 0 <= i < len(candidates) for i in initial_indices)):
        raise ValueError('Initialization must be unique and inside the paid budget')
    x = features(candidates)
    if raw_target:
        if source_model is not None:
            raise ValueError('The two-target control cannot receive historical source posteriors')
        raw_base = ClassificationKernel(x.shape[1])
        raw_model = copy.deepcopy(raw_base)
        source_names,source_prior = ['target_physical_gp'],np.ones(1)
    else:
        if (source_model is None or source_model.observed
                or source_model.mean.shape[1] != len(candidates)):
            raise ValueError('A fresh source posterior must match the public candidate pool')
        source_model.fixed_weights = True
        source_names,source_prior = source_model.component_names,source_model.weights
    names = [*source_names,'target_fsbo']
    # Retain the original source/physical proportions and add one target prior.
    prior = np.r_[source_prior,.25] if len(source_prior) > 1 else np.array([1.,1.])
    prior /= prior.sum()
    log_weight = np.broadcast_to(np.log(prior[:,None]),(len(prior),len(x))).copy()
    target_base = pretrained
    target_model = copy.deepcopy(target_base)
    selected,labels,queries = [],[],[]
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = time.perf_counter()
        with torch.no_grad():
            target_p = collision_probabilities(target_model,x[selected],labels,x).numpy()
            source_p = (collision_probabilities(raw_model,x[selected],labels,x).numpy()[None]
                        if raw_target else source_model.component_probabilities())
        component_p = np.vstack((source_p,target_p))
        probability = (np.exp(log_weight)*component_p).sum(0)
        if step < len(initial_indices):
            index = initial_indices[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min((i for i in remaining if probability[i] >= maximum-1e-12),
                        key=lambda i:candidates[i].scenario_id)
        selection_s = time.perf_counter()-started
        feedback = oracle.query(index,purpose='initial' if step < len(initial_indices) else 'target')
        if not isinstance(feedback,QueryFeedback) or feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError('Adaptive target mixture received a different paid scenario')
        selected.append(index)
        labels.append(feedback.collision)
        remaining.remove(index)
        before = np.exp(log_weight[:,index]).copy()
        log_weight,likelihood = reliability_update(log_weight,x,index,component_p[:,index],
                                                  feedback.collision,bandwidth)
        started = time.perf_counter()
        if not raw_target:
            source_model.observe(index,feedback.collision)
        else:
            raw_model = copy.deepcopy(raw_base)
            fit_target(raw_model,x[selected],labels,steps=10,learning_rate=.01)
        target_model = copy.deepcopy(target_base)
        fit_target(target_model,x[selected],labels,steps=10,learning_rate=.003)
        queries.append({'query':step+1,'index':index,'feedback':feedback.to_dict(),
                        'prequery_probability':float(probability[index]),
                        'component_names':names,'prequery_component_probabilities':component_p[:,index].tolist(),
                        'prequery_component_weights':before.tolist(),
                        'prequery_component_log_likelihood':likelihood.tolist(),
                        'component_weights_after_query':np.exp(log_weight[:,index]).tolist(),
                        'selection_s':selection_s,'adaptation_s':time.perf_counter()-started,
                        'reason':'common_initialization' if step < len(initial_indices) else 'adaptive_target_mixture'})
    return {'method':'adaptive_target_mixture','queries':queries,'selected_indices':selected,
            'actual_queries':len(queries),'maximum_queries':budget,'termination':'budget_exhausted',
            'component_prior_weights':prior.tolist(),'component_names':names,
            'feedback_used':'paid collision labels only'}
