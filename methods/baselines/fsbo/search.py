"""Original FSBO fit and acquisition rules with an explicit inference device."""

import copy
import time

import torch

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.common.gp_classifier import collision_probabilities, fit_target


def select(candidates,oracle,*,pretrained,initial_indices,budget=128,device='cuda',feature_fn=features):
    torch.set_num_threads(1)
    if (not 0 < len(initial_indices) <= budget <= len(candidates)
            or len(set(initial_indices)) != len(initial_indices)
            or any(type(i) is not int or not 0 <= i < len(candidates) for i in initial_indices)):
        raise ValueError('FSBO initialization must be unique and included in the paid budget')
    base = pretrained.to(device)
    model = copy.deepcopy(base)
    x = feature_fn(candidates)
    selected,labels,queries = [],[],[]
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = time.perf_counter()
        with torch.no_grad():
            probability = collision_probabilities(model,x[selected],labels,x).cpu().numpy()
        if step < len(initial_indices):
            index = initial_indices[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min((i for i in remaining if probability[i] >= maximum-1e-12),
                        key=lambda i:candidates[i].scenario_id)
        selection_s = time.perf_counter()-started
        feedback = oracle.query(index,purpose='initial' if step < len(initial_indices) else 'target')
        if not isinstance(feedback,QueryFeedback) or feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError('FSBO received a different paid scenario')
        selected.append(index)
        labels.append(feedback.collision)
        remaining.remove(index)
        started = time.perf_counter()
        model = copy.deepcopy(base)
        fit_target(model,x[selected],labels,steps=10,learning_rate=.003)
        if model.mean.is_cuda:
            torch.cuda.synchronize()
        queries.append({'query':step+1,'index':index,'feedback':feedback.to_dict(),
                        'prequery_probability':float(probability[index]),'selection_s':selection_s,
                        'adaptation_s':time.perf_counter()-started,
                        'reason':'common_initialization' if step < len(initial_indices) else 'fsbo_cuda'})
    return {'method':'fsbo_cuda','queries':queries,'selected_indices':selected,'actual_queries':len(queries),
            'maximum_queries':budget,'termination':'budget_exhausted','feedback_used':'paid collision labels only',
            'device':str(model.mean.device)}
