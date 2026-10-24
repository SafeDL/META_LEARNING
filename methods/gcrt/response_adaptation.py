"""Target-adapted similarity between independent historical response fields."""

import copy
import math
import time

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from torch import nn

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.common.gp_classifier import collision_probabilities, fit_target


SURROGATE_CONFIG = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=15,
                        min_samples_leaf=12, l2_regularization=1, random_state=1601)


def response_coordinates(history, x):
    if not history or any(archive.role != 'history' for archive in history):
        raise ValueError('Response predictors train on historical archives only')
    columns = []
    for archive in history:
        if len(np.unique(archive.collision)) == 1:
            probability = np.full(len(x), (archive.collision.sum() + 0.5) /
                                  (len(archive.collision) + 1))
        else:
            model = HistGradientBoostingClassifier(**SURROGATE_CONFIG)
            model.fit(archive.x, archive.collision)
            probability = model.predict_proba(x)[:, 1]
        columns.append(logit(np.clip(probability, 0.01, 0.99)) / logit(0.99))
    return np.column_stack(columns)


class ResponseKernel(nn.Module):
    def __init__(self, target_kernel, sources, *, adaptive=True, response=True):
        super().__init__()
        self.target_kernel = target_kernel
        self.dimension = target_kernel.dimension
        self.sources = sources
        self.response = response
        self.log_response_scale = nn.Parameter(
            torch.full((sources,), -0.5 * math.log(sources), dtype=torch.float64),
            requires_grad=adaptive)
        self.log_response_lengthscale = nn.Parameter(
            torch.full((sources,), math.log(0.3), dtype=torch.float64),
            requires_grad=adaptive)

    @property
    def mean(self):
        return self.target_kernel.mean

    def covariance(self, x, other=None):
        x = torch.as_tensor(x, dtype=torch.float64, device=self.mean.device)
        other = x if other is None else torch.as_tensor(
            other, dtype=torch.float64, device=self.mean.device)
        covariance = self.target_kernel.covariance(
            x[:, :self.dimension], other[:, :self.dimension])
        if self.response:
            difference = x[:, None, self.dimension:] - other[None, :, self.dimension:]
            lengthscale = self.log_response_lengthscale.exp().clamp_min(0.01)
            source_kernels = (-0.5 * (difference / lengthscale).square()).exp()
            covariance = covariance + (
                source_kernels * self.log_response_scale.exp().square()).sum(-1)
        return covariance


def select(candidates, oracle, *, initial_indices, pretrained, responses,
           adaptive=True, budget=128):
    torch.set_num_threads(1)
    if (not 0 < len(initial_indices) <= budget <= len(candidates)
            or len(set(initial_indices)) != len(initial_indices)
            or any(type(i) is not int or not 0 <= i < len(candidates)
                   for i in initial_indices)):
        raise ValueError('Common initialization must be unique and inside the paid budget')
    x = np.column_stack((features(candidates), responses))
    base = ResponseKernel(pretrained, responses.shape[1], adaptive=adaptive)
    model = copy.deepcopy(base)
    selected, labels, queries = [], [], []
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = time.perf_counter()
        with torch.no_grad():
            probability = collision_probabilities(model, x[selected], labels, x).numpy()
        if step < len(initial_indices):
            index = initial_indices[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min((i for i in remaining if probability[i] >= maximum - 1e-12),
                        key=lambda i: candidates[i].scenario_id)
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose='initial' if step < len(initial_indices) else 'target')
        if not isinstance(feedback, QueryFeedback) or feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError('Response adaptation received feedback for a different paid query')
        selected.append(index)
        labels.append(feedback.collision)
        remaining.remove(index)
        started = time.perf_counter()
        model = copy.deepcopy(base)
        fit_target(model, x[selected], labels, steps=10, learning_rate=0.003)
        queries.append({'query':step + 1, 'index':index, 'feedback':feedback.to_dict(),
                        'prequery_probability':float(probability[index]),
                        'selection_s':selection_s,
                        'adaptation_s':time.perf_counter() - started,
                        'response_variances':model.log_response_scale.exp().square().detach().tolist(),
                        'response_lengthscales':model.log_response_lengthscale.exp().detach().tolist(),
                        'reason':'common_initialization' if step < len(initial_indices) else
                                 'response_adaptive' if adaptive else 'response_fixed'})
    return {'method':'response_adaptive' if adaptive else 'response_fixed', 'queries':queries,
            'selected_indices':selected, 'actual_queries':len(queries), 'maximum_queries':budget,
            'termination':'budget_exhausted', 'feedback_used':'paid collision labels'}
