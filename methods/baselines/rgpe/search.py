"""Ranking-weighted GP ensemble adapted explicitly to collision classification."""

import copy
import time

import torch

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.common.classification import torch_moments
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.common.gp_joint import bernoulli_posterior


MODES = ('rgpe_classifier', 'physical_rgpe_classifier')
RANK_SAMPLES = 512


def ranking_losses(samples, labels):
    """Use ordered unequal-label pairs; the middle target axis indexes LOO folds."""
    labels = torch.as_tensor(labels, dtype=samples.dtype, device=samples.device)
    unequal = labels[:, None] != labels[None, :]
    desired = labels[:, None] < labels[None, :]
    if samples.ndim == 3:
        predicted = samples.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) < samples
    else:
        predicted = samples.unsqueeze(-1) < samples.unsqueeze(-2)
    return ((predicted ^ desired) & unequal).sum(dim=(-1, -2))


def weights_from_losses(losses, generator):
    """Paper's median/95th-percentile pruning and target-first tie rule."""
    threshold = torch.quantile(losses[-1].double(), .95)
    medians = torch.quantile(losses.double(), .5, dim=1)
    pruned = medians > threshold
    pruned[-1] = False
    eligible = ~pruned[:, None] & (losses == losses.masked_fill(pruned[:, None],
                                                               torch.iinfo(losses.dtype).max).min(0).values)
    noise = torch.rand(losses.shape, dtype=torch.float64, device=losses.device, generator=generator)
    winners = noise.masked_fill(~eligible, -1).argmax(0)
    winners[eligible[-1]] = len(losses) - 1
    weights = torch.bincount(winners, minlength=len(losses)).to(torch.float64) / losses.shape[-1]
    return weights, pruned, medians, threshold


def gaussian_samples(mean, covariance, count, generator):
    eye = torch.eye(mean.shape[-1], dtype=mean.dtype, device=mean.device)
    factor = torch.linalg.cholesky(covariance + 1e-8 * eye)
    noise = torch.randn((count, *mean.shape), dtype=mean.dtype, device=mean.device,
                        generator=generator)
    return mean.unsqueeze(0) + (factor.unsqueeze(0) @ noise.unsqueeze(-1)).squeeze(-1)


def rank_weights(source_mean, source_covariance, target_mean, target_covariance,
                 observed, labels, *, seed, samples=RANK_SAMPLES):
    count = len(labels)
    components = len(source_mean) + 1
    y = torch.as_tensor(labels, dtype=target_mean.dtype, device=target_mean.device)
    if count < 2 or y.unique().numel() < 2:
        weights = target_mean.new_zeros(components)
        weights[-1] = 1.
        return weights, {'pruned': [False] * components, 'median_losses': [0.] * components,
                         'target_loss_percentile_95': 0., 'informative_ordered_pairs': 0}
    indices = torch.as_tensor(observed, dtype=torch.long, device=target_mean.device)
    generator = torch.Generator(device=target_mean.device).manual_seed(seed)
    means = source_mean[:, indices]
    covariances = source_covariance[:, indices][:, :, indices]
    source_draws = gaussian_samples(means, covariances, samples, generator)
    losses = [ranking_losses(source_draws[:, s], y) for s in range(len(source_mean))]
    target_m = target_mean[indices]
    target_k = target_covariance[indices][:, indices]
    positions = torch.arange(count, device=target_mean.device)
    loo = positions.expand(count, -1)[~torch.eye(count, device=target_mean.device, dtype=torch.bool)]
    loo = loo.reshape(count, count - 1)
    loo_mean, loo_covariance = bernoulli_posterior(target_m, target_k, loo, y[loo])
    target_draws = gaussian_samples(loo_mean, loo_covariance, samples, generator)
    losses.append(ranking_losses(target_draws, y))
    weights, pruned, medians, threshold = weights_from_losses(torch.stack(losses), generator)
    return weights, {'pruned': pruned.cpu().tolist(), 'median_losses': medians.cpu().tolist(),
                     'target_loss_percentile_95': float(threshold),
                     'informative_ordered_pairs': int((y[:, None] != y[None, :]).sum())}


def ensemble_probability(means, variances, weights):
    mean = (weights[:, None] * means).sum(0)
    variance = (weights[:, None].square() * variances).sum(0).clamp_min(1e-10)
    return torch_moments(mean, variance)[0]


def probabilities(model, x, observed, labels, source_mean, source_covariance, *, seed):
    mean = model.mean.expand(len(x))
    covariance = model.covariance(x)
    weights, diagnostics = rank_weights(source_mean, source_covariance, mean, covariance,
                                       observed, labels, seed=seed)
    target_mean, target_covariance = bernoulli_posterior(mean, covariance, observed, labels)
    means = torch.cat((source_mean, target_mean[None]))
    variances = torch.cat((source_covariance.diagonal(dim1=-2, dim2=-1),
                           target_covariance.diagonal()[None]))
    return ensemble_probability(means, variances, weights), weights, diagnostics


def select(candidates, oracle, *, pretrained, sources, initial_indices, sampling_seed,
           mode='rgpe_classifier', budget=128, device=None, feature_fn=features):
    torch.set_num_threads(1)
    if mode not in MODES:
        raise ValueError('Unknown RGPE collision comparison')
    if (not 0 < len(initial_indices) <= budget <= len(candidates)
            or len(set(initial_indices)) != len(initial_indices)
            or any(type(i) is not int or not 0 <= i < len(candidates) for i in initial_indices)):
        raise ValueError('RGPE initialization must be unique and included in the paid budget')
    names = sources['names']
    if not names or len(set(names)) != len(names):
        raise ValueError('RGPE requires uniquely named historical sources')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    base = copy.deepcopy(pretrained).to(device)
    model = copy.deepcopy(base)
    x = feature_fn(candidates)
    order = sorted(range(len(names)), key=lambda s: names[s])
    names = [names[s] for s in order]
    source_mean = torch.as_tensor(sources['means'], dtype=torch.float64, device=device)[order]
    source_covariance = torch.as_tensor(sources['covariances'], dtype=torch.float64, device=device)[order]
    physical_mass = .1 if mode == 'physical_rgpe_classifier' else 0.
    physical_base = ClassificationKernel(x.shape[1]).to(device) if physical_mass else None
    physical = copy.deepcopy(physical_base)
    selected, labels, queries = [], [], []
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = time.perf_counter()
        with torch.no_grad():
            p, weights, diagnostics = probabilities(model, x, selected, labels, source_mean,
                                                    source_covariance, seed=sampling_seed + 1009 * step)
            learned_p = p
            raw_p = None
            if physical is not None:
                raw_p = collision_probabilities(physical, x[selected], labels, x)
                p = (1 - physical_mass) * p + physical_mass * raw_p
            probability = p.cpu().numpy()
        if step < len(initial_indices):
            index = initial_indices[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min((i for i in remaining if probability[i] >= maximum - 1e-12),
                        key=lambda i: candidates[i].scenario_id)
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose='initial' if step < len(initial_indices) else 'target')
        if not isinstance(feedback, QueryFeedback) or feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError('RGPE received a different paid scenario')
        row = {'query': step + 1, 'index': index, 'feedback': feedback.to_dict(),
               'prequery_probability': float(probability[index]),
               'prequery_ensemble_probability': float(learned_p[index]),
               'prequery_physical_probability': None if raw_p is None else float(raw_p[index]),
               'prequery_rank_weights': weights.cpu().tolist(),
               'prequery_ranking': diagnostics, 'selection_s': selection_s,
               'reason': 'common_initialization' if step < len(initial_indices) else mode}
        selected.append(index)
        labels.append(feedback.collision)
        remaining.remove(index)
        started = time.perf_counter()
        model = copy.deepcopy(base)
        fit_target(model, x[selected], labels, steps=10, learning_rate=.003)
        if model.mean.is_cuda:
            torch.cuda.synchronize()
        row['target_adaptation_s'] = time.perf_counter() - started
        row['physical_adaptation_s'] = 0.
        if physical_base is not None:
            started = time.perf_counter()
            physical = copy.deepcopy(physical_base)
            fit_target(physical, x[selected], labels, steps=10, learning_rate=.01)
            if physical.mean.is_cuda:
                torch.cuda.synchronize()
            row['physical_adaptation_s'] = time.perf_counter() - started
        row['adaptation_s'] = row['target_adaptation_s'] + row['physical_adaptation_s']
        queries.append(row)
    return {'method': mode, 'queries': queries, 'selected_indices': selected,
            'actual_queries': len(queries), 'maximum_queries': budget, 'termination': 'budget_exhausted',
            'component_names': [*names, 'target_fsbo'], 'ranking_samples': RANK_SAMPLES,
            'sampling_seed': sampling_seed, 'physical_mass': physical_mass,
            'feedback_used': 'paid collision labels only',
            'ranking_ties': 'equal labels excluded; loss ties prefer target, otherwise random',
            'target_cross_validation': 'independent Laplace LOO modes with fixed adapted kernel'}
