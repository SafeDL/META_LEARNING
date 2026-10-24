"""ScaML-GP prior structure adapted to collision classification on a fixed pool."""

import copy
import math
import time

import numpy as np
from scipy.optimize import minimize
import torch

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.common.classification import torch_moments
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.common.gp_laplace import laplace, posterior


SOURCE_MODES = ('scaml_classifier', 'physical_scaml_classifier')
TARGET_MODES = ('scaml_target', 'physical_scaml_target')
MODES = (*SOURCE_MODES, *TARGET_MODES)
OPTIMIZER = {'maxiter': 100, 'maxfun': 500, 'ftol': 1e-9, 'gtol': 1e-5, 'maxls': 30}
KERNEL_BOUNDS = (math.log(1e-4), math.log(1e2))
WEIGHT_FLOOR = 1e-8


def residual_covariance(encoded, log_lengthscale, log_variance):
    distance = torch.cdist(encoded, encoded) / log_lengthscale.exp()
    r = math.sqrt(5) * distance
    return log_variance.exp() * (1 + r + r.square() / 3) * (-r).exp()


def source_prior(target_mean, target_covariance, source_mean, source_covariance,
                 weights, source_reference):
    mean = target_mean + (weights[:, None] * (source_mean - source_reference)).sum(0)
    covariance = target_covariance + (weights[:, None, None].square() * source_covariance).sum(0)
    return mean, covariance


def fit_prior(encoded, target_mean, source_mean, source_covariance, source_reference,
              observed, labels, initial_log_lengthscale, initial_log_variance):
    h = len(source_mean)
    initial = np.array([initial_log_lengthscale, initial_log_variance,
                        *([1 / h] * h if h else [])], dtype=float)
    initial[:2] = np.clip(initial[:2], *KERNEL_BOUNDS)
    bounds = [KERNEL_BOUNDS, KERNEL_BOUNDS, *([(WEIGHT_FLOOR, None)] * h)]
    if not observed:
        parameters = torch.as_tensor(initial, dtype=encoded.dtype, device=encoded.device)
        info = {'status': 'no_feedback', 'success': True, 'iterations': 0, 'evaluations': 0,
                'initial_objective': None, 'objective': None, 'projected_gradient_max': None}
    else:
        indices = torch.as_tensor(observed, dtype=torch.long, device=encoded.device)
        z = encoded[indices]
        means = source_mean[:, indices]
        covariances = source_covariance[:, indices][:, :, indices]
        evaluations = 0
        mode_iterations, mode_gradients = [], []

        def objective(values):
            nonlocal evaluations
            parameters = torch.tensor(values, dtype=encoded.dtype, device=encoded.device, requires_grad=True)
            residual = residual_covariance(z, parameters[0], parameters[1])
            mean, covariance = source_prior(target_mean, residual, means, covariances,
                                             parameters[2:], source_reference)
            # Gamma(1, 1) weight priors; residual kernel fits Bernoulli evidence.
            state = laplace(mean, covariance, labels)
            mode_iterations.append(state.iterations)
            mode_gradients.append(state.gradient_max)
            loss = state.nll + parameters[2:].sum()
            gradient = torch.autograd.grad(loss, parameters)[0]
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite ScaML collision evidence or gradient')
            evaluations += 1
            return float(loss.detach()), gradient.detach().cpu().numpy()

        initial_loss, _ = objective(initial)
        fitted = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=OPTIMIZER)
        if (not fitted.success and fitted.status != 1
                or not np.isfinite(fitted.fun) or not np.isfinite(fitted.x).all()
                or fitted.fun > initial_loss + 1e-7):
            raise FloatingPointError(f'ScaML collision optimization failed: {fitted.message}')
        parameters = torch.as_tensor(fitted.x, dtype=encoded.dtype, device=encoded.device)
        projected = fitted.jac.copy()
        for i, (lower, upper) in enumerate(bounds):
            if fitted.x[i] <= lower + 1e-10 and projected[i] > 0:
                projected[i] = 0
            if upper is not None and fitted.x[i] >= upper - 1e-10 and projected[i] < 0:
                projected[i] = 0
        info = {'status': str(fitted.message), 'success': bool(fitted.success),
                'iterations': int(fitted.nit), 'evaluations': evaluations,
                'initial_objective': initial_loss, 'objective': float(fitted.fun),
                'projected_gradient_max': float(np.max(np.abs(projected))),
                'maximum_mode_iterations': max(mode_iterations),
                'maximum_mode_gradient': max(mode_gradients)}
    with torch.no_grad():
        mean, covariance = source_prior(target_mean,
            residual_covariance(encoded, parameters[0], parameters[1]), source_mean, source_covariance,
            parameters[2:], source_reference)
    return mean, covariance, parameters, info


def predict(model, x, observed, labels, sources, source_reference):
    with torch.no_grad():
        encoded = model.features(x).detach()
        target_mean = model.mean.detach()
        initial_l = float(model.log_lengthscale)
        initial_v = 2 * float(model.log_scale)
    started = time.perf_counter()
    mean, covariance, parameters, info = fit_prior(encoded, target_mean, sources['means'],
        sources['covariances'], source_reference, observed, labels, initial_l, initial_v)
    if mean.is_cuda:
        torch.cuda.synchronize()
    optimization_s = time.perf_counter() - started
    with torch.no_grad():
        posterior_mean, posterior_covariance, mode_info = posterior(mean, covariance, observed, labels)
        p = torch_moments(posterior_mean, posterior_covariance.diagonal().clamp_min(1e-10))[0]
    info['prediction_mode'] = mode_info
    return p, parameters, info, optimization_s


def select(candidates, oracle, *, pretrained, sources, source_reference=None, initial_indices,
           mode='scaml_classifier', budget=128, device=None, feature_fn=features):
    torch.set_num_threads(1)
    if mode not in MODES:
        raise ValueError('Unknown ScaML collision comparison')
    if (not 0 < len(initial_indices) <= budget <= len(candidates)
            or len(set(initial_indices)) != len(initial_indices)
            or any(type(i) is not int or not 0 <= i < len(candidates) for i in initial_indices)):
        raise ValueError('ScaML initialization must be unique and included in the paid budget')
    target_only = mode in TARGET_MODES
    if target_only:
        if sources is not None or source_reference is not None:
            raise ValueError('ScaML target controls cannot receive explicit source posteriors')
        names = []
    else:
        if sources is None or source_reference is None:
            raise ValueError('ScaML source versions require explicit source posteriors and reference mean')
        names = sources['names']
        if not names or len(names) != len(set(names)) or not np.isfinite(source_reference):
            raise ValueError('ScaML requires named historical sources and their finite reference mean')
        order = sorted(range(len(names)), key=lambda i: names[i])
        names = [names[i] for i in order]
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    base = copy.deepcopy(pretrained).to(device)
    model = copy.deepcopy(base)
    x = feature_fn(candidates)
    if target_only:
        sources = {'means': torch.empty((0, len(x)), dtype=torch.float64, device=device),
                   'covariances': torch.empty((0, len(x), len(x)), dtype=torch.float64, device=device)}
    else:
        sources = {k: torch.as_tensor(sources[k], dtype=torch.float64, device=device)[order]
                   for k in ('means', 'covariances')}
    mass = .1 if mode.startswith('physical_') else 0.
    raw_base = ClassificationKernel(x.shape[1]).to(device) if mass else None
    raw = copy.deepcopy(raw_base)
    selected, labels, queries = [], [], []
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = time.perf_counter()
        learned, parameters, fitted, optimization_s = predict(model, x, selected, labels,
                                                             sources, 0. if target_only else source_reference)
        with torch.no_grad():
            physical = None if raw is None else collision_probabilities(raw, x[selected], labels, x)
            p = learned if physical is None else (1 - mass) * learned + mass * physical
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
            raise ValueError('ScaML received a different paid scenario')
        row = {'query': step + 1, 'index': index, 'feedback': feedback.to_dict(),
               'prequery_probability': float(probability[index]),
               'prequery_learned_probability': float(learned[index]),
               'prequery_physical_probability': None if physical is None else float(physical[index]),
               'prequery_source_weights': parameters[2:].cpu().tolist(),
               'prequery_log_lengthscale': float(parameters[0]),
               'prequery_log_variance': float(parameters[1]),
               'prequery_target_mean': float(model.mean.detach()), 'optimizer': fitted,
               'prior_optimization_s': optimization_s, 'selection_s': selection_s,
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
        if raw_base is not None:
            started = time.perf_counter()
            raw = copy.deepcopy(raw_base)
            fit_target(raw, x[selected], labels, steps=10, learning_rate=.01)
            if raw.mean.is_cuda:
                torch.cuda.synchronize()
            row['physical_adaptation_s'] = time.perf_counter() - started
        row['adaptation_s'] = row['target_adaptation_s'] + row['physical_adaptation_s']
        queries.append(row)
    return {'method': mode, 'queries': queries, 'selected_indices': selected,
            'actual_queries': len(queries), 'maximum_queries': budget, 'termination': 'budget_exhausted',
            'source_names': names, 'source_reference_mean': source_reference, 'physical_mass': mass,
            'source_weights_normalized': False,
            'source_weight_prior': None if target_only else 'independent Gamma(1,1)',
            'uses_source_posteriors': not target_only,
            'residual_hyperparameters': 'Bernoulli evidence; no continuous-response kernel priors',
            'target_representation': 'reset FSBO; collision-only adaptation; fixed during scalar optimization',
            'optimizer_settings': OPTIMIZER, 'feedback_used': 'paid collision labels only'}
