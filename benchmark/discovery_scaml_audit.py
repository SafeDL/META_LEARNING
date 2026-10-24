"""Historical-only numerical and optimization checks for collision ScaML-GP."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import root, root_scalar
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.baselines.scaml_gp.search import (KERNEL_BOUNDS, WEIGHT_FLOOR,
                                              fit_prior, residual_covariance, source_prior)
from methods.common.gp_classifier import fit_target
from methods.common.gp_classifier import laplace as legacy_laplace
from methods.common.gp_laplace import laplace, posterior
from methods.common.classification import torch_moments
from methods.gcrt.joint_training import load_model

from .construction import load_construction
from .discovery_config import read_json, write_json
from .discovery_physical_mixture import MAIN, checkpoint_path
from .discovery_physical_mixture_audit import classifier_probabilities_numpy, raw_banks
from .discovery_residual_scale import ARTIFACTS, REPORT as LEARNING
from .protocol import ROOT


REPORT = ROOT / 'results/baselines/scaml_gp/collision/preparation'


def prior_numpy(encoded, target_mean, source_mean, source_covariance, source_reference, parameters):
    distance = np.linalg.norm(encoded[:, None, :] - encoded[None, :, :], axis=-1)
    r = math.sqrt(5) * distance / np.exp(parameters[0])
    residual = np.exp(parameters[1]) * (1 + r + r**2 / 3) * np.exp(-r)
    weights = parameters[2:]
    mean = target_mean + weights @ (source_mean - source_reference)
    covariance = residual + np.einsum('h,hij->ij', weights**2, source_covariance)
    return mean, covariance


def mode_numpy(mean, covariance, labels):
    labels = np.asarray(labels, dtype=float)
    k = covariance + 1e-5 * np.eye(len(labels))
    factor = np.linalg.cholesky(k)

    def gradient(z):
        return z + factor.T @ (expit(mean + factor @ z) - labels)

    def hessian(z):
        p = expit(mean + factor @ z)
        return np.eye(len(labels)) + factor.T @ ((p * (1 - p))[:, None] * factor)

    solved = root(gradient, np.zeros(len(labels)), jac=hessian, method='hybr',
                  options={'xtol': 1e-11, 'maxfev': 1000})
    maximum = float(np.max(np.abs(gradient(solved.x))))
    if maximum > 5e-9:
        raise FloatingPointError(f'Independent Bernoulli root failed: gradient {maximum}; {solved.message}')
    latent = mean + factor @ solved.x
    p = expit(latent)
    curvature = np.sqrt(np.maximum(p * (1 - p), 1e-12))
    posterior_factor = np.linalg.cholesky(np.eye(len(labels)) + curvature[:, None] * k * curvature[None, :])
    nll = (.5 * np.square(solved.x).sum() + np.sum(np.logaddexp(0, latent) - labels * latent)
           + np.log(np.diag(posterior_factor)).sum())
    return nll, np.linalg.solve(factor.T, solved.x), curvature, posterior_factor, maximum


def evidence_numpy(mean, covariance, labels):
    return mode_numpy(mean, covariance, labels)[0]


def posterior_numpy(mean, covariance, observed, labels):
    indices = np.asarray(observed, dtype=int)
    state = mode_numpy(mean[indices], covariance[np.ix_(indices, indices)], labels)
    _, alpha, curvature, factor, _ = state
    cross = covariance[indices]
    solved = np.linalg.solve(factor, curvature[:, None] * cross)
    return mean + cross.T @ alpha, covariance - solved.T @ solved


def finite_gradient(objective, parameters):
    """Fourth-order differences; stay within the fitted parameter bounds."""
    result = []
    bounds = [KERNEL_BOUNDS, KERNEL_BOUNDS, *([(WEIGHT_FLOOR, None)] * (len(parameters) - 2))]
    for i, (lower, upper) in enumerate(bounds):
        step = 1e-3 * max(1, abs(parameters[i]))

        def shifted(multiplier):
            value = parameters.copy()
            value[i] += multiplier * step
            return objective(value)

        if parameters[i] - 2 * step < lower:
            derivative = (-25 * shifted(0) + 48 * shifted(1) - 36 * shifted(2)
                          + 16 * shifted(3) - 3 * shifted(4)) / (12 * step)
        elif upper is not None and parameters[i] + 2 * step > upper:
            derivative = (25 * shifted(0) - 48 * shifted(-1) + 36 * shifted(-2)
                          - 16 * shifted(-3) + 3 * shifted(-4)) / (12 * step)
        else:
            derivative = (shifted(-2) - 8 * shifted(-1) + 8 * shifted(1) - shifted(2)) / (12 * step)
        result.append(derivative)
    return np.asarray(result)


def historical_check(*, target_only=False):
    torch.set_num_threads(1)
    receipt = read_json(LEARNING / 'selection_audit.json')
    if receipt['status'] != 'passed' or receipt['unique_historical_cache_tasks'] != 16:
        raise ValueError('Complete the registered historical source construction before ScaML checks')
    started = time.perf_counter()
    checks = []
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            history, _ = load_construction(kind)
            if len(history) != 8 or any(a.role != 'history' for a in history):
                raise ValueError('ScaML numerical preparation uses only the eight historical objects')
            if any(not np.array_equal(a.x, history[0].x) for a in history):
                raise ValueError('Historical ScaML checks require the shared historical pool')
            if target_only:
                all_m, all_k = np.empty((0, 512)), np.empty((0, 512, 512))
                reference = 0.
            else:
                with np.load(ARTIFACTS / 'priors' / f'{kind}.npz', allow_pickle=False) as saved:
                    setting = json.loads(str(saved['specification']))
                    if setting['source_ids'] != [a.source_id for a in history]:
                        raise ValueError('ScaML historical source order differs')
                    np.testing.assert_array_equal(setting['x'], history[0].x)
                    all_m, all_k = saved['means'].copy(), saved['covariances'].copy()
                teacher, _ = load_model(checkpoint_path(kind, MAIN))
                reference = float(teacher.mean.detach())
            base, _ = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            base = base.to(device)
            ordered = np.random.default_rng(1301).permutation(512)
            for pseudo_target in sorted(history, key=lambda a: a.source_id):
                order = ([] if target_only else
                         sorted((i for i, a in enumerate(history) if a.source_id != pseudo_target.source_id),
                                key=lambda i: history[i].source_id))
                source_m, source_k = all_m[order], all_k[order]
                sm = torch.as_tensor(source_m, dtype=torch.float64, device=device)
                sk = torch.as_tensor(source_k, dtype=torch.float64, device=device)
                for count in (8, 128):
                    model = copy.deepcopy(base)
                    paid, labels = ordered[:count].tolist(), pseudo_target.collision[ordered[:count]].tolist()
                    fit_target(model, pseudo_target.x[paid], labels, steps=10, learning_rate=.003)
                    with torch.no_grad():
                        z = model.features(pseudo_target.x).detach()
                        target_mean = model.mean.detach()
                    mean, covariance, parameters, info = fit_prior(z, target_mean, sm, sk, reference,
                        paid, labels, float(model.log_lengthscale), 2 * float(model.log_scale))
                    values, encoded = parameters.cpu().numpy(), z.cpu().numpy()
                    expected_m, expected_k = prior_numpy(encoded, float(target_mean), source_m,
                                                        source_k, reference, values)
                    np.testing.assert_allclose(mean.cpu(), expected_m, atol=2e-8, rtol=2e-8)
                    np.testing.assert_allclose(covariance.cpu(), expected_k, atol=2e-8, rtol=2e-8)
                    with torch.no_grad():
                        pm, pk, mode_info = posterior(mean, covariance, paid, labels)
                        actual_p = torch_moments(pm, pk.diagonal().clamp_min(1e-10))[0].cpu().numpy()
                    nm, nk = posterior_numpy(expected_m, expected_k, paid, labels)
                    nodes, weights = np.polynomial.hermite.hermgauss(32)
                    expected_p = expit(nm[:, None] + np.sqrt(2 * np.maximum(np.diag(nk), 1e-10))[:, None] * nodes)
                    expected_p = expected_p @ (weights / np.sqrt(np.pi))
                    np.testing.assert_allclose(actual_p, expected_p, atol=2e-7, rtol=2e-6)
                    zp, smp, skp = z[paid], sm[:, paid], sk[:, paid][:, :, paid]
                    parameters_t = parameters.clone().requires_grad_(True)
                    m, k = source_prior(target_mean, residual_covariance(zp, *parameters_t[:2]),
                                         smp, skp, parameters_t[2:], reference)
                    loss = laplace(m, k, labels).nll + parameters_t[2:].sum()
                    gradient = torch.autograd.grad(loss, parameters_t)[0].cpu().numpy()

                    def objective(v):
                        m, k = prior_numpy(encoded[paid], float(target_mean), source_m[:, paid],
                                           source_k[:, paid][:, :, paid], reference, v)
                        return evidence_numpy(m, k, labels) + v[2:].sum()

                    independent_loss = objective(values)
                    np.testing.assert_allclose(float(loss.detach()), independent_loss, atol=2e-6, rtol=2e-6)
                    independent_gradient = finite_gradient(objective, values)
                    np.testing.assert_allclose(gradient, independent_gradient, atol=2e-5, rtol=2e-4)
                    checks.append({'kind': kind, 'historical_pseudo_target': pseudo_target.source_id,
                        'source_ids': [history[i].source_id for i in order], 'support_count': count,
                        'full_pool_size': 512, 'source_reference': None if target_only else reference,
                        'probability_max_error': float(np.max(np.abs(actual_p - expected_p))),
                        'prior_covariance_max_error': float(np.max(np.abs(covariance.cpu().numpy() - expected_k))),
                        'objective_error': abs(float(loss.detach()) - independent_loss),
                        'gradient_max_error': float(np.max(np.abs(gradient - independent_gradient))),
                        'parameters': values.tolist(), 'optimizer': info, 'prediction_mode': mode_info})
                    print('scaml_historical_numerics', kind, pseudo_target.source_id, count,
                          'converged', info['success'], flush=True)
    result = {'status': 'passed', 'checks': checks, 'full_pool_predictions': len(checks),
              'independent_objectives': len(checks), 'independent_gradient_vectors': len(checks),
              'converged_optimizations': sum(c['optimizer']['success'] for c in checks),
              'source_context': ('no explicit source posterior; teacher uses all history, not a held-system evaluation'
                                 if target_only else
                                 'own pseudo source excluded; teacher uses all history, not a held-system evaluation'),
              'wall_s': time.perf_counter() - started, 'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(REPORT / ('target_numeric.json' if target_only else 'historical_numeric.json'), result)
    return result


def numerical_correction():
    torch.set_num_threads(1)
    started = time.perf_counter()
    old = read_json(REPORT / 'undamped_failure.json')
    with np.load(REPORT / 'undamped_failure_inputs.npz', allow_pickle=False) as data:
        encoded = data['encoded']
        target_mean, reference = float(data['target_mean']), float(data['source_reference'])
        sm, sk, labels = data['source_mean'], data['source_covariance'], data['labels']
        parameters = data['parameters']
    selected = old['perturbations'][2]
    rows = []
    with threadpool_limits(1):
        for item in selected['values']:
            values = parameters.copy()
            values[2] += item['step']
            mean, covariance = prior_numpy(encoded, target_mean, sm, sk, reference, values)
            state = laplace(torch.tensor(mean), torch.tensor(covariance), labels)
            expected = evidence_numpy(mean, covariance, labels) + values[2:].sum()
            actual = float(state.nll.detach()) + values[2:].sum()
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
            rows.append({'increment': item['step'], 'undamped_objective': item['torch_loss'],
                         'converged_objective': actual, 'independent_objective': expected,
                         'mode_gradient_max': state.gradient_max})
        simple = []
        for mean in np.linspace(-8, -1, 29):
            m, k = torch.tensor([mean]), torch.tensor([[100.]], dtype=torch.float64)
            prior = SimpleNamespace(mean=m, covariance=lambda _: k)
            old_loss = legacy_laplace(prior, torch.empty((1, 0)), [True], iterations=10)[0]
            state = laplace(m, k, [True])
            optimum = root_scalar(lambda f: (f - mean) / 100.00001 + expit(f) - 1,
                                  bracket=(mean, 100), method='brentq').root
            expected = (.5 * (optimum - mean)**2 / 100.00001 + np.logaddexp(0, -optimum)
                        + .5 * np.log1p(100.00001 * expit(optimum) * (1 - expit(optimum))))
            np.testing.assert_allclose(float(state.nll), expected, atol=1e-9, rtol=1e-9)
            simple.append({'prior_mean': float(mean), 'undamped_nll': float(old_loss),
                           'converged_nll': float(state.nll), 'closed_form_mode_nll': expected})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    axes[0].plot([r['increment'] for r in rows], [r['undamped_objective'] for r in rows],
                 'o-', color='#c24136', label='10 undamped updates')
    axes[0].plot([r['increment'] for r in rows], [r['converged_objective'] for r in rows],
                 's-', color='#2563a6', label='Damped, mode converged')
    axes[0].set(xscale='symlog', yscale='log', xlabel='Increment to one source coefficient',
                ylabel='Penalized Laplace negative log evidence',
                title='Historical DQN, cut-in, 128 fixed labels')
    axes[0].set_xscale('symlog', linthresh=1e-6)
    axes[0].set_xlim(0, 1.12 * max(r['increment'] for r in rows))
    axes[1].plot([r['prior_mean'] for r in simple], [r['undamped_nll'] for r in simple],
                 color='#c24136', label='10 undamped updates')
    axes[1].plot([r['prior_mean'] for r in simple], [r['converged_nll'] for r in simple],
                 color='#2563a6', label='Damped, mode converged')
    axes[1].set(xlabel='Prior latent mean', ylabel='Laplace negative log evidence',
                title='One positive label, prior variance = 100')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=9)
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(REPORT / f'numerical_correction.{extension}', dpi=170, bbox_inches='tight')
    plt.close(figure)
    result = {'status': 'passed', 'historical_fixed_parameters': rows, 'one_point_checks': simple,
              'historical_input': 'undamped_failure_inputs.npz',
              'scope': 'historical numerical correction; no development collision discovery claims',
              'wall_s': time.perf_counter() - started, 'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(REPORT / 'numerical_correction.json', result)
    return result


def check_run(row, candidates, feedbacks, output=None):
    from methods.baselines.scaml_gp.search import OPTIMIZER, SOURCE_MODES
    from methods.common.gp_classifier import ClassificationKernel
    from methods.common.gp_laplace import MODE_ITERATIONS, MODE_TOLERANCE
    from methods.common.response import archive_fingerprints
    from . import discovery_scaml as study
    from . import discovery_residual_online as residual
    from .discovery_config import development_tasks
    from .initialization import initial_indices
    from .scenarios import dedup_key, features
    from .schema import QueryFeedback

    torch.set_num_threads(1)
    started = time.perf_counter()
    with threadpool_limits(1):
        output = read_json(ROOT / row['path']) if output is None else output
        setting, result, metrics = output['specification'], output['result'], output['metrics']
        frozen = read_json(study.REPORT / 'protocol.json')
        task = {k: row[k] for k in development_tasks()[0]}
        method = row['method']
        if (setting['protocol'] != frozen or task not in frozen['tasks'] or setting['task'] != task
                or method not in study.METHODS or setting['method'] != method or result['method'] != method
                or setting['candidate_ids'] != [s.scenario_id for s in candidates]):
            raise ValueError('ScaML run differs from its registered method, task or candidates')
        queries, ledger = result['queries'], output['query_ledger']
        selected = [q['index'] for q in queries]
        source_mode = method in SOURCE_MODES
        if (len(queries) != 128 or len(ledger) != 128 or len(set(selected)) != 128
                or selected[:8] != initial_indices(512, 8, task['algorithm_seed'])
                or result['selected_indices'] != selected or result['actual_queries'] != 128
                or result['maximum_queries'] != 128 or result['termination'] != 'budget_exhausted'
                or result['physical_mass'] != frozen['physical_mass'][method]
                or result['uses_source_posteriors'] != source_mode
                or result['optimizer_settings'] != OPTIMIZER
                or result['source_weights_normalized'] is not False):
            raise ValueError('ScaML budget, initialization, source control or optimization configuration differs')
        failures = {dedup_key(s) for s, f in zip(candidates, feedbacks) if f['collision']}
        found, curve = set(), []
        for n, (q, record) in enumerate(zip(queries, ledger), 1):
            index = q['index']
            if (q['query'] != n or record['call'] != n or record['index'] != index
                    or record['status'] != 'complete' or record['account'] != 'target'
                    or record['scenario_id'] != candidates[index].scenario_id
                    or record['purpose'] != ('initial' if n <= 8 else 'target')
                    or record['feedback'] != q['feedback']
                    or QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[index]):
                raise ValueError('ScaML feedback differs from its original complete execution')
            if q['feedback']['collision']:
                found.add(dedup_key(candidates[index]))
            curve.append(len(found))
        recall = [v / len(failures) if failures else None for v in curve]
        if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != recall
                or metrics['unique_collisions'] != curve[-1] or metrics['collision_recall'] != recall[-1]
                or metrics['pool_unique_collisions'] != len(failures)
                or any(row[k] != v for k, v in metrics.items())):
            raise ValueError('ScaML metrics differ from the complete deduplicated collision pool')
        np.testing.assert_allclose(metrics['discovery_auc'], np.mean(curve), atol=1e-12)
        for budget in (16, 32, 64, 128):
            if metrics['checkpoints'][str(budget)] != {'unique_collisions': curve[budget - 1],
                                                      'collision_recall': recall[budget - 1]}:
                raise ValueError('ScaML checkpoint differs from its full discovery curve')
        history, development = load_construction(task['scenario'])
        archive = next(a for a in development if a.source_id == task['target_id'])
        if (archive.role != 'development' or any(a.role != 'history' for a in history)
                or archive.scenarios != candidates or setting['bank_contract'] != archive.provenance['contract']
                or setting['history_fingerprints'] != archive_fingerprints(history)):
            raise ValueError('ScaML archive roles, public scenarios or qualification contract differs')
        base, metadata = load_fsbo(checkpoint_path(task['scenario'], 'physical_fsbo'))
        if (metadata['history_fingerprints'] != archive_fingerprints(history)
                or metadata['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('ScaML representation used a different training or selection split')
        base = base.to(frozen['device'])
        x = features(candidates)
        if source_mode:
            sources = residual.source_priors(task['scenario'], candidates, history)
            order = sorted(range(len(sources['names'])), key=lambda i: sources['names'][i])
            names = [sources['names'][i] for i in order]
            sm, sk = sources['means'][order], sources['covariances'][order]
            teacher, _ = load_model(checkpoint_path(task['scenario'], MAIN))
            reference = float(teacher.mean.detach())
            if result['source_reference_mean'] != reference or result['source_weight_prior'] != 'independent Gamma(1,1)':
                raise ValueError('ScaML historical reference mean or coefficient prior differs')
        else:
            names, sm, sk, reference = [], np.empty((0, 512)), np.empty((0, 512, 512)), 0.
            if result['source_reference_mean'] is not None or result['source_weight_prior'] is not None:
                raise ValueError('ScaML target controls received explicit historical source information')
        if result['source_names'] != names:
            raise ValueError('ScaML source identity order differs')
        numerical = []
        for n, q in enumerate(queries, 1):
            weights = np.asarray(q['prequery_source_weights'], dtype=float)
            values = np.array([q['prequery_log_lengthscale'], q['prequery_log_variance'], *weights])
            fit = q['optimizer']
            if (weights.shape != (len(names),) or not np.isfinite(values).all()
                    or np.any(weights < WEIGHT_FLOOR)
                    or any(not KERNEL_BOUNDS[0] - 1e-12 <= v <= KERNEL_BOUNDS[1] + 1e-12 for v in values[:2])
                    or not np.isfinite(q['prequery_target_mean'])):
                raise ValueError('ScaML prequery source coefficients or residual kernel are invalid')
            mode = fit['prediction_mode']
            if (not np.isfinite(mode['gradient_max']) or mode['gradient_max'] > MODE_TOLERANCE
                    or not 0 <= mode['iterations'] <= MODE_ITERATIONS or mode['backtracks'] < 0):
                raise ValueError('ScaML prediction mode did not meet its registered convergence rule')
            if n == 1:
                if (fit['status'] != 'no_feedback' or fit['iterations'] or fit['evaluations']
                        or fit['objective'] is not None or fit['initial_objective'] is not None):
                    raise ValueError('ScaML initial prior was optimized on unavailable target feedback')
                initial = np.clip([float(base.log_lengthscale), 2 * float(base.log_scale)], *KERNEL_BOUNDS)
                np.testing.assert_allclose(values[:2], initial, atol=1e-12)
                np.testing.assert_allclose(weights, np.full(len(names), 1 / len(names)) if names else [], atol=1e-12)
                np.testing.assert_allclose(q['prequery_target_mean'], float(base.mean.detach()), atol=1e-12)
            elif (not np.isfinite(fit['objective']) or not np.isfinite(fit['initial_objective'])
                    or fit['objective'] > fit['initial_objective'] + 1e-7
                    or fit['maximum_mode_gradient'] > MODE_TOLERANCE
                    or not np.isfinite(fit['maximum_mode_gradient'])
                    or not np.isfinite(fit['projected_gradient_max']) or fit['projected_gradient_max'] < 0
                    or not 0 <= fit['maximum_mode_iterations'] <= MODE_ITERATIONS
                    or not 0 <= fit['iterations'] <= OPTIMIZER['maxiter'] or fit['evaluations'] < 2
                    or (not fit['success'] and 'LIMIT' not in fit['status'])):
                raise ValueError('ScaML scalar optimization or internal modes failed their registered rules')
            mass = result['physical_mass']
            learned, raw = q['prequery_learned_probability'], q['prequery_physical_probability']
            if (not np.isfinite(learned) or not 0 <= learned <= 1 or (raw is None) != (mass == 0)
                    or raw is not None and (not np.isfinite(raw) or not 0 <= raw <= 1)):
                raise ValueError('ScaML probability does not match its component configuration')
            np.testing.assert_allclose(q['prequery_probability'], (1 - mass) * learned + mass * (raw or 0), atol=2e-12)
            if n == 1:
                index = q['index']
                mean = float(base.mean.detach()) + weights @ (sm[:, index] - reference)
                variance = np.exp(values[1]) + (weights**2) @ sk[:, index, index]
                nodes, quadrature = np.polynomial.hermite.hermgauss(32)
                initial_p = expit(mean + np.sqrt(2 * variance) * nodes) @ (quadrature / np.sqrt(np.pi))
                np.testing.assert_allclose(learned, initial_p, atol=2e-12, rtol=2e-12)
                if mass:
                    np.testing.assert_allclose(raw, .5, atol=2e-12)
            if n in (9, 65, 128):
                paid = selected[:n - 1]
                labels = [queries[i]['feedback']['collision'] for i in range(n - 1)]
                model = copy.deepcopy(base)
                fit_target(model, x[paid], labels, steps=10, learning_rate=.003)
                with torch.no_grad():
                    z = model.features(x).detach()
                    target_mean = float(model.mean.detach())
                np.testing.assert_allclose(q['prequery_target_mean'], target_mean, atol=2e-8, rtol=2e-8)
                repeated_m, repeated_k, repeated, repeated_info = fit_prior(z, model.mean.detach(),
                    torch.as_tensor(sm, dtype=torch.float64, device=z.device),
                    torch.as_tensor(sk, dtype=torch.float64, device=z.device), reference, paid, labels,
                    float(model.log_lengthscale), 2 * float(model.log_scale))
                np.testing.assert_allclose(values, repeated.cpu(), atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(fit['objective'], repeated_info['objective'], atol=2e-6, rtol=2e-6)
                mean, covariance = prior_numpy(z.cpu().numpy(), target_mean, sm, sk, reference, values)
                np.testing.assert_allclose(repeated_m.cpu(), mean, atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(repeated_k.cpu(), covariance, atol=2e-6, rtol=2e-6)
                pm, pk = posterior_numpy(mean, covariance, paid, labels)
                nodes, quadrature = np.polynomial.hermite.hermgauss(32)
                p = expit(pm[:, None] + np.sqrt(2 * np.maximum(np.diag(pk), 1e-10))[:, None] * nodes)
                p = p @ (quadrature / np.sqrt(np.pi))
                independent_objective = evidence_numpy(mean[paid], covariance[np.ix_(paid, paid)], labels) + values[2:].sum()
                np.testing.assert_allclose(fit['objective'], independent_objective, atol=2e-6, rtol=2e-6)
                index = q['index']
                np.testing.assert_allclose(learned, p[index], atol=2e-7, rtol=2e-6)
                if mass:
                    physical = ClassificationKernel(x.shape[1]).to(frozen['device'])
                    fit_target(physical, x[paid], labels, steps=10, learning_rate=.01)
                    raw_p = classifier_probabilities_numpy(physical, x, paid, labels)
                    np.testing.assert_allclose(raw, raw_p[index], atol=2e-9, rtol=2e-9)
                    p = (1 - mass) * p + mass * raw_p
                remaining = set(range(512)) - set(paid)
                maximum = max(p[j] for j in remaining)
                if p[index] < maximum - 2e-7:
                    raise ValueError('ScaML choice differs from independent full-pool probabilities')
                numerical.append({'query': n, 'probability': float(p[index]),
                                  'acquisition_gap': float(maximum - p[index]),
                                  'objective_error': abs(fit['objective'] - independent_objective)})
            if (not all(np.isfinite(q[k]) and q[k] >= 0 for k in ('selection_s', 'prior_optimization_s',
                                                               'target_adaptation_s', 'physical_adaptation_s'))
                    or q['prior_optimization_s'] > q['selection_s'] or not mass and q['physical_adaptation_s'] != 0):
                raise ValueError('ScaML costs differ from its model configuration')
            np.testing.assert_allclose(q['adaptation_s'], q['target_adaptation_s'] + q['physical_adaptation_s'], atol=1e-12)
        if (output['cost']['equivalent_queries'] != 128 or output['cost']['new_physical_calls'] != 0
                or not source_mode and output['cost']['source_construction_s'] != 0):
            raise ValueError('ScaML replay or source-construction costs differ from the registered control')
        return {'path': row['path'], 'numerical_prefixes': numerical, 'prequery_metadata_checks': 128,
                'wall_s': time.perf_counter() - started}


def audit(scope='complete'):
    from . import discovery_scaml as study
    from .discovery_config import development_tasks

    frozen = read_json(study.REPORT / 'protocol.json')
    if frozen != study.protocol():
        raise ValueError('ScaML audit requires its unchanged frozen protocol')
    if scope == 'smoke':
        saved = read_json(study.REPORT / 'smoke.json')
        rows = saved['runs']
        if (saved['protocol'] != frozen or len(rows) != 4 or {r['method'] for r in rows} != set(study.METHODS)
                or any(any(r[k] != v for k, v in development_tasks()[0].items()) for r in rows)):
            raise ValueError('Check all four registered actual ScaML smoke cases')
    elif scope == 'available':
        rows = []
        for method in study.METHODS:
            for task in development_tasks():
                path = study.RUNS / method / task['target_id'] / task['scenario'] / f"{task['algorithm_seed']}.json"
                if path.exists():
                    rows.append({**task, 'method': method, 'path': str(path.relative_to(ROOT)), **read_json(path)['metrics']})
    else:
        index = read_json(study.REPORT / 'run_index.json')
        if index['status'] != 'complete' or index['protocol'] != frozen:
            raise ValueError('Complete the frozen ScaML matrix before full audit')
        rows = index['runs']
        study.validate_rows(rows)
    raw = raw_banks(rows)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sources = {p: digest(ROOT / p) for p in
               ('benchmark/discovery_scaml_audit.py', 'benchmark/discovery_physical_mixture_audit.py')}
    checked = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {}
        for row in rows:
            key = row['target_id'], row['scenario']
            identity = {'audit_sources': sources, 'protocol': digest(study.REPORT / 'protocol.json'),
                        'run': digest(ROOT / row['path']),
                        'raw_feedback': hashlib.sha256(json.dumps(raw[key][1], sort_keys=True).encode()).hexdigest()}
            path = study.REPORT / 'audit/runs' / row['method'] / row['target_id'] / row['scenario'] / f"{row['algorithm_seed']}.json"
            if path.exists():
                saved = read_json(path)
                if saved['status'] == 'passed' and saved['identity'] == identity:
                    checked.append(saved['result'])
                    continue
            futures[pool.submit(check_run, row, *raw[key])] = path, identity
        for future in as_completed(futures):
            path, identity = futures[future]
            checked_run = future.result()
            write_json(path, {'status': 'passed', 'identity': identity, 'result': checked_run})
            checked.append(checked_run)
            print('scaml_audit', len(checked), '/', len(rows), flush=True)
    result = {'status': 'passed' if scope != 'available' else 'available_completed_runs_checked',
              'scope': scope, 'runs': len(checked), 'expected_runs': 160, 'queries': 128 * len(checked),
              'independent_full_pool_choices': 3 * len(checked), 'prequery_metadata_checks': 128 * len(checked),
              'kernel_and_weight_refits_per_run': 3, 'whole_matrix_audited': scope == 'complete',
              'main_method_promoted': False, 'new_physical_calls': 0,
              'checks': sorted(checked, key=lambda r: r['path'])}
    write_json(study.REPORT / 'audit' / ('report.json' if scope == 'complete' else f'{scope}.json'), result)
    return result


def negative_controls():
    from . import discovery_scaml as study

    smoke = read_json(study.REPORT / 'smoke.json')
    positive = read_json(study.REPORT / 'audit/smoke.json')
    if positive['status'] != 'passed' or positive['runs'] != 4:
        raise ValueError('Independently audit all four actual ScaML smoke cases before corruption controls')
    raw = raw_banks(smoke['runs'])
    row = next(r for r in smoke['runs'] if r['method'] == 'scaml_classifier')
    path = ROOT / row['path']
    original = path.read_bytes()
    saved = json.loads(original)
    rejected = []
    for mutation in ('feedback', 'denominator', 'duplicate', 'physical_mass', 'weights',
                     'mode_gradient', 'objective', 'wrong_gp', 'timing'):
        output = copy.deepcopy(saved)
        q = output['result']['queries']
        if mutation == 'feedback':
            q[0]['feedback']['collision'] = not q[0]['feedback']['collision']
        elif mutation == 'denominator':
            output['metrics']['pool_unique_collisions'] += 1
        elif mutation == 'duplicate':
            q[8]['index'] = q[0]['index']
        elif mutation == 'physical_mass':
            output['result']['physical_mass'] = .1
        elif mutation == 'weights':
            q[8]['prequery_source_weights'][0] += .5
        elif mutation == 'mode_gradient':
            q[8]['optimizer']['prediction_mode']['gradient_max'] = .01
        elif mutation == 'objective':
            q[8]['optimizer']['objective'] = q[8]['optimizer']['initial_objective'] + 1.
        elif mutation == 'wrong_gp':
            p = q[8]['prequery_learned_probability']
            p = p + .1 if p < .8 else p - .1
            q[8]['prequery_learned_probability'] = q[8]['prequery_probability'] = p
        else:
            q[8]['physical_adaptation_s'] = 1.
            q[8]['adaptation_s'] += 1.
        try:
            check_run(row, *raw[(row['target_id'], row['scenario'])], output=output)
        except (ValueError, AssertionError):
            rejected.append(mutation)
        else:
            raise AssertionError(f'ScaML auditor accepted corrupted {mutation}')
    if path.read_bytes() != original:
        raise AssertionError('ScaML corruption checks changed the original run')
    target_row = next(r for r in smoke['runs'] if r['method'] == 'scaml_target')
    target_path = ROOT / target_row['path']
    target_original = target_path.read_bytes()
    changed = json.loads(target_original)
    changed['result']['source_reference_mean'] = 0.
    try:
        check_run(target_row, *raw[(target_row['target_id'], target_row['scenario'])], output=changed)
    except (ValueError, AssertionError):
        rejected.append('target_source_context')
    else:
        raise AssertionError('ScaML auditor accepted source context in its source-free control')
    if target_path.read_bytes() != target_original:
        raise AssertionError('ScaML target corruption check changed its original run')
    result = {'status': 'passed', 'positive_controls': 4, 'rejected_corruptions': len(rejected),
              'corruptions': rejected, 'original_unchanged': True, 'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(study.REPORT / 'audit/negative_controls.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('historical', 'target', 'correction', 'smoke', 'available', 'complete', 'negative'),
                        nargs='?', default='historical')
    stage = parser.parse_args().stage
    if stage in ('historical', 'target'):
        historical_check(target_only=stage == 'target')
    elif stage == 'correction':
        numerical_correction()
    elif stage == 'negative':
        negative_controls()
    else:
        audit(stage)
