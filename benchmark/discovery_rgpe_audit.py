"""Independent numerical checks for collision RGPE, using historical data only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.baselines.rgpe.search import RANK_SAMPLES, probabilities
from methods.common.gp_classifier import fit_target
from methods.common.gp_joint import bernoulli_posterior

from .construction import load_construction
from .discovery_config import read_json, write_json
from .discovery_physical_mixture import checkpoint_path
from .discovery_physical_mixture_audit import classifier_probabilities_numpy, raw_banks
from .discovery_residual_scale import ARTIFACTS, REPORT as LEARNING
from .protocol import ROOT


REPORT = ROOT / 'results/baselines/rgpe/collision/preparation'


def posterior_numpy(mean, covariance, observed, labels):
    """Newton solve in precision form, followed by direct Gaussian prediction."""
    observed = np.asarray(observed, dtype=int)
    labels = np.asarray(labels, dtype=float)
    if not len(observed):
        return mean.copy(), covariance.copy()
    support = covariance[np.ix_(observed, observed)] + 1e-5 * np.eye(len(observed))
    precision = cho_solve(cho_factor(support, lower=True), np.eye(len(observed)))
    prior_mean = mean[observed]
    latent = prior_mean.copy()
    for _ in range(10):
        p = expit(latent)
        weight = np.maximum(p * (1 - p), 1e-9)
        gradient = precision @ (latent - prior_mean) + p - labels
        hessian = precision + np.diag(weight)
        latent -= cho_solve(cho_factor(hessian, lower=True), gradient)
    p = expit(latent)
    weight = np.maximum(p * (1 - p), 1e-9)
    cross = covariance[observed]
    posterior_mean = mean + cross.T @ (precision @ (latent - prior_mean))
    posterior_covariance = covariance - cross.T @ cho_solve(
        cho_factor(support + np.diag(1 / weight), lower=True), cross)
    return posterior_mean, .5 * (posterior_covariance + posterior_covariance.T)


def normal_noise(shape, generator, device):
    # Share random numbers only; all GP inference, ranking and voting below use NumPy/SciPy.
    return torch.randn(shape, dtype=torch.float64, device=device, generator=generator).cpu().numpy()


def probability_numpy(source_mean, source_covariance, target_mean, target_covariance,
                      observed, labels, *, seed, device, samples=RANK_SAMPLES):
    n, h = len(labels), len(source_mean)
    indices, y = np.asarray(observed, dtype=int), np.asarray(labels, dtype=float)
    target_m, target_k = posterior_numpy(target_mean, target_covariance, indices, y)
    weights = np.zeros(h + 1)
    if n < 2 or len(np.unique(y)) < 2:
        weights[-1] = 1.
        diagnostics = {'pruned': [False] * (h + 1), 'median_losses': [0.] * (h + 1),
                       'target_loss_percentile_95': 0., 'informative_ordered_pairs': 0}
        loo_mean, loo_covariance = None, None
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        means = source_mean[:, indices]
        covariances = source_covariance[:, indices][:, :, indices]
        factor = np.linalg.cholesky(covariances + 1e-8 * np.eye(n))
        draws = means[None] + np.einsum('hij,shj->shi', factor,
                                       normal_noise((samples, h, n), generator, device))
        unequal = y[:, None] != y[None, :]
        desired = y[:, None] < y[None, :]
        losses = []
        for source in range(h):
            misranked = (draws[:, source, :, None] < draws[:, source, None, :]) ^ desired
            losses.append((misranked & unequal).sum(axis=(1, 2)))
        paid_mean, paid_covariance = target_mean[indices], target_covariance[np.ix_(indices, indices)]
        loo_mean, loo_covariance = [], []
        for fold in range(n):
            support = np.delete(np.arange(n), fold)
            m, k = posterior_numpy(paid_mean, paid_covariance, support, y[support])
            loo_mean.append(m)
            loo_covariance.append(k)
        loo_mean, loo_covariance = np.asarray(loo_mean), np.asarray(loo_covariance)
        factor = np.linalg.cholesky(loo_covariance + 1e-8 * np.eye(n))
        draws = loo_mean[None] + np.einsum('hij,shj->shi', factor,
                                         normal_noise((samples, n, n), generator, device))
        held = draws[:, np.arange(n), np.arange(n)]
        losses.append((((held[:, :, None] < draws) ^ desired) & unequal).sum(axis=(1, 2)))
        losses = np.asarray(losses)
        medians = np.quantile(losses, .5, axis=1)
        threshold = float(np.quantile(losses[-1], .95))
        pruned = medians > threshold
        pruned[-1] = False
        best = np.min(np.where(pruned[:, None], np.iinfo(np.int64).max, losses), axis=0)
        tied = (losses == best) & ~pruned[:, None]
        random = torch.rand(losses.shape, dtype=torch.float64, device=device,
                            generator=generator).cpu().numpy()
        winners = np.argmax(np.where(tied, random, -1), axis=0)
        winners[tied[-1]] = h
        weights = np.bincount(winners, minlength=h + 1) / samples
        diagnostics = {'pruned': pruned.tolist(), 'median_losses': medians.tolist(),
                       'target_loss_percentile_95': threshold,
                       'informative_ordered_pairs': int(unequal.sum())}
    mean = weights[:-1] @ source_mean + weights[-1] * target_m
    variance = np.einsum('h,hn->n', weights[:-1]**2,
                         np.diagonal(source_covariance, axis1=-2, axis2=-1))
    variance += weights[-1]**2 * np.diag(target_k)
    nodes, quadrature_weights = np.polynomial.hermite.hermgauss(32)
    p = expit(mean[:, None] + np.sqrt(2 * np.maximum(variance, 1e-10))[:, None] * nodes)
    p = p @ (quadrature_weights / np.sqrt(np.pi))
    return p, weights, diagnostics, (target_m, target_k), (loo_mean, loo_covariance)


def historical_check():
    torch.set_num_threads(1)
    receipt = read_json(LEARNING / 'selection_audit.json')
    if receipt['status'] != 'passed' or receipt['unique_historical_cache_tasks'] != 16:
        raise ValueError('Verify historical construction and adaptation data before RGPE checks')
    started = time.perf_counter()
    checks = []
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            history, _ = load_construction(kind)
            if len(history) != 8 or any(a.role != 'history' for a in history):
                raise ValueError('RGPE preparation uses only the eight historical systems')
            if any(not np.array_equal(a.x, history[0].x) for a in history):
                raise ValueError('Historical-coordinate checks require the shared historical pool')
            with np.load(ARTIFACTS / 'priors' / f'{kind}.npz', allow_pickle=False) as saved:
                specification = json.loads(str(saved['specification']))
                if specification['source_ids'] != [a.source_id for a in history]:
                    raise ValueError('Historical source identity order differs')
                np.testing.assert_array_equal(specification['x'], history[0].x)
                all_source_m = saved['means'].copy()
                all_source_k = saved['covariances'].copy()
            base, _ = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            base = base.to(device)
            ordered = np.random.default_rng(1301).permutation(512)
            for pseudo_target in sorted(history, key=lambda a: a.source_id):
                order = sorted((i for i, a in enumerate(history) if a.source_id != pseudo_target.source_id),
                               key=lambda i: history[i].source_id)
                source_m, source_k = all_source_m[order], all_source_k[order]
                source_m_t = torch.as_tensor(source_m, dtype=torch.float64, device=device)
                source_k_t = torch.as_tensor(source_k, dtype=torch.float64, device=device)
                for count in (8, 128):
                    model = copy.deepcopy(base)
                    paid = ordered[:count].tolist()
                    labels = pseudo_target.collision[paid].tolist()
                    fit_target(model, pseudo_target.x[paid], labels, steps=10, learning_rate=.003)
                    with torch.no_grad():
                        mean = model.mean.expand(512)
                        covariance = model.covariance(pseudo_target.x)
                        actual_p, actual_weights, actual_details = probabilities(
                            model, pseudo_target.x, paid, labels, source_m_t, source_k_t, seed=1301 + count)
                        target_m, target_k = bernoulli_posterior(mean, covariance, paid, labels)
                        expected = probability_numpy(source_m, source_k, mean.cpu().numpy(),
                            covariance.cpu().numpy(), paid, labels, seed=1301 + count, device=device)
                        expected_p, expected_weights, expected_details, target, loo = expected
                        np.testing.assert_allclose(actual_p.cpu(), expected_p, atol=2e-7, rtol=2e-6)
                        np.testing.assert_array_equal(actual_weights.cpu(), expected_weights)
                        if actual_details != expected_details:
                            raise ValueError('Independent RGPE pruning or voting differs')
                        np.testing.assert_allclose(target_m.cpu(), target[0], atol=2e-6, rtol=2e-6)
                        np.testing.assert_allclose(target_k.cpu(), target[1], atol=2e-7, rtol=2e-6)
                        if loo[0] is not None:
                            positions = torch.arange(count, device=device)
                            indices = positions.expand(count, -1)[~torch.eye(count, dtype=torch.bool, device=device)]
                            indices = indices.reshape(count, count - 1)
                            paid_tensor = torch.as_tensor(paid, device=device)
                            y = torch.as_tensor(labels, dtype=torch.float64, device=device)
                            loo_m, loo_k = bernoulli_posterior(mean[paid_tensor],
                                covariance[paid_tensor][:, paid_tensor], indices, y[indices])
                            np.testing.assert_allclose(loo_m.cpu(), loo[0], atol=2e-6, rtol=2e-6)
                            np.testing.assert_allclose(loo_k.cpu(), loo[1], atol=2e-7, rtol=2e-6)
                        else:
                            loo_m = loo_k = None
                        checks.append({'kind': kind, 'historical_pseudo_target': pseudo_target.source_id,
                            'source_ids': [history[i].source_id for i in order], 'support_count': count,
                            'full_pool_size': 512, 'ranking_samples': RANK_SAMPLES,
                            'independent_loo_modes': 0 if loo[0] is None else count,
                            'probability_max_error': float(np.max(np.abs(actual_p.cpu().numpy() - expected_p))),
                            'target_mean_max_error': float(np.max(np.abs(target_m.cpu().numpy() - target[0]))),
                            'target_covariance_max_error': float(np.max(np.abs(target_k.cpu().numpy() - target[1]))),
                            'loo_mean_max_error': None if loo_m is None else float(np.max(np.abs(loo_m.cpu().numpy() - loo[0]))),
                            'loo_covariance_max_error': None if loo_k is None else float(np.max(np.abs(loo_k.cpu().numpy() - loo[1]))),
                            'weights': expected_weights.tolist(), 'ranking': expected_details})
                    print('rgpe_historical_numerics', kind, pseudo_target.source_id, count, flush=True)
    result = {'status': 'passed', 'checks': checks, 'full_pool_predictions': len(checks),
              'independent_loo_modes': sum(c['independent_loo_modes'] for c in checks),
              'source_context': 'exclude pseudo target; teacher includes all history, not a held-system evaluation',
              'randomness': 'shared standard-normal and uniform draws only; independent NumPy/SciPy inference',
              'wall_s': time.perf_counter() - started, 'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(REPORT / 'historical_numeric.json', result)
    return result


def check_run(row, candidates, feedbacks, output=None):
    from methods.common.gp_classifier import ClassificationKernel
    from methods.common.response import archive_fingerprints
    from . import discovery_rgpe as study
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
            raise ValueError('RGPE run differs from its registered task, method or candidates')
        queries, ledger = result['queries'], output['query_ledger']
        selected = [q['index'] for q in queries]
        if (len(queries) != 128 or len(ledger) != 128 or len(set(selected)) != 128
                or selected[:8] != initial_indices(512, 8, task['algorithm_seed'])
                or result['selected_indices'] != selected or result['actual_queries'] != 128
                or result['maximum_queries'] != 128 or result['termination'] != 'budget_exhausted'
                or result['sampling_seed'] != task['algorithm_seed']
                or result['ranking_samples'] != RANK_SAMPLES
                or result['physical_mass'] != frozen['physical_mass'][method]):
            raise ValueError('RGPE budget, initialization, sampling or physical mass differs')
        failures = {dedup_key(s) for s, f in zip(candidates, feedbacks) if f['collision']}
        found, curve = set(), []
        for n, (q, paid) in enumerate(zip(queries, ledger), 1):
            index = q['index']
            if (q['query'] != n or paid['call'] != n or paid['index'] != index
                    or paid['status'] != 'complete' or paid['account'] != 'target'
                    or paid['scenario_id'] != candidates[index].scenario_id
                    or paid['purpose'] != ('initial' if n <= 8 else 'target')
                    or paid['feedback'] != q['feedback']
                    or QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[index]):
                raise ValueError('RGPE disclosure differs from its complete paid execution')
            if q['feedback']['collision']:
                found.add(dedup_key(candidates[index]))
            curve.append(len(found))
        recall = [v / len(failures) if failures else None for v in curve]
        if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != recall
                or metrics['unique_collisions'] != curve[-1] or metrics['collision_recall'] != recall[-1]
                or metrics['pool_unique_collisions'] != len(failures)
                or any(row[k] != v for k, v in metrics.items())):
            raise ValueError('RGPE collision metrics differ from the full deduplicated pool')
        np.testing.assert_allclose(metrics['discovery_auc'], np.mean(curve), atol=1e-12)
        for budget in (16, 32, 64, 128):
            if metrics['checkpoints'][str(budget)] != {'unique_collisions': curve[budget - 1],
                                                      'collision_recall': recall[budget - 1]}:
                raise ValueError('RGPE checkpoint differs from its discovery curve')
        history, development = load_construction(task['scenario'])
        archive = next(a for a in development if a.source_id == task['target_id'])
        if (archive.role != 'development' or any(a.role != 'history' for a in history)
                or archive.scenarios != candidates or setting['bank_contract'] != archive.provenance['contract']
                or setting['history_fingerprints'] != archive_fingerprints(history)):
            raise ValueError('RGPE archives differ in role, scenario or execution contract')
        base, metadata = load_fsbo(checkpoint_path(task['scenario'], 'physical_fsbo'))
        if (metadata['history_fingerprints'] != archive_fingerprints(history)
                or metadata['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('RGPE target teacher differs from its allowed training and selection data')
        base = base.to(frozen['device'])
        x = features(candidates)
        sources = residual.source_priors(task['scenario'], candidates, history)
        order = sorted(range(len(sources['names'])), key=lambda i: sources['names'][i])
        names = [sources['names'][i] for i in order]
        source_m, source_k = sources['means'][order], sources['covariances'][order]
        if result['component_names'] != [*names, 'target_fsbo']:
            raise ValueError('RGPE source identities or order differs')
        numerical = []
        for n, q in enumerate(queries, 1):
            weights = np.asarray(q['prequery_rank_weights'], dtype=float)
            details = q['prequery_ranking']
            paid = selected[:n - 1]
            labels = [queries[i]['feedback']['collision'] for i in range(n - 1)]
            pairs = 2 * sum(labels) * (len(labels) - sum(labels))
            medians = np.asarray(details['median_losses'], dtype=float)
            pruned = np.asarray(details['pruned'], dtype=bool)
            threshold = details['target_loss_percentile_95']
            if (weights.shape != (9,) or medians.shape != (9,) or pruned.shape != (9,)
                    or not np.all(np.isfinite(weights)) or np.any(weights < 0)
                    or abs(weights.sum() - 1) > 1e-12
                    or np.max(np.abs(weights * RANK_SAMPLES - np.round(weights * RANK_SAMPLES))) > 1e-12
                    or np.any(weights[pruned] != 0) or pruned[-1]
                    or details['informative_ordered_pairs'] != pairs
                    or not 0 <= threshold <= pairs or np.any(medians < 0) or np.any(medians > pairs)
                    or not np.array_equal(pruned, medians > threshold)):
                raise ValueError('RGPE prequery weights, pruning or ranking pairs are invalid')
            if not pairs and (not np.array_equal(weights, [0.] * 8 + [1.])
                              or np.any(medians != 0) or threshold != 0):
                raise ValueError('RGPE no-ranking-information target tie rule differs')
            mass = result['physical_mass']
            learned, raw = q['prequery_ensemble_probability'], q['prequery_physical_probability']
            if (not np.isfinite(learned) or not 0 <= learned <= 1
                    or (raw is None) != (mass == 0)
                    or (raw is not None and (not np.isfinite(raw) or not 0 <= raw <= 1))):
                raise ValueError('RGPE probability is invalid for its model configuration')
            np.testing.assert_allclose(q['prequery_probability'], (1 - mass) * learned + mass * (raw or 0), atol=2e-12)
            if n in (9, 65, 128):
                model = copy.deepcopy(base)
                fit_target(model, x[paid], labels, steps=10, learning_rate=.003)
                with torch.no_grad():
                    mean = model.mean.expand(512).cpu().numpy()
                    covariance = model.covariance(x).cpu().numpy()
                prediction = probability_numpy(source_m, source_k, mean, covariance, paid, labels,
                    seed=task['algorithm_seed'] + 1009 * (n - 1), device=frozen['device'])
                p, expected_weights, expected_details = prediction[:3]
                np.testing.assert_array_equal(weights, expected_weights)
                if details != expected_details:
                    raise ValueError('Independent RGPE ranking, pruning or voting differs')
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
                    raise ValueError('RGPE selected point differs from independent full-pool probabilities')
                numerical.append({'query': n, 'probability': float(p[index]),
                                  'acquisition_gap': float(maximum - p[index])})
            if (not all(np.isfinite(q[k]) and q[k] >= 0
                        for k in ('selection_s', 'target_adaptation_s', 'physical_adaptation_s'))
                    or (not mass and q['physical_adaptation_s'] != 0)):
                raise ValueError('RGPE timing differs from its model configuration')
            np.testing.assert_allclose(q['adaptation_s'], q['target_adaptation_s'] + q['physical_adaptation_s'], atol=1e-12)
        if output['cost']['equivalent_queries'] != 128 or output['cost']['new_physical_calls'] != 0:
            raise ValueError('RGPE replay costs differ from its paid budget')
        return {'path': row['path'], 'numerical_prefixes': numerical,
                'prequery_metadata_checks': 128, 'wall_s': time.perf_counter() - started}


def audit(scope='complete'):
    from . import discovery_rgpe as study
    from .discovery_config import development_tasks

    frozen = read_json(study.REPORT / 'protocol.json')
    if frozen != study.protocol():
        raise ValueError('RGPE audit requires the unchanged frozen protocol')
    if scope == 'smoke':
        saved = read_json(study.REPORT / 'smoke.json')
        rows = saved['runs']
        if (saved['protocol'] != frozen or len(rows) != 2 or {r['method'] for r in rows} != set(study.METHODS)
                or any(any(r[k] != v for k, v in development_tasks()[0].items()) for r in rows)):
            raise ValueError('Check both registered actual RGPE smoke cases')
    elif scope == 'available':
        rows = []
        for method in study.METHODS:
            for task in development_tasks():
                path = study.RUNS / method / task['target_id'] / task['scenario'] / f"{task['algorithm_seed']}.json"
                if path.exists():
                    rows.append({**task, 'method': method, 'path': str(path.relative_to(ROOT)),
                                 **read_json(path)['metrics']})
    else:
        index = read_json(study.REPORT / 'run_index.json')
        if index['status'] != 'complete' or index['protocol'] != frozen:
            raise ValueError('Complete the frozen RGPE matrix before full audit')
        rows = index['runs']
        study.validate_rows(rows)
    raw = raw_banks(rows)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit_sources = {p: digest(ROOT / p) for p in
                     ('benchmark/discovery_rgpe_audit.py', 'benchmark/discovery_physical_mixture_audit.py')}
    checked = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {}
        for row in rows:
            key = row['target_id'], row['scenario']
            identity = {'audit_sources': audit_sources, 'protocol': digest(study.REPORT / 'protocol.json'),
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
            result = future.result()
            write_json(path, {'status': 'passed', 'identity': identity, 'result': result})
            checked.append(result)
            print('rgpe_audit', len(checked), '/', len(rows), flush=True)
    result = {'status': 'passed' if scope != 'available' else 'available_completed_runs_checked',
              'scope': scope, 'runs': len(checked), 'expected_runs': 80,
              'queries': 128 * len(checked), 'independent_full_pool_choices': 3 * len(checked),
              'prequery_metadata_checks': 128 * len(checked),
              'weights_recomputed_from_independent_GP_at_queries': [9, 65, 128],
              'whole_matrix_audited': scope == 'complete', 'main_method_promoted': False,
              'new_physical_calls': 0, 'checks': sorted(checked, key=lambda r: r['path'])}
    write_json(study.REPORT / 'audit' / ('report.json' if scope == 'complete' else f'{scope}.json'), result)
    return result


def negative_controls():
    from . import discovery_rgpe as study

    smoke = read_json(study.REPORT / 'smoke.json')
    positive = read_json(study.REPORT / 'audit/smoke.json')
    if positive['status'] != 'passed' or positive['runs'] != 2:
        raise ValueError('Independently audit both actual smoke runs before corruption controls')
    raw = raw_banks(smoke['runs'])
    row = next(r for r in smoke['runs'] if r['method'] == 'rgpe_classifier')
    path = ROOT / row['path']
    original = path.read_bytes()
    saved = json.loads(original)
    rejected = []
    for mutation in ('feedback', 'denominator', 'duplicate', 'seed', 'weights', 'pairs', 'wrong_gp', 'timing'):
        output = copy.deepcopy(saved)
        q = output['result']['queries']
        if mutation == 'feedback':
            q[0]['feedback']['collision'] = not q[0]['feedback']['collision']
        elif mutation == 'denominator':
            output['metrics']['pool_unique_collisions'] += 1
        elif mutation == 'duplicate':
            q[8]['index'] = q[0]['index']
        elif mutation == 'seed':
            output['result']['sampling_seed'] += 1
        elif mutation == 'weights':
            q[8]['prequery_rank_weights'][0] += .25
        elif mutation == 'pairs':
            q[8]['prequery_ranking']['informative_ordered_pairs'] += 1
        elif mutation == 'wrong_gp':
            p = q[8]['prequery_ensemble_probability']
            p = p + .1 if p < .8 else p - .1
            q[8]['prequery_ensemble_probability'] = q[8]['prequery_probability'] = p
        else:
            q[8]['physical_adaptation_s'] = 1.
            q[8]['adaptation_s'] += 1.
        try:
            check_run(row, *raw[(row['target_id'], row['scenario'])], output=output)
        except (ValueError, AssertionError):
            rejected.append(mutation)
        else:
            raise AssertionError(f'RGPE auditor accepted corrupted {mutation}')
    if path.read_bytes() != original:
        raise AssertionError('RGPE corruption checks changed the original run')
    result = {'status': 'passed', 'positive_controls': 2, 'rejected_corruptions': len(rejected),
              'corruptions': rejected, 'original_unchanged': True, 'new_selector_queries': 0,
              'new_physical_calls': 0}
    write_json(study.REPORT / 'audit/negative_controls.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scope', choices=('historical', 'smoke', 'available', 'complete', 'negative'),
                        nargs='?', default='historical')
    scope = parser.parse_args().scope
    if scope == 'historical':
        historical_check()
    elif scope == 'negative':
        negative_controls()
    else:
        audit(scope)
