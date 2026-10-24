"""Historical-only separation of conditional convergence and reliability scores."""

import argparse
import hashlib
import json
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.common.classification import logistic_moments
from methods.common.response import source_weights
from methods.gcrt.conditional_adaptation import component_probabilities as legacy_probabilities
from methods.gcrt.converged_response import component_probabilities
from methods.gcrt.residual_scale import cached_covariance, residual_prior, update_reliability

from .discovery_config import OUTPUT, read_json, write_json
from .discovery_counterfactual_adaptation import REPORT as VIRTUAL
from .discovery_residual_scale import BETAS, COUNTS, REPORT as LEARNING, cache_path
from .discovery_residual_scale import protocol as learning_protocol
from .discovery_response_locality import history_inputs
from .discovery_scaml_audit import posterior_numpy
from .protocol import ROOT


REPORT = OUTPUT/'converged_response'
VARIANTS = ('legacy', 'converged_old_weights', 'converged_prequential')


def protocol():
    return {'purpose': 'historical conditional convergence correction; no new development search',
        'scenarios': ['cut_in', 'cut_out'], 'historical_tasks': 16, 'beta_grid': list(BETAS),
        'counts': list(COUNTS), 'support_count': 128, 'query_count': 32, 'permutation_seed': 1301,
        'variants': list(VARIANTS), 'reliability': ['local', 'global'], 'bandwidth': .3,
        'source_prior_mass': .75, 'target_prior_mass': .25, 'physical_mass': .1,
        'mode_tolerance': 1e-10, 'mode_maximum_iterations': 100, 'device': 'cpu',
        'selection': 'converged_prequential local query NLL; Brier; distance to beta=1; beta',
        'teacher_scope': 'all historical systems; own source posterior excluded; not held-system validation',
        'old_historical_protocol': learning_protocol(),
        'input_evidence': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            LEARNING/'selection.json', LEARNING/'selection_audit.json', VIRTUAL/'mode_audit/cached.json')},
        'source_files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'benchmark/discovery_converged_response.py', 'methods/gcrt/converged_response.py',
            'methods/common/gp_laplace.py')},
        'new_selector_queries': 0, 'new_physical_calls': 0}


def prepare():
    mode = read_json(VIRTUAL/'mode_audit/cached.json')
    if mode['status'] != 'complete_modes_checked' or mode['mode_checks'] != 368 or mode['nonstationary_modes'] != 18:
        raise ValueError('Preserve the independently diagnosed conditional mode failure first')
    if read_json(LEARNING/'selection.json')['protocol'] != learning_protocol():
        raise ValueError('The historical states and old parameter selection must remain frozen')
    current = protocol()
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != current:
        raise ValueError('The convergence-correction study is already registered differently')
    write_json(path, current)
    return current


def cache_inputs(kind, archive):
    with np.load(cache_path(kind, archive.source_id), allow_pickle=False) as saved:
        spec = json.loads(str(saved['specification']))
        if spec['protocol'] != learning_protocol() or spec['role'] != 'history' or spec['source_id'] != archive.source_id:
            raise ValueError('Convergence correction accepts only the checked historical adaptation cache')
        ordered, labels = saved['ordered_indices'].copy(), saved['labels'].copy()
        np.testing.assert_array_equal(ordered, np.random.default_rng(1301).permutation(512)[:160])
        np.testing.assert_array_equal(labels, archive.collision[ordered])
        np.testing.assert_array_equal(saved['x'], archive.x[ordered])
        return {'ordered': ordered, 'labels': labels, 'encoded': torch.tensor(saved['encoded']),
            'parameters': torch.tensor(saved['parameters']), 'raw_probability': saved['raw_probabilities'].copy(),
            'x': torch.tensor(saved['x'])}


def query_metrics(probability, truth):
    p = np.clip(np.asarray(probability), 1e-12, 1-1e-12)
    return {'query_nll': float(-np.mean(truth*np.log(p)+(1-truth)*np.log1p(-p))),
            'query_brier': float(np.square(p-truth).mean())}


def independent_probabilities(means, covariances, paid, labels):
    result = []
    for m, c in zip(means, covariances):
        if len(paid):
            mean, covariance = posterior_numpy(m, c, paid, labels)
            result.append(logistic_moments(mean, np.maximum(np.diag(covariance), 1e-10))[0])
        else:
            result.append(logistic_moments(m, np.maximum(np.diag(c), 1e-10))[0])
    return np.asarray(result)


def prefix():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol() or (REPORT/'prefix.json').exists():
        raise ValueError('Run only the unchanged and not already completed prefix comparison')
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows, checks = [], []
    with threadpool_limits(1):
        for kind in specification['scenarios']:
            history, means, covariances = history_inputs(kind)
            beta = read_json(LEARNING/'selection.json')['selected_beta'][kind]
            for archive in history:
                saved = cache_inputs(kind, archive)
                ordered = saved['ordered']
                positions = [i for i, a in enumerate(history) if a.source_id != archive.source_id]
                sm = torch.tensor(means[positions][:, ordered])
                sc = torch.tensor(covariances[positions][:, ordered, :][:, :, ordered])
                parameter = saved['parameters'][64]
                covariance = cached_covariance(saved['encoded'][64], parameter[1], parameter[2])
                m, c = residual_prior(parameter[0], covariance, sm, sc, saved['parameters'][0, 0], beta)
                with torch.no_grad():
                    corrected, info = component_probabilities(m, c, list(range(64)), saved['labels'][:64])
                independent = independent_probabilities(m.numpy(), c.numpy(), list(range(64)), saved['labels'][:64])
                np.testing.assert_allclose(corrected, independent, atol=2e-7, rtol=2e-6)
                with np.load(VIRTUAL/'histories'/kind/f'{archive.source_id}.npz', allow_pickle=False) as before:
                    score = before['prequery_log_weights'][:, ordered]
                    weight = np.exp(score-score.max(0));weight /= weight.sum(0)
                    legacy = before['prequery_probability'][ordered]
                    p_old = before['prequery_components'][:, ordered]
                actual_legacy = .9*(weight*p_old).sum(0)+.1*saved['raw_probability'][64]
                np.testing.assert_allclose(legacy, actual_legacy, atol=2e-12, rtol=2e-12)
                fixed = .9*(weight*corrected.numpy()).sum(0)+.1*saved['raw_probability'][64]
                rows.append({'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
                    'beta': beta, 'support_count': 64, 'source_ids': [history[i].source_id for i in positions],
                    'legacy': query_metrics(legacy[128:], saved['labels'][128:]),
                    'converged_old_weights': query_metrics(fixed[128:], saved['labels'][128:]),
                    'component_maximum_difference': float(np.max(np.abs(corrected.numpy()-p_old))),
                    'mixture_maximum_difference': float(np.max(np.abs(fixed-legacy))),
                    'corrected_mode_maximum_gradient': max(r['gradient_max'] for r in info)})
                checks.append(float(np.max(np.abs(corrected.numpy()-independent))))
    result = {'status': 'complete_independently_checked_prefix_comparison', 'protocol': specification,
        'rows': rows, 'independent_prediction_groups': len(checks), 'independent_maximum_error': max(checks),
        'wall_s': time.perf_counter()-started, 'new_selector_queries': 0, 'new_physical_calls': 0,
        'discovery_admission_claimed': False}
    write_json(REPORT/'prefix.json', result)
    return result


def history_task(kind, archive, history, all_means, all_covariances, specification):
    destination = REPORT/'histories'/kind/f'{archive.source_id}.json'
    if destination.exists():
        old = read_json(destination)
        if old['protocol'] != specification or old['status'] != 'complete':
            raise ValueError('A saved historical correction task differs from its registered setting')
        return old
    started = time.perf_counter()
    saved = cache_inputs(kind, archive)
    ordered, truth = saved['ordered'], saved['labels']
    positions = [i for i, a in enumerate(history) if a.source_id != archive.source_id]
    sources = [history[i] for i in positions]
    source_mean = torch.tensor(all_means[positions][:, ordered])
    source_covariance = torch.tensor(all_covariances[positions][:, ordered, :][:, :, ordered])
    initial = torch.tensor(np.r_[.75*source_weights(sources), .25]).log()[:, None].expand(-1, 160).clone()
    rows, independent, convergence = [], [], []
    for beta in BETAS:
        scores = {variant: {mode: initial.clone() for mode in ('local', 'global')}
                  for variant in ('legacy', 'converged_prequential')}
        iterations, backtracks, gradient_max, maximum_iterations = 0, 0, 0., 0
        for count in range(129):
            parameters = saved['parameters'][count]
            covariance = cached_covariance(saved['encoded'][count], parameters[1], parameters[2])
            m, c = residual_prior(parameters[0], covariance, source_mean, source_covariance,
                                   saved['parameters'][0, 0], beta)
            paid, labels = list(range(count)), truth[:count]
            with torch.no_grad():
                old_p = legacy_probabilities(m, c, paid, labels)
                corrected, info = component_probabilities(m, c, paid, labels)
            iterations += sum(r['iterations'] for r in info)
            backtracks += sum(r['backtracks'] for r in info)
            gradient_max = max(gradient_max, max(r['gradient_max'] for r in info))
            maximum_iterations = max(maximum_iterations, max(r['iterations'] for r in info))
            if count in (0, 8, 64, 128):
                expected = independent_probabilities(m.numpy(), c.numpy(), paid, labels)
                np.testing.assert_allclose(corrected, expected, atol=2e-7, rtol=2e-6)
                independent.append({'beta': beta, 'support_count': count,
                    'probability_maximum_error': float(np.max(np.abs(corrected.numpy()-expected)))})
            if count in COUNTS:
                for mode in ('local', 'global'):
                    combinations = [('legacy', old_p, scores['legacy'][mode]),
                        ('converged_old_weights', corrected, scores['legacy'][mode]),
                        ('converged_prequential', corrected, scores['converged_prequential'][mode])]
                    for variant, probability, score in combinations:
                        mixture = (.9*(score.softmax(0)*probability).sum(0)).numpy()+.1*saved['raw_probability'][count]
                        rows.append({'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
                            'source_ids': [s.source_id for s in sources], 'beta': beta, 'reliability': mode,
                            'support_count': count, 'variant': variant,
                            **query_metrics(mixture[128:], truth[128:])})
            if count < 128:
                for variant, p in (('legacy', old_p), ('converged_prequential', corrected)):
                    for mode in ('local', 'global'):
                        scores[variant][mode] = update_reliability(scores[variant][mode], p, count,
                            bool(truth[count]), saved['x'], .3 if mode == 'local' else None)
        convergence.append({'beta': beta, 'gradient_max': gradient_max, 'maximum_iterations': maximum_iterations,
                            'sum_mode_iterations': iterations, 'sum_backtracks': backtracks,
                            'nonempty_mode_solves': 128*8})
        print('converged_history', kind, archive.source_id, beta, flush=True)
    result = {'status': 'complete', 'protocol': specification, 'scenario': kind, 'pseudo_target': archive.source_id,
        'rows': rows, 'independent_checks': independent, 'convergence': convergence,
        'wall_s': time.perf_counter()-started, 'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(destination, result)
    return result


def aggregate(rows):
    return {metric: float(np.mean([np.mean([r[metric] for r in rows if r['lineage'] == lineage])
                                  for lineage in sorted({r['lineage'] for r in rows})]))
            for metric in ('query_nll', 'query_brier')}


def history():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol() or (REPORT/'history.json').exists():
        raise ValueError('Run only the unchanged and not already completed historical correction')
    torch.set_num_threads(1)
    started = time.perf_counter()
    records = []
    with threadpool_limits(1):
        for kind in specification['scenarios']:
            histories, means, covariances = history_inputs(kind)
            for archive in histories:
                try:
                    records.append(history_task(kind, archive, histories, means, covariances, specification))
                except Exception:
                    failure = {'scenario': kind, 'pseudo_target': archive.source_id, 'wall_s': time.perf_counter()-started,
                        'complete_tasks_this_invocation': len(records), 'error': traceback.format_exc(),
                        'new_selector_queries': 0, 'new_physical_calls': 0}
                    with (REPORT/'failed_attempts.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(failure, ensure_ascii=False)+'\n')
                    raise
    rows = [r for record in records for r in record['rows']]
    checks = [r for record in records for r in record['independent_checks']]
    if len(rows) != 2400 or len(checks) != 320 or specification != protocol():
        raise ValueError('Complete all historical amplitudes, supports, source contexts and numerical checks')
    legacy = {(r['scenario'], r['pseudo_target'], r['beta'], r['reliability'], r['support_count']): r
              for r in read_json(LEARNING/'selection.json')['rows']}
    errors = []
    for r in rows:
        if r['variant'] != 'legacy':
            continue
        reference = legacy[r['scenario'], r['pseudo_target'], r['beta'], r['reliability'], r['support_count']]
        error = max(abs(r[m]-reference[m]) for m in ('query_nll', 'query_brier'))
        np.testing.assert_allclose(error, 0, atol=2e-8, rtol=0)
        errors.append(error)
    summaries = []
    for kind in specification['scenarios']:
        for beta in BETAS:
            for mode in ('local', 'global'):
                for variant in VARIANTS:
                    group = [r for r in rows if (r['scenario'], r['beta'], r['reliability'], r['variant']) == (kind, beta, mode, variant)]
                    summaries.append({'scenario': kind, 'beta': beta, 'reliability': mode, 'variant': variant, **aggregate(group)})
    selected = {kind: min((r for r in summaries if r['scenario'] == kind and r['variant'] == 'converged_prequential'
                          and r['reliability'] == 'local'),
                         key=lambda r: (r['query_nll'], r['query_brier'], abs(r['beta']-1), r['beta']))['beta']
                for kind in specification['scenarios']}
    result = {'status': 'complete_independently_checked_historical_correction', 'protocol': specification,
        'rows': rows, 'summary': summaries, 'selected_beta': selected,
        'old_selected_beta': read_json(LEARNING/'selection.json')['selected_beta'],
        'independent_prediction_groups': len(checks), 'independent_probability_maximum_error': max(r['probability_maximum_error'] for r in checks),
        'legacy_scores_checked': len(errors), 'legacy_score_maximum_error': max(errors),
        'maximum_converged_gradient': max(c['gradient_max'] for r in records for c in r['convergence']),
        'nonempty_conditional_mode_solves': sum(c['nonempty_mode_solves'] for r in records for c in r['convergence']),
        'sum_task_wall_s': sum(r['wall_s'] for r in records), 'wall_s': time.perf_counter()-started,
        'new_selector_queries': 0, 'new_physical_calls': 0, 'development_admission_claimed': False}
    write_json(REPORT/'history.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'prefix', 'history'))
    {'prepare': prepare, 'prefix': prefix, 'history': history}[parser.parse_args().stage]()
