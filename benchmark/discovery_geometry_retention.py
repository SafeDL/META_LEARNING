"""Historical-only prequential comparison of retained and adapted collision kernels."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.common.response import source_weights
from methods.gcrt.converged_response import component_probabilities
from methods.gcrt.residual_scale import cached_covariance, residual_prior

from . import discovery_converged_response as correction
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_residual_scale import COUNTS


REPORT = OUTPUT / 'geometry_retention'
VARIANTS = ('adapted', 'frozen', 'fixed_half', 'learned_views')
MODES = ('local', 'global')


def protocol():
    return {'purpose': 'historical screening of retaining and adapting collision geometry',
        'scenarios': ['cut_in', 'cut_out'], 'historical_tasks': 16,
        'variants': list(VARIANTS), 'reliability': list(MODES), 'counts': list(COUNTS),
        'support_count': 128, 'query_count': 32, 'permutation_seed': 1301,
        'beta': 1., 'bandwidth': .3, 'source_mass': .75, 'target_mass': .25,
        'physical_mass': .1, 'view_prior': [.5, .5],
        'mean': 'same paid adapted mean and historical source offsets in both views',
        'variance': 'frozen initial collision kernel scaled to current kernel mean diagonal',
        'fixed_half': 'average view probabilities within source, then score sources prequentially',
        'learned_views': 'joint source-view reliability scored before each paid label',
        'target_adaptation': 'reuse original 129 CUDA states; no new training',
        'teacher_scope': 'all historical systems; own source posterior excluded; not held-system validation',
        'screening': 'local learned views lower NLL and no higher Brier than all three local controls '
                     'and global learned views in both scenarios; two positive historical lineages '
                     'against strongest local control by NLL then Brier in each scenario',
        'historical_protocol': correction.protocol(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_selector_queries': 0, 'new_physical_calls': 0}


def prepare():
    assert read_json(correction.REPORT / 'summary_audit.json')['status'] == 'passed'
    specification = protocol()
    path = REPORT / 'protocol.json'
    if path.exists():
        assert read_json(path) == specification
    else:
        write_json(path, specification)
    return specification


def task(kind, archive, history, all_means, all_covariances, specification):
    destination = REPORT / 'histories' / kind / f'{archive.source_id}.json'
    if destination.exists():
        previous = read_json(destination)
        assert previous['status'] == 'complete' and previous['protocol'] == specification
        return previous
    started = time.perf_counter()
    saved = correction.cache_inputs(kind, archive)
    ordered, truth = saved['ordered'], saved['labels']
    positions = [i for i, source in enumerate(history) if source.source_id != archive.source_id]
    sources = [history[i] for i in positions]
    source_means = torch.tensor(all_means[positions][:, ordered])
    source_covariances = torch.tensor(all_covariances[positions][:, ordered, :][:, :, ordered])
    prior = np.r_[.75 * source_weights(sources), .25]
    frozen = cached_covariance(saved['encoded'][0], *saved['parameters'][0, 1:])
    distance = torch.square(saved['x'][:, None] - saved['x'][None]).mean(-1).numpy()
    influences = {'local': np.exp(-distance / (2 * .3**2)), 'global': np.ones((160, 160))}
    priors = {v: np.tile(prior / 2, 2) if v == 'learned_views' else prior for v in VARIANTS}
    scores = {(v, m): np.tile(np.log(priors[v])[:, None], (1, 160)) for v in VARIANTS for m in MODES}
    likelihoods = {v: np.empty((len(priors[v]), 128)) for v in VARIANTS}
    rows, checks, arrays = [], [], {'ordered_indices': ordered, 'labels': truth,
                                   'x': saved['x'].numpy(), 'source_prior': prior}
    maximum_gradient, weight_error, mixture_error = 0., 0., 0.
    checkpoint_components = []
    for count in range(129):
        parameters = saved['parameters'][count]
        adapted = cached_covariance(saved['encoded'][count], *parameters[1:])
        retained = frozen * (adapted.diagonal().mean() / frozen.diagonal().mean())
        np.testing.assert_allclose(retained.diagonal().mean(), adapted.diagonal().mean(), atol=1e-12)
        views = []
        for kernel in (adapted, retained):
            means, covariances = residual_prior(parameters[0], kernel, source_means,
                source_covariances, saved['parameters'][0, 0], 1.)
            with torch.no_grad():
                probability, diagnostics = component_probabilities(means, covariances, list(range(count)), truth[:count])
            maximum_gradient = max(maximum_gradient, max(i['gradient_max'] for i in diagnostics))
            views.append(probability.numpy())
            if count in (0, 8, 64, 128):
                independent = correction.independent_probabilities(means.numpy(), covariances.numpy(),
                                                                  list(range(count)), truth[:count])
                np.testing.assert_allclose(probability, independent, atol=2e-7, rtol=2e-6)
                checks.append({'support_count': count, 'view': len(views) - 1,
                    'maximum_probability_error': float(np.max(np.abs(probability.numpy() - independent)))})
        probabilities = {'adapted': views[0], 'frozen': views[1],
                         'fixed_half': .5 * (views[0] + views[1]),
                         'learned_views': np.concatenate(views)}
        if count in COUNTS:
            checkpoint_components.append(np.asarray(views))
            arrays[f'raw_{count}'] = saved['raw_probability'][count]
            for variant in VARIANTS:
                for mode in MODES:
                    score = scores[variant, mode]
                    weight = np.exp(score - logsumexp(score, axis=0))
                    expected_score = (np.log(priors[variant])[:, None]
                        + likelihoods[variant][:, :count] @ influences[mode][:count])
                    expected = np.exp(expected_score - logsumexp(expected_score, axis=0))
                    weight_error = max(weight_error, float(np.max(np.abs(expected - weight))))
                    np.testing.assert_allclose(weight, expected, atol=2e-12, rtol=2e-12)
                    probability = .9 * (weight * probabilities[variant]).sum(0) + .1 * saved['raw_probability'][count]
                    sequential = np.asarray([.9 * sum(float(expected[k, j]) * float(probabilities[variant][k, j])
                        for k in range(len(expected))) + .1 * float(saved['raw_probability'][count, j])
                        for j in range(160)])
                    mixture_error = max(mixture_error, float(np.max(np.abs(probability - sequential))))
                    np.testing.assert_allclose(probability, sequential, atol=2e-12, rtol=2e-12)
                    rows.append({'scenario': kind, 'pseudo_target': archive.source_id,
                        'lineage': archive.lineage, 'variant': variant, 'reliability': mode,
                        'support_count': count, **correction.query_metrics(probability[128:], truth[128:])})
                    arrays[f'{variant}_{mode}_{count}_weights'] = weight
                    arrays[f'{variant}_{mode}_{count}_probability'] = probability
        if count < 128:
            for variant, probability in probabilities.items():
                likelihood = np.log(np.maximum(probability[:, count] if truth[count]
                                               else 1 - probability[:, count], 1e-15))
                likelihoods[variant][:, count] = likelihood
                for mode in MODES:
                    scores[variant, mode] += likelihood[:, None] * influences[mode][count]
    arrays['checkpoint_components'] = np.asarray(checkpoint_components)
    arrays.update({f'{v}_prequery_log_likelihoods': value for v, value in likelihoods.items()})
    result = {'status': 'complete', 'protocol': specification, 'scenario': kind,
        'pseudo_target': archive.source_id, 'lineage': archive.lineage,
        'source_ids': [s.source_id for s in sources], 'rows': rows, 'independent_checks': checks,
        'nonempty_mode_solves': 128 * 8 * 2, 'maximum_final_gradient': maximum_gradient,
        'maximum_weight_error': weight_error, 'maximum_mixture_error': mixture_error,
        'wall_s': time.perf_counter() - started, 'new_selector_queries': 0, 'new_physical_calls': 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination.with_suffix('.npz'), **arrays)
    write_json(destination, result)
    print('geometry_retention', kind, archive.source_id, flush=True)
    return result


def run():
    specification = read_json(REPORT / 'protocol.json')
    assert specification == protocol() and not (REPORT / 'report.json').exists()
    started = time.perf_counter()
    torch.set_num_threads(1)
    records = []
    with threadpool_limits(1):
        for kind in specification['scenarios']:
            histories, means, covariances = correction.history_inputs(kind)
            for archive in histories:
                try:
                    records.append(task(kind, archive, histories, means, covariances, specification))
                except Exception:
                    with (REPORT / 'failed_attempts.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'scenario': kind, 'pseudo_target': archive.source_id,
                            'wall_s': time.perf_counter() - started, 'error': traceback.format_exc(),
                            'new_selector_queries': 0, 'new_physical_calls': 0}) + '\n')
                    raise
    rows = [r for record in records for r in record['rows']]
    assert len(records) == 16 and len(rows) == 640 and specification == protocol()
    controls = read_json(correction.REPORT / 'history.json')['rows']
    control_error = 0.
    for row in rows:
        if row['variant'] != 'adapted':
            continue
        old = next(r for r in controls if (r['scenario'], r['pseudo_target'], r['support_count'],
            r['beta'], r['reliability'], r['variant']) == (row['scenario'], row['pseudo_target'],
            row['support_count'], 1., row['reliability'], 'converged_prequential'))
        for metric in ('query_nll', 'query_brier'):
            control_error = max(control_error, abs(row[metric] - old[metric]))
            np.testing.assert_allclose(row[metric], old[metric], atol=2e-8, rtol=0)
    summary, screening = [], []
    for kind in specification['scenarios']:
        group = [r for r in rows if r['scenario'] == kind]
        values = {(v, m): correction.aggregate([r for r in group if (r['variant'], r['reliability']) == (v, m)])
                  for v in VARIANTS for m in MODES}
        summary += [{'scenario': kind, 'variant': v, 'reliability': m, **value}
                    for (v, m), value in values.items()]
        local_controls = [(v, 'local') for v in VARIANTS if v != 'learned_views']
        best = min(local_controls, key=lambda key: (values[key]['query_nll'], values[key]['query_brier'], key))
        candidate = values['learned_views', 'local']
        comparisons = [{'variant': v, 'reliability': m,
            'nll_lower': candidate['query_nll'] < values[v, m]['query_nll'] - 1e-12,
            'brier_not_higher': candidate['query_brier'] <= values[v, m]['query_brier'] + 1e-12}
            for v, m in (*local_controls, ('learned_views', 'global'))]
        gains = {}
        for lineage in sorted({r['lineage'] for r in group}):
            def loss(key):
                return correction.aggregate([r for r in group if r['lineage'] == lineage
                    and (r['variant'], r['reliability']) == key])['query_nll']
            gains[lineage] = loss(best) - loss(('learned_views', 'local'))
        screening.append({'scenario': kind, 'strongest_local_control': best[0],
            'comparisons': comparisons, 'lineage_nll_gains': gains,
            'passed': all(c['nll_lower'] and c['brier_not_higher'] for c in comparisons)
                      and sum(v > 1e-12 for v in gains.values()) >= 2})
    result = {'status': 'complete_historical_screening', 'protocol': specification,
        'historical_tasks': 16, 'rows': rows, 'summary': summary, 'screening': screening,
        'screening_passed': all(s['passed'] for s in screening),
        'nonempty_mode_solves': sum(r['nonempty_mode_solves'] for r in records),
        'maximum_final_gradient': max(r['maximum_final_gradient'] for r in records),
        'independent_prediction_groups': sum(len(r['independent_checks']) for r in records),
        'maximum_independent_probability_error': max(c['maximum_probability_error']
            for r in records for c in r['independent_checks']),
        'original_control_scores_reproduced': 160, 'maximum_control_error': control_error,
        'maximum_weight_error': max(r['maximum_weight_error'] for r in records),
        'maximum_mixture_error': max(r['maximum_mixture_error'] for r in records),
        'wall_s': time.perf_counter() - started, 'new_selector_queries': 0,
        'new_physical_calls': 0, 'development_admission_claimed': False}
    write_json(REPORT / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    globals()[parser.parse_args().stage]()
