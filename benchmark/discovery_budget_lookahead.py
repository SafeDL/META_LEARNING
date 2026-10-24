"""All-candidate fixed-kernel lookahead on paid historical prefixes only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.gp_classifier import ClassificationKernel, fit_target
from methods.common.response import source_weights
from methods.gcrt.converged_lookahead import acquisitions, choose, mix_virtual, virtual_components
from methods.gcrt.converged_response import component_probabilities
from methods.gcrt.residual_scale import residual_prior

from . import discovery_converged_response as correction
from . import discovery_feature_reliability as reliability
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_converged_search_audit import probabilities_numpy
from .discovery_physical_mixture import checkpoint_path
from .protocol import ROOT
from .scenarios import dedup_key


REPORT = OUTPUT / 'budget_lookahead'
MODES = ('local', 'global')
HORIZONS = (5, 64)


def protocol():
    return {'purpose': 'full-candidate historical lookahead diagnostic; no development stage C',
        'historical_tasks': 16, 'support_count': 64, 'candidate_count': 512,
        'unqueried_count': 448, 'ordered_seed': 1301, 'modes': list(MODES), 'horizons': list(HORIZONS),
        'beta': {'cut_in': 1., 'cut_out': 1.}, 'bandwidth': .3, 'physical_mass': .1,
        'virtual_model': 'fixed current kernel and mean; recondition all paid labels plus one virtual Boolean',
        'solver': 'unchanged converged CPU float64; gradient at most 1e-10',
        'reliability': 'past prequery likelihoods unchanged; virtual likelihood before observing virtual label',
        'teacher_scope': 'existing all-history teachers; own source posterior excluded; not strict held-system validation',
        'retention': 'a local window improves lineage-equal actual batch potential; neither scene lower; two positive lineages',
        'probe_candidates': 8, 'maximum_projected_serial_virtual_s': 1800,
        'historical_context': correction.protocol(), 'reliability_context': reliability.protocol(),
        'source_files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'benchmark/discovery_budget_lookahead.py', 'methods/gcrt/converged_lookahead.py')},
        'new_development_queries': 0, 'new_physical_calls': 0, 'formal_phases_entered': False}


def prepare():
    assert read_json(correction.REPORT / 'summary_audit.json')['status'] == 'passed'
    assert read_json(reliability.REPORT / 'summary_audit.json')['status'] == 'passed'
    specification = protocol()
    path = REPORT / 'protocol.json'
    if path.exists():
        assert read_json(path) == specification
    else:
        write_json(path, specification)
    return specification


def state(kind, archive, history, all_means, all_covariances):
    started = time.perf_counter()
    assert len({dedup_key(s) for s in archive.scenarios}) == 512
    cache = correction.cache_inputs(kind, archive)
    ordered = cache['ordered']
    paid = ordered[:64].tolist()
    labels = archive.collision[paid].tolist()
    positions = [i for i, source in enumerate(history) if source.source_id != archive.source_id]
    sources = [history[i] for i in positions]
    assert len(sources) == 7
    path = reliability.REPORT / 'histories' / kind / f'{archive.source_id}.json'
    metadata = read_json(path)
    assert metadata['protocol'] == reliability.protocol() and metadata['source_ids'] == [s.source_id for s in sources]
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['ordered_indices'], ordered)
        likelihoods = saved['prequery_log_likelihoods'][:, :64].copy()
        cached_before = saved['checkpoint_probabilities'][3].copy()
    base, _ = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
    base = base.to('cuda')
    current = copy.deepcopy(base)
    physical = ClassificationKernel(archive.x.shape[1]).to('cuda')
    fit_target(current, archive.x[paid], labels, steps=10, learning_rate=.003)
    fit_target(physical, archive.x[paid], labels, steps=10, learning_rate=.01)
    with torch.no_grad():
        torch.testing.assert_close(current.features(archive.x)[ordered].cpu(), cache['encoded'][64], atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose([float(current.mean), float(current.log_lengthscale), float(current.log_scale)],
                                  cache['parameters'][64], atol=2e-8, rtol=2e-7)
        means, covariances = residual_prior(current.mean.cpu(), current.covariance(archive.x).cpu(),
            torch.tensor(all_means[positions]), torch.tensor(all_covariances[positions]), float(base.mean), 1.)
        means = torch.cat((means, physical.mean.cpu().expand(1, 512)))
        covariances = torch.cat((covariances, physical.covariance(archive.x)[None].cpu()))
        before, information = component_probabilities(means, covariances, paid, labels)
    before = before.numpy()
    np.testing.assert_allclose(before[:-1, ordered], cached_before, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(before[-1, ordered], cache['raw_probability'][64], atol=2e-9, rtol=2e-9)
    prior = np.r_[.75 * source_weights(sources), .25]
    distances = np.square(archive.x[paid, None] - archive.x[None]).mean(-1)
    scores = {'local': np.log(prior)[:, None] + likelihoods @ np.exp(-distances / .18),
              'global': np.log(prior)[:, None] + likelihoods @ np.ones((64, 512))}
    probability = {}
    for mode in MODES:
        weight = np.exp(scores[mode] - logsumexp(scores[mode], axis=0))
        probability[mode] = .9 * (weight * before[:-1]).sum(0) + .1 * before[-1]
        observed = correction.query_metrics(probability[mode][ordered[128:]], archive.collision[ordered[128:]])
        old = next(r for r in metadata['rows'] if r['setting'] == ('physical_0.3' if mode == 'local' else 'global')
                   and r['support_count'] == 64)
        for key, value in observed.items():
            np.testing.assert_allclose(value, old[key], atol=2e-8, rtol=0)
    return {'means': means, 'covariances': covariances, 'before': before, 'paid': paid, 'labels': labels,
        'scores': scores, 'probability': probability, 'available': np.asarray(sorted(set(range(512)) - set(paid))),
        'x': archive.x, 'source_ids': [s.source_id for s in sources],
        'real_gradient_max': max(r['gradient_max'] for r in information),
        'preparation_s': time.perf_counter() - started}


def enumerate_candidates(current, candidates):
    started = time.perf_counter()
    result = np.empty((2, len(candidates), 2, 512))
    gradients = np.empty((len(candidates), 2, 9))
    iterations = np.empty_like(gradients, dtype=np.int16)
    before_means, before_covariances = current['means'].clone(), current['covariances'].clone()
    scores_before = {k: value.copy() for k, value in current['scores'].items()}
    for position, index in enumerate(candidates):
        local = np.exp(-np.square(current['x'] - current['x'][index]).mean(-1) / .18)
        for label in (False, True):
            component, info = virtual_components(current['means'], current['covariances'],
                current['paid'], current['labels'], int(index), label)
            gradients[position, int(label)] = [r['gradient_max'] for r in info]
            iterations[position, int(label)] = [r['iterations'] for r in info]
            for mode, name in enumerate(MODES):
                result[mode, position, int(label)] = mix_virtual(component, current['before'],
                    current['scores'][name], local if name == 'local' else np.ones(512), int(index), label)
    torch.testing.assert_close(current['means'], before_means, atol=0, rtol=0)
    torch.testing.assert_close(current['covariances'], before_covariances, atol=0, rtol=0)
    for name, value in scores_before.items():
        np.testing.assert_array_equal(current['scores'][name], value)
    return result, gradients, iterations, time.perf_counter() - started


def independent_branch(current, index, label, actual):
    component = probabilities_numpy(current['means'].numpy(), current['covariances'].numpy(),
                                    [*current['paid'], index], [*current['labels'], label])
    errors = []
    for mode, name in enumerate(MODES):
        evidence = np.log(np.maximum(current['before'][:-1, index] if label
                                    else 1 - current['before'][:-1, index], 1e-15))
        weights = np.empty_like(current['scores'][name])
        for j in range(512):
            influence = (np.exp(-np.mean((current['x'][j] - current['x'][index])**2) / .18)
                         if name == 'local' else 1.)
            score = current['scores'][name][:, j] + evidence * influence
            weights[:, j] = np.exp(score - logsumexp(score))
        expected = .9 * (weights * component[:-1]).sum(0) + .1 * component[-1]
        error = float(np.max(np.abs(expected - actual[mode])))
        errors.append(error)
        np.testing.assert_allclose(actual[mode], expected, atol=2e-7, rtol=2e-6)
    return max(errors)


def probe():
    specification = read_json(REPORT / 'protocol.json')
    assert specification == protocol()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, means, covariances = correction.history_inputs('cut_in')
        current = state('cut_in', history[0], history, means, covariances)
        candidates = current['available'][:8]
        values, gradients, _, elapsed = enumerate_candidates(current, candidates)
        error = max(independent_branch(current, int(candidates[i]), y, values[:, i, int(y)])
                    for i in (0, 7) for y in (False, True))
    result = {'status': 'passed', 'historical_state': history[0].source_id, 'candidates': 8,
        'virtual_component_modes': int(gradients.size), 'maximum_gradient': float(gradients.max()),
        'independent_branch_groups': 4, 'independent_maximum_probability_error': error,
        'state_preparation_s': current['preparation_s'], 'virtual_inference_s': elapsed,
        'projected_serial_full_virtual_s': elapsed * 448 / 8 * 16,
        'actual_state_unchanged': True, 'new_development_queries': 0, 'new_physical_calls': 0}
    write_json(REPORT / 'probe.json', result)
    print(result)
    return result


def execute(kind, pseudo_target):
    torch.set_num_threads(1)
    started = time.perf_counter()
    destination = REPORT / 'histories' / kind / f'{pseudo_target}.json'
    specification = read_json(REPORT / 'protocol.json')
    if destination.exists():
        saved = read_json(destination)
        assert saved['protocol'] == specification and saved['status'] == 'complete'
        return saved
    try:
        with threadpool_limits(1):
            history, means, covariance = correction.history_inputs(kind)
            archive = next(a for a in history if a.source_id == pseudo_target)
            current = state(kind, archive, history, means, covariance)
            available = current['available']
            conditional, gradients, iterations, elapsed = enumerate_candidates(current, available)
            identifiers = [s.scenario_id for s in archive.scenarios]
            records, candidates_to_check = [], {int(available[0])}
            for mode, name in enumerate(MODES):
                probability = current['probability'][name]
                greedy = choose(probability[available], available, identifiers)
                acquisition = acquisitions(probability, conditional[mode], available, HORIZONS)
                for horizon in HORIZONS:
                    chosen = choose(acquisition[horizon], available, identifiers)
                    candidates_to_check.update((greedy, chosen))
                    for rule, index in (('greedy', greedy), ('lookahead', chosen)):
                        position = int(np.flatnonzero(available == index)[0])
                        # Complete historical labels enter only this evaluation of fixed choices.
                        label = bool(archive.collision[index])
                        after = conditional[mode, position, int(label)]
                        future = sorted((i for i in available if i != index), key=lambda i: (-after[i], identifiers[i]))
                        records.append({'scenario': kind, 'pseudo_target': pseudo_target, 'lineage': archive.lineage,
                            'reliability': name, 'horizon': horizon, 'rule': rule, 'candidate': index,
                            'actual_label': label, 'expected_batch_utility': float(acquisition[horizon][position]),
                            'actual_batch_collisions': int(label) + int(archive.collision[future[:horizon - 1]].sum())})
            errors = []
            for index in sorted(candidates_to_check):
                position = int(np.flatnonzero(available == index)[0])
                for label in (False, True):
                    errors.append(independent_branch(current, index, label, conditional[:, position, int(label)]))
            result = {'status': 'complete', 'protocol': specification, 'scenario': kind,
                'pseudo_target': pseudo_target, 'lineage': archive.lineage, 'source_ids': current['source_ids'],
                'records': records, 'enumerated_candidates': 448, 'virtual_component_modes': int(gradients.size),
                'maximum_virtual_gradient': float(gradients.max()), 'maximum_virtual_iterations': int(iterations.max()),
                'independent_branch_groups': len(errors), 'independent_maximum_probability_error': max(errors),
                'real_state_unchanged': True, 'state_preparation_s': current['preparation_s'],
                'virtual_inference_s': elapsed, 'wall_s': time.perf_counter() - started,
                'new_development_queries': 0, 'new_physical_calls': 0}
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination.with_suffix('.npz'), available=available, paid=np.asarray(current['paid']),
                before=current['before'], local_scores=current['scores']['local'], global_scores=current['scores']['global'],
                probability=np.asarray([current['probability'][m] for m in MODES]),
                conditional=conditional, gradients=gradients, iterations=iterations)
            write_json(destination, result)
            print('budget_lookahead', kind, pseudo_target, flush=True)
            return result
    except Exception:
        with (REPORT / 'failed_attempts.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'scenario': kind, 'pseudo_target': pseudo_target,
                'wall_s': time.perf_counter() - started, 'error': traceback.format_exc(),
                'new_development_queries': 0, 'new_physical_calls': 0}) + '\n')
        raise


def run():
    specification = read_json(REPORT / 'protocol.json')
    checked = read_json(REPORT / 'probe.json')
    assert specification == protocol() and checked['status'] == 'passed'
    assert checked['projected_serial_full_virtual_s'] <= 1800
    tasks = [(kind, a.source_id) for kind in ('cut_in', 'cut_out') for a in correction.history_inputs(kind)[0]]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute, *task) for task in tasks]):
            rows.append(future.result())
    assert len(rows) == 16 and sum(r['virtual_component_modes'] for r in rows) == 129024
    write_json(REPORT / 'all_states.json', {'status': 'complete', 'protocol': specification, 'states': rows,
        'new_development_queries': 0, 'new_physical_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'probe', 'run'))
    globals()[parser.parse_args().stage]()
