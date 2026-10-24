"""Evaluator-only paired failure sets and causal paid-prefix ranking diagnostics."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.gp_classifier import ClassificationKernel, fit_target

from . import discovery_converged_search as study
from . import discovery_physical_mixture as physical
from .construction import load_construction
from .discovery_config import development_tasks, read_json, write_json
from .discovery_converged_search_audit import probabilities_numpy
from .discovery_residual_online import source_priors
from .executions import ExecutionBank
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key, features


REPORT = study.REPORT / 'diagnosis'
PREFIXES = (9, 65, 128)
VARIANTS = ('original', 'prior_only', 'recent32', 'target_floor25',
            'physical_only', 'target_and_physical')


def group_key(scenario):
    return json.dumps(dedup_key(scenario), separators=(',', ':'))


def sets():
    path = REPORT / 'sets.json'
    if path.exists():
        raise ValueError('Preserve the completed paired failure diagnosis')
    certificate = read_json(study.REPORT / 'completed_main_necessary_condition.json')
    if certificate['main'] != study.MAIN or not certificate['main_recall_below_lower_bound']:
        raise ValueError('This diagnosis follows the fully audited failed designated main')
    reference = certificate['fixed_completed_reference']
    references = {(r['target_id'], r['scenario'], r['algorithm_seed']): r
                  for r in read_json(physical.REPORT / 'run_index.json')['runs']
                  if r['method'] == reference}
    archives = {(a.source_id, kind): a for kind in ('cut_in', 'cut_out')
                for a in load_construction(kind)[1]}
    totals, tasks, unique = Counter(), defaultdict(Counter), defaultdict(lambda: defaultdict(set))
    records = []
    for task in development_tasks():
        key = task['target_id'], task['scenario'], task['algorithm_seed']
        archive = archives[key[:2]]
        main_path = study.RUNS / study.MAIN / task['target_id'] / task['scenario'] / f"{task['algorithm_seed']}.json"
        main, baseline = read_json(main_path), read_json(ROOT / references[key]['path'])
        candidates = archive.scenarios
        groups = defaultdict(list)
        positive = defaultdict(list)
        for index, (scenario, label) in enumerate(zip(candidates, archive.collision)):
            groups[group_key(scenario)].append(index)
            if label:
                positive[group_key(scenario)].append(index)
        pool = set(positive)

        def found(output):
            queries = output['result']['queries']
            assert len(queries) == 128 and len({q['index'] for q in queries}) == 128
            result = {group_key(candidates[q['index']]) for q in queries if q['feedback']['collision']}
            assert len(result) == output['metrics']['unique_collisions']
            assert len(pool) == output['metrics']['pool_unique_collisions']
            return result

        left, right = found(main), found(baseline)
        parts = {'both_found': left & right, 'main_only': left - right,
                 'reference_only': right - left, 'both_missed': pool - (left | right)}
        assert sum(map(len, parts.values())) == len(pool)
        assert len(left) - len(right) == len(parts['main_only']) - len(parts['reference_only'])
        cases = []
        for category, values in parts.items():
            totals[category] += len(values)
            tasks[(task['lineage'], task['scenario'])][category] += len(values)
            unique[(task['target_id'], task['scenario'])][category].update(values)
            if category != 'both_found':
                for value in sorted(values):
                    cases.append({'group': value, 'category': category, 'indices': groups[value],
                                  'positive_indices': positive[value],
                                  'parameters': candidates[positive[value][0]].parameters})
        records.append({**task, 'main_path': str(main_path.relative_to(ROOT)),
                        'reference_path': references[key]['path'], 'pool_unique_collisions': len(pool),
                        'partition_counts': {k: len(v) for k, v in parts.items()}, 'cases': cases})
    result = {'status': 'complete_paired_failure_partition', 'main_runs': len(records),
              'main': study.MAIN, 'reference': reference, 'runwise_totals': dict(totals),
              'by_task': [{'lineage': k[0], 'scenario': k[1], **dict(v)} for k, v in tasks.items()],
              'distinct_groups_by_task_category': [
                  {'target_id': k[0], 'scenario': k[1], **{n: len(s) for n, s in v.items()}}
                  for k, v in unique.items()],
              'distinct_group_interpretation': 'categories are per paired seed; their unions across seeds may overlap',
              'records': records, 'new_selector_queries': 0, 'new_physical_calls': 0,
              'frozen_search_changed': False}
    write_json(path, result)
    return result


def weights_for_prefix(queries, public_x, prior, paid):
    likelihoods = np.asarray([q['prequery_group_log_likelihood'] for q in queries[:paid]]).T
    selected = [q['index'] for q in queries[:paid]]
    influence = np.exp(-np.square(public_x[selected, None] - public_x[None]).mean(-1) / (2 * .3**2))
    score = np.log(prior)[:, None] + likelihoods @ influence
    original = np.exp(score - logsumexp(score, axis=0))
    recent_score = np.log(prior)[:, None] + likelihoods[:, -32:] @ influence[-32:]
    recent = np.exp(recent_score - logsumexp(recent_score, axis=0))
    if paid <= 32:
        np.testing.assert_allclose(original, recent, atol=1e-14)
    target = np.maximum(original[-1], .25)
    history = np.exp(score[:-1] - logsumexp(score[:-1], axis=0))
    floor = np.vstack((history * (1 - target), target))
    for weight in (original, recent, floor):
        assert np.isfinite(weight).all() and np.min(weight) >= 0
        np.testing.assert_allclose(weight.sum(0), 1., atol=1e-12)
    return original, recent, floor


def reconstruct(output, candidates, history):
    """Predict using paid feedback only; endpoint failure labels enter evaluation later."""
    setting, result = output['specification'], output['result']
    frozen = read_json(study.REPORT / 'protocol.json')
    assert setting['protocol'] == frozen == study.protocol()
    assert setting['method'] == study.MAIN
    assert setting['candidate_ids'] == [s.scenario_id for s in candidates]
    task, queries = setting['task'], result['queries']
    base, _ = load_fsbo(physical.checkpoint_path(task['scenario'], 'physical_fsbo'))
    base = base.to(frozen['kernel_and_adaptation_device'])
    x = features(candidates)
    sources = source_priors(task['scenario'], candidates, history)
    prior = np.r_[.75 * sources['weights'], .25]
    assert result['component_names'] == [*sources['names'], 'target_fsbo', 'target_physical']
    predictions = []
    for query_number in PREFIXES:
        paid_queries = queries[:query_number - 1]
        paid = [q['index'] for q in paid_queries]
        labels = np.asarray([q['feedback']['collision'] for q in paid_queries], dtype=bool)
        model = copy.deepcopy(base)
        fit_target(model, x[paid], labels, steps=10, learning_rate=.003)
        with torch.no_grad():
            covariance = model.covariance(x).cpu().numpy()
            center = float(model.mean.detach())
        means = np.concatenate((sources['means'] + center - float(base.mean.detach()),
                                np.full((1, len(x)), center)))
        covariances = np.concatenate((sources['covariances'] + result['beta']**2 * covariance[None],
                                      covariance[None]))
        components = probabilities_numpy(means, covariances, paid, labels)
        raw = ClassificationKernel(x.shape[1]).to(frozen['kernel_and_adaptation_device'])
        fit_target(raw, x[paid], labels, steps=10, learning_rate=.01)
        with torch.no_grad():
            raw_mean = np.full((1, len(x)), float(raw.mean.detach()))
            raw_covariance = raw.covariance(x).cpu().numpy()[None]
        physical_p = probabilities_numpy(raw_mean, raw_covariance, paid, labels)[0]
        original, recent, floor = weights_for_prefix(queries, x, prior, len(paid))
        pooled = lambda weight: .9 * (weight * components).sum(0) + .1 * physical_p
        scores = np.stack((pooled(original), pooled(prior[:, None]), pooled(recent), pooled(floor),
                           physical_p, .9 * components[-1] + .1 * physical_p))
        actual = queries[query_number - 1]
        index = actual['index']
        np.testing.assert_allclose(original[:, index], actual['prequery_group_weights'], atol=2e-12)
        np.testing.assert_allclose(components[:, index], actual['prequery_component_probabilities'][:-1],
                                   atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(physical_p[index], actual['prequery_physical_probability'], atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(scores[0, index], actual['prequery_probability'], atol=2e-7, rtol=2e-6)
        remaining = np.asarray(sorted(set(range(len(x))) - set(paid)), dtype=int)
        gap = float(scores[0, remaining].max() - scores[0, index])
        assert gap <= 2e-7 and np.isfinite(scores).all()
        predictions.append({'query': query_number, 'selected_index': index, 'paid': np.asarray(paid),
                            'remaining': remaining, 'scores': scores,
                            'components': np.vstack((components, physical_p)),
                            'weights': original, 'recent_weights': recent, 'floor_weights': floor,
                            'acquisition_gap': gap})
    return predictions


def prefix_job(row):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        path = REPORT / 'prefixes' / row['target_id'] / row['scenario'] / f"{row['algorithm_seed']}.npz"
        identity = {'source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'run': hashlib.sha256((ROOT / row['main_path']).read_bytes()).hexdigest(),
                    'sets': hashlib.sha256((REPORT / 'sets.json').read_bytes()).hexdigest(),
                    'protocol': hashlib.sha256((study.REPORT / 'protocol.json').read_bytes()).hexdigest(),
                    'prefixes': list(PREFIXES), 'variants': list(VARIANTS)}
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                assert json.loads(str(saved['identity'])) == identity
                return json.loads(str(saved['summary']))
        started = time.perf_counter()
        output = read_json(ROOT / row['main_path'])
        bank = ExecutionBank(DATA_ROOT / 'physical/development' / row['scenario'] / row['target_id'])
        history, _ = load_construction(row['scenario'])
        predictions = reconstruct(output, bank.candidates, history)
        records = []
        for prediction in predictions:
            selected, remaining = prediction['selected_index'], prediction['remaining']
            for case in row['cases']:
                if case['category'] not in ('reference_only', 'both_missed'):
                    continue
                positive = np.asarray(case['positive_indices'], dtype=int)
                assert set(positive) <= set(remaining)
                scores = prediction['scores']
                index = int(positive[np.argmax(scores[0, positive])])
                ranks = {variant: 1 + int(np.sum(scores[i, remaining] > scores[i, positive].max() + 1e-12))
                         for i, variant in enumerate(VARIANTS)}
                records.append({'query': prediction['query'], 'group': case['group'],
                    'category': case['category'], 'index': index, 'ranks': ranks,
                    'original_probability': float(scores[0, index]),
                    'selected_probability': float(scores[0, selected]),
                    'physical_probability': float(scores[4, index]),
                    'selected_physical_probability': float(scores[4, selected]),
                    'target_weight': float(prediction['weights'][-1, index]),
                    'target_probability': float(prediction['components'][-2, index]),
                    'historical_max_probability': float(prediction['components'][:-2, index].max()),
                    'component_probabilities': prediction['components'][:, index].tolist(),
                    'group_weights': prediction['weights'][:, index].tolist(),
                    'physical_ranks_missed_group_above_selected': bool(
                        scores[4, positive].max() > scores[4, selected] + 1e-6
                        and scores[0, positive].max() < scores[0, selected] - 1e-6)})
        summary = {k: row[k] for k in development_tasks()[0]}
        summary.update(independently_reconstructed_prefixes=len(predictions),
                       maximum_acquisition_gap=max(p['acquisition_gap'] for p in predictions),
                       records=records, wall_s=time.perf_counter() - started)
        arrays = {'identity': np.asarray(json.dumps(identity)), 'summary': np.asarray(json.dumps(summary))}
        for prediction in predictions:
            for name, value in prediction.items():
                if isinstance(value, np.ndarray):
                    arrays[f"q{prediction['query']}_{name}"] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return summary


def run_prefixes(smoke=False):
    diagnosis = read_json(REPORT / 'sets.json')
    assert diagnosis['main_runs'] == 40
    rows = diagnosis['records'][:1] if smoke else diagnosis['records']
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prefix_job, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
            print('converged_paid_prefix', len(results), '/', len(rows), flush=True)
    result = {'status': 'complete_fixed_support_diagnosis', 'scope': 'smoke' if smoke else 'complete',
              'runs': len(results), 'prefixes': len(results) * len(PREFIXES),
              'query_checkpoints': list(PREFIXES), 'counterfactual_rankings': list(VARIANTS),
              'counterfactuals_are_online_results': False, 'records': results,
              'new_selector_queries': 0, 'new_physical_calls': 0,
              'cost_scope': 'diagnostic worker timing overlaps live selectors; not exclusive algorithm timing'}
    write_json(REPORT / ('smoke.json' if smoke else 'prefix.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('sets', 'smoke', 'prefix'))
    stage = parser.parse_args().stage
    sets() if stage == 'sets' else run_prefixes(smoke=stage == 'smoke')
