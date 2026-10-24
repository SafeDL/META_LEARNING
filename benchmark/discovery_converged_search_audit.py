"""Independent collision, causal reliability and converged-mode checks for source search."""

import argparse
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
from methods.common.response import archive_fingerprints
from methods.gcrt.joint_training import load_model

from . import discovery_physical_mixture as physical
from .construction import load_construction
from .discovery_config import development_tasks, read_json, write_json
from .discovery_physical_mixture_audit import raw_banks
from .discovery_converged_search import MAIN, METHODS, REPORT, RUNS, gate, protocol
from .discovery_residual_online import source_priors
from .discovery_scaml_audit import mode_numpy
from .initialization import initial_indices
from .protocol import ROOT
from .scenarios import dedup_key, features
from .schema import QueryFeedback


def probabilities_numpy(means, covariances, paid, labels):
    from scipy.special import expit

    nodes, weights = np.polynomial.hermite.hermgauss(32)
    probabilities = []
    for mean, covariance in zip(means, covariances):
        if len(paid):
            _, alpha, curvature, factor, _ = mode_numpy(mean[paid], covariance[np.ix_(paid, paid)], labels)
            cross = covariance[np.asarray(paid)]
            solved = np.linalg.solve(factor, curvature[:, None]*cross)
            m = mean+cross.T@alpha
            variance = np.diag(covariance)-np.square(solved).sum(0)
        else:
            m, variance = mean, np.diag(covariance)
        probabilities.append(expit(m[:, None]+np.sqrt(2*np.maximum(variance, 1e-10))[:, None]*nodes)
                             @ (weights/np.sqrt(np.pi)))
    return np.asarray(probabilities)


def check_run(row, candidates, feedbacks, output=None):
    torch.set_num_threads(1)
    started = time.perf_counter()
    with threadpool_limits(1):
        output = read_json(ROOT/row['path']) if output is None else output
        setting, result = output['specification'], output['result']
        task = {k: row[k] for k in development_tasks()[0]}
        method = row['method']
        frozen = read_json(REPORT/'protocol.json')
        if (setting['protocol'] != frozen or setting['task'] != task or task not in frozen['tasks']
                or method not in METHODS or setting['method'] != method or result['method'] != method
                or setting['candidate_ids'] != [s.scenario_id for s in candidates]):
            raise ValueError('Converged run differs from the registered task and public candidates')
        queries, ledger = result['queries'], output['query_ledger']
        selected = [q['index'] for q in queries]
        if (len(queries) != 128 or len(ledger) != 128 or len(set(selected)) != 128
                or selected[:8] != initial_indices(512, 8, task['algorithm_seed'])
                or result['selected_indices'] != selected or result['actual_queries'] != 128
                or result['maximum_queries'] != 128 or result['termination'] != 'budget_exhausted'):
            raise ValueError('Converged budget, initialization or query uniqueness differs')
        failure_pool = {dedup_key(s) for s, f in zip(candidates, feedbacks) if f['collision']}
        found, curve = set(), []
        for n, (q, record) in enumerate(zip(queries, ledger), 1):
            if (q['query'] != n or record['call'] != n or record['index'] != q['index']
                    or record['status'] != 'complete' or record['account'] != 'target'
                    or record['scenario_id'] != candidates[q['index']].scenario_id
                    or record['purpose'] != ('initial' if n <= 8 else 'target')
                    or record['feedback'] != q['feedback']
                    or QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[q['index']]):
                raise ValueError('Converged paid disclosure differs from its original complete execution')
            if q['feedback']['collision']:
                found.add(dedup_key(candidates[q['index']]))
            curve.append(len(found))
        recall = [v/len(failure_pool) if failure_pool else None for v in curve]
        metrics = output['metrics']
        if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != recall
                or metrics['unique_collisions'] != curve[-1] or metrics['collision_recall'] != recall[-1]
                or metrics['pool_unique_collisions'] != len(failure_pool)
                or any(row[k] != value for k, value in metrics.items())):
            raise ValueError('Converged discovery metrics differ from the complete deduplicated collision pool')
        np.testing.assert_allclose(metrics['discovery_auc'], np.mean(curve), atol=1e-12)
        for budget in (16, 32, 64, 128):
            if metrics['checkpoints'][str(budget)] != {'unique_collisions': curve[budget-1], 'collision_recall': recall[budget-1]}:
                raise ValueError('Converged checkpoint differs from its true discovery curve')
        history, development = load_construction(task['scenario'])
        archive = next(a for a in development if a.source_id == task['target_id'])
        if (archive.scenarios != candidates or setting['history_fingerprints'] != archive_fingerprints(history)
                or setting['bank_contract'] != archive.provenance['contract']):
            raise ValueError('Converged run differs from the qualified archive roles and execution contract')
        base, metadata = load_fsbo(physical.checkpoint_path(task['scenario'], 'physical_fsbo'))
        joint_base, joint_metadata = load_model(physical.checkpoint_path(task['scenario'], physical.MAIN))
        for construction in (metadata, joint_metadata['specification']):
            if (construction['history_fingerprints'] != archive_fingerprints(history)
                    or construction['development_fingerprints'] != archive_fingerprints(development)):
                raise ValueError('Converged teachers used a different allowed history/development split')
        base, joint_base = base.to(frozen['kernel_and_adaptation_device']), joint_base.to(frozen['kernel_and_adaptation_device'])
        uses_sources = method not in ('converged_target', 'converged_dual_target')
        sources = source_priors(task['scenario'], candidates, history) if uses_sources else None
        beta = frozen['selected_beta'][task['scenario']] if uses_sources else None
        bandwidth = .3 if method in ('converged_local', 'converged_dual_target') else None
        if uses_sources:
            prior = np.r_[.75*sources['weights'], .25]
            names = [*sources['names'], 'target_fsbo', 'target_physical']
        elif method == 'converged_dual_target':
            prior, names = np.array([.75, .25]), ['target_joint', 'target_fsbo', 'target_physical']
        else:
            prior, names = np.ones(1), ['target_fsbo', 'target_physical']
        if (result['beta'] != beta or result['reliability_bandwidth'] != bandwidth
                or result['component_names'] != names or result['physical_mass'] != .1
                or result['uses_source_posteriors'] != uses_sources
                or result['feedback_used'] != 'paid collision labels only'):
            raise ValueError('Converged scale, information control or component identities differ')
        np.testing.assert_allclose(result['group_prior_weights'], prior, atol=1e-14)
        np.testing.assert_allclose(result['reference_target_mean'], float(base.mean.detach()), atol=1e-14)
        if (result['conditional_inference_device'] != 'cpu'
                or result['kernel_and_adaptation_device'] != frozen['kernel_and_adaptation_device']
                or result['conditional_mode_solver'] != 'whitened damped Newton; gradient at most 1e-10'):
            raise ValueError('Converged conditional inference changed device or solver')
        x = features(candidates)
        score = np.log(prior)[:, None]+np.zeros((len(prior), 512))
        numerical = []
        for n, q in enumerate(queries, 1):
            index, label = q['index'], q['feedback']['collision']
            infos = [*q['conditional_modes'], q['physical_mode']]
            if (len(infos) != len(prior)+1 or any(not np.isfinite(i['gradient_max'])
                    or not 0 <= i['gradient_max'] <= 1e-10 or not 0 <= i['iterations'] <= 100
                    or i['backtracks'] < 0 for i in infos)):
                raise ValueError('A reported source or target conditional mode did not converge')
            probabilities = np.asarray(q['prequery_component_probabilities'])
            if probabilities.shape != (len(prior)+1,) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError('Converged component probabilities have invalid shape or values')
            weights = np.exp(score[:, index]-logsumexp(score[:, index]))
            np.testing.assert_allclose(q['prequery_group_weights'], weights, atol=2e-12)
            np.testing.assert_allclose(q['prequery_component_weights'], np.r_[.9*weights, .1], atol=2e-12)
            np.testing.assert_allclose(q['prequery_physical_probability'], probabilities[-1], atol=2e-12)
            np.testing.assert_allclose(q['prequery_group_probability'], weights @ probabilities[:-1], atol=2e-12)
            np.testing.assert_allclose(q['prequery_probability'], .9*(weights @ probabilities[:-1])+.1*probabilities[-1], atol=2e-12)
            if n in (9, 65, 128):
                paid = selected[:n-1]
                labels = np.array([feedbacks[i]['collision'] for i in paid], dtype=bool)
                model = copy.deepcopy(base)
                fit_target(model, x[paid], labels, steps=10, learning_rate=.003)
                with torch.no_grad():
                    covariance = model.covariance(x).cpu().numpy()
                    center = float(model.mean.detach())
                if uses_sources:
                    means = np.concatenate((sources['means']+(center-float(base.mean.detach())), np.full((1, 512), center)))
                    covariances = np.concatenate((sources['covariances']+beta**2*covariance[None], covariance[None]))
                elif method == 'converged_dual_target':
                    second = copy.deepcopy(joint_base)
                    fit_target(second, x[paid], labels, steps=10, learning_rate=.003)
                    with torch.no_grad():
                        means = np.array([np.full(512, float(second.mean)), np.full(512, center)])
                        covariances = np.stack((second.covariance(x).cpu().numpy(), covariance))
                else:
                    means, covariances = np.full((1, 512), center), covariance[None]
                independent_p = probabilities_numpy(means, covariances, paid, labels)
                np.testing.assert_allclose(probabilities[:-1], independent_p[:, index], atol=2e-7, rtol=2e-6)
                raw = ClassificationKernel(x.shape[1]).to(frozen['kernel_and_adaptation_device'])
                fit_target(raw, x[paid], labels, steps=10, learning_rate=.01)
                with torch.no_grad():
                    raw_mean = np.full((1, 512), float(raw.mean.detach()))
                    raw_covariance = raw.covariance(x).cpu().numpy()[None]
                independent_raw = probabilities_numpy(raw_mean, raw_covariance, paid, labels)[0]
                np.testing.assert_allclose(probabilities[-1], independent_raw[index], atol=2e-9, rtol=2e-9)
                all_weights = np.exp(score-logsumexp(score, axis=0))
                complete_p = .9*(all_weights*independent_p).sum(0)+.1*independent_raw
                available = set(range(512))-set(paid)
                maximum = max(complete_p[i] for i in available)
                if complete_p[index] < maximum-2e-7:
                    raise ValueError('Converged acquisition differs from independent full-pool probabilities')
                numerical.append({'query': n, 'probability': float(complete_p[index]),
                    'acquisition_gap': float(maximum-complete_p[index]),
                    'component_probability_max_error': float(np.max(np.abs(probabilities[:-1]-independent_p[:, index])))})
            likelihood = np.log(np.maximum(probabilities[:-1] if label else 1-probabilities[:-1], 1e-15))
            np.testing.assert_allclose(q['prequery_group_log_likelihood'], likelihood, atol=2e-12)
            influence = np.exp(-np.square(x-x[index]).mean(1)/(2*.3**2)) if bandwidth is not None else np.ones(512)
            score += likelihood[:, None]*influence
            score -= logsumexp(score, axis=0)
            np.testing.assert_allclose(q['group_weights_after_query'], np.exp(score[:, index]), atol=2e-12)
            if (method != 'converged_dual_target' and q['joint_adaptation_s'] != 0
                    or not all(np.isfinite(q[k]) and q[k] >= 0 for k in ('fsbo_adaptation_s', 'joint_adaptation_s', 'physical_adaptation_s', 'selection_s'))):
                raise ValueError('Converged target-control adaptation costs differ from their model configuration')
            np.testing.assert_allclose(q['adaptation_s'], sum(q[k] for k in ('fsbo_adaptation_s', 'joint_adaptation_s', 'physical_adaptation_s')), atol=1e-12)
        if output['cost']['equivalent_queries'] != 128 or output['cost']['new_physical_calls'] != 0:
            raise ValueError('Converged replay costs differ from its actual paid budget')
        return {'path': row['path'], 'numerical_prefixes': numerical, 'reliability_updates': 128,
                'wall_s': time.perf_counter()-started}


def audit_rows(rows):
    raw = raw_banks(rows)
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
               (Path(__file__).relative_to(ROOT).as_posix(), 'benchmark/discovery_scaml_audit.py',
                'benchmark/discovery_physical_mixture_audit.py')}
    frozen_digest = hashlib.sha256((REPORT/'protocol.json').read_bytes()).hexdigest()
    feedback_digests = {key: hashlib.sha256(json.dumps(value[1], sort_keys=True).encode()).hexdigest() for key, value in raw.items()}
    checked = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {}
        for row in rows:
            key = row['target_id'], row['scenario']
            identity = {'audit_sources': sources, 'protocol_sha256': frozen_digest,
                        'run_sha256': hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),
                        'raw_feedback_sha256': feedback_digests[key]}
            path = REPORT/'audit/runs'/row['method']/row['target_id']/row['scenario']/f"{row['algorithm_seed']}.json"
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
            print('converged_search_audit', len(checked), '/', len(rows), flush=True)
    return sorted(checked, key=lambda r: r['path'])


def available_rows():
    rows = []
    for method in METHODS:
        for task in development_tasks():
            path = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
            if path.exists():
                saved = read_json(path)
                rows.append({**task, 'method': method, 'path': str(path.relative_to(ROOT)), **saved['metrics']})
    return rows


def audit(scope='complete'):
    if read_json(REPORT/'protocol.json') != protocol():
        raise ValueError('Converged audit requires the unchanged frozen comparison')
    if scope == 'smoke':
        smoke = read_json(REPORT/'smoke.json')
        rows = smoke['runs']
        if (smoke['status'] != 'complete' or len(rows) != 4 or {r['method'] for r in rows} != set(METHODS)
                or any(any(r[k] != v for k, v in development_tasks()[0].items()) for r in rows)):
            raise ValueError('Independently check all four actual smoke conditions')
    elif scope == 'available':
        rows = available_rows()
    else:
        index = read_json(REPORT/'run_index.json')
        rows = index['runs']
        if index['status'] != 'complete' or index['protocol'] != protocol() or gate(rows) != read_json(REPORT/'gate.json'):
            raise ValueError('Converged full matrix or admission differs from its frozen result')
    checked = audit_rows(rows)
    result = {'status': 'passed' if scope != 'available' else 'available_completed_runs_checked',
        'scope': scope, 'runs': len(checked), 'expected_runs': 160, 'queries': len(checked)*128,
        'independent_full_pool_choices': len(checked)*3, 'reliability_updates_checked': len(checked)*128,
        'all_reported_modes_converged': True,
        'whole_matrix_audited': scope == 'complete',
        'whole_matrix_admission_claimed': scope == 'complete' and read_json(REPORT/'gate.json')['status'] == 'passed',
        'new_physical_calls': 0, 'checks': checked}
    destination = 'report.json' if scope == 'complete' else f'{scope}.json'
    write_json(REPORT/'audit'/destination, result)
    return result

def negative_controls():
    positive = audit('smoke')
    rows = read_json(REPORT/'smoke.json')['runs']
    row = next(r for r in rows if r['method'] == MAIN)
    candidates, feedback = raw_banks([row])[(row['target_id'], row['scenario'])]
    original = read_json(ROOT/row['path'])
    identities = {r['path']: hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() for r in rows}
    mutations = []

    def changed(name, edit):
        output = copy.deepcopy(original)
        edit(output)
        mutations.append((name, output))

    changed('collision_label', lambda o: o['result']['queries'][0]['feedback'].__setitem__('collision', not o['result']['queries'][0]['feedback']['collision']))
    changed('full_pool_denominator', lambda o: o['metrics'].__setitem__('pool_unique_collisions', o['metrics']['pool_unique_collisions']+1))
    changed('duplicate_query', lambda o: o['result']['queries'][1].__setitem__('index', o['result']['queries'][0]['index']))
    changed('wrong_scale', lambda o: o['result'].__setitem__('beta', o['result']['beta']+.1))
    changed('incorrect_prequery_source_score', lambda o: o['result']['queries'][0]['prequery_group_log_likelihood'].__setitem__(
        0, o['result']['queries'][0]['prequery_group_log_likelihood'][0]+.25))
    changed('historical_query_account', lambda o: o['query_ledger'][0].__setitem__('account', 'history_certification'))
    changed('budget_not_complete', lambda o: o['result'].__setitem__('actual_queries', 127))
    changed('prior_mass', lambda o: o['result']['group_prior_weights'].__setitem__(0, .9))

    def change_probability(output, physical):
        q = output['result']['queries'][8]
        p = np.array(q['prequery_component_probabilities'])
        position = -1 if physical else 0
        p[position] += .02 if p[position] < .97 else -.02
        weights = np.array(q['prequery_group_weights'])
        group = float(weights @ p[:-1])
        q.update(prequery_component_probabilities=p.tolist(), prequery_physical_probability=float(p[-1]),
                 prequery_group_probability=group, prequery_probability=.9*group+.1*p[-1])
        likelihood = np.log(np.maximum(p[:-1] if q['feedback']['collision'] else 1-p[:-1], 1e-15))
        after = weights*np.exp(likelihood)
        q.update(prequery_group_log_likelihood=likelihood.tolist(), group_weights_after_query=(after/after.sum()).tolist())

    changed('internally_consistent_wrong_learned_gp', lambda o: change_probability(o, False))
    changed('internally_consistent_wrong_physical_gp', lambda o: change_probability(o, True))
    changed('nonconverged_mode_receipt', lambda o: o['result']['queries'][1]['conditional_modes'][0].__setitem__('gradient_max', .01))
    changed('conditional_inference_device', lambda o: o['result'].__setitem__('conditional_inference_device', 'cuda'))
    rejected = []
    for name, output in mutations:
        try:
            check_run(row, candidates, feedback, output)
        except (ValueError, AssertionError) as error:
            rejected.append({'name': name, 'error': str(error)[:240]})
        else:
            raise AssertionError(f'Converged audit accepted corruption: {name}')
    if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != identity for path, identity in identities.items()):
        raise ValueError('An in-memory corruption test modified a saved actual run')
    result = {'status': 'passed', 'positive_controls': positive['runs'], 'rejected_corruptions': len(rejected),
              'negative_controls': rejected, 'original_run_identities_unchanged': True, 'new_physical_calls': 0}
    write_json(REPORT/'audit/negative_controls.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scope', choices=('smoke', 'available', 'complete', 'negative'), nargs='?', default='complete')
    chosen = parser.parse_args().scope
    negative_controls() if chosen == 'negative' else audit(chosen)
