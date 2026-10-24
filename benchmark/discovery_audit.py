"""Recompute completed development evidence from raw executions and paid ledgers."""

import math

import numpy as np
from scipy.special import logsumexp
import torch

from methods.common.classification import logistic_moments, site_derivatives
from methods.gcrt.discovery import discovery_posterior
from methods.gcrt.train import load_encoder

from .build_collision_models import model_path
from .construction import load_construction
from .discovery import experiment_contract, pilot_gate, summary
from .discovery_config import (DATA, OUTPUT, STRONG_BASELINES, SUPPLEMENT_LIMIT,
                               read_json, write_json)
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key, features
from .schema import QueryFeedback


def audit_source_evidence(payload, candidates, history):
    specification = payload['specification']
    method = specification['construction']['method']
    bandwidth = specification['construction'].get('bandwidth')
    encoder, metadata = load_encoder(model_path('history_response', 'all', candidates[0].kind))
    model = discovery_posterior(encoder, features(candidates), history,
                                independent=method != 'shared_local', bandwidth=bandwidth,
                                target_weight=metadata['config']['target_weight'])
    return reconstruct_evidence(payload['result']['queries'], model, features(candidates))


def reconstruct_evidence(queries, model, x):
    """Independent array updates for a fresh public prior and its recorded queries."""
    if model.observed or model.sites:
        raise ValueError('Evidence reconstruction requires an unconditioned initial prior')
    bandwidth = getattr(model, 'bandwidth', None)
    mean, covariance = model.mean.copy(), model.covariance.copy()
    prior = np.log(model.weights)
    evidence = np.zeros_like(mean)
    traces = []
    for query in queries:
        i, y = query['index'], query['feedback']['collision']
        logits = prior + evidence[:, i]
        log_weights = logits - logsumexp(logits)
        p = logistic_moments(mean[:, i], covariance[:, i, i])[0]
        np.testing.assert_allclose(np.exp(log_weights) @ p, query['prequery_probability'], atol=2e-12)
        likelihood, gradient, curvature = site_derivatives(mean[:, i], covariance[:, i, i], y)
        expected_evidence = logsumexp(log_weights + likelihood)
        np.testing.assert_allclose(expected_evidence, query['log_evidence'], atol=2e-12)
        influence = (np.ones(len(x)) if bandwidth is None else
                     np.exp(-((x - x[i])**2).mean(1) / (2 * bandwidth**2)))
        evidence += likelihood[:, None] * influence
        weights = prior + evidence[:, i]
        weights = np.exp(weights - logsumexp(weights))
        np.testing.assert_allclose(weights, query['source_weights'], atol=2e-12)
        cross = covariance[:, :, i].copy()
        mean += gradient[:, None] * cross
        covariance += curvature[:, None, None] * cross[:, :, None] * cross[:, None, :]
        traces.append({'query': query['query'], 'index': i,
                       'prequery_component_log_likelihood': likelihood.tolist()})
    return traces


def audit():
    torch.set_num_threads(1)
    index = read_json(OUTPUT / 'pilot/run_index.json')
    tuning = read_json(OUTPUT / 'pilot/bandwidth.json')
    gate = read_json(OUTPUT / 'pilot/gate.json')
    rows_by_path = {r['path']: r for r in (*index['runs'], *tuning['runs'])}
    if len(index['runs']) != 440 or len(tuning['runs']) != 120 or len(rows_by_path) != 520:
        raise ValueError('Audit requires 440 main runs plus exactly 80 nonselected bandwidth runs')
    raw_banks = {}
    for row in index['runs']:
        key = row['target_id'], row['scenario']
        if key not in raw_banks:
            bank = ExecutionBank(DATA_ROOT / 'physical/development' / row['scenario'] / row['target_id'])
            raw_banks[key] = (bank.candidates, [bank(s).feedback.to_dict() for s in bank.candidates])
    checked, source_checked = [], 0
    histories = {kind: load_construction(kind)[0] for _, kind in raw_banks}
    for path, row in sorted(rows_by_path.items()):
        payload = read_json(ROOT / path)
        if (payload['cost']['execution_mode'] != 'replay'
                or payload['cost']['actual_new_physical_calls'] != 0):
            raise ValueError('Development replay must not hide new physical executions')
        if payload['specification']['metadata']['discovery_contract'] != experiment_contract():
            raise ValueError('Completed run differs from the current recorded experiment source')
        candidates, feedbacks = raw_banks[row['target_id'], row['scenario']]
        queries = payload['result']['queries']
        selected = [q['index'] for q in queries]
        if (len(queries) != 128 or len(set(selected)) != 128
                or selected[:8] != initial_indices(512, 8, row['algorithm_seed'])
                or payload['result']['selected_indices'] != selected
                or len(payload['query_ledger']) != 128):
            raise ValueError(f'Incomplete, repeated or unmatched initialization in {path}')
        failed = {dedup_key(s) for s, f in zip(candidates, feedbacks) if f['collision']}
        found, curve = set(), []
        for n, (q, ledger) in enumerate(zip(queries, payload['query_ledger']), 1):
            if (q['query'] != n or ledger['call'] != n or ledger['status'] != 'complete'
                    or QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[q['index']]
                    or ledger['feedback'] != q['feedback']
                    or ledger['index'] != q['index']
                    or ledger['purpose'] != ('initial' if n <= 8 else 'target')):
                raise ValueError(f'Ledger differs from original paid feedback in {path}')
            if q['feedback']['collision']:
                found.add(dedup_key(candidates[q['index']]))
            curve.append(len(found))
        expected_recall = [n / len(failed) if failed else None for n in curve]
        metrics = payload['metrics']
        if any(row.get(key) != value for key, value in metrics.items()):
            raise ValueError(f'Run index differs from its saved execution metrics in {path}')
        if any(row[key] != payload['specification']['metadata'][key]
               for key in ('target_id', 'lineage', 'scenario', 'algorithm_seed', 'method')):
            raise ValueError(f'Run index differs from its declared development task in {path}')
        if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != expected_recall
                or metrics['pool_unique_collisions'] != len(failed)
                or metrics['unique_collisions'] != curve[-1]
                or metrics['collision_recall'] != expected_recall[-1]
                or not math.isclose(metrics['discovery_auc'], float(np.mean(curve)))):
            raise ValueError(f'Discovery/recall metrics differ from exhaustive raw executions in {path}')
        for b in (16, 32, 64, 128):
            if metrics['checkpoints'][str(b)] != {'unique_collisions': curve[b-1],
                                                 'collision_recall': expected_recall[b-1]}:
                raise ValueError(f'Checkpoint metric differs in {path}')
        if row['method'] in ('source_global', 'shared_local', 'source_local'):
            trace = audit_source_evidence(payload, candidates, histories[row['scenario']])
            trace_path = OUTPUT / 'audit/evidence' / row['target_id'] / row['scenario'] / row['method']
            write_json(trace_path / f"bandwidth_{row['bandwidth']}_{row['algorithm_seed']}.json", trace)
            source_checked += 1
        checked.append(path)
        if len(checked) % 40 == 0:
            print('audit', len(checked), '/', 520, flush=True)
    if pilot_gate(index['runs']) != gate:
        raise ValueError('Stored development gate differs from its complete paired matrix')
    baseline_recall = gate['summary'][gate['strongest_baseline']]['collision_recall']['mean']
    budget_summaries = {}
    for budget in (16, 32, 64, 128):
        checkpoint_rows = [{**r, **r['checkpoints'][str(budget)]} for r in index['runs']]
        values = {method: {metric: summary([r for r in checkpoint_rows if r['method'] == method], metric)
                            for metric in ('unique_collisions', 'collision_recall')}
                  for method in ('source_local', *STRONG_BASELINES)}
        budget_summaries[str(budget)] = values
    result = {'status': 'passed', 'runs': len(checked), 'queries': len(checked)*128,
              'raw_banks': len(raw_banks), 'raw_pool_executions': 8*512,
              'source_evidence_reconstructions': source_checked,
              'new_physical_calls': 0, 'checked_paths': checked,
              'gate_status': gate['status'], 'strongest_baseline_recall': baseline_recall,
              'maximum_possible_recall_gain': 1 - baseline_recall,
              'five_percentage_point_gate_feasible': baseline_recall <= 0.95,
              'prespecified_budget_summaries': budget_summaries,
              'discovery_contract': experiment_contract()}
    write_json(OUTPUT / 'audit/report.json', result)
    write_json(DATA / 'supplement_ledger.json', {
        'attempt_limit': SUPPLEMENT_LIMIT, 'new_physical_attempts': 0,
        'reason': 'All required construction and development execution banks were complete and audited',
        'reused_development_banks': len(raw_banks), 'equivalent_queries': len(checked)*128,
        'audit_report': str((OUTPUT / 'audit/report.json').relative_to(ROOT))})
    lines = ['# 开发实验独立复核', '',
             f"核对 {len(checked)} 项运行、{len(checked)*128:,} 次查询；原始开发执行库 8 份。",
             f"重建 {source_checked} 项来源核/可靠性运行的全部查询前似然、混合概率与更新后权重。", '',
             f"最强基线召回率 {baseline_recall:.4%}，理论剩余提升空间 {1-baseline_recall:.4%}。",
             f"原定提高 5 个百分点门槛在此候选库上的可达性：{'可达' if baseline_recall <= 0.95 else '不可达'}。",
             f"实际阶段 B 门槛结果：{gate['status']}。原门槛及所有运行均未修改。", '',
             '16/32/64/128 预算点均为预先规定的辅助分析，完整谱系汇总见 report.json；不将其中任一结果替代128次主门槛。', '']
    (OUTPUT / 'audit/report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({k: v for k, v in result.items() if k not in (
        'checked_paths', 'discovery_contract', 'prespecified_budget_summaries')}, flush=True)
    return result


if __name__ == '__main__':
    audit()
