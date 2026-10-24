"""Separate, paired development comparison of the two collision RGPE adaptations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
import traceback

import torch
from threadpoolctl import threadpool_limits

from methods.baselines.fsbo.method import load as load_fsbo
from methods.baselines.rgpe.search import MODES, RANK_SAMPLES, select
from methods.common.response import archive_fingerprints

from . import discovery_residual_online as residual
from . import discovery_physical_mixture as physical
from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import method_summary
from .discovery_config import OUTPUT, development_tasks, read_json, write_json
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated


REPORT = OUTPUT / 'rgpe'
RUNS = ROOT / 'results/baselines/rgpe/collision/discovery/runs'
PREPARATION = ROOT / 'results/baselines/rgpe/collision/preparation'
METHODS = MODES
REFERENCES = (*residual.PREVIOUS, *residual.METHODS)


def protocol():
    return {'purpose': 'closest published transfer-GP classification comparison; no designated-main promotion',
        'methods': list(METHODS), 'tasks': development_tasks(), 'candidate_count': 512,
        'budget': 128, 'initial_count': 8, 'ranking_samples': RANK_SAMPLES,
        'physical_mass': {'rgpe_classifier': 0., 'physical_rgpe_classifier': .1},
        'sampling_seed': 'algorithm_seed + 1009 * paid_count',
        'target_adaptation': {'steps': 10, 'learning_rate': .003},
        'physical_adaptation': {'steps': 10, 'learning_rate': .01},
        'feedback': 'paid collision Boolean only', 'reference_methods': list(REFERENCES),
        'foundation': residual.protocol(), 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'source_files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
            'benchmark/discovery_rgpe.py',
            'methods/baselines/rgpe/search.py', 'methods/common/gp_joint.py')},
        'preparation': {p: hashlib.sha256((PREPARATION / p).read_bytes()).hexdigest()
                        for p in ('checks.json', 'historical_numeric.json')}}


def previous_rows():
    return residual.previous_rows() + read_json(residual.REPORT / 'run_index.json')['runs']


def prepare():
    checks = read_json(PREPARATION / 'checks.json')
    numerical = read_json(PREPARATION / 'historical_numeric.json')
    if (checks['status'] != 'passed' or checks['rgpe_specific_checks'] != 15
            or checks['failures'] or checks['errors'] or numerical['status'] != 'passed'
            or numerical['full_pool_predictions'] != 32 or numerical['independent_loo_modes'] < 128):
        raise ValueError('Complete collision RGPE unit and historical numerical checks first')
    expected = {(kind, source.source_id, count) for kind in ('cut_in', 'cut_out')
                for source in load_construction(kind)[0] for count in (8, 128)}
    actual = [(c['kind'], c['historical_pseudo_target'], c['support_count']) for c in numerical['checks']]
    if (len(actual) != 32 or set(actual) != expected
            or any(c['historical_pseudo_target'] in c['source_ids'] or len(c['source_ids']) != 7
                   or c['full_pool_size'] != 512 for c in numerical['checks'])):
        raise ValueError('Historical RGPE checks must exclude each pseudo target and include all tasks')
    specification = protocol()
    destination = REPORT / 'design.json'
    if destination.exists() and read_json(destination) != specification:
        raise ValueError('The RGPE design was already registered differently')
    write_json(destination, specification)
    return specification


def freeze():
    audit = read_json(residual.REPORT / 'audit/report.json')
    foundation = read_json(residual.REPORT / 'run_index.json')
    if (audit['status'] != 'passed' or audit['runs'] != 240
            or audit['independent_full_pool_choices'] != 720
            or len(audit['equivalent_parameter_pairs']) != 40
            or foundation['status'] != 'complete' or foundation['protocol'] != residual.protocol()
            or residual.gate(foundation['runs']) != read_json(residual.REPORT / 'gate.json')):
        raise ValueError('Finish all residual controls and their independent audit before freezing RGPE')
    validate_rows(previous_rows(), REFERENCES)
    specification = protocol()
    if read_json(REPORT / 'design.json') != specification:
        raise ValueError('RGPE implementation or preparation differs from the registered design')
    destination = REPORT / 'protocol.json'
    if destination.exists() and read_json(destination) != specification:
        raise ValueError('The RGPE comparison was already frozen differently')
    write_json(destination, specification)
    return specification


def validate_rows(rows, methods=METHODS):
    expected = {(m, t['target_id'], t['scenario'], t['algorithm_seed'])
                for m in methods for t in development_tasks()}
    actual = [(r['method'], r['target_id'], r['scenario'], r['algorithm_seed']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Every method requires all 40 complete paired tasks without duplicates')
    denominators = {}
    for row in rows:
        key = row['target_id'], row['scenario'], row['algorithm_seed']
        total, found = row['pool_unique_collisions'], row['unique_collisions']
        if (row['queries'] != 128 or denominators.setdefault(key, total) != total
                or not 0 <= found <= min(128, total)
                or row['collision_recall'] != (found / total if total else None)):
            raise ValueError('RGPE comparison has an inconsistent budget, denominator or recall')


def execute(task, method, specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        started = time.perf_counter()
        history, development = load_construction(task['scenario'])
        target = next(a for a in development if a.source_id == task['target_id'])
        if (target.role != 'development' or any(a.role != 'history' for a in history)
                or target.source_id in {a.source_id for a in history}):
            raise ValueError('RGPE construction mixes historical and development roles')
        pretrained, metadata = load_fsbo(physical.checkpoint_path(task['scenario'], 'physical_fsbo'))
        if (metadata['history_fingerprints'] != archive_fingerprints(history)
                or metadata['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('RGPE target representation has a different construction split')
        loading_s = time.perf_counter() - started
        bank = ExecutionBank(DATA_ROOT / 'physical/development' / task['scenario'] / task['target_id'])
        setting = {'task': task, 'method': method, 'protocol': specification,
                   'history_fingerprints': archive_fingerprints(history),
                   'bank_contract': bank.manifest['contract'],
                   'candidate_ids': [s.scenario_id for s in bank.candidates]}
        destination = RUNS / method / task['target_id'] / task['scenario'] / f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != setting or output['result']['actual_queries'] != 128:
                raise ValueError('Saved RGPE run differs from the frozen setting')
        else:
            constructed = time.perf_counter()
            sources = residual.source_priors(task['scenario'], bank.candidates, history)
            construction_s = time.perf_counter() - constructed
            ledger = QueryLedger(bank.candidates, 128, bank)
            public = {'candidates': bank.candidates, 'pretrained': pretrained, 'sources': sources,
                      'initial_indices': initial_indices(512, 8, task['algorithm_seed']),
                      'sampling_seed': task['algorithm_seed'], 'mode': method, 'budget': 128,
                      'device': specification['device']}
            try:
                result = run_isolated(select, public, ledger)
            except Exception:
                path = REPORT / 'failed_runs' / method / task['target_id'] / task['scenario'] / f"{task['algorithm_seed']}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'specification': setting, 'status': 'failed',
                        'query_ledger': ledger.records, 'paid_attempts': len(ledger.records),
                        'error': traceback.format_exc(), 'wall_s': time.perf_counter() - started,
                        'new_physical_calls': 0}, ensure_ascii=False, allow_nan=False) + '\n')
                raise
            metrics = summarize(bank.candidates, result['queries'], ledger.trajectories)
            pool = next(b for b in read_json(OUTPUT / 'diagnosis/report.json')['banks']
                        if b['source_id'] == task['target_id'] and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates, pool['pool_collisions'], result['queries']))
            output = {'specification': setting, 'result': result, 'metrics': metrics,
                      'query_ledger': ledger.records,
                      'cost': {'equivalent_queries': 128, 'new_physical_calls': 0,
                               'model_loading_s': loading_s, 'source_construction_s': construction_s,
                               'wall_s': time.perf_counter() - started, 'new_historical_training': False}}
            write_json(destination, output)
    return {**task, 'method': method, 'path': str(destination.relative_to(ROOT)), **output['metrics']}


def smoke():
    specification = read_json(REPORT / 'protocol.json')
    if specification != protocol():
        raise ValueError('RGPE code or preparation changed after freezing')
    rows = [execute(development_tasks()[0], method, specification) for method in METHODS]
    write_json(REPORT / 'smoke.json', {'status': 'complete', 'runs': rows, 'protocol': specification})


def run():
    specification = read_json(REPORT / 'protocol.json')
    checked = read_json(REPORT / 'audit/smoke.json')
    negative = read_json(REPORT / 'audit/negative_controls.json')
    if (specification != protocol() or checked['status'] != 'passed' or checked['runs'] != 2
            or checked['independent_full_pool_choices'] != 6 or negative['status'] != 'passed'
            or negative['positive_controls'] != 2 or negative['rejected_corruptions'] < 8):
        raise ValueError('Check both actual RGPE smoke cases and corruption controls before running')
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, task, method, specification)
                   for method in METHODS for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('rgpe_development', len(rows), '/ 80', flush=True)
    validate_rows(rows)
    write_json(REPORT / 'run_index.json', {'status': 'complete', 'protocol': specification,
               'runs': rows, 'equivalent_queries': 10240, 'new_physical_calls': 0})


def report():
    from .discovery_physical_mixture_report import aggregate_curves

    index = read_json(REPORT / 'run_index.json')
    audit = read_json(REPORT / 'audit/report.json')
    if (index['status'] != 'complete' or index['protocol'] != protocol()
            or audit['status'] != 'passed' or audit['runs'] != 80
            or audit['independent_full_pool_choices'] != 240):
        raise ValueError('Finish and independently audit all 80 RGPE runs before reporting')
    validate_rows(index['runs'])
    prior = previous_rows()
    validate_rows(prior, REFERENCES)
    rows = index['runs'] + prior
    validate_rows(rows, (*METHODS, *REFERENCES))
    stats = method_summary(rows)
    reference = max(REFERENCES, key=lambda m: (stats[m]['collision_recall']['mean'],
                    stats[m]['unique_collisions']['mean'], stats[m]['discovery_auc']['mean'], m))
    curves = aggregate_curves(rows, stats)
    comparisons = []
    for method, baseline in [(m, reference) for m in METHODS] + [(METHODS[1], METHODS[0])]:
        for metric in ('unique_collisions', 'collision_recall', 'discovery_auc'):
            pair = [r for r in rows if r['method'] in (method, baseline) and r[metric] is not None]
            comparisons.append(paired_lineage_bootstrap(pair, method, baseline, metric))
    costs = {}
    for method in METHODS:
        outputs = [read_json(ROOT / r['path']) for r in index['runs'] if r['method'] == method]
        costs[method] = {'runs': len(outputs), 'equivalent_queries': 128 * len(outputs),
                        'new_physical_calls': 0,
                        **{k: sum(o['cost'][k] for o in outputs)
                           for k in ('wall_s', 'model_loading_s', 'source_construction_s')},
                        **{k: sum(q[k] for o in outputs for q in o['result']['queries'])
                           for k in ('selection_s', 'target_adaptation_s', 'physical_adaptation_s')},
                        'replay_equivalent_simulation_s': sum(q['feedback']['execution_s']
                                                              for o in outputs for q in o['query_ledger'])}
    failures = [json.loads(line) for path in (REPORT / 'failed_runs').rglob('*.jsonl')
                for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {'status': 'complete_audited_comparison', 'published_algorithm': '2018 RGPE classification adaptation',
              'strongest_previous_reference': reference, 'summary': stats, 'comparisons': comparisons,
              'curves': curves,
              'checkpoints': {m: {str(b): {'unique_collisions': c['discovery_curve'][b - 1],
                                         'collision_recall': c['recall_curve'][b - 1]}
                                    for b in (16, 32, 64, 128)} for m, c in curves.items()},
              'costs': costs, 'failed_attempts': failures,
              'numerical_preparation': read_json(PREPARATION / 'historical_numeric.json'),
              'initial_numerical_preparation': read_json(PREPARATION / 'first_bank_numeric.json'),
              'reused_foundation_costs': read_json(residual.REPORT / 'costs.json'),
              'main_method_promoted': False, 'prior_failed_gates_changed': False,
              'formal_phases_started': False, 'new_physical_calls': 0}
    write_json(REPORT / 'report.json', result)
    lines = ['# RGPE碰撞分类开发比较', '', f'完整80项、10,240次等效查询通过独立审计，新增物理调用为零。最强既有参考为`{reference}`。', '',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现曲线AUC |', '|---|---:|---:|---:|']
    for method in (*METHODS, reference, residual.MAIN):
        values = stats[method]
        lines.append(f"| {method} | {values['unique_collisions']['mean']:.6f} | {values['collision_recall']['mean']:.6%} | {values['discovery_auc']['mean']:.6f} |")
    lines += ['', '完整预算点、曲线、配对分层区间及分项成本见`report.json`。该比较不替换已失败的指定主方法，不自动通过阶段B或启动C/D/E。', '',
              '原论文为连续响应GP与期望改进；这里采用Bernoulli/Laplace、不同标签对、共享FSBO表示及碰撞概率贪心，详细差异见`methods/baselines/rgpe/README.md`。', '']
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'freeze', 'smoke', 'run', 'report'))
    globals()[parser.parse_args().stage]()
