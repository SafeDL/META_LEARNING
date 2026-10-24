"""Development-first discovery experiment with metered, isolated selector replay."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time

import numpy as np
import torch

from .build_collision_models import model_path
from .collision_metrics import paired_lineage_bootstrap
from .collision_runs import DISCOVERY_METHODS, MAIN_METHODS, run_task
from .construction import load_construction
from .discovery_config import (BANDWIDTHS, BASELINES, CONTRACT_PATHS, METHODS, OUTPUT,
                               STRONG_BASELINES, WORKERS, development_tasks,
                               read_json, run_path, write_json)
from .discovery_diagnosis import diagnose
from .executions import ExecutionBank
from .protocol import DATA_ROOT, DEFAULT, ROOT


def experiment_contract():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in CONTRACT_PATHS}


def summary(rows, metric):
    lineages = sorted({r['lineage'] for r in rows})
    grouped = {lineage: [r[metric] for r in rows if r['lineage'] == lineage and r[metric] is not None]
               for lineage in lineages}
    values = {name: float(np.mean(items)) for name, items in grouped.items() if items}
    return {'mean': float(np.mean(list(values.values()))) if values else None,
            'by_lineage': values, 'valid_lineages': len(values),
            'valid_runs': sum(len(items) for items in grouped.values())}


def method_summary(rows):
    return {method: {metric: summary([r for r in rows if r['method'] == method], metric)
                     for metric in ('unique_collisions', 'collision_recall', 'discovery_auc',
                                    'selection_s', 'adaptation_s')}
            for method in sorted({r['method'] for r in rows})}


def pilot_gate(rows):
    expected = {(t['target_id'], t['scenario'], t['algorithm_seed'], m)
                for t in development_tasks() for m in METHODS}
    actual = [(r['target_id'], r['scenario'], r['algorithm_seed'], r['method']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('The gate requires the exact complete 440-run pilot matrix')
    if any(r['queries'] != DEFAULT.budget for r in rows):
        raise ValueError('A partial paid budget cannot enter the development gate')
    totals = {}
    for row in rows:
        key = (row['target_id'], row['scenario'], row['algorithm_seed'])
        previous = totals.setdefault(key, row['pool_unique_collisions'])
        if row['pool_unique_collisions'] != previous:
            raise ValueError('Paired methods must have the same sealed denominator')
    stats = method_summary(rows)
    baseline = sorted(STRONG_BASELINES, key=lambda m: (
        -(stats[m]['collision_recall']['mean'] if stats[m]['collision_recall']['mean'] is not None else -1),
        -stats[m]['unique_collisions']['mean'], -stats[m]['discovery_auc']['mean'], m))[0]
    main, ref, global_ref = stats['source_local'], stats[baseline], stats['source_global']
    recall = main['collision_recall']['mean']
    ref_recall = ref['collision_recall']['mean']
    global_recall = global_ref['collision_recall']['mean']
    gains = {name: value - ref['collision_recall']['by_lineage'][name]
             for name, value in main['collision_recall']['by_lineage'].items()}
    conditions = {
        'recall_gain_at_least_5pp': recall is not None and ref_recall is not None and recall - ref_recall >= 0.05 - 1e-12,
        'collision_count_not_lower': main['unique_collisions']['mean'] >= ref['unique_collisions']['mean'] - 1e-12,
        'at_least_two_positive_lineages': sum(v > 1e-12 for v in gains.values()) >= 2,
        'local_recall_exceeds_global': recall is not None and global_recall is not None and recall > global_recall + 1e-12,
        'local_collision_count_not_lower': main['unique_collisions']['mean'] >= global_ref['unique_collisions']['mean'] - 1e-12,
    }
    comparisons = []
    for other in (*STRONG_BASELINES, 'source_global', 'shared_local', 'greedy'):
        for metric in ('unique_collisions', 'collision_recall', 'discovery_auc'):
            paired = [r for r in rows if r['method'] in ('source_local', other) and r[metric] is not None]
            if paired:
                comparisons.append(paired_lineage_bootstrap(paired, 'source_local', other, metric))
    return {'status': 'passed' if all(conditions.values()) else 'not_passed',
            'strongest_baseline': baseline, 'conditions': conditions,
            'lineage_recall_gains': gains, 'summary': stats, 'comparisons': comparisons,
            'runs': len(rows), 'gate_is_statistical_significance': False,
            'next_stage': 'selection' if all(conditions.values()) else 'report',
            'counts_include_zero_failure_tasks': True,
            'recall_excludes_zero_denominators_without_imputation': True}


def execute_job(job):
    task, method, bandwidth, contract, diagnosis_path = job
    torch.set_num_threads(1)
    history, _ = load_construction(task['scenario'])
    bank = ExecutionBank(DATA_ROOT / 'physical/development' / task['scenario'] / task['target_id'])
    diagnosis = read_json(diagnosis_path)
    record = next(b for b in diagnosis['banks'] if b['source_id'] == task['target_id']
                  and b['scenario'] == task['scenario'])
    if record['role'] != 'development' or len(record['pool_collisions']) != len(bank.candidates):
        raise ValueError('Pilot outcomes must come from the audited complete development bank')
    if method in (*MAIN_METHODS, *DISCOVERY_METHODS):
        checkpoint = model_path('history_response', 'all', task['scenario'])
    elif method in ('fsbo', 'pooled_ft'):
        checkpoint = model_path(method, 'all', task['scenario'])
    else:
        checkpoint = None
    used_history = () if method in ('random', 'gp_collision') else history
    horizon = DEFAULT.budget if method.endswith('_full') else 5
    destination = run_path(task['target_id'], task['scenario'], method, task['algorithm_seed'], bandwidth)
    metadata = {**task, 'method': method, 'experiment': 'discovery',
                'discovery_contract': contract}
    result = run_task(bank, used_history, method=method, algorithm_seed=task['algorithm_seed'],
                      budget=DEFAULT.budget, initial_count=DEFAULT.initial_count,
                      destination=destination, metadata=metadata, checkpoint=checkpoint,
                      horizon=horizon, bandwidth=bandwidth, pool_collisions=record['pool_collisions'])
    return {**task, 'method': method, 'bandwidth': bandwidth, 'path': str(destination.relative_to(ROOT)),
            **result['metrics']}


def run_batch(settings):
    diagnosis_path = OUTPUT / 'diagnosis/report.json'
    diagnosis = read_json(diagnosis_path)
    if diagnosis['status'] != 'passed':
        raise ValueError('A successful exhaustive development audit is required')
    contract = experiment_contract()
    jobs = [(task, method, bandwidth, contract, str(diagnosis_path))
            for method, bandwidth in settings for task in development_tasks()]
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pending = {pool.submit(execute_job, job): job for job in jobs}
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            if len(rows) % 8 == 0 or len(rows) == len(jobs):
                print('discovery', len(rows), '/', len(jobs), row['method'], row['bandwidth'],
                      row['target_id'], row['scenario'], 'elapsed_s', round(time.perf_counter()-started), flush=True)
    return sorted(rows, key=lambda r: (r['method'], r['bandwidth'] or 0, r['target_id'],
                                       r['scenario'], r['algorithm_seed']))


def pilot():
    tuning = run_batch([('source_local', b) for b in BANDWIDTHS])
    scores = {b: method_summary([r for r in tuning if r['bandwidth'] == b])['source_local']
              for b in BANDWIDTHS}
    best = max(BANDWIDTHS, key=lambda b: (
        scores[b]['collision_recall']['mean'] if scores[b]['collision_recall']['mean'] is not None else -1,
        scores[b]['unique_collisions']['mean'], scores[b]['discovery_auc']['mean'], b))
    write_json(OUTPUT / 'pilot/bandwidth.json', {'selected_bandwidth': best,
               'scores': scores, 'runs': tuning, 'extra_tuning_runs': 80,
               'discovery_contract': experiment_contract()})
    rows = run_batch([(m, None) for m in BASELINES] + [('shared_local', best)])
    rows.extend(r for r in tuning if r['bandwidth'] == best)
    gate = pilot_gate(rows)
    write_json(OUTPUT / 'pilot/run_index.json', {'status': 'complete', 'runs': rows,
               'main_runs': 440, 'extra_tuning_runs': 80, 'equivalent_queries': 520*128,
               'new_physical_calls': 0, 'selected_bandwidth': best,
               'discovery_contract': experiment_contract()})
    write_json(OUTPUT / 'pilot/gate.json', gate)
    report()
    print(json.dumps({'pilot_gate': gate['status'], 'conditions': gate['conditions'],
                      'strongest_baseline': gate['strongest_baseline']}, ensure_ascii=False), flush=True)
    return gate


def require_gate():
    gate = read_json(OUTPUT / 'pilot/gate.json')
    if gate['status'] != 'passed':
        raise ValueError('Stage B development gate did not pass; preserve evidence and report, do not expand')
    return gate


def selection():
    require_gate()
    bandwidth = read_json(OUTPUT / 'pilot/bandwidth.json')['selected_bandwidth']
    rows = run_batch([(m, bandwidth if 'local' in m else None)
                      for m in ('source_global_h5', 'source_global_full', 'source_local_h5', 'source_local_full')])
    pilot_rows = read_json(OUTPUT / 'pilot/run_index.json')['runs']
    stats = method_summary(rows + pilot_rows)
    decision = {}
    for prefix in ('source_global', 'source_local'):
        greedy = stats[prefix]
        eligible = [prefix]
        for method in (prefix+'_h5', prefix+'_full'):
            current = stats[method]
            if (current['discovery_auc']['mean'] > greedy['discovery_auc']['mean']
                    and current['unique_collisions']['mean'] >= greedy['unique_collisions']['mean']
                    and current['collision_recall']['mean'] >= greedy['collision_recall']['mean']):
                eligible.append(method)
        decision[prefix] = max(eligible, key=lambda m: (stats[m]['discovery_auc']['mean'],
                               stats[m]['collision_recall']['mean'], -stats[m]['selection_s']['mean']))
    result = {'status': 'complete', 'runs': rows, 'summary': stats, 'selected_methods': decision,
              'selected_bandwidth': bandwidth, 'discovery_contract': experiment_contract()}
    write_json(OUTPUT / 'selection/decision.json', result)
    report()
    return result


def report():
    diagnosis = read_json(OUTPUT / 'diagnosis/report.json')
    lines = ['# 历史响应驱动的碰撞发现：执行结果', '',
             f"阶段 A：{diagnosis['status']}；核对 {len(diagnosis['banks'])} 份完整历史/开发执行档案；新增物理调用 0。", '']
    if not (OUTPUT / 'pilot/gate.json').exists():
        lines.append('阶段 B 尚未完成，不能据阶段 A 的失效互补性判断方法收益。')
    else:
        gate = read_json(OUTPUT / 'pilot/gate.json')
        index = read_json(OUTPUT / 'pilot/run_index.json')
        lines.extend([f"阶段 B：{gate['status']}。主矩阵 440 项，额外带宽试验 80 项，共 66,560 次等效查询。",
                      f"选择带宽：{index['selected_bandwidth']}；最强开发基线：{gate['strongest_baseline']}。", '',
                      '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现 AUC |', '|---|---:|---:|---:|'])
        for method, stats in gate['summary'].items():
            recall = stats['collision_recall']['mean']
            label = f'{recall:.2%}' if recall is not None else '不适用'
            lines.append(f"| {method} | {stats['unique_collisions']['mean']:.3f} | {label} | {stats['discovery_auc']['mean']:.3f} |")
        lines.extend(['', '门槛结果：', ''])
        lines.extend(f"- {name}：{'通过' if value else '未通过'}。" for name, value in gate['conditions'].items())
        lines.extend(['', '数量与召回率在同一候选库内成比例；此处不将其作为两份独立证据。',
                      '谱系内先汇总，再对谱系等权；零分母任务的召回率不填值。配对分层区间见 pilot/gate.json。', ''])
        if gate['status'] != 'passed':
            lines.append('按预定门槛，本轮不进入阶段 C、D、E。该结果是开发证据，不是独立目标上的效果结论。')
        elif (OUTPUT / 'selection/decision.json').exists():
            lines.append('阶段 C 完成；选择器决策和实际计算开销见 selection/decision.json。阶段 D、E 须另经冻结与验证。')
        else:
            lines.append('阶段 B 通过；阶段 C 尚待执行，阶段 D、E 尚未执行。')
        plot_curves(index['runs'])
    lines.append('')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return lines


def plot_curves(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method in ('random', 'gp_collision', 'fsbo', 'pooled_ft', 'greedy', 'source_global', 'source_local'):
        chosen = [r for r in rows if r['method'] == method]
        for axis, metric in zip(axes, ('discovery_curve', 'recall_curve')):
            lineages = sorted({r['lineage'] for r in chosen})
            curves = []
            for lineage in lineages:
                values = [r[metric] for r in chosen if r['lineage'] == lineage and r[metric][0] is not None]
                if values:
                    curves.append(np.mean(values, axis=0))
            axis.plot(np.arange(1, 129), np.mean(curves, axis=0), label=method)
    axes[0].set(xlabel='Paid queries (including 8 initial)', ylabel='Unique collision scenarios')
    axes[1].set(xlabel='Paid queries (including 8 initial)', ylabel='Recall over complete finite pool', ylim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(OUTPUT / f'development_curves.{extension}', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('diagnose', 'pilot', 'selection', 'expand', 'freeze', 'evaluate', 'report'))
    stage = parser.parse_args().stage
    if stage == 'diagnose':
        diagnose()
    elif stage == 'pilot':
        pilot()
    elif stage == 'selection':
        selection()
    elif stage == 'report':
        report()
    else:
        require_gate()
        raise NotImplementedError(f'{stage} must be implemented and verified after the development gate passes')


if __name__ == '__main__':
    main()
