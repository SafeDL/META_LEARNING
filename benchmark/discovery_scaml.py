"""Paired collision ScaML-GP comparison with matching source-free controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
import traceback

from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.baselines.scaml_gp.search import (KERNEL_BOUNDS, MODES, OPTIMIZER, SOURCE_MODES,
                                              TARGET_MODES, WEIGHT_FLOOR, select)
from methods.common.gp_laplace import MODE_ITERATIONS, MODE_TOLERANCE
from methods.common.response import archive_fingerprints
from methods.gcrt.joint_training import load_model

from . import discovery_rgpe as rgpe
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


REPORT = OUTPUT / 'scaml'
PREPARATION = ROOT / 'results/baselines/scaml_gp/collision/preparation'
RUNS = ROOT / 'results/baselines/scaml_gp/collision/discovery/runs'
METHODS = MODES
REFERENCES = (*rgpe.REFERENCES, *rgpe.METHODS)


def protocol():
    return {'purpose': 'ScaML classification comparator and same-inference target controls; no main promotion',
        'methods': list(METHODS), 'tasks': development_tasks(), 'candidate_count': 512,
        'budget': 128, 'initial_count': 8, 'reference_methods': list(REFERENCES),
        'physical_mass': {m: .1 if m.startswith('physical_') else 0. for m in METHODS},
        'target_adaptation': {'steps': 10, 'learning_rate': .003},
        'physical_adaptation': {'steps': 10, 'learning_rate': .01},
        'source_weight_prior': 'independent Gamma(1,1); target controls omit source terms entirely',
        'source_weight_floor': WEIGHT_FLOOR, 'kernel_log_bounds': list(KERNEL_BOUNDS),
        'scalar_optimizer': dict(OPTIMIZER), 'mode_gradient_tolerance': MODE_TOLERANCE,
        'mode_maximum_iterations': MODE_ITERATIONS, 'feedback': 'paid collision Boolean only',
        'target_representation': 'same reset FSBO adaptation, fixed during scalar kernel/weight fitting',
        'foundation': rgpe.protocol(), 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'source_files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
            'benchmark/discovery_scaml.py', 'methods/baselines/scaml_gp/search.py',
            'methods/common/gp_laplace.py')},
        'preparation': {p: hashlib.sha256((PREPARATION / p).read_bytes()).hexdigest()
                        for p in ('checks.json', 'historical_numeric.json', 'target_numeric.json',
                                  'numerical_correction.json')}}


def validate_rows(rows, methods=METHODS):
    expected = {(m, t['target_id'], t['scenario'], t['algorithm_seed'])
                for m in methods for t in development_tasks()}
    actual = [(r['method'], r['target_id'], r['scenario'], r['algorithm_seed']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Each ScaML comparison method needs all 40 paired tasks without duplicates')
    denominators = {}
    for row in rows:
        key = row['target_id'], row['scenario'], row['algorithm_seed']
        total, found = row['pool_unique_collisions'], row['unique_collisions']
        if (row['queries'] != 128 or denominators.setdefault(key, total) != total
                or not 0 <= found <= min(128, total)
                or row['collision_recall'] != (found / total if total else None)):
            raise ValueError('ScaML comparison differs in budget, deduplicated pool or recall')


def previous_rows():
    return rgpe.previous_rows() + read_json(rgpe.REPORT / 'run_index.json')['runs']


def prepare():
    checks = read_json(PREPARATION / 'checks.json')
    if (checks['status'] != 'passed' or checks['dedicated_unit_checks'] != 17
            or checks['failures'] or checks['errors']):
        raise ValueError('Pass all four ScaML modes and matched target-control checks before registration')
    expected = {(kind, a.source_id, count) for kind in ('cut_in', 'cut_out')
                for a in load_construction(kind)[0] for count in (8, 128)}
    for filename, source_count in (('historical_numeric.json', 7), ('target_numeric.json', 0)):
        result = read_json(PREPARATION / filename)
        actual = [(c['kind'], c['historical_pseudo_target'], c['support_count']) for c in result['checks']]
        if (result['status'] != 'passed' or len(actual) != 32 or set(actual) != expected
                or result['independent_gradient_vectors'] != 32
                or any(len(c['source_ids']) != source_count or c['historical_pseudo_target'] in c['source_ids']
                       or c['full_pool_size'] != 512
                       or source_count == 0 and c['source_reference'] is not None
                       for c in result['checks'])):
            raise ValueError('Complete every source and no-source historical numerical condition')
    specification = protocol()
    destination = REPORT / 'design.json'
    if destination.exists() and read_json(destination) != specification:
        raise ValueError('ScaML comparison design was already registered differently')
    write_json(destination, specification)
    return specification


def freeze():
    index = read_json(rgpe.REPORT / 'run_index.json')
    audit = read_json(rgpe.REPORT / 'audit/report.json')
    report = read_json(rgpe.REPORT / 'report.json')
    negative = read_json(rgpe.REPORT / 'audit/negative_controls.json')
    if (index['status'] != 'complete' or index['protocol'] != rgpe.protocol()
            or audit['status'] != 'passed' or audit['runs'] != 80
            or audit['independent_full_pool_choices'] != 240
            or report['status'] != 'complete_audited_comparison'
            or negative['status'] != 'passed' or negative['positive_controls'] != 2
            or negative['rejected_corruptions'] < 8):
        raise ValueError('Complete and independently audit/report the registered RGPE comparison first')
    validate_rows(previous_rows(), REFERENCES)
    specification = protocol()
    if read_json(REPORT / 'design.json') != specification:
        raise ValueError('ScaML runtime or preparation differs from its registered design')
    destination = REPORT / 'protocol.json'
    if destination.exists() and read_json(destination) != specification:
        raise ValueError('ScaML comparison was already frozen differently')
    write_json(destination, specification)
    return specification


def execute(task, method, specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        started = time.perf_counter()
        history, development = load_construction(task['scenario'])
        target = next(a for a in development if a.source_id == task['target_id'])
        if (target.role != 'development' or any(a.role != 'history' for a in history)
                or target.source_id in {a.source_id for a in history}):
            raise ValueError('ScaML construction mixes source and development roles')
        pretrained, metadata = load_fsbo(physical.checkpoint_path(task['scenario'], 'physical_fsbo'))
        if (metadata['history_fingerprints'] != archive_fingerprints(history)
                or metadata['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('ScaML representation differs from the allowed construction data')
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
                raise ValueError('Saved ScaML run differs from its frozen task')
        else:
            construction_s, sources, reference = 0., None, None
            if method in SOURCE_MODES:
                constructed = time.perf_counter()
                sources = residual.source_priors(task['scenario'], bank.candidates, history)
                teacher, teacher_metadata = load_model(physical.checkpoint_path(task['scenario'], physical.MAIN))
                expected = teacher_metadata['specification']
                if (expected['history_fingerprints'] != archive_fingerprints(history)
                        or expected['development_fingerprints'] != archive_fingerprints(development)):
                    raise ValueError('ScaML source teacher differs from the allowed construction data')
                reference = float(teacher.mean.detach())
                construction_s = time.perf_counter() - constructed
            ledger = QueryLedger(bank.candidates, 128, bank)
            public = {'candidates': bank.candidates, 'pretrained': pretrained, 'sources': sources,
                      'source_reference': reference, 'mode': method, 'budget': 128,
                      'initial_indices': initial_indices(512, 8, task['algorithm_seed']),
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
        raise ValueError('ScaML runtime changed after freezing')
    rows = [execute(development_tasks()[0], method, specification) for method in METHODS]
    write_json(REPORT / 'smoke.json', {'status': 'complete', 'protocol': specification, 'runs': rows})


def run():
    specification = read_json(REPORT / 'protocol.json')
    checked = read_json(REPORT / 'audit/smoke.json')
    negative = read_json(REPORT / 'audit/negative_controls.json')
    if (specification != protocol() or checked['status'] != 'passed' or checked['runs'] != 4
            or checked['independent_full_pool_choices'] != 12 or negative['status'] != 'passed'
            or negative['positive_controls'] != 4 or negative['rejected_corruptions'] < 8):
        raise ValueError('Check all four actual ScaML smoke cases and reject corrupted data before full running')
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, task, method, specification)
                   for method in METHODS for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('scaml_development', len(rows), '/ 160', flush=True)
    validate_rows(rows)
    write_json(REPORT / 'run_index.json', {'status': 'complete', 'protocol': specification,
               'runs': rows, 'equivalent_queries': 20480, 'new_physical_calls': 0})


def report():
    from .discovery_physical_mixture_report import aggregate_curves

    index = read_json(REPORT / 'run_index.json')
    audit = read_json(REPORT / 'audit/report.json')
    if (index['status'] != 'complete' or index['protocol'] != protocol()
            or audit['status'] != 'passed' or audit['runs'] != 160
            or audit['independent_full_pool_choices'] != 480):
        raise ValueError('Finish all 160 ScaML conditions and their independent audit before reporting')
    prior = previous_rows()
    rows = index['runs'] + prior
    validate_rows(rows, (*METHODS, *REFERENCES))
    stats = method_summary(rows)
    reference = max(REFERENCES, key=lambda m: (stats[m]['collision_recall']['mean'],
                    stats[m]['unique_collisions']['mean'], stats[m]['discovery_auc']['mean'], m))
    curves = aggregate_curves(rows, stats)
    pairs = [(m, reference) for m in METHODS] + list(zip(SOURCE_MODES, TARGET_MODES))
    comparisons = [paired_lineage_bootstrap([r for r in rows if r['method'] in (m, b) and r[metric] is not None],
                                           m, b, metric)
                   for m, b in pairs for metric in ('unique_collisions', 'collision_recall', 'discovery_auc')]
    costs = {}
    for method in METHODS:
        outputs = [read_json(ROOT / r['path']) for r in index['runs'] if r['method'] == method]
        costs[method] = {'runs': len(outputs), 'equivalent_queries': 128 * len(outputs), 'new_physical_calls': 0,
            **{k: sum(o['cost'][k] for o in outputs) for k in ('wall_s', 'model_loading_s', 'source_construction_s')},
            **{k: sum(q[k] for o in outputs for q in o['result']['queries'])
               for k in ('selection_s', 'prior_optimization_s', 'target_adaptation_s', 'physical_adaptation_s')},
            'replay_equivalent_simulation_s': sum(q['feedback']['execution_s'] for o in outputs for q in o['query_ledger']),
            'scalar_optimizations_not_converged': sum(not q['optimizer']['success'] for o in outputs for q in o['result']['queries'])}
    failures = [json.loads(line) for p in (REPORT / 'failed_runs').rglob('*.jsonl')
                for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {'status': 'complete_audited_comparison', 'strongest_previous_reference': reference,
              'summary': stats, 'comparisons': comparisons, 'curves': curves,
              'checkpoints': {m: {str(b): {'unique_collisions': c['discovery_curve'][b - 1],
                                         'collision_recall': c['recall_curve'][b - 1]}
                                    for b in (16, 32, 64, 128)} for m, c in curves.items()},
              'costs': costs, 'failed_attempts': failures,
              'preparation_checks': read_json(PREPARATION / 'checks.json'),
              'reused_foundation_costs': read_json(rgpe.REPORT / 'report.json')['reused_foundation_costs'],
              'main_method_promoted': False, 'prior_failed_gates_changed': False,
              'formal_phases_started': False, 'new_physical_calls': 0}
    write_json(REPORT / 'report.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = {'scaml_classifier': 'ScaML classification',
              'physical_scaml_classifier': 'ScaML classification + physical GP',
              'scaml_target': 'Same inference, target GP only',
              'physical_scaml_target': 'Same inference, target GP + physical GP'}
    labels.setdefault(reference, f'Previous reference: {reference}')
    labels.setdefault(residual.MAIN, 'Previous residual main')
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    figure.subplots_adjust(left=.07, right=.98, top=.95, bottom=.24, wspace=.22)
    colors = plt.get_cmap('tab10')
    for i, (method, label) in enumerate(labels.items()):
        for axis, name in zip(axes, ('discovery_curve', 'recall_curve')):
            axis.plot(range(1, 129), curves[method][name], label=label, color=colors(i),
                      linewidth=2 if method in METHODS else 1.5,
                      linestyle='-' if method in SOURCE_MODES else '--')
    axes[0].set_ylabel('Unique collision scenarios')
    axes[1].set(ylabel='Full-pool collision recall', ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.set_xlabel('Paid queries (8 initial queries included)')
        axis.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, fontsize=8)
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(REPORT / f'curves.{extension}', dpi=180, bbox_inches='tight', pad_inches=.15)
    plt.close(figure)
    lines = ['# ScaML-GP碰撞分类与同推断目标控制', '',
             f'完整160项、20,480次等效查询通过独立审计；最强既有参考为`{reference}`。', '',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现曲线AUC |', '|---|---:|---:|---:|']
    for method in (*METHODS, reference, residual.MAIN):
        values = stats[method]
        lines.append(f"| {method} | {values['unique_collisions']['mean']:.6f} | {values['collision_recall']['mean']:.6%} | {values['discovery_auc']['mean']:.6f} |")
    lines += ['', '![完整发现曲线](curves.png)', '',
              '所有新模式共用FSBO表示适配、收敛Laplace模式及标量核优化。无来源控制不接收显式来源后验和来源均值。',
              '原论文为连续响应高斯过程；本项目改编使用Bernoulli分类、均值中心化及固定候选的碰撞概率贪心。',
              '新旧数值推断差异不单独归因为来源机制。完整预算点、配对分层区间和分项成本见report.json。',
              '本比较不指定新主方法，不改既有失败结论，也不自动进入C/D/E。', '']
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'freeze', 'smoke', 'run', 'report'))
    globals()[parser.parse_args().stage]()
