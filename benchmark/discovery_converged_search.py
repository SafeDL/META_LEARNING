"""Registered search comparison after correcting source-conditional mode inference."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
import traceback

from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.response import archive_fingerprints
from methods.gcrt.converged_search import MODES, select
from methods.gcrt.joint_training import load_model

from . import discovery_converged_response as historical
from . import discovery_physical_mixture as physical
from . import discovery_residual_online as residual
from . import discovery_scaml as scaml
from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import method_summary
from .discovery_config import OUTPUT, development_tasks, read_json, write_json
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated


REPORT = OUTPUT / 'converged_search'
RUNS = ROOT / 'results/gcrt/collision/discovery/converged_search/runs'
METHODS = MODES
MAIN = 'converged_local'
REFERENCES = (*scaml.REFERENCES, *scaml.METHODS)
STRONG = (*REFERENCES, 'converged_target', 'converged_dual_target')


def protocol():
    selected = read_json(historical.REPORT / 'history.json')['selected_beta']
    return {'purpose': 'source-conditional convergence correction with matched target controls',
        'main': MAIN, 'methods': list(METHODS), 'tasks': development_tasks(),
        'reference_methods': list(REFERENCES), 'strong_baselines': list(STRONG),
        'candidate_count': 512, 'budget': 128, 'initial_count': 8,
        'selected_beta': selected, 'local_bandwidth': .3, 'source_prior_mass': .75,
        'independent_target_prior_mass': .25, 'physical_mass': .1,
        'miss_reduction_threshold': .5, 'minimum_positive_lineages': 2,
        'target_adaptation': {'steps': 10, 'learning_rate': .003},
        'physical_adaptation': {'steps': 10, 'learning_rate': .01},
        'mode_gradient_tolerance': 1e-10, 'mode_maximum_iterations': 100,
        'conditional_inference_device': 'cpu',
        'kernel_and_adaptation_device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'feedback': 'paid collision Boolean only; no unqueried trajectories or labels',
        'selection': 'greedy probability; tie within 1e-12 by scenario_id',
        'historical_evidence': {p: hashlib.sha256((historical.REPORT / p).read_bytes()).hexdigest()
            for p in ('protocol.json', 'history.json', 'summary_audit.json', 'prefix.json', 'search_validation.json')},
        'source_files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
            'benchmark/discovery_converged_search.py', 'benchmark/discovery_converged_search_audit.py',
            'methods/gcrt/converged_response.py', 'methods/gcrt/converged_search.py',
            'methods/common/gp_laplace.py')},
        'foundation': scaml.protocol(),
        'unit_scale_control': 'selected beta=1 for both scenarios; reuse old residual_unit local/global',
        'original_failed_gates_changed': False, 'numerical_fix_claimed_as_originality': False}


def previous_rows():
    return scaml.previous_rows() + read_json(scaml.REPORT / 'run_index.json')['runs']


def strongest(stats, methods):
    return max(methods, key=lambda m: (stats[m]['collision_recall']['mean']
        if stats[m]['collision_recall']['mean'] is not None else -1,
        stats[m]['unique_collisions']['mean'], stats[m]['discovery_auc']['mean'], m))


def feasibility(rows):
    scaml.validate_rows(rows, REFERENCES)
    stats = method_summary(rows)
    reference = strongest(stats, REFERENCES)
    recall = stats[reference]['collision_recall']['mean']
    headroom = [k for k, v in stats[reference]['collision_recall']['by_lineage'].items()
                if v is not None and v < 1-1e-12]
    return {'reference': reference, 'reference_recall': recall,
        'required_recall': .5*(1+recall) if recall is not None else None,
        'lineages_with_recall_headroom': headroom,
        'attainable_necessary_conditions': recall is not None and recall < 1-1e-12 and len(headroom) >= 2,
        'scope': 'necessary arithmetic conditions; no success or statistical significance claim'}


def prepare():
    history = read_json(historical.REPORT / 'history.json')
    checked = read_json(historical.REPORT / 'summary_audit.json')
    validation = read_json(historical.REPORT / 'search_validation.json')
    if (history['status'] != 'complete_independently_checked_historical_correction'
            or history['protocol'] != historical.protocol() or checked['status'] != 'passed'
            or checked['history_records'] != 16 or checked['score_rows'] != 2400
            or checked['independent_probability_groups'] != 320
            or checked['legacy_score_matches'] != 800
            or checked['nonempty_mode_solves'] != 81920
            or not checked['all_final_gradients_at_most_1e_10']
            or checked['selected_beta_independently_recomputed'] != history['selected_beta']
            or history['selected_beta'] != {'cut_in': 1., 'cut_out': 1.}
            or validation['status'] != 'passed' or validation['dedicated_and_related_checks'] < 25
            or validation['full_regression_failures'] or validation['full_regression_errors']
            or not validation['includes_metadrive_regression']):
        raise ValueError('Complete the registered historical correction and independent summary audit first')
    specification = protocol()
    destination = REPORT / 'design.json'
    if destination.exists() and read_json(destination) != specification:
        raise ValueError('The converged search design was already registered differently')
    write_json(destination, specification)
    return specification


def freeze():
    index = read_json(scaml.REPORT / 'run_index.json')
    audit = read_json(scaml.REPORT / 'audit/report.json')
    report = read_json(scaml.REPORT / 'report.json')
    negative = read_json(scaml.REPORT / 'audit/negative_controls.json')
    if (index['status'] != 'complete' or index['protocol'] != scaml.protocol()
            or audit['status'] != 'passed' or audit['runs'] != 160
            or audit['independent_full_pool_choices'] != 480
            or report['status'] != 'complete_audited_comparison'
            or negative['status'] != 'passed' or negative['positive_controls'] != 4
            or negative['rejected_corruptions'] < 8):
        raise ValueError('Complete all ScaML conditions, independent audits and report before new search')
    prior = previous_rows()
    certificate = feasibility(prior)
    write_json(REPORT / 'feasibility.json', certificate)
    if not certificate['attainable_necessary_conditions']:
        raise ValueError('The registered gate lacks mathematical headroom; retain the certificate before requesting a decision')
    specification = protocol()
    if read_json(REPORT / 'design.json') != specification:
        raise ValueError('Converged runtime or historical evidence differs from its registered design')
    destination = REPORT / 'protocol.json'
    if destination.exists() and read_json(destination) != specification:
        raise ValueError('Converged search already frozen differently')
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
            raise ValueError('Source and development archive roles differ from the registered split')
        base, metadata = load_fsbo(physical.checkpoint_path(task['scenario'], 'physical_fsbo'))
        teacher, teacher_metadata = load_model(physical.checkpoint_path(task['scenario'], physical.MAIN))
        for construction in (metadata, teacher_metadata['specification']):
            if (construction['history_fingerprints'] != archive_fingerprints(history)
                    or construction['development_fingerprints'] != archive_fingerprints(development)):
                raise ValueError('Converged-search teachers differ from the registered construction data')
        loading_s = time.perf_counter()-started
        bank = ExecutionBank(DATA_ROOT / 'physical/development' / task['scenario'] / task['target_id'])
        setting = {'task': task, 'method': method, 'protocol': specification,
            'history_fingerprints': archive_fingerprints(history), 'bank_contract': bank.manifest['contract'],
            'candidate_ids': [s.scenario_id for s in bank.candidates]}
        destination = RUNS / method / task['target_id'] / task['scenario'] / f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != setting or output['result']['actual_queries'] != 128:
                raise ValueError('Saved converged-search run differs from the frozen comparison')
        else:
            started_source = time.perf_counter()
            sources = (None if method in METHODS[2:] else
                       residual.source_priors(task['scenario'], bank.candidates, history))
            construction_s = time.perf_counter()-started_source
            ledger = QueryLedger(bank.candidates, 128, bank)
            public = {'candidates': bank.candidates, 'pretrained': base, 'sources': sources,
                'auxiliary': teacher if method == 'converged_dual_target' else None,
                'mode': method, 'beta': specification['selected_beta'][task['scenario']],
                'initial_indices': initial_indices(512, 8, task['algorithm_seed']), 'budget': 128,
                'device': specification['kernel_and_adaptation_device']}
            try:
                result = run_isolated(select, public, ledger)
            except Exception:
                failed = REPORT / 'failed_runs' / method / task['target_id'] / task['scenario'] / f"{task['algorithm_seed']}.jsonl"
                failed.parent.mkdir(parents=True, exist_ok=True)
                with failed.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'specification': setting, 'status': 'failed',
                        'query_ledger': ledger.records, 'paid_attempts': len(ledger.records),
                        'error': traceback.format_exc(), 'wall_s': time.perf_counter()-started,
                        'new_physical_calls': 0}, ensure_ascii=False, allow_nan=False)+'\n')
                raise
            metrics = summarize(bank.candidates, result['queries'], ledger.trajectories)
            pool = next(b for b in read_json(OUTPUT / 'diagnosis/report.json')['banks']
                if b['source_id'] == task['target_id'] and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates, pool['pool_collisions'], result['queries']))
            output = {'specification': setting, 'result': result, 'metrics': metrics, 'query_ledger': ledger.records,
                'cost': {'equivalent_queries': 128, 'new_physical_calls': 0, 'model_loading_s': loading_s,
                    'source_construction_s': construction_s, 'wall_s': time.perf_counter()-started,
                    'new_historical_training': False}}
            write_json(destination, output)
    return {**task, 'method': method, 'path': str(destination.relative_to(ROOT)), **output['metrics']}


def gate(rows, baselines=None):
    scaml.validate_rows(rows, METHODS)
    baselines = previous_rows() if baselines is None else baselines
    scaml.validate_rows(baselines, REFERENCES)
    combined = rows+baselines
    scaml.validate_rows(combined, (*METHODS, *REFERENCES))
    stats = method_summary(combined)
    baseline = strongest(stats, STRONG)
    main, ref = stats[MAIN], stats[baseline]
    recall, reference = main['collision_recall']['mean'], ref['collision_recall']['mean']
    reduction = ((recall-reference)/(1-reference)
        if recall is not None and reference is not None and reference < 1-1e-12 else None)
    gains = {k: v-ref['collision_recall']['by_lineage'][k]
        for k, v in main['collision_recall']['by_lineage'].items()}

    def exceeds(control):
        other = stats[control]['collision_recall']['mean']
        return recall is not None and other is not None and recall > other+1e-12

    def count_not_lower(control):
        return main['unique_collisions']['mean'] >= stats[control]['unique_collisions']['mean']-1e-12

    conditions = {'remaining_misses_reduced_by_half': reduction is not None and reduction >= .5-1e-12,
        'collision_count_not_lower_than_reference': count_not_lower(baseline),
        'two_positive_lineages': sum(v > 1e-12 for v in gains.values()) >= 2,
        'local_recall_exceeds_global': exceeds('converged_global'),
        'local_count_not_lower_than_global': count_not_lower('converged_global'),
        'recall_exceeds_same_information_dual_target': exceeds('converged_dual_target'),
        'count_not_lower_than_dual_target': count_not_lower('converged_dual_target')}
    pairs = list(dict.fromkeys((MAIN, m) for m in (*STRONG, 'converged_global',
                                                   'residual_unit_local', 'residual_unit_global')))
    comparisons = [paired_lineage_bootstrap([r for r in combined if r['method'] in (a, b) and r[metric] is not None],
        a, b, metric) for a, b in pairs for metric in ('unique_collisions', 'collision_recall', 'discovery_auc')
        if any(r['method'] in (a, b) and r[metric] is not None for r in combined)]
    return {'status': 'passed' if all(conditions.values()) else 'not_passed', 'main': MAIN,
        'strongest_baseline': baseline, 'remaining_miss_reduction': reduction,
        'required_recall': .5*(1+reference) if reference is not None else None,
        'conditions': conditions, 'lineage_recall_gains': gains, 'summary': stats,
        'comparisons': comparisons, 'previous_failed_gates_changed': False,
        'numerical_correction_claimed_as_originality': False, 'new_physical_calls': 0}


def smoke():
    specification = read_json(REPORT / 'protocol.json')
    if specification != protocol():
        raise ValueError('Converged-search runtime changed after freezing')
    rows = [execute(development_tasks()[0], m, specification) for m in METHODS]
    write_json(REPORT / 'smoke.json', {'status': 'complete', 'protocol': specification,
        'runs': rows, 'part_of_full_matrix': True, 'equivalent_queries': 512, 'new_physical_calls': 0})


def run():
    specification = read_json(REPORT / 'protocol.json')
    checked = read_json(REPORT / 'audit/smoke.json')
    negative = read_json(REPORT / 'audit/negative_controls.json')
    if (specification != protocol() or checked['status'] != 'passed' or checked['runs'] != 4
            or checked['independent_full_pool_choices'] != 12 or negative['status'] != 'passed'
            or negative['positive_controls'] != 4 or negative['rejected_corruptions'] < 10):
        raise ValueError('Check every actual smoke method and reject corruptions before the full matrix')
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, t, m, specification) for m in METHODS for t in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('converged_search', len(rows), '/ 160', flush=True)
    scaml.validate_rows(rows, METHODS)
    rows.sort(key=lambda r: (r['method'], r['target_id'], r['scenario'], r['algorithm_seed']))
    write_json(REPORT / 'run_index.json', {'status': 'complete', 'protocol': specification,
        'runs': rows, 'equivalent_queries': 20480, 'new_physical_calls': 0})
    write_json(REPORT / 'gate.json', gate(rows))


def report():
    from .discovery_physical_mixture_report import aggregate_curves

    index = read_json(REPORT / 'run_index.json')
    audit = read_json(REPORT / 'audit/report.json')
    negative = read_json(REPORT / 'audit/negative_controls.json')
    if (index['status'] != 'complete' or index['protocol'] != protocol()
            or audit['status'] != 'passed' or audit['runs'] != 160
            or audit['independent_full_pool_choices'] != 480
            or negative['status'] != 'passed' or negative['positive_controls'] != 4
            or negative['rejected_corruptions'] < 10):
        raise ValueError('Complete all 160 converged-search conditions and their audit before reporting')
    result = gate(index['runs'])
    if result != read_json(REPORT / 'gate.json'):
        raise ValueError('Final admission differs from the saved frozen-matrix result')
    rows = index['runs']+previous_rows()
    curves = aggregate_curves(rows, result['summary'])
    costs = {}
    for method in METHODS:
        outputs = [read_json(ROOT / r['path']) for r in index['runs'] if r['method'] == method]
        infos = [i for o in outputs for q in o['result']['queries']
                 for i in (*q['conditional_modes'], q['physical_mode'])]
        costs[method] = {'runs': len(outputs), 'equivalent_queries': len(outputs)*128,
            **{k: sum(o['cost'][k] for o in outputs) for k in ('wall_s', 'model_loading_s', 'source_construction_s')},
            **{k: sum(q[k] for o in outputs for q in o['result']['queries'])
               for k in ('selection_s', 'fsbo_adaptation_s', 'joint_adaptation_s', 'physical_adaptation_s')},
            'mode_solves_including_empty': len(infos), 'maximum_final_gradient': max(i['gradient_max'] for i in infos),
            'maximum_mode_iterations': max(i['iterations'] for i in infos),
            'total_backtracks': sum(i['backtracks'] for i in infos),
            'replay_equivalent_simulation_s': sum(q['feedback']['execution_s'] for o in outputs for q in o['query_ledger']),
            'new_physical_calls': 0}
    failures = [json.loads(line) for p in (REPORT / 'failed_runs').rglob('*.jsonl')
                for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    result.update(status='complete_audited_search', admission=read_json(REPORT / 'gate.json')['status'],
        curves=curves, checkpoints={m: {str(b): {'unique_collisions': c['discovery_curve'][b-1],
            'collision_recall': c['recall_curve'][b-1]} for b in (16, 32, 64, 128)} for m, c in curves.items()},
        costs=costs, failed_attempts=failures, formal_phases_started=False,
        historical_correction=read_json(historical.REPORT / 'summary_audit.json'),
        reused_foundation_costs=read_json(scaml.REPORT / 'report.json')['reused_foundation_costs'])
    write_json(REPORT / 'report.json', result)
    labels = list(dict.fromkeys((*METHODS, result['strongest_baseline'], 'residual_unit_local', 'residual_unit_global')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(left=.07, right=.98, top=.95, bottom=.25, wspace=.22)
    for m in labels:
        for ax, key in zip(axes, ('discovery_curve', 'recall_curve')):
            ax.plot(range(1, 129), curves[m][key], label=m, linewidth=2 if m == MAIN else 1.4,
                    linestyle='-' if m in METHODS else '--')
    axes[0].set_ylabel('Unique collision scenarios')
    axes[1].set(ylabel='Full-pool collision recall', ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel('Paid queries (8 initial queries included)')
        ax.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, fontsize=8)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(REPORT / f'curves.{extension}', dpi=180, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    lines = ['# 收敛来源推断的碰撞发现比较', '',
        f"完整160项开发运行通过独立审计。主方法门槛：`{result['admission']}`；最强参考：`{result['strongest_baseline']}`。", '',
        '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现曲线AUC |', '|---|---:|---:|---:|']
    for method in labels:
        values = result['summary'][method]
        r = values['collision_recall']['mean']
        recall = f'{r:.6%}' if r is not None else '不适用'
        lines.append(f"| {method} | {values['unique_collisions']['mean']:.6f} | {recall} | {values['discovery_auc']['mean']:.6f} |")
    lines += ['', '![完整发现曲线](curves.png)', '',
        '主方法在运行前指定为局部权重版。全局版、目标控制和旧失败结论完整保留；不按结果改换主方法。',
        '数值求解修正不单独作为创新。完整38种方法、预算点、配对分层区间、条件与成本见report.json。',
        '所有查询来自隔离执行档案回放；全库标签和召回分母只在评估端读取。未自动进入C/D/E。', '']
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'freeze', 'smoke', 'run', 'report'))
    globals()[parser.parse_args().stage]()
