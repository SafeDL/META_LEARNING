"""Audited budget curves, paired comparisons and costs for residual search."""

import hashlib
import json

import numpy as np

from . import discovery_physical_mixture as physical
from .collision_metrics import paired_lineage_bootstrap
from .discovery_config import read_json, write_json
from .discovery_physical_mixture_report import aggregate_curves, recall_text
from .discovery_residual_online import MAIN, METHODS, REPORT, STRONG, gate, previous_rows, protocol
from .discovery_residual_scale import ARTIFACTS, REPORT as LEARNING
from .protocol import ROOT


def report():
    index = read_json(REPORT/'run_index.json')
    audit = read_json(REPORT/'audit/report.json')
    negative = read_json(REPORT/'audit/negative_controls.json')
    if (index['protocol'] != protocol() or audit['status'] != 'passed' or audit['runs'] != 240
            or audit['independent_full_pool_choices'] != 720 or negative['status'] != 'passed'
            or negative['positive_controls'] != 6 or negative['rejected_corruptions'] < 8
            or len(audit['equivalent_parameter_pairs']) != 40):
        raise ValueError('Finish the complete residual matrix, independent audit and corruption controls before reporting')
    result = gate(index['runs'])
    if result != read_json(REPORT/'gate.json'):
        raise ValueError('Residual admission differs from its saved complete result')
    rows = index['runs']+previous_rows()
    curves = aggregate_curves(rows, result['summary'])
    checkpoints = {m: {str(b): {'unique_collisions': c['discovery_curve'][b-1], 'collision_recall': c['recall_curve'][b-1]}
                       for b in (16, 32, 64, 128)} for m, c in curves.items()}
    budget_references, intervals = {}, []
    for budget in (16, 32, 64, 128):
        reference = max(STRONG, key=lambda m: (
            checkpoints[m][str(budget)]['collision_recall'] if checkpoints[m][str(budget)]['collision_recall'] is not None else -1,
            checkpoints[m][str(budget)]['unique_collisions'], float(np.mean(curves[m]['discovery_curve'][:budget])), m))
        budget_references[str(budget)] = reference
        for metric, name in (('unique_collisions', 'discovery_curve'), ('collision_recall', 'recall_curve')):
            paired = [{**r, metric: r[name][budget-1]} for r in rows if r['method'] in (MAIN, reference) and r[name][budget-1] is not None]
            if paired:
                intervals.append({'budget': budget, **paired_lineage_bootstrap(paired, MAIN, reference, metric)})
    labels = {'residual_local': 'Main: learned residual scale + local reliability',
        'residual_global': 'Learned scale + global reliability', 'residual_unit_local': 'Unit scale + local reliability',
        'residual_unit_global': 'Unit scale + global reliability', 'residual_target': 'FSBO targets only',
        'residual_dual_target': 'Dual-teacher targets only'}
    for name in dict.fromkeys([result['strongest_baseline'], *budget_references.values()]):
        labels.setdefault(name, name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    figure.subplots_adjust(left=.07, right=.98, top=.95, bottom=.28, wspace=.22)
    colors = plt.get_cmap('tab20')
    for i, (method, label) in enumerate(labels.items()):
        for axis, name in zip(axes, ('discovery_curve', 'recall_curve')):
            if curves[method][name][0] is not None:
                axis.plot(range(1, 129), curves[method][name], label=label, color=colors(i),
                          linewidth=2.8 if method == MAIN else 1.5,
                          linestyle='-' if method in METHODS else '--')
    axes[0].set_ylabel('Unique collision scenarios')
    axes[1].set(ylabel='Full-pool collision recall', ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.set_xlabel('Paid queries (8 initial queries included)')
        axis.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, fontsize=8)
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(REPORT/f'curves.{extension}', dpi=180, bbox_inches='tight', pad_inches=.15)
    plt.close(figure)
    learning_audit = read_json(LEARNING/'selection_audit.json')
    costs = {'new_physical_calls': 0,
        'reused_teacher_and_physical_pool_costs': read_json(physical.REPORT/'costs.json')['reused_foundation_costs'],
        'historical_scale_learning': {
            'selected_beta': learning_audit['selected_beta_recomputed'],
            'historical_prior_construction_wall_s': sum(read_json(p)['wall_s'] for p in (ARTIFACTS/'priors').glob('*.json')),
            'adaptation_cache_sum_worker_wall_s': learning_audit['historical_cache_sum_worker_wall_s'],
            'parameter_fit_wall_s': learning_audit['parameter_fit_wall_s'],
            'new_selector_queries': 0, 'new_physical_calls': 0},
        'by_method': {}, 'failed_attempts': []}
    for method in METHODS:
        group = [read_json(ROOT/r['path']) for r in index['runs'] if r['method'] == method]
        costs['by_method'][method] = {'runs': 40, 'equivalent_queries': 5120, 'new_physical_calls': 0,
            'mean_selection_s': result['summary'][method]['selection_s']['mean'],
            'mean_adaptation_s': result['summary'][method]['adaptation_s']['mean'],
            'sum_run_wall_s': sum(r['cost']['wall_s'] for r in group),
            'sum_model_loading_s': sum(r['cost']['model_loading_s'] for r in group),
            'sum_source_construction_s': sum(r['cost']['source_construction_s'] for r in group),
            'mean_fsbo_adaptation_s': float(np.mean([sum(q['fsbo_adaptation_s'] for q in r['result']['queries']) for r in group])),
            'mean_joint_adaptation_s': float(np.mean([sum(q['joint_adaptation_s'] for q in r['result']['queries']) for r in group])),
            'mean_physical_adaptation_s': float(np.mean([sum(q['physical_adaptation_s'] for q in r['result']['queries']) for r in group])),
            'replay_equivalent_simulation_s': sum(q['feedback']['execution_s'] for r in group for q in r['query_ledger'])}
    for path in (REPORT/'failed_runs').rglob('*.jsonl'):
        attempts = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        costs['failed_attempts'].append({'path': str(path.relative_to(ROOT)), 'attempts': len(attempts),
            'paid_attempts': sum(a['paid_attempts'] for a in attempts), 'wall_s': sum(a['wall_s'] for a in attempts)})
    boundary = read_json(physical.REPORT/'curves.json')
    totals = {'prior_valid_search_runs': boundary['valid_total_search_runs'], 'new_valid_search_runs': 240,
        'valid_total_search_runs': boundary['valid_total_search_runs']+240,
        'valid_total_equivalent_queries': boundary['valid_total_equivalent_queries']+30720,
        'new_physical_calls': 0, 'failed_attempts_included_in_valid_totals': False}
    write_json(REPORT/'costs.json', costs)
    write_json(REPORT/'curves.json', {'curves': curves, 'checkpoints': checkpoints, 'summary': result['summary'],
        'strongest_reference_by_budget': budget_references, 'checkpoint_comparisons': intervals,
        'complete_matrix_audited': True, 'audit_sha256': hashlib.sha256((REPORT/'audit/report.json').read_bytes()).hexdigest(), **totals})
    lines = ['# 历史残差尺度：完整开发结果', '',
        f"开发准入：{result['status']}；主方法：{MAIN}；最强参考：{result['strongest_baseline']}。",
        '240项冻结运行已全部完成并独立复核；每项128次查询包括8次共同初始化。',
        '此前22种参考和两个目标控制共同参与最强参考选择，主配置不依据辅助版本结果更换。', '',
        '| 方法 | 平均真实碰撞数 | 平均召回率 | 发现AUC |', '|---|---:|---:|---:|']
    for method, s in result['summary'].items():
        lines.append(f"| {method} | {s['unique_collisions']['mean']:.3f} | {recall_text(s['collision_recall']['mean'])} | {s['discovery_auc']['mean']:.3f} |")
    lines += ['', '![完整发现曲线](curves.png)', '', '| 方法 | 查询预算 | 平均碰撞数 | 平均召回率 |', '|---|---:|---:|---:|']
    for method in labels:
        for budget, m in checkpoints[method].items():
            lines.append(f"| {labels[method]} | {budget} | {m['unique_collisions']:.3f} | {recall_text(m['collision_recall'])} |")
    lines += ['', '先在谱系内汇总再等权平均；零碰撞库保留数量及成本、召回不适用。',
        '数量与召回不视为两份独立证据。每个预算点对其最强预登记参考另算配对分层区间，见curves.json；终点机制区间和门槛见gate.json。',
        '独立审计逐项核对完整碰撞分母、全部查询和720次完整候选选点，并检查40对相同参数的切出曲线一致。',
        '双教师控制取得相同历史训练信息，只对已付费碰撞适配；不把额外表示或历史间距辅助训练单独归因为来源后验收益。',
        '尺度由历史内部学习，基础教师曾用全部历史对象，不能将其称为严格留系统验证。尺度学习不作为单独原创贡献。',
        'costs.json分开记录历史建设及基础训练、历史尺度学习、模型加载、来源构建、在线选例和三类适配。工作进程耗时合计不解释为并行经过时间。',
        '失败尝试另列，开发证据不替代新交互任务、独立目标或严格留家族结果，有限池召回不外推连续空间。',
        f"累计{totals['valid_total_search_runs']:,}项有效搜索、{totals['valid_total_equivalent_queries']:,}次等效查询；新增物理调用为零。"]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    report()
