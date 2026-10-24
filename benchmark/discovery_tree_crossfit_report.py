"""Publish the complete coordinate-cross-fitted historical forest comparison."""

from . import discovery_tree_crossfit as study
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_tree_screening_report import mean_by_lineage


def report():
    index = read_json(study.REPORT / 'run_index.json')
    checked = read_json(study.REPORT / 'audit/report.json')
    source_check = read_json(study.REPORT / 'audit/sources.json')
    assert index['protocol'] == study.protocol() and checked['status'] == source_check['status'] == 'passed'
    assert checked['runs'] == 48 and checked['raw_feedbacks_checked'] == 6144
    assert checked['independent_full_pool_choices'] == 144 and checked['rejected_corruptions'] == 10
    assert source_check['source_fold_fits'] == 80 and source_check['source_coordinate_exclusion_checked']
    rows = index['runs']
    summary = {method: {metric: mean_by_lineage([r for r in rows if r['method'] == method], metric)
        for metric in ('unique_collisions', 'collision_recall', 'discovery_auc')} for method in study.METHODS}
    comparisons = []
    for control in ('physical', 'pooled_response'):
        gains = {lineage: mean_by_lineage([r for r in rows if r['lineage'] == lineage and r['method'] == 'source_response'],
            'collision_recall') - mean_by_lineage([r for r in rows if r['lineage'] == lineage and r['method'] == control],
            'collision_recall') for lineage in sorted({r['lineage'] for r in rows})}
        a, b = summary['source_response'], summary[control]
        conditions = {'recall_higher': a['collision_recall'] > b['collision_recall'] + 1e-12,
            'count_not_lower': a['unique_collisions'] >= b['unique_collisions'] - 1e-12,
            'auc_not_lower': a['discovery_auc'] >= b['discovery_auc'] - 1e-12,
            'two_positive_lineages': sum(v > 1e-12 for v in gains.values()) >= 2}
        comparisons.append({'control': control, 'conditions': conditions, 'lineage_recall_gains': gains})
    curves, costs = {}, {}
    for method in study.METHODS:
        group = [r for r in rows if r['method'] == method]
        curves[method] = {key: [mean_by_lineage([{'lineage': r['lineage'], 'value': r[key][i]}
            for r in group], 'value') for i in range(128)] for key in ('discovery_curve', 'recall_curve')}
        outputs = [read_json(OUTPUT / r['path']) for r in group]
        costs[method] = {'historical_runs': len(outputs), 'historical_replay_queries': len(outputs) * 128,
            'sum_run_wall_s': sum(o['wall_s'] for o in outputs),
            'target_fitting_s': sum(q['adaptation_s'] for o in outputs for q in o['result']['queries']),
            'selection_s': sum(q['selection_s'] for o in outputs for q in o['result']['queries']),
            'replay_equivalent_physical_s': sum(q['feedback']['execution_s'] for o in outputs for q in o['result']['queries'])}
    old = read_json(study.original.REPORT / 'report.json')['summary']
    result = {'status': 'complete_audited_crossfitted_historical_screening', 'summary': summary,
        'comparisons': comparisons, 'screening_passed': all(all(c['conditions'].values()) for c in comparisons),
        'original_screening_differences': {m: {metric: summary[m][metric] - old[m][metric]
            for metric in summary[m]} for m in study.METHODS}, 'curves': curves, 'method_costs': costs,
        'source_preparation': read_json(study.REPORT / 'preparation.json'),
        'source_audit_wall_s': source_check['wall_s'], 'choice_audit_wall_s': checked['wall_s'],
        'historical_runs': 48, 'historical_replay_queries': 6144, 'new_development_queries': 0,
        'new_physical_calls': 0, 'development_admission_claimed': False}
    write_json(study.REPORT / 'report.json', result)
    lines = ['# 来源坐标交叉拟合的历史树模型筛选', '',
        '每个来源预测排除其整个坐标折，自身来源也排除。80次来源重建、48项隔离回放、'
        '6,144条反馈、144次全候选选点及10项损坏控制通过。', '',
        '| 输入 | 平均去重碰撞数 | 平均召回 | 发现曲线AUC |', '|---|---:|---:|---:|']
    for method, values in summary.items():
        lines.append(f"| {method} | {values['unique_collisions']:.6f} | {values['collision_recall']:.6%} | "
                     f"{values['discovery_auc']:.6f} |")
    lines += ['', f"预登记历史筛选：{'通过' if result['screening_passed'] else '未通过'}。",
        '三个版本都重新执行，未复用坐标重合矩阵的物理控制；全部配置及原失败结论保持。',
        '这项历史单种子检查不代表开发门槛、严格留家族或最终独立对象验证。',
        '新增神经网络训练、开发选例及物理调用均为零；未自动进入C/D/E。', '',
        '![完整历史发现曲线](curves.png)', '']
    (study.REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for method, value in curves.items():
        for ax, key in zip(axes, ('discovery_curve', 'recall_curve')):
            ax.plot(range(1, 129), value[key], label=method)
    axes[0].set_ylabel('Unique collision scenarios')
    axes[1].set(ylabel='Full-pool collision recall', ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel('Historical replay queries (8 initial queries included)')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(study.REPORT / f'curves.{extension}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return result


if __name__ == '__main__':
    result = report()
    print({k: result[k] for k in ('status', 'screening_passed', 'summary')})
