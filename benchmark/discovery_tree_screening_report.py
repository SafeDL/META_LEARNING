"""Historical forest summary using native scalar types for JSON publication."""

from collections import defaultdict
import math

from . import discovery_tree_screening as study
from .discovery_config import read_json, write_json


def mean_by_lineage(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            groups[row['lineage']].append(row[metric])
    if not groups:
        return None
    return math.fsum(math.fsum(v) / len(v) for v in groups.values()) / len(groups)


def report():
    index = read_json(study.REPORT / 'run_index.json')
    checked = read_json(study.REPORT / 'audit/report.json')
    assert index['protocol'] == study.protocol() and checked['status'] == 'passed'
    assert checked['runs'] == 48 and checked['raw_feedbacks_checked'] == 6144
    assert checked['independent_full_pool_choices'] == 144 and checked['rejected_corruptions'] == 10
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
    result = {'status': 'complete_audited_historical_screening', 'summary': summary,
        'comparisons': comparisons, 'screening_passed': all(all(c['conditions'].values()) for c in comparisons),
        'historical_runs': 48, 'historical_replay_queries': 6144, 'new_development_queries': 0,
        'new_physical_calls': 0, 'development_admission_claimed': False}
    write_json(study.REPORT / 'report.json', result)
    lines = ['# 逐来源响应的历史树模型筛选', '',
        '完整48项历史隔离回放、6,144条反馈、144组独立全候选选点及10项损坏控制通过。',
        '每项排除自身来源，只使用其余7个历史森林，不调用旧深核教师或开发数据。', '',
        '| 输入 | 平均去重碰撞数 | 平均召回 | 发现曲线AUC |', '|---|---:|---:|---:|']
    for method, values in summary.items():
        lines.append(f"| {method} | {values['unique_collisions']:.6f} | {values['collision_recall']:.6%} | "
                     f"{values['discovery_auc']:.6f} |")
    lines += ['', f"预登记历史筛选：{'通过' if result['screening_passed'] else '未通过'}。",
        '只含一个配对种子；历史对象共用来源训练坐标，不代表新坐标泛化、开发准入或最终独立验证。',
        '树模型、逐来源特征与随机分区核均不单独宣称原创性。',
        '冻结选择器和全部查询保持原样。审计与原始测量按规范反馈对象比较，报告使用原生数值及布尔类型。',
        '首次审计和首次报告的序列化失败及恢复成本分别保留，不重跑历史选点。',
        '新增神经网络训练、开发选例及物理调用均为零；未自动进入C/D/E。', '']
    (study.REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    print(report())
