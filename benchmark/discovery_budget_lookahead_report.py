"""Recompute all lookahead decisions and historical batch counts from saved arrays."""

from collections import defaultdict
import math

import numpy as np
from scipy.special import logsumexp

from . import discovery_budget_lookahead as study
from .discovery_config import read_json, write_json


def mean_lineages(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[row['lineage']].append(row[metric])
    return math.fsum(math.fsum(values) / len(values) for values in groups.values()) / len(groups)


def report():
    index = read_json(study.REPORT / 'all_states.json')
    assert index['status'] == 'complete' and index['protocol'] == study.protocol()
    states = index['states']
    expected = {(kind, a.source_id) for kind in ('cut_in', 'cut_out')
                for a in study.correction.history_inputs(kind)[0]}
    assert len(states) == 16 and {(s['scenario'], s['pseudo_target']) for s in states} == expected
    records, maximum_acquisition_error, maximum_probability_error, comparisons = [], 0., 0., []
    for state in states:
        assert state['protocol'] == index['protocol'] and state['status'] == 'complete'
        history = study.correction.history_inputs(state['scenario'])[0]
        archive = next(a for a in history if a.source_id == state['pseudo_target'])
        assert state['source_ids'] == [a.source_id for a in history if a.source_id != archive.source_id]
        path = study.REPORT / 'histories' / state['scenario'] / f"{state['pseudo_target']}.npz"
        with np.load(path, allow_pickle=False) as packed:
            saved = {key: packed[key] for key in packed.files}
            paid = np.random.default_rng(1301).permutation(512)[:64]
            available = np.asarray(sorted(set(range(512)) - set(paid)))
            np.testing.assert_array_equal(saved['paid'], paid)
            np.testing.assert_array_equal(saved['available'], available)
            assert saved['gradients'].shape == saved['iterations'].shape == (448, 2, 9)
            assert np.isfinite(saved['conditional']).all() and np.all((saved['conditional'] >= 0) & (saved['conditional'] <= 1))
            assert saved['gradients'].max() <= 1e-10 and state['virtual_component_modes'] == 8064
            identifiers = [s.scenario_id for s in archive.scenarios]
            for mode, name in enumerate(study.MODES):
                scores = saved[f'{name}_scores']
                weight = np.exp(scores - logsumexp(scores, axis=0))
                before = .9 * np.sum(weight * saved['before'][:-1], axis=0) + .1 * saved['before'][-1]
                maximum_probability_error = max(maximum_probability_error, float(np.max(np.abs(before - saved['probability'][mode]))))
                np.testing.assert_allclose(before, saved['probability'][mode], atol=2e-12)
                greedy = min((i for i in available if before[i] >= max(before[available]) - 1e-12),
                             key=lambda i: identifiers[i])
                for horizon in study.HORIZONS:
                    values = []
                    for position, candidate in enumerate(available):
                        future = available[available != candidate]
                        gains = [float(np.sort(saved['conditional'][mode, position, label, future])[-(horizon - 1):].sum())
                                 for label in (0, 1)]
                        values.append(float(before[candidate] + (1 - before[candidate]) * gains[0] + before[candidate] * gains[1]))
                    best = max(values)
                    lookahead = min((i for i, value in zip(available, values) if value >= best - 1e-12),
                                    key=lambda i: identifiers[i])
                    by_rule = {}
                    for rule, candidate in (('greedy', greedy), ('lookahead', lookahead)):
                        original = next(r for r in state['records'] if (r['reliability'], r['horizon'], r['rule']) == (name, horizon, rule))
                        assert original['candidate'] == candidate
                        position = int(np.flatnonzero(available == candidate)[0])
                        error = abs(values[position] - original['expected_batch_utility'])
                        maximum_acquisition_error = max(maximum_acquisition_error, error)
                        assert error < 1e-10
                        actual = bool(archive.collision[candidate])
                        assert original['actual_label'] == actual
                        conditional = saved['conditional'][mode, position, int(actual)]
                        future = sorted((i for i in available if i != candidate),
                                        key=lambda i: (-conditional[i], identifiers[i]))
                        hits = int(actual) + int(archive.collision[future[:horizon - 1]].sum())
                        assert hits == original['actual_batch_collisions']
                        records.append(original)
                        by_rule[rule] = hits
                    comparisons.append({'scenario': state['scenario'], 'lineage': state['lineage'],
                        'pseudo_target': state['pseudo_target'], 'reliability': name, 'horizon': horizon,
                        'actual_batch_gain': by_rule['lookahead'] - by_rule['greedy'],
                        'candidate_changed': bool(greedy != lookahead)})
    assert len(records) == 128 and len(comparisons) == 64
    summary, screening = [], []
    for horizon in study.HORIZONS:
        for mode in study.MODES:
            values = [r for r in comparisons if r['horizon'] == horizon and r['reliability'] == mode]
            scenes = {kind: mean_lineages([r for r in values if r['scenario'] == kind], 'actual_batch_gain')
                      for kind in ('cut_in', 'cut_out')}
            by = {lineage: mean_lineages([r for r in values if r['lineage'] == lineage], 'actual_batch_gain')
                  for lineage in sorted({r['lineage'] for r in values})}
            entry = {'horizon': horizon, 'reliability': mode, 'mean_actual_batch_gain': mean_lineages(values, 'actual_batch_gain'),
                'scene_gains': scenes, 'lineage_gains': by, 'candidate_changes': sum(r['candidate_changed'] for r in values)}
            summary.append(entry)
            if mode == 'local':
                screening.append({'horizon': horizon, 'passed': entry['mean_actual_batch_gain'] > 1e-12
                    and all(v >= -1e-12 for v in scenes.values()) and sum(v > 1e-12 for v in by.values()) >= 2})
    result = {'status': 'complete_independently_checked_historical_lookahead', 'historical_states': 16,
        'full_pool_virtual_component_modes': sum(s['virtual_component_modes'] for s in states),
        'maximum_gradient': max(s['maximum_virtual_gradient'] for s in states),
        'independent_branch_groups': sum(s['independent_branch_groups'] for s in states),
        'maximum_independent_probability_error': max(s['independent_maximum_probability_error'] for s in states),
        'maximum_array_probability_error': maximum_probability_error,
        'maximum_independent_acquisition_error': maximum_acquisition_error,
        'records': records, 'comparisons': comparisons, 'summary': summary, 'screening': screening,
        'further_comparison_indicated': any(r['passed'] for r in screening),
        'virtual_inference_worker_sum_s': sum(s['virtual_inference_s'] for s in states),
        'state_preparation_worker_sum_s': sum(s['state_preparation_s'] for s in states),
        'total_worker_sum_s': sum(s['wall_s'] for s in states), 'actual_states_unchanged': True,
        'new_development_queries': 0, 'new_physical_calls': 0, 'formal_stage_C_started': False,
        'online_gain_or_admission_claimed': False}
    write_json(study.REPORT / 'report.json', result)
    write_json(study.REPORT / 'summary_audit.json', {'status': 'passed', 'states': 16,
        'actual_batch_records': 128, 'acquisition_arrays_and_selected_candidates_checked': 64,
        'maximum_acquisition_error': maximum_acquisition_error, 'maximum_probability_error': maximum_probability_error,
        'all_virtual_modes_stationary': True, 'new_development_queries': 0, 'new_physical_calls': 0})
    lines = ['# 收敛后验上的全候选预算选例诊断', '',
        '16个历史64点状态，逐状态枚举448个未查询候选及两种虚拟标签，129,024个分量模式均满足收敛阈值。',
        '来源均值、协方差及已保存可靠性得分不变；虚拟分支仅固定核条件化，不重新适配网络。', '',
        '| 窗口 | 可靠性 | 切入实际批量差 | 切出实际批量差 | 谱系等权差 | 选点改变数 |', '|---:|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['horizon']} | {row['reliability']} | {row['scene_gains']['cut_in']:.6f} | "
            f"{row['scene_gains']['cut_out']:.6f} | {row['mean_actual_batch_gain']:.6f} | {row['candidate_changes']} |")
    lines += ['', f"预定后续比较线索：{'满足' if result['further_comparison_indicated'] else '不满足'}。",
        '实际批量潜力只评估选定点的真实历史标签及固定后续排序，不等同完整在线搜索、128次召回或AUC。',
        '教师使用全部历史对象，自身来源后验排除；不作为严格留系统或最终目标泛化证据。',
        '原阶段B及50%漏检减少门槛均未改写，未自动进入开发阶段C。', '']
    (study.REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({k: result[k] for k in ('status', 'further_comparison_indicated', 'summary', 'total_worker_sum_s')})
    return result


if __name__ == '__main__':
    report()
