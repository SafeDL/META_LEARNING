"""Independently review saved paid-prefix arrays and report all ranking controls."""

from collections import defaultdict
import hashlib
import json

import numpy as np
from scipy.special import logsumexp

from . import discovery_converged_search as study
from .discovery_config import development_tasks, read_json, write_json
from .executions import ExecutionBank
from .protocol import DATA_ROOT, ROOT
from .scenarios import features


REPORT = study.REPORT / 'diagnosis'
VARIANTS = ('original', 'prior_only', 'recent32', 'target_floor25',
            'physical_only', 'target_and_physical')


def run():
    inputs, predictions = read_json(REPORT / 'sets.json'), read_json(REPORT / 'prefix.json')
    assert inputs['main_runs'] == predictions['runs'] == 40 and predictions['prefixes'] == 120
    assert predictions['counterfactual_rankings'] == list(VARIANTS)
    assert read_json(study.REPORT / 'protocol.json') == study.protocol()
    expected = {(t['target_id'], t['scenario'], t['algorithm_seed']) for t in development_tasks()}
    assert {(r['target_id'], r['scenario'], r['algorithm_seed']) for r in predictions['records']} == expected
    sets_by_task = {(r['target_id'], r['scenario'], r['algorithm_seed']): r for r in inputs['records']}
    grouped, component_groups = defaultdict(list), defaultdict(list)
    checked_prefixes, checked_records = 0, 0
    maximum_score_error = maximum_weight_error = 0.
    own_input_source = ROOT / 'benchmark/discovery_converged_diagnosis.py'
    source_digest = hashlib.sha256(own_input_source.read_bytes()).hexdigest()
    for row in predictions['records']:
        key = row['target_id'], row['scenario'], row['algorithm_seed']
        path = REPORT / 'prefixes' / row['target_id'] / row['scenario'] / f"{row['algorithm_seed']}.npz"
        original = read_json(ROOT / sets_by_task[key]['main_path'])
        x = features(ExecutionBank(DATA_ROOT / 'physical/development' / row['scenario'] / row['target_id']).candidates)
        prior = np.asarray(original['result']['group_prior_weights'])
        case_lookup = {c['group']: c for c in sets_by_task[key]['cases']}
        with np.load(path, allow_pickle=False) as saved:
            identity = json.loads(str(saved['identity']))
            assert identity['source'] == source_digest
            assert identity['run'] == hashlib.sha256((ROOT / sets_by_task[key]['main_path']).read_bytes()).hexdigest()
            assert identity['sets'] == hashlib.sha256((REPORT / 'sets.json').read_bytes()).hexdigest()
            assert identity['protocol'] == hashlib.sha256((study.REPORT / 'protocol.json').read_bytes()).hexdigest()
            assert json.loads(str(saved['summary'])) == row
            for query in (9, 65, 128):
                arrays = {name: saved[f'q{query}_{name}'] for name in
                          ('scores', 'components', 'weights', 'recent_weights', 'floor_weights', 'paid', 'remaining')}
                q = original['result']['queries'][query - 1]
                expected_paid = [p['index'] for p in original['result']['queries'][:query - 1]]
                np.testing.assert_array_equal(arrays['paid'], expected_paid)
                assert set(arrays['remaining']) == set(range(512)) - set(expected_paid)

                # Independent sequential accumulation, rather than the saved
                # matrix multiplication, and never rescore any fitted label.
                all_score = np.log(prior)[:, None] + np.zeros((len(prior), 512))
                recent_score = all_score.copy()
                for n, past in enumerate(original['result']['queries'][:query - 1]):
                    influence = np.exp(-np.square(x - x[past['index']]).mean(1) / (2 * .3**2))
                    increment = np.asarray(past['prequery_group_log_likelihood'])[:, None] * influence
                    all_score += increment
                    all_score -= logsumexp(all_score, axis=0)
                    if n >= max(0, query - 1 - 32):
                        recent_score += increment
                        recent_score -= logsumexp(recent_score, axis=0)
                weights = np.exp(all_score)
                recent = np.exp(recent_score)
                floor = weights.copy()
                low = weights[-1] < .25
                floor[:-1, low] *= .75 / (1 - weights[-1, low])
                floor[-1, low] = .25
                for name, expected_weights in (('weights', weights), ('recent_weights', recent), ('floor_weights', floor)):
                    error = float(np.max(np.abs(expected_weights - arrays[name])))
                    maximum_weight_error = max(maximum_weight_error, error)
                    np.testing.assert_allclose(expected_weights, arrays[name], atol=2e-12)
                component, physical = arrays['components'][:-1], arrays['components'][-1]
                scores = np.stack((.9 * np.sum(weights * component, axis=0) + .1 * physical,
                                   .9 * np.sum(prior[:, None] * component, axis=0) + .1 * physical,
                                   .9 * np.sum(recent * component, axis=0) + .1 * physical,
                                   .9 * np.sum(floor * component, axis=0) + .1 * physical,
                                   physical, .9 * component[-1] + .1 * physical))
                maximum_score_error = max(maximum_score_error, float(np.max(np.abs(scores - arrays['scores']))))
                np.testing.assert_allclose(scores, arrays['scores'], atol=2e-12)
                np.testing.assert_allclose(scores[0, q['index']], q['prequery_probability'], atol=2e-7, rtol=2e-6)
                for record in (r for r in row['records'] if r['query'] == query):
                    case = case_lookup[record['group']]
                    positive = np.asarray(case['positive_indices'])
                    ranks = {name: 1 + int(np.count_nonzero(scores[i, arrays['remaining']]
                             > np.max(scores[i, positive]) + 1e-12)) for i, name in enumerate(VARIANTS)}
                    assert ranks == record['ranks']
                    grouped[(row['lineage'], row['scenario'], query, record['category'])].append(record)
                    component_ranks = [1 + int(np.count_nonzero(p[arrays['remaining']]
                                       > np.max(p[positive]) + 1e-12)) for p in arrays['components']]
                    component_groups[(row['lineage'], row['scenario'], query, record['category'])].append(
                        {'best_history_rank': min(component_ranks[:-2]),
                         'target_rank': component_ranks[-2], 'physical_rank': component_ranks[-1]})
                    checked_records += 1
                checked_prefixes += 1
    aggregates = []
    for key, records in sorted(grouped.items()):
        aggregates.append({'lineage': key[0], 'scenario': key[1], 'query': key[2], 'category': key[3],
            'paired_case_records': len(records),
            'median_rank': {name: float(np.median([r['ranks'][name] for r in records])) for name in VARIANTS},
            'median_target_weight': float(np.median([r['target_weight'] for r in records])),
            'target_weights_below_quarter': sum(r['target_weight'] < .25 for r in records),
            'physical_above_actual_selected_count': sum(r['physical_ranks_missed_group_above_selected'] for r in records),
            'median_target_probability': float(np.median([r['target_probability'] for r in records])),
            'median_physical_probability': float(np.median([r['physical_probability'] for r in records])),
            'median_historical_max_probability': float(np.median([r['historical_max_probability'] for r in records]))})
    component_aggregates = [
        {'lineage': k[0], 'scenario': k[1], 'query': k[2], 'category': k[3],
         'paired_case_records': len(values),
         **{name: float(np.median([v[name] for v in values])) for name in
            ('best_history_rank', 'target_rank', 'physical_rank')}}
        for k, values in sorted(component_groups.items())]
    result = {'status': 'complete_independently_reviewed_paid_prefix_diagnosis',
        'paired_runs': 40, 'prefixes_checked': checked_prefixes, 'paired_case_prefix_records_checked': checked_records,
        'rankings': list(VARIANTS), 'maximum_weight_error': maximum_weight_error,
        'maximum_score_error': maximum_score_error, 'runwise_partition_counts': inputs['runwise_totals'],
        'by_task_partition': inputs['by_task'], 'distinct_groups_by_task_category': inputs['distinct_groups_by_task_category'],
        'aggregates': aggregates, 'component_rank_aggregates': component_aggregates,
        'best_history_rank_is_casewise_oracle_envelope': True,
        'causal_checks': read_json(REPORT / 'validation.json'),
        'costs': read_json(REPORT / 'progress.json'), 'counterfactuals_are_online_results': False,
        'new_selector_queries': 0, 'new_physical_calls': 0, 'frozen_search_changed': False}
    assert checked_prefixes == 120 and checked_records == 78
    write_json(REPORT / 'report.json', result)
    lines = ['# 收敛来源主方法的已付费状态诊断', '',
        '局部主方法40项已完整审计并未达召回准入下界。本报告复核与固定已完成参考的碰撞集合，'
        '在查询9/65/128之前的原付费支持集上重建120个状态；仅改变排序权重，不执行新查询。', '',
        '配对运行累计：共同发现1,499，主方法独有5，参考独有16，共同遗漏10。主方法独有记录来自一个IDM切入参数组；'
        '共同遗漏来自IDM切出和PPO切出各一个参数组。跨种子重复不是独立参数案例，各分类的跨种子并集还可能重叠。', '',
        '| 谱系/场景 | 查询 | 类别 | 配对遗漏数 | 原模型 | 先验权重 | 近32次 | 目标权重下限0.25 | 物理独立 | 目标+物理 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in aggregates:
        ranks = row['median_rank']
        lines.append(f"| {row['lineage']}/{row['scenario']} | {row['query']} | {row['category']} | "
                     f"{row['paired_case_records']} | " + ' | '.join(f'{ranks[name]:.0f}' for name in VARIANTS) + ' |')
    lines += ['', '表中为遗漏组在当时剩余候选中的中位排名，越小越靠前；三种查询时刻的剩余候选数分别为504、448、385。'
        '所有列共享同一付费支持集和分量概率，近32次只保留原查询前似然，目标权重下限保持历史分量之间的相对权重。', '',
        '目标分量权重在遗漏位置很低，但提高其下限或缩短可靠性记忆没有一致改善：查询128时，PPO切出的中位排名'
        '由32改善至19或11，IDM切出与MPC切入却变差。物理独立排序改善IDM切出，同时不改善其余三类遗漏。'
        '因此这些固定状态证据不足以支持直接扩大权重变体实验。', '',
        'MPC切入遗漏位置在查询128时的目标深核预测中位概率约0.0134、历史分量最大预测约0.0098；'
        '这两类响应均给出很低概率，但最优历史分量的中位排名为79，独立目标为268、物理GP为264。'
        '六种固定权重方案未恢复该遗漏的优先级；历史分量仍有相对排序信息，不能仅凭低绝对概率断言经验无用。'
        '按已知失效逐例挑选的最佳历史排名是诊断上界，不是可执行组合策略的排名。'
        '后续先检验响应几何或排序组合，不将固定状态排名改善解释为在线发现收益。', '',
        f'独立顺序复算检查40项、120个状态及{checked_records}条配对案例记录，权重最大差{maximum_weight_error:.3g}、'
        f'排序分数最大差{maximum_score_error:.3g}。三项因果权重检查通过，重建额外开销单列于`progress.json`；与在途选择器共享机器。', '',
        '当前160项冻结矩阵继续执行，旧主方法失败结论、准入标准和独立目标保持不变。']
    (REPORT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    run()
