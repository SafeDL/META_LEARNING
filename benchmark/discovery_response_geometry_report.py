"""Review saved geometry diagnostics without changing their frozen computation."""

from collections import defaultdict
import json

import numpy as np

from . import discovery_converged_search as study
from .discovery_config import development_tasks, read_json, write_json


REPORT = study.REPORT / 'diagnosis/geometry'


def run():
    data, specification = read_json(REPORT / 'report.json'), read_json(REPORT / 'protocol.json')
    inputs = read_json(study.REPORT / 'diagnosis/sets.json')
    assert data['runs'] == 40 and data['prefixes'] == 120 and data['views'] == specification['views']
    expected = {(t['target_id'], t['scenario'], t['algorithm_seed']) for t in development_tasks()}
    assert {(r['target_id'], r['scenario'], r['algorithm_seed']) for r in data['records']} == expected
    cases = {(r['target_id'], r['scenario'], r['algorithm_seed']): r for r in inputs['records']}
    grouped, count = defaultdict(list), 0
    maximum_score_error = maximum_variance_error = 0.
    for row in data['records']:
        key = row['target_id'], row['scenario'], row['algorithm_seed']
        path = REPORT / 'prefixes' / row['target_id'] / row['scenario'] / f"{row['algorithm_seed']}.npz"
        old_path = study.REPORT / 'diagnosis/prefixes' / row['target_id'] / row['scenario'] / f"{row['algorithm_seed']}.npz"
        with np.load(path, allow_pickle=False) as saved, np.load(old_path, allow_pickle=False) as original:
            assert json.loads(str(saved['identity']))['protocol'] == specification
            assert json.loads(str(saved['summary'])) == row
            by_group = {c['group']: c for c in cases[key]['cases']}
            for query in (9, 65, 128):
                components, scores = saved[f'q{query}_components'], saved[f'q{query}_scores']
                assert components.shape == (4, 9, 512) and scores.shape == (4, 512)
                weights, physical = original[f'q{query}_weights'], original[f'q{query}_components'][-1]
                rebuilt = np.stack([.9 * np.sum(p * weights, axis=0) + .1 * physical for p in components])
                np.testing.assert_allclose(rebuilt, scores, atol=2e-12)
                maximum_score_error = max(maximum_score_error, float(np.max(np.abs(rebuilt - scores))))
                variance = saved[f'q{query}_kernel_mean_diagonal']
                np.testing.assert_allclose(variance, variance[0], atol=1e-12)
                maximum_variance_error = max(maximum_variance_error, float(np.max(np.abs(variance - variance[0]))))
                remaining = original[f'q{query}_remaining']
                for record in (r for r in row['records'] if r['query'] == query):
                    positive = np.asarray(by_group[record['group']]['positive_indices'])
                    ranks = {name: 1 + int(np.count_nonzero(scores[i, remaining]
                             > np.max(scores[i, positive]) + 1e-12)) for i, name in enumerate(data['views'])}
                    assert ranks == record['ranks']
                    grouped[(row['lineage'], row['scenario'], query, record['category'])].append(ranks)
                    count += 1
    aggregates = [{'lineage': k[0], 'scenario': k[1], 'query': k[2], 'category': k[3],
                   'paired_case_records': len(rows),
                   'median_rank': {name: float(np.median([r[name] for r in rows])) for name in data['views']}}
                  for k, rows in sorted(grouped.items())]
    assert aggregates == data['aggregates'] and count == 78
    checks = {'status': 'passed', 'runs': 40, 'prefixes': 120, 'case_prefix_records': count,
              'maximum_score_error': maximum_score_error, 'maximum_variance_error': maximum_variance_error,
              'counterfactuals_are_online_results': False, 'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(REPORT / 'summary_audit.json', checks)
    lines = ['# 固定支持集上的历史响应几何诊断', '',
        '四种核使用相同的付费支持集、均值、可靠性权重和平均边际方差。'
        '冻结编码核未作目标适配，组合核仅将适配核与冻结安全响应核各取一半。'
        '所有列均为原轨迹上固定状态的排序对照，不是新的在线发现结果。', '',
        '| 谱系/场景 | 查询 | 类别 | 配对遗漏数 | 当前适配核 | 冻结碰撞核 | 冻结安全响应核 | 各半安全响应组合 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['lineage']}/{row['scenario']} | {row['query']} | {row['category']} | "
                     f"{row['paired_case_records']} | " +
                     ' | '.join(f"{row['median_rank'][name]:.0f}" for name in data['views']) + ' |')
    lines += ['', '数值为遗漏参数组在当时剩余候选中的中位排名，越小越靠前。跨种子重复案例不作为独立参数样本。', '',
        '查询128时，冻结原碰撞核将IDM切出共同遗漏的排名由123改善至16、MPC切入遗漏由188改善至17；'
        'PPO共同遗漏却由32恶化至284。冻结安全响应核在相应三类上的排名为17、35、201，'
        '并未优于冻结碰撞核。因此本结果支持检验选择性保留与适配，不支持宣称安全响应核全面胜出。', '',
        '适配核和冻结核在不同任务上各有优势。下一项验证应先在历史任务上比较保留两种几何及其可靠性学习，'
        '而非依据这些开发遗漏挑选单一固定核。新模型仍须重新生成自己的因果预测记录，并通过完整在线比较；'
        '本诊断使用原可靠性权重，不替代该验证。', '',
        f"40项、120个状态及78条案例记录的数组复核通过；原核最大复现差"
        f"{data['maximum_original_score_error']:.3g}。核的正定性、方差与尺度控制有两项专门检查。", '',
        '完整配置见`protocol.json`，共享机器复现成本见`progress.json`；没有新增查询或物理调用。']
    (REPORT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return checks


if __name__ == '__main__':
    run()
