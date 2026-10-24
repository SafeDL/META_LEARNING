"""Re-derive the original admission gate's feasibility from raw development pools."""

import numpy as np

from .discovery import summary
from .discovery_config import OUTPUT, read_json, write_json
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key
from .schema import QueryFeedback


def certify():
    index = read_json(OUTPUT / 'pilot/run_index.json')
    baseline = [r for r in index['runs'] if r['method'] == 'fsbo']
    if len(baseline) != 40:
        raise ValueError('The frozen baseline must contain all 40 paired development runs')
    banks, cases = {}, []
    for row in baseline:
        key = row['target_id'], row['scenario']
        if key not in banks:
            bank = ExecutionBank(DATA_ROOT / 'physical/development' / row['scenario'] / row['target_id'])
            banks[key] = bank.candidates, [bank(s).feedback.to_dict() for s in bank.candidates]
        candidates, feedbacks = banks[key]
        payload = read_json(ROOT / row['path'])
        queries = payload['result']['queries']
        selected = [q['index'] for q in queries]
        initial = initial_indices(512, 8, row['algorithm_seed'])
        if len(selected) != 128 or len(set(selected)) != 128 or selected[:8] != initial:
            raise ValueError('Baseline budget or common initialization differs from the frozen task')
        found = set()
        for q in queries:
            if QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[q['index']]:
                raise ValueError('Baseline observation differs from its raw physical execution')
            if feedbacks[q['index']]['collision']:
                found.add(dedup_key(candidates[q['index']]))
        failures = {dedup_key(s) for s, f in zip(candidates, feedbacks) if f['collision']}
        initial_found = {dedup_key(candidates[i]) for i in initial if feedbacks[i]['collision']}
        oracle_count = min(len(failures), len(initial_found) + 128 - 8)
        cases.append({**{k: row[k] for k in ('target_id', 'lineage', 'scenario', 'algorithm_seed')},
                      'pool_unique_collisions': len(failures), 'baseline_collisions': len(found),
                      'initial_collisions': len(initial_found), 'oracle_collisions_bound': oracle_count,
                      'baseline_recall': len(found) / len(failures) if failures else None,
                      'oracle_recall_bound': oracle_count / len(failures) if failures else None})
    baseline_recall = summary(cases, 'baseline_recall')['mean']
    oracle_recall = summary(cases, 'oracle_recall_bound')['mean']
    required_recall = baseline_recall + 0.05
    feasible = required_recall <= oracle_recall + 1e-12
    reported = read_json(OUTPUT / 'pilot/gate.json')['summary']['fsbo']['collision_recall']['mean']
    np.testing.assert_allclose(baseline_recall, reported, atol=1e-15)
    result = {'status': 'feasible' if feasible else 'infeasible',
              'requirement_source': 'docs/collision_discovery_plan.md section 3.3, condition 1',
              'baseline': 'fsbo', 'raw_banks_checked': len(banks), 'raw_executions_checked': len(banks)*512,
              'paired_baseline_runs_checked': len(cases), 'zero_failure_runs': sum(
                  r['pool_unique_collisions'] == 0 for r in cases),
              'baseline_recall': baseline_recall, 'oracle_recall_upper_bound': oracle_recall,
              'required_absolute_gain': 0.05, 'required_recall': required_recall,
              'maximum_possible_gain': oracle_recall-baseline_recall,
              'shortfall_even_with_oracle': max(0.0, required_recall-oracle_recall),
              'maximum_collision_count_mean': summary(cases, 'oracle_collisions_bound')['mean'],
              'new_physical_calls': 0, 'cases': cases,
              'proof': 'Each discovered failure set is a subset of the full failure set. '
                       'Each later query discovers at most one group. Equal positive-weight averaging '
                       'over the same nonempty tasks preserves this bound. Required recall exceeds the oracle bound.',
              'unresolved_requirements': ['original_stage_B_admission', 'stage_C_selection_experiment',
                                          'stage_D_new_scenario_targets_and_freeze', 'stage_E_independent_validation'],
              'overall_objective_achieved': False,
              'recovery_requires': 'User clarification of the acceptance condition, query budget, or candidate pool; '
                                   'do not silently relax the original criterion or mark the failed gate passed'}
    write_json(OUTPUT / 'feasibility/report.json', result)
    lines = ['# 原开发门槛的可达性复核', '',
             '从8份原始开发执行库重新核对4,096次物理执行和40项FSBO配对运行，不重新运行驾驶系统。', '',
             f'- FSBO平均召回率：{baseline_recall:.6%}。',
             f'- 相同候选、初始化和128次预算下的理想算法召回率上界：{oracle_recall:.6%}。',
             f'- 原方案要求的召回率：{required_recall:.6%}。',
             f'- 最大可提升空间：{100*(oracle_recall-baseline_recall):.6f}个百分点，小于所需5个百分点。', '',
             '每项非空任务中，发现的失效集合包含于全库失效集合，因此召回率不超过1；相同任务掩码下的正权重平均仍不超过1。',
             '即使给算法全部未查询标签并允许理想选点，也无法满足原门槛。修改模型、增加训练或重复运行不能解除这一约束。', '',
             '该结论不证明当前原型已优于基线。当前最好原型仍低于FSBO，整体目标未完成。',
             '需要明确下一轮验收标准或实验条件。原B门槛保留未通过，C/D/E不越过门槛执行。', '']
    (OUTPUT / 'feasibility/report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({k: result[k] for k in ('status', 'baseline_recall', 'oracle_recall_upper_bound',
                                 'required_recall', 'maximum_possible_gain')}, flush=True)
    return result


if __name__ == '__main__':
    certify()
