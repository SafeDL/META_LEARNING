"""Evaluator-only paired discovery overlap; never a 128-query selector result."""

from .discovery import method_summary
from .discovery_config import OUTPUT, read_json, write_json
from .executions import ExecutionBank
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key


def diagnose():
    original = read_json(OUTPUT/'pilot/run_index.json')['runs']
    additional = (read_json(OUTPUT/'support/search/report.json')['runs'] +
                  read_json(OUTPUT/'recovery/run_index.json')['runs'] +
                  read_json(OUTPUT/'geometry_search/run_index.json')['runs'])
    baselines = [r for r in original if r['method'] == 'fsbo']
    methods = ('gp_collision','nearest_residual','history_response','source_global',
               'source_local','wide_source_global','wide_source_local',
               'response_fixed','geometry_response','kinematic_kernel')
    paired = {(r['target_id'],r['scenario'],r['algorithm_seed']):r for r in baselines}
    raw,found,records = {},{},[]
    for row in baselines + [r for r in original + additional if r['method'] in methods]:
        bank_key = row['target_id'],row['scenario']
        if bank_key not in raw:
            bank = ExecutionBank(DATA_ROOT/'physical/development'/row['scenario']/row['target_id'])
            labels = [bank(s).feedback.collision for s in bank.candidates]
            keys = [dedup_key(s) for s in bank.candidates]
            raw[bank_key] = keys,labels
        keys,labels = raw[bank_key]
        queries = read_json(ROOT/row['path'])['result']['queries']
        if len(queries) != 128 or len({q['index'] for q in queries}) != 128:
            raise ValueError('Complementarity requires two complete distinct-query experiments')
        if any(q['feedback']['collision'] != labels[q['index']] for q in queries):
            raise ValueError('Complementarity query differs from original collision feedback')
        found[(row['method'],row['target_id'],row['scenario'],row['algorithm_seed'])] = {
            keys[q['index']] for q in queries if labels[q['index']]}
    for row in original + additional:
        if row['method'] not in methods:
            continue
        task_key = row['target_id'],row['scenario'],row['algorithm_seed']
        a = found[(row['method'],*task_key)]
        b = found[('fsbo',*task_key)]
        denominator = paired[task_key]['pool_unique_collisions']
        if denominator != row['pool_unique_collisions']:
            raise ValueError('Complementary methods have different candidate pools')
        records.append({**{k:row[k] for k in ('target_id','scenario','lineage','family',
                       'condition','environment_seed','algorithm_seed')},
                       'method':f"fsbo_union_{row['method']}",'queries':256,
                       'unique_collisions':len(a|b),'collision_recall':len(a|b)/denominator if denominator else None,
                       'pool_unique_collisions':denominator,'fsbo_only':len(b-a),'other_only':len(a-b),
                       'both':len(a&b),'discovery_auc':0,'selection_s':0,'adaptation_s':0})
    stats = method_summary(records)
    result = {'status':'complete','records':records,'summary':stats,
              'cost':'two separate 128-query experiments, 256 paid queries in total',
              'attainable_at_128_claimed':False,'bound_on_adaptive_fusion_claimed':False,
              'uses_evaluator_only_labels':True,'raw_banks':len(raw),'new_physical_calls':0}
    destination = OUTPUT/'complementarity'
    write_json(destination/'report.json',result)
    lines = ['# 目标适配模型与已有预测器的发现互补性','',
             '每个并集来自两项独立128查询运行，总成本256次；它不是128次预算的可实现结果，也不是重新自适应融合的性能上界。',
             '所有四个开发系统、两类场景和五个种子均保留。此诊断只读评估端反馈，不向选择器传递并集或未查询标签。','',
             '| 与FSBO合并的预测器 | 平均碰撞并集数 | 平均并集召回率 |','|---|---:|---:|']
    for method,stat in stats.items():
        lines.append(f"| {method.removeprefix('fsbo_union_')} | {stat['unique_collisions']['mean']:.3f} | {stat['collision_recall']['mean']:.5%} |")
    lines += ['','逐任务的双向独有发现、交集和完整分母保存在report.json。任何融合方法仍须重新使用共同8次初始化、128次付费查询验证。','']
    (destination/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({method:{k:s[k]['mean'] for k in ('unique_collisions','collision_recall')}
           for method,s in stats.items()},flush=True)
    return result


if __name__ == '__main__':
    diagnose()
