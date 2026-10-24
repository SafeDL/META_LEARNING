"""Frozen 128-query development experiment after the original infeasible gate."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import time

from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.response import archive_fingerprints
from methods.gcrt.response_adaptation import SURROGATE_CONFIG, response_coordinates, select

from .build_collision_models import model_path
from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import experiment_contract, method_summary
from .discovery_config import (OUTPUT, STRONG_BASELINES, development_tasks,
                               read_json, write_json)
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features


REPORT = OUTPUT / 'recovery'
RUNS = ROOT / 'results/gcrt/collision/discovery/recovery/runs'
METHODS = ('response_adaptive', 'response_fixed')
MISS_REDUCTION = 0.5


def protocol():
    return {'budget':128, 'initial_count':8, 'candidates':512,
            'miss_reduction_threshold':MISS_REDUCTION,
            'user_decision':'保留128次预算；采用漏检减少比例；保留首轮失败结论',
            'original_gate_preserved':True, 'tasks':development_tasks(),
            'methods':list(METHODS), 'strong_baselines':list(STRONG_BASELINES),
            'surrogate_config':SURROGATE_CONFIG, 'adaptation_steps':10,
            'adaptation_rate':0.003, 'baseline_contract':experiment_contract(),
            'source_files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                            for p in ('methods/gcrt/response_adaptation.py',
                                      'benchmark/discovery_recovery.py')},
            'sklearn_version':version('scikit-learn')}


def freeze():
    path = REPORT / 'protocol.json'
    specification = protocol()
    if path.exists():
        if read_json(path) != specification:
            raise ValueError('Recovery protocol already frozen with different settings')
    else:
        write_json(path, specification)
    print('recovery_protocol_frozen', flush=True)
    return specification


def execute(task, method, specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, _ = load_construction(task['scenario'])
        checkpoint = model_path('fsbo', 'all', task['scenario'])
        pretrained, metadata = load_fsbo(checkpoint)
        if metadata['history_fingerprints'] != archive_fingerprints(history):
            raise ValueError('Recovery initialization does not match allowed historical sources')
        bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
        setting = {'task':task, 'method':method, 'protocol':specification,
                   'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'history_fingerprints':archive_fingerprints(history),
                   'bank_contract':bank.manifest['contract'],
                   'candidate_ids':[s.scenario_id for s in bank.candidates]}
        destination = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != setting or output['result']['actual_queries'] != 128:
                raise ValueError('Saved recovery run differs from the frozen specification')
        else:
            started = time.perf_counter()
            responses = response_coordinates(history, features(bank.candidates))
            construction_s = time.perf_counter() - started
            ledger = QueryLedger(bank.candidates, 128, bank)
            result = run_isolated(select, {
                'candidates':bank.candidates, 'initial_indices':initial_indices(512,8,task['algorithm_seed']),
                'pretrained':pretrained, 'responses':responses,
                'adaptive':method == 'response_adaptive', 'budget':128}, ledger)
            metrics = summarize(bank.candidates, result['queries'], ledger.trajectories)
            diagnosis = read_json(OUTPUT/'diagnosis/report.json')
            pool = next(b for b in diagnosis['banks'] if b['source_id'] == task['target_id']
                        and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates,pool['pool_collisions'],result['queries']))
            output = {'specification':setting, 'result':result, 'metrics':metrics,
                      'history_response_coordinates':responses.tolist(),
                      'query_ledger':ledger.records, 'cost':{'new_physical_calls':0,
                      'equivalent_queries':128, 'response_construction_s':construction_s,
                      'wall_s':time.perf_counter() - started,
                      'equivalent_execution_s':metrics['equivalent_execution_s']}}
            write_json(destination, output)
    return {**task, 'method':method, 'path':str(destination.relative_to(ROOT)), **output['metrics']}


def recovery_gate(rows, baselines):
    expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                for t in development_tasks() for method in METHODS}
    actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Recovery gate requires the exact complete 80-run development matrix')
    baseline_expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                         for t in development_tasks() for method in STRONG_BASELINES}
    baseline_actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in baselines]
    if len(baseline_actual) != len(baseline_expected) or set(baseline_actual) != baseline_expected:
        raise ValueError('Recovery comparison requires every registered strong baseline')
    denominators = {}
    for row in rows + baselines:
        if row['queries'] != 128:
            raise ValueError('Initialization and all paid queries must fit the unchanged budget')
        key = (row['target_id'], row['scenario'], row['algorithm_seed'])
        if denominators.setdefault(key,row['pool_unique_collisions']) != row['pool_unique_collisions']:
            raise ValueError('Methods do not share the same evaluator-only collision denominator')
    stats = method_summary(rows + baselines)
    baseline = max(STRONG_BASELINES, key=lambda m:(stats[m]['collision_recall']['mean'],
                       stats[m]['unique_collisions']['mean'], stats[m]['discovery_auc']['mean'],m))
    main, ref, fixed = (stats[m] for m in ('response_adaptive',baseline,'response_fixed'))
    recall, ref_recall = main['collision_recall']['mean'], ref['collision_recall']['mean']
    reduction = (recall - ref_recall)/(1 - ref_recall) if ref_recall < 1 - 1e-12 else None
    gains = {name:value - ref['collision_recall']['by_lineage'][name]
             for name,value in main['collision_recall']['by_lineage'].items()}
    conditions = {'remaining_misses_reduced_by_half':reduction is not None and reduction >= 0.5 - 1e-12,
                  'collision_count_not_lower':main['unique_collisions']['mean'] >= ref['unique_collisions']['mean'] - 1e-12,
                  'at_least_two_positive_lineages':sum(v > 1e-12 for v in gains.values()) >= 2,
                  'adaptation_recall_exceeds_fixed':recall > fixed['collision_recall']['mean'] + 1e-12,
                  'adaptation_count_not_lower':main['unique_collisions']['mean'] >= fixed['unique_collisions']['mean'] - 1e-12}
    comparisons = []
    for other in (*STRONG_BASELINES,'response_fixed'):
        for metric in ('unique_collisions','collision_recall','discovery_auc'):
            paired = [r for r in rows + baselines if r['method'] in ('response_adaptive',other)
                      and r[metric] is not None]
            comparisons.append(paired_lineage_bootstrap(paired,'response_adaptive',other,metric))
    return {'status':'passed' if all(conditions.values()) else 'not_passed',
            'strongest_baseline':baseline, 'remaining_miss_reduction':reduction,
            'required_recall':ref_recall + 0.5 * (1 - ref_recall), 'conditions':conditions,
            'lineage_recall_gains':gains, 'summary':stats, 'comparisons':comparisons,
            'original_gate_changed':False, 'statistical_significance_claimed':False}


def report(rows=None):
    if rows is None:
        rows = read_json(REPORT/'run_index.json')['runs']
    baselines = [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs']
                 if r['method'] in STRONG_BASELINES]
    gate = recovery_gate(rows,baselines)
    write_json(REPORT/'gate.json',gate)
    lines = ['# 历史响应相似性的目标适配：开发结果','',
             '128次预算、8次共同初始化、80项完整开发回放；新增物理调用为零。',
             '原B轮5个百分点门槛及其失败结论保持原样。','',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现AUC |',
             '|---|---:|---:|---:|']
    for method,stat in gate['summary'].items():
        lines.append(f"| {method} | {stat['unique_collisions']['mean']:.3f} | {stat['collision_recall']['mean']:.4%} | {stat['discovery_auc']['mean']:.3f} |")
    lines += ['',f"后续门槛：{gate['status']}。漏检减少比例：{gate['remaining_miss_reduction']:.4%}；要求至少50%。",
              f"要求召回率至少{gate['required_recall']:.8%}，且数量不降低、至少两个谱系正收益、自适应机制优于固定消融。",
              '逐项门槛、配对区间、谱系结果见gate.json。开发结果不能替代独立目标验证。','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({k:gate[k] for k in ('status','remaining_miss_reduction','conditions')},flush=True)
    return gate


def run():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Freeze the unchanged recovery protocol before running')
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute,task,method,specification)
                   for method in METHODS for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('response_adaptation',len(rows),'/ 80',flush=True)
    rows.sort(key=lambda r:(r['method'],r['target_id'],r['scenario'],r['algorithm_seed']))
    write_json(REPORT/'run_index.json',{'status':'complete','runs':rows,
               'equivalent_queries':80*128,'new_physical_calls':0,'protocol':specification})
    return report(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','run','report'))
    {'freeze':freeze,'run':run,'report':report}[parser.parse_args().stage]()
