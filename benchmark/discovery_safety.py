"""Fixed historical safety geometry after collision-only kernel adaptation fails."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import time

from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.response import archive_fingerprints
from methods.gcrt.response_adaptation import SURROGATE_CONFIG, select
from methods.gcrt.safety_response import safety_coordinates

from .build_collision_models import model_path
from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import method_summary
from .discovery_config import OUTPUT, STRONG_BASELINES, development_tasks, read_json, write_json
from .discovery_recovery import protocol as previous_protocol
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features


REPORT = OUTPUT/'safety_response'
RUNS = ROOT/'results/gcrt/collision/discovery/safety_response/runs'
METHODS = ('safety_fixed','safety_adaptive')


def protocol():
    return {'budget':128,'initial_count':8,'candidates':512,'tasks':development_tasks(),
            'methods':list(METHODS),'main':'safety_fixed','miss_reduction_threshold':.5,
            'mechanism_control':'response_fixed','surrogate_config':SURROGATE_CONFIG,
            'regression_targets':'archive.auxiliary: clipped clearance/100 and TTC/10',
            'normalization':'(prediction-history_mean)/max(3*history_std,0.1), clipped [-1,1]',
            'previous_protocol':previous_protocol(),
            'source_files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                            for p in ('methods/gcrt/safety_response.py','benchmark/discovery_safety.py')}}


def freeze():
    if read_json(OUTPUT/'recovery/gate.json')['status'] != 'not_passed':
        raise ValueError('This diagnosis follows the recorded failed collision-only mechanism')
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != protocol():
        raise ValueError('Safety-response development protocol already frozen differently')
    write_json(path,protocol())
    print('safety_response_protocol_frozen',flush=True)


def execute(task,method,specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history,_ = load_construction(task['scenario'])
        checkpoint = model_path('fsbo','all',task['scenario'])
        pretrained,metadata = load_fsbo(checkpoint)
        if metadata['history_fingerprints'] != archive_fingerprints(history):
            raise ValueError('Historical safety fields and FSBO must use the same allowed sources')
        bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
        setting = {'task':task,'method':method,'protocol':specification,
                   'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'history_fingerprints':archive_fingerprints(history),
                   'bank_contract':bank.manifest['contract'],
                   'candidate_ids':[s.scenario_id for s in bank.candidates]}
        destination = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != setting or output['result']['actual_queries'] != 128:
                raise ValueError('Safety-response run differs from the frozen specification')
        else:
            started = time.perf_counter()
            response = safety_coordinates(history,features(bank.candidates))
            construction_s = time.perf_counter() - started
            ledger = QueryLedger(bank.candidates,128,bank)
            result = run_isolated(select,{'candidates':bank.candidates,
                     'initial_indices':initial_indices(512,8,task['algorithm_seed']),
                     'pretrained':pretrained,'responses':response,
                     'adaptive':method == 'safety_adaptive','budget':128},ledger)
            metrics = summarize(bank.candidates,result['queries'],ledger.trajectories)
            pool = next(b for b in read_json(OUTPUT/'diagnosis/report.json')['banks']
                        if b['source_id'] == task['target_id'] and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates,pool['pool_collisions'],result['queries']))
            output = {'specification':setting,'result':result,'metrics':metrics,
                      'history_response_coordinates':response.tolist(),'query_ledger':ledger.records,
                      'cost':{'new_physical_calls':0,'equivalent_queries':128,
                              'response_construction_s':construction_s,
                              'wall_s':time.perf_counter() - started}}
            write_json(destination,output)
    return {**task,'method':method,'path':str(destination.relative_to(ROOT)),**output['metrics']}


def report(rows=None):
    if rows is None:
        rows = read_json(REPORT/'run_index.json')['runs']
    expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                for t in development_tasks() for method in METHODS}
    actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected or any(r['queries'] != 128 for r in rows):
        raise ValueError('Safety-response gate requires all 80 paired 128-query runs')
    baselines = [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs']
                 if r['method'] in STRONG_BASELINES]
    control = [r for r in read_json(OUTPUT/'recovery/run_index.json')['runs']
               if r['method'] == 'response_fixed']
    combined = rows + baselines + control
    totals = {}
    for row in combined:
        key = row['target_id'],row['scenario'],row['algorithm_seed']
        if totals.setdefault(key,row['pool_unique_collisions']) != row['pool_unique_collisions']:
            raise ValueError('Safety-response comparisons have different sealed denominators')
    stats = method_summary(combined)
    baseline = max(STRONG_BASELINES,key=lambda m:(stats[m]['collision_recall']['mean'],
                   stats[m]['unique_collisions']['mean'],stats[m]['discovery_auc']['mean'],m))
    main,ref,binary = (stats[m] for m in ('safety_fixed',baseline,'response_fixed'))
    recall,reference = main['collision_recall']['mean'],ref['collision_recall']['mean']
    reduction = (recall - reference)/(1 - reference)
    gains = {k:v - ref['collision_recall']['by_lineage'][k]
             for k,v in main['collision_recall']['by_lineage'].items()}
    conditions = {'remaining_misses_reduced_by_half':reduction >= .5 - 1e-12,
                  'collision_count_not_lower':main['unique_collisions']['mean'] >= ref['unique_collisions']['mean'] - 1e-12,
                  'two_positive_lineages':sum(v > 1e-12 for v in gains.values()) >= 2,
                  'safety_recall_exceeds_collision_only':recall > binary['collision_recall']['mean'] + 1e-12,
                  'safety_count_not_lower':main['unique_collisions']['mean'] >= binary['unique_collisions']['mean'] - 1e-12}
    comparisons = []
    for other in (*STRONG_BASELINES,'response_fixed','safety_adaptive'):
        for metric in ('unique_collisions','collision_recall','discovery_auc'):
            paired = [r for r in combined if r['method'] in ('safety_fixed',other) and r[metric] is not None]
            comparisons.append(paired_lineage_bootstrap(paired,'safety_fixed',other,metric))
    gate = {'status':'passed' if all(conditions.values()) else 'not_passed',
            'summary':stats,'strongest_baseline':baseline,'remaining_miss_reduction':reduction,
            'conditions':conditions,'lineage_recall_gains':gains,'comparisons':comparisons,
            'original_gate_changed':False,'previous_recovery_gate_changed':False}
    write_json(REPORT/'gate.json',gate)
    lines = ['# 连续历史安全响应的开发验证','',
             '80项完整开发回放，预算128，新增物理调用为零。固定来源核为预先指定主方法；适配版本为辅助对照。','',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现AUC |','|---|---:|---:|---:|']
    for method,stat in stats.items():
        lines.append(f"| {method} | {stat['unique_collisions']['mean']:.3f} | {stat['collision_recall']['mean']:.4%} | {stat['discovery_auc']['mean']:.3f} |")
    lines += ['',f"新门槛：{gate['status']}；剩余漏检减少{reduction:.4%}，要求至少50%。",
              '全部逐项门槛、谱系与配对区间保留在gate.json。原B和碰撞响应适配两轮失败记录保持原样。','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'status':gate['status'],'remaining_miss_reduction':reduction,'conditions':conditions},flush=True)
    return gate


def run():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Safety-response source differs from the frozen protocol')
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute,task,method,specification)
                   for method in METHODS for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('safety_response',len(rows),'/ 80',flush=True)
    rows.sort(key=lambda r:(r['method'],r['target_id'],r['scenario'],r['algorithm_seed']))
    write_json(REPORT/'run_index.json',{'status':'complete','runs':rows,'protocol':specification,
                                      'new_physical_calls':0,'equivalent_queries':10240})
    return report(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','run','report'))
    {'freeze':freeze,'run':run,'report':report}[parser.parse_args().stage]()
