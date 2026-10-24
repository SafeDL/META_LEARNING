"""Paired search test of structured historical fields and a physical-only control."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import time

from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.response import archive_fingerprints
from methods.gcrt.response_adaptation import select
from methods.gcrt.response_geometry import kinematic_features, response_fields

from .build_collision_models import model_path
from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import method_summary
from .discovery_config import OUTPUT, STRONG_BASELINES, development_tasks, read_json, write_json
from .discovery_safety import protocol as safety_protocol
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features


REPORT = OUTPUT/'geometry_search'
RUNS = ROOT/'results/gcrt/collision/discovery/geometry/runs'
METHODS = ('geometry_response','kinematic_kernel')


def protocol():
    return {'tasks':development_tasks(),'budget':128,'initial_count':8,'candidates':512,
            'methods':list(METHODS),'main':'geometry_response','miss_reduction_threshold':.5,
            'strong_baselines':[*STRONG_BASELINES,'kinematic_kernel'],
            'response_parameters_adaptive':False,'previous_protocol':safety_protocol(),
            'source_files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                            for p in ('methods/gcrt/response_geometry.py','benchmark/discovery_geometry_search.py')},
            'historical_cv':read_json(OUTPUT/'geometry_diagnosis/report.json')['summary']}


def freeze():
    if read_json(OUTPUT/'safety_response/gate.json')['status'] != 'not_passed':
        raise ValueError('Preserve the complete continuous-field diagnosis before this search')
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != protocol():
        raise ValueError('Structured response search already frozen differently')
    write_json(path,protocol())
    print('geometry_search_protocol_frozen',flush=True)


def execute(task,method,specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history,_ = load_construction(task['scenario'])
        checkpoint = model_path('fsbo','all',task['scenario'])
        pretrained,metadata = load_fsbo(checkpoint)
        if metadata['history_fingerprints'] != archive_fingerprints(history):
            raise ValueError('Structured fields use a different historical source set')
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
                raise ValueError('Structured response search differs from frozen inputs')
        else:
            started = time.perf_counter()
            response = (response_fields(history,bank.candidates) if method == 'geometry_response'
                        else kinematic_features(bank.candidates)[:,features(bank.candidates).shape[1]:])
            construction_s = time.perf_counter() - started
            ledger = QueryLedger(bank.candidates,128,bank)
            result = run_isolated(select,{'candidates':bank.candidates,'pretrained':pretrained,
                     'responses':response,'adaptive':False,'budget':128,
                     'initial_indices':initial_indices(512,8,task['algorithm_seed'])},ledger)
            metrics = summarize(bank.candidates,result['queries'],ledger.trajectories)
            pool = next(b for b in read_json(OUTPUT/'diagnosis/report.json')['banks']
                        if b['source_id'] == task['target_id'] and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates,pool['pool_collisions'],result['queries']))
            output = {'specification':setting,'result':result,'metrics':metrics,
                      'history_response_coordinates':response.tolist(),'query_ledger':ledger.records,
                      'cost':{'new_physical_calls':0,'equivalent_queries':128,
                              'response_construction_s':construction_s,'wall_s':time.perf_counter()-started}}
            write_json(destination,output)
    return {**task,'method':method,'path':str(destination.relative_to(ROOT)),**output['metrics']}


def report(rows=None):
    if rows is None:
        rows = read_json(REPORT/'run_index.json')['runs']
    expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                for t in development_tasks() for method in METHODS}
    actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected or any(r['queries'] != 128 for r in rows):
        raise ValueError('Structured response gate requires all 80 paired 128-query runs')
    baselines = [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs'] if r['method'] in STRONG_BASELINES]
    raw = [r for r in read_json(OUTPUT/'safety_response/run_index.json')['runs'] if r['method'] == 'safety_fixed']
    combined = rows + baselines + raw
    totals = {}
    for row in combined:
        key = row['target_id'],row['scenario'],row['algorithm_seed']
        if totals.setdefault(key,row['pool_unique_collisions']) != row['pool_unique_collisions']:
            raise ValueError('Structured response comparisons have different collision denominators')
    stats = method_summary(combined)
    baseline = max((*STRONG_BASELINES,'kinematic_kernel'),key=lambda m:(stats[m]['collision_recall']['mean'],
                   stats[m]['unique_collisions']['mean'],stats[m]['discovery_auc']['mean'],m))
    main,ref,unstructured = (stats[m] for m in ('geometry_response',baseline,'safety_fixed'))
    recall,reference = main['collision_recall']['mean'],ref['collision_recall']['mean']
    reduction = (recall-reference)/(1-reference) if reference < 1-1e-12 else None
    gains = {k:v-ref['collision_recall']['by_lineage'][k] for k,v in main['collision_recall']['by_lineage'].items()}
    conditions = {'remaining_misses_reduced_by_half':reduction is not None and reduction >= .5-1e-12,
                  'collision_count_not_lower':main['unique_collisions']['mean'] >= ref['unique_collisions']['mean']-1e-12,
                  'two_positive_lineages':sum(v > 1e-12 for v in gains.values()) >= 2,
                  'structured_recall_exceeds_raw':recall > unstructured['collision_recall']['mean']+1e-12,
                  'structured_count_not_lower':main['unique_collisions']['mean'] >= unstructured['unique_collisions']['mean']-1e-12}
    comparisons = []
    for other in (*STRONG_BASELINES,'kinematic_kernel','safety_fixed'):
        for metric in ('unique_collisions','collision_recall','discovery_auc'):
            paired = [r for r in combined if r['method'] in ('geometry_response',other) and r[metric] is not None]
            comparisons.append(paired_lineage_bootstrap(paired,'geometry_response',other,metric))
    gate = {'status':'passed' if all(conditions.values()) else 'not_passed','summary':stats,
            'strongest_baseline':baseline,'remaining_miss_reduction':reduction,
            'conditions':conditions,'lineage_recall_gains':gains,'comparisons':comparisons,
            'previous_gates_changed':False,'new_physical_calls':0}
    write_json(REPORT/'gate.json',gate)
    lines = ['# 相对运动条件下的历史响应迁移','',
             '80项开发回放；对照直接使用同一组相对运动特征，不读取额外历史响应。该对照参与最强基线选择。','',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现AUC |','|---|---:|---:|---:|']
    for method,stat in stats.items():
        lines.append(f"| {method} | {stat['unique_collisions']['mean']:.3f} | {stat['collision_recall']['mean']:.4%} | {stat['discovery_auc']['mean']:.3f} |")
    lines += ['',f"门槛：{gate['status']}；最强基线：{baseline}；漏检减少比例：{reduction}。",
              '历史交叉验证的预测改善不能单独证明发现收益。全部已失败阶段保留，不覆盖旧协议。','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'status':gate['status'],'remaining_miss_reduction':reduction,'conditions':conditions},flush=True)
    return gate


def run():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Structured response source differs from the frozen protocol')
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute,task,method,specification)
                   for method in METHODS for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('geometry_search',len(rows),'/ 80',flush=True)
    rows.sort(key=lambda r:(r['method'],r['target_id'],r['scenario'],r['algorithm_seed']))
    write_json(REPORT/'run_index.json',{'status':'complete','runs':rows,'protocol':specification,
                                      'new_physical_calls':0,'equivalent_queries':10240})
    return report(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','run','report'))
    {'freeze':freeze,'run':run,'report':report}[parser.parse_args().stage]()
