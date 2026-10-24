"""Gated development comparison of joint transfer and an independent physical GP."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
import traceback

from threadpoolctl import threadpool_limits
import torch

from methods.common.response import archive_fingerprints
from methods.baselines.fsbo.method import load as load_fsbo
from methods.baselines.fsbo.search import select as select_fsbo
from methods.gcrt.joint_training import load_model
from methods.gcrt.physical_mixture import MODES, PHYSICAL_MASS, select

from . import discovery_joint_response as joint
from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .build_collision_models import model_path
from .construction import load_construction
from .discovery import method_summary
from .discovery_config import OUTPUT, STRONG_BASELINES, development_tasks, read_json, write_json
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated


REPORT = OUTPUT/'physical_mixture'
RUNS = ROOT/'results/gcrt/collision/discovery/physical_mixture/runs'
METHODS = ('fsbo_cuda','physical_fsbo','physical_target',
           *(m for m in MODES if m not in ('physical_fsbo','physical_target')))
MAIN = 'physical_global'
PREVIOUS = tuple(dict.fromkeys((*joint.STRONG,*joint.METHODS)))
STRONG = (*PREVIOUS,'physical_target','physical_fsbo','fsbo_cuda')


def checkpoint_path(kind,method):
    return model_path('fsbo','all',kind) if method in ('physical_fsbo','fsbo_cuda') else joint.model_path(kind)


def protocol():
    prerequisites = ('run_index.json','gate.json','audit/report.json')
    if any(not (joint.REPORT/path).exists() for path in prerequisites):
        raise ValueError('Complete and independently audit the entire 280-run joint matrix first')
    foundation = joint.protocol()
    return {'tasks':development_tasks(),'methods':list(METHODS),'main':MAIN,'strong_baselines':list(STRONG),
            'budget':128,'initial_count':8,'candidate_count':512,'physical_mass':PHYSICAL_MASS,
            'physical_mass_by_method':{m:0. if m == 'fsbo_cuda' else PHYSICAL_MASS for m in METHODS},
            'miss_reduction_threshold':.5,'local_bandwidth':.3,
            'joint_adaptation':{'steps':10,'learning_rate':.003},
            'physical_adaptation':{'steps':10,'learning_rate':.01},
            'online_feedback':{m:['collision','clearance'] if m == 'physical_margin' else ['collision'] for m in METHODS},
            'device':foundation['device'],'foundation':foundation,
            'fsbo_checkpoints':{kind:hashlib.sha256(checkpoint_path(kind,'physical_fsbo').read_bytes()).hexdigest()
                                for kind in ('cut_in','cut_out')},
            'foundation_evidence':{path:hashlib.sha256((joint.REPORT/path).read_bytes()).hexdigest()
                                   for path in prerequisites},
            'source_files':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                            for path in ('benchmark/discovery_physical_mixture.py','methods/gcrt/physical_mixture.py',
                                         'methods/baselines/fsbo/search.py')}}


def freeze():
    index = read_json(joint.REPORT/'run_index.json') if (joint.REPORT/'run_index.json').exists() else None
    audit = read_json(joint.REPORT/'audit/report.json') if (joint.REPORT/'audit/report.json').exists() else None
    if (not index or not audit or index['status'] != 'complete' or len(index['runs']) != 280
            or audit['status'] != 'passed' or audit['runs'] != 280
            or index['protocol'] != joint.protocol()):
        raise ValueError('Complete and independently audit the entire 280-run joint matrix first')
    if joint.gate(index['runs']) != read_json(joint.REPORT/'gate.json'):
        raise ValueError('The joint-response result no longer matches its frozen admission evidence')
    specification = protocol()
    destination = REPORT/'protocol.json'
    if destination.exists() and read_json(destination) != specification:
        raise ValueError('Physical-mixture protocol already frozen differently')
    write_json(destination,specification)
    return specification


def execute(task,method,specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history,development = load_construction(task['scenario'])
        checkpoint = checkpoint_path(task['scenario'],method)
        if method in ('physical_fsbo','fsbo_cuda'):
            model,construction = load_fsbo(checkpoint)
        else:
            model,metadata = load_model(checkpoint)
            construction = metadata['specification']
        if (construction['history_fingerprints'] != archive_fingerprints(history)
                or construction['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('Physical-mixture model differs from the allowed construction archives')
        bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
        setting = {'task':task,'method':method,'protocol':specification,
                   'checkpoint_path':str(checkpoint.relative_to(ROOT)),
                   'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'history_fingerprints':archive_fingerprints(history),'bank_contract':bank.manifest['contract'],
                   'candidate_ids':[s.scenario_id for s in bank.candidates]}
        destination = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != setting or output['result']['actual_queries'] != 128:
                raise ValueError('Saved physical-mixture run differs from the frozen experiment')
        else:
            started = time.perf_counter()
            sources = (None if method in ('physical_target','physical_fsbo','fsbo_cuda') else
                       joint.read_prior(task['scenario'],bank.candidates,history,setting['checkpoint_sha256']))
            construction_s = time.perf_counter()-started
            ledger = QueryLedger(bank.candidates,128,bank)
            try:
                public = {'candidates':bank.candidates,'pretrained':model,
                          'initial_indices':initial_indices(512,8,task['algorithm_seed']),
                          'budget':128,'device':specification['device']}
                if method != 'fsbo_cuda':
                    public.update(sources=sources,mode=method)
                result = run_isolated(select_fsbo if method == 'fsbo_cuda' else select,public,ledger)
            except Exception:
                path = REPORT/'failed_runs'/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.jsonl"
                path.parent.mkdir(parents=True,exist_ok=True)
                failure = {'specification':setting,'status':'failed','query_ledger':ledger.records,
                           'paid_attempts':len(ledger.records),'error':traceback.format_exc(),
                           'wall_s':time.perf_counter()-started,'new_physical_calls':0}
                with path.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(failure,ensure_ascii=False,allow_nan=False)+'\n')
                raise
            metrics = summarize(bank.candidates,result['queries'],ledger.trajectories)
            pool = next(b for b in read_json(OUTPUT/'diagnosis/report.json')['banks']
                        if b['source_id'] == task['target_id'] and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates,pool['pool_collisions'],result['queries']))
            output = {'specification':setting,'result':result,'metrics':metrics,'query_ledger':ledger.records,
                      'cost':{'equivalent_queries':128,'new_physical_calls':0,'source_construction_s':construction_s,
                              'wall_s':time.perf_counter()-started,'new_historical_training':False}}
            write_json(destination,output)
    return {**task,'method':method,'path':str(destination.relative_to(ROOT)),**output['metrics']}


def smoke():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Physical-mixture source differs from the frozen protocol')
    task = development_tasks()[0]
    rows,new_runs = [],0
    for method in METHODS:
        path = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
        existed = path.exists()
        rows.append(execute(task,method,specification))
        new_runs += not existed
        print('physical_smoke',method,'complete',flush=True)
    result = {'status':'complete','runs':rows,'valid_matrix_runs':len(rows),
              'new_runs_this_invocation':new_runs,'equivalent_queries_this_invocation':128*new_runs,
              'part_of_full_matrix':True,'new_physical_calls':0}
    write_json(REPORT/'smoke.json',result)
    return result


def previous_rows():
    rows = [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs'] if r['method'] in STRONG_BASELINES]
    rows += [r for r in read_json(OUTPUT/'geometry_search/run_index.json')['runs'] if r['method'] == 'kinematic_kernel']
    rows += [r for r in read_json(OUTPUT/'adaptive_target/run_index.json')['runs'] if r['method'] in ('target_local','gp_target_local')]
    return rows+read_json(joint.REPORT/'run_index.json')['runs']


def gate(rows):
    expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],m) for t in development_tasks() for m in METHODS}
    actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Physical-mixture admission requires every complete paired run in the 320-run matrix')
    previous = previous_rows()
    required_previous = {(t['target_id'],t['scenario'],t['algorithm_seed'],m)
                         for t in development_tasks() for m in PREVIOUS}
    actual_previous = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in previous]
    if len(actual_previous) != len(required_previous) or set(actual_previous) != required_previous:
        raise ValueError('Every previous strong method must have its full 40-run counterpart')
    combined = rows+previous
    denominators = {}
    for r in combined:
        key = r['target_id'],r['scenario'],r['algorithm_seed']
        if r['queries'] != 128 or denominators.setdefault(key,r['pool_unique_collisions']) != r['pool_unique_collisions']:
            raise ValueError('Physical-mixture baselines differ in budget or evaluator-only collision denominators')
    stats = method_summary(combined)
    baseline = max(STRONG,key=lambda m:((stats[m]['collision_recall']['mean']
                    if stats[m]['collision_recall']['mean'] is not None else -1),
                    stats[m]['unique_collisions']['mean'],stats[m]['discovery_auc']['mean'],m))
    main,ref,adaptive = stats[MAIN],stats[baseline],stats['physical_adaptive']
    recall,reference = main['collision_recall']['mean'],ref['collision_recall']['mean']
    reduction = ((recall-reference)/(1-reference)
                 if recall is not None and reference is not None and reference < 1-1e-12 else None)
    gains = {k:v-ref['collision_recall']['by_lineage'][k]
             for k,v in main['collision_recall']['by_lineage'].items()}
    conditions = {'remaining_misses_reduced_by_half':reduction is not None and reduction >= .5-1e-12,
                  'collision_count_not_lower':main['unique_collisions']['mean'] >= ref['unique_collisions']['mean']-1e-12,
                  'two_positive_lineages':sum(v > 1e-12 for v in gains.values()) >= 2,
                  'fixed_mass_recall_exceeds_adaptive':recall is not None and recall > adaptive['collision_recall']['mean']+1e-12,
                  'fixed_mass_count_not_lower':main['unique_collisions']['mean'] >= adaptive['unique_collisions']['mean']-1e-12}
    comparisons = []
    for control in (*STRONG,'physical_local','physical_mean','physical_adaptive','physical_margin'):
        for metric in ('unique_collisions','collision_recall','discovery_auc'):
            paired = [r for r in combined if r['method'] in (MAIN,control) and r[metric] is not None]
            if paired:
                comparisons.append(paired_lineage_bootstrap(paired,MAIN,control,metric))
    return {'status':'passed' if all(conditions.values()) else 'not_passed','main':MAIN,'summary':stats,
            'strongest_baseline':baseline,'required_recall':.5*(1+reference) if reference is not None else None,
            'remaining_miss_reduction':reduction,'conditions':conditions,'lineage_recall_gains':gains,
            'comparisons':comparisons,'previous_gates_changed':False,
            'local_reliability_claimed_as_necessary':False,'new_physical_calls':0}


def report(rows=None):
    if rows is None:
        rows = read_json(REPORT/'run_index.json')['runs']
    result = gate(rows)
    write_json(REPORT/'gate.json',result)
    lines = ['# 联合响应与独立物理GP的开发结果','','320项冻结运行，预算128包含8次共同初始化。',
             '全部此前强对照与7个联合响应版本，以及physical_target和FSBO物理混合共同参与最强基线选择。','',
             '| 方法 | 平均碰撞数 | 平均召回率 | 发现AUC |','|---|---:|---:|---:|']
    for method,values in result['summary'].items():
        r = values['collision_recall']['mean']
        recall = f'{r:.6%}' if r is not None else '不适用'
        lines.append(f"| {method} | {values['unique_collisions']['mean']:.3f} | {recall} | {values['discovery_auc']['mean']:.3f} |")
    lines += ['',f"准入：{result['status']}；最强基线：{result['strongest_baseline']}。",
              '主方法仅用已付费碰撞标签更新，physical_margin单独使用在线间距；历史来源均保留原联合响应后验。',
              '物理分量固定保留0.10预测质量；自适应外层对照从同一0.10开始，仅用查询前二值得分更新。',
              '主方法使用全局来源可靠性，不把局部门控或常见混合规则本身解释成原创贡献。',
              '这些是开发证据；首轮与联合响应失败记录不改变，C/D/E仍需独立准入和冻结。','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def run():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Physical-mixture source differs from the frozen protocol')
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute,t,m,specification) for m in METHODS for t in development_tasks()]
        try:
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 8 == 0:
                    print('physical_mixture',len(rows),'/ 320',flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    rows.sort(key=lambda r:(r['method'],r['target_id'],r['scenario'],r['algorithm_seed']))
    write_json(REPORT/'run_index.json',{'status':'complete','protocol':specification,'runs':rows,
                                      'equivalent_queries':40960,'new_physical_calls':0})
    return report(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','smoke','run','report','audit'))
    stage = parser.parse_args().stage
    if stage == 'audit':
        from .discovery_physical_mixture_audit import audit
        audit()
    else:
        {'freeze':freeze,'smoke':smoke,'run':run,'report':report}[stage]()
