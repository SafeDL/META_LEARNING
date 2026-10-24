"""Strong target component with source-kernel and target-ensemble controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import time

from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.response import archive_fingerprints
from methods.gcrt.adaptive_target import select
from methods.gcrt.discovery import discovery_posterior
from methods.gcrt.posterior import CollisionPosterior
from methods.gcrt.train import load_encoder

from .build_collision_models import model_path
from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import experiment_contract, method_summary
from .discovery_config import OUTPUT, STRONG_BASELINES, development_tasks, read_json, write_json
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features


REPORT = OUTPUT/'adaptive_target'
RUNS = ROOT/'results/gcrt/collision/discovery/adaptive_target/runs'
METHODS = ('hybrid_local','hybrid_global','hybrid_shared','target_local','gp_target_local')
STRONG = (*STRONG_BASELINES,'kinematic_kernel','target_local','gp_target_local')


def protocol():
    return {'tasks':development_tasks(),'methods':list(METHODS),'main':'hybrid_local',
            'strong_baselines':list(STRONG),'budget':128,'initial_count':8,'candidates':512,
            'miss_reduction_threshold':.5,'local_bandwidth':.3,'fsbo_additional_prior':.25,
            'target_adaptation':{'steps':10,'fsbo_rate':.003,'raw_gp_rate':.01},
            'historical_encoder':'same original shared-trained frozen encoder in all source-kernel arms',
            'source_updates':'ADF Bernoulli; per-component source weights disabled internally',
            'reliability':'flat prequery likelihoods for source and target components; no rescoring',
            'baseline_contract':experiment_contract(),
            'source_files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                            for p in ('methods/gcrt/adaptive_target.py','benchmark/discovery_adaptive_target.py')}}


def freeze():
    if read_json(OUTPUT/'geometry_search/audit/report.json')['status'] != 'passed':
        raise ValueError('Complete the preceding development diagnosis and audit first')
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != protocol():
        raise ValueError('Adaptive-target protocol already frozen differently')
    write_json(path,protocol())
    print('adaptive_target_protocol_frozen',flush=True)


def source_prior(kind,candidates,history,method):
    if method == 'gp_target_local':
        return None
    encoder,metadata = load_encoder(model_path('history_response','all',kind))
    if metadata['history_fingerprints'] != archive_fingerprints(history):
        raise ValueError('Adaptive-target source kernel uses different historical archives')
    prior = discovery_posterior(encoder,features(candidates),history,
                                independent=method != 'hybrid_shared')
    if method == 'target_local':
        return CollisionPosterior(prior.mean[-1:],prior.covariance[-1:],[1.],
                                  fixed_weights=True,component_names=['target_only'])
    return prior


def execute(task,method,specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history,_ = load_construction(task['scenario'])
        checkpoint = model_path('fsbo','all',task['scenario'])
        pretrained,metadata = load_fsbo(checkpoint)
        if metadata['history_fingerprints'] != archive_fingerprints(history):
            raise ValueError('Strong target and source kernels require the same allowed histories')
        bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
        checkpoints = {'fsbo':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        if method != 'gp_target_local':
            checkpoints['history_response'] = hashlib.sha256(
                model_path('history_response','all',task['scenario']).read_bytes()).hexdigest()
        setting = {'task':task,'method':method,'protocol':specification,'checkpoints':checkpoints,
                   'history_fingerprints':archive_fingerprints(history),
                   'bank_contract':bank.manifest['contract'],
                   'candidate_ids':[s.scenario_id for s in bank.candidates]}
        destination = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != setting or output['result']['actual_queries'] != 128:
                raise ValueError('Adaptive-target run differs from frozen inputs')
        else:
            started = time.perf_counter()
            model = source_prior(task['scenario'],bank.candidates,history,method)
            construction_s = time.perf_counter()-started
            ledger = QueryLedger(bank.candidates,128,bank)
            result = run_isolated(select,{'candidates':bank.candidates,'source_model':model,
                     'pretrained':pretrained,'budget':128,'raw_target':method == 'gp_target_local',
                     'bandwidth':None if method == 'hybrid_global' else .3,
                     'initial_indices':initial_indices(512,8,task['algorithm_seed'])},ledger)
            metrics = summarize(bank.candidates,result['queries'],ledger.trajectories)
            pool = next(b for b in read_json(OUTPUT/'diagnosis/report.json')['banks']
                        if b['source_id'] == task['target_id'] and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates,pool['pool_collisions'],result['queries']))
            output = {'specification':setting,'result':result,'metrics':metrics,
                      'query_ledger':ledger.records,'cost':{'new_physical_calls':0,
                      'equivalent_queries':128,'source_construction_s':construction_s,
                      'wall_s':time.perf_counter()-started}}
            write_json(destination,output)
    return {**task,'method':method,'path':str(destination.relative_to(ROOT)),**output['metrics']}


def gate(rows):
    expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                for t in development_tasks() for method in METHODS}
    actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Strong-target gate requires the exact complete 200-run development matrix')
    baselines = [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs'] if r['method'] in STRONG_BASELINES]
    baselines += [r for r in read_json(OUTPUT/'geometry_search/run_index.json')['runs'] if r['method'] == 'kinematic_kernel']
    baseline_expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                         for t in development_tasks() for method in (*STRONG_BASELINES,'kinematic_kernel')}
    baseline_actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in baselines]
    if len(baseline_actual) != len(baseline_expected) or set(baseline_actual) != baseline_expected:
        raise ValueError('Strong-target gate requires every registered baseline counterpart')
    combined = rows+baselines
    totals = {}
    for row in combined:
        if row['queries'] != 128:
            raise ValueError('Adaptive-target comparison requires the full common paid budget')
        key = row['target_id'],row['scenario'],row['algorithm_seed']
        if totals.setdefault(key,row['pool_unique_collisions']) != row['pool_unique_collisions']:
            raise ValueError('Adaptive-target comparisons have different sealed denominators')
    stats = method_summary(combined)
    baseline = max(STRONG,key=lambda m:(stats[m]['collision_recall']['mean'],
                   stats[m]['unique_collisions']['mean'],stats[m]['discovery_auc']['mean'],m))
    main,ref = stats['hybrid_local'],stats[baseline]
    recall,reference = main['collision_recall']['mean'],ref['collision_recall']['mean']
    reduction = (recall-reference)/(1-reference) if reference < 1-1e-12 else None
    gains = {k:v-ref['collision_recall']['by_lineage'][k] for k,v in main['collision_recall']['by_lineage'].items()}
    conditions = {'remaining_misses_reduced_by_half':reduction is not None and reduction >= .5-1e-12,
                  'collision_count_not_lower':main['unique_collisions']['mean'] >= ref['unique_collisions']['mean']-1e-12,
                  'two_positive_lineages':sum(v > 1e-12 for v in gains.values()) >= 2}
    for control in ('hybrid_global','hybrid_shared'):
        conditions[f'recall_exceeds_{control}'] = recall > stats[control]['collision_recall']['mean']+1e-12
        conditions[f'count_not_lower_than_{control}'] = main['unique_collisions']['mean'] >= stats[control]['unique_collisions']['mean']-1e-12
    comparisons = []
    for other in (*STRONG,'hybrid_global','hybrid_shared'):
        for metric in ('unique_collisions','collision_recall','discovery_auc'):
            paired = [r for r in combined if r['method'] in ('hybrid_local',other) and r[metric] is not None]
            comparisons.append(paired_lineage_bootstrap(paired,'hybrid_local',other,metric))
    return {'status':'passed' if all(conditions.values()) else 'not_passed','summary':stats,
            'strongest_baseline':baseline,'required_recall':.5*(1+reference),
            'remaining_miss_reduction':reduction,'conditions':conditions,
            'lineage_recall_gains':gains,'comparisons':comparisons,'previous_gates_changed':False}


def report(rows=None):
    if rows is None:
        rows = read_json(REPORT/'run_index.json')['runs']
    result = gate(rows)
    write_json(REPORT/'gate.json',result)
    lines = ['# 强目标适配与历史来源核的开发比较','',
             '完整200项运行；目标深核每轮从历史初始化出发做10步适配，全部分量用查询前概率更新同一层可靠性权重。',
             'target_local只组合冻结物理GP与目标深核；gp_target_local只组合从零适配的物理GP与目标深核。两者均参与最强基线选择。','',
             '| 方法 | 平均碰撞数 | 平均召回率 | 发现AUC |','|---|---:|---:|---:|']
    for method,s in result['summary'].items():
        lines.append(f"| {method} | {s['unique_collisions']['mean']:.3f} | {s['collision_recall']['mean']:.5%} | {s['discovery_auc']['mean']:.3f} |")
    lines += ['',f"门槛：{result['status']}；最强基线：{result['strongest_baseline']}；要求召回率至少{result['required_recall']:.8%}。",
              '该轮只改变目标独立分量与权重组织，仍使用原历史编码器；正式独立验证前须重新训练与最终推断匹配的历史模型。',
              '原首轮与后续失败结果保持原样；开发互补性并集并不代表128次查询能实现的收益。','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({k:result[k] for k in ('status','remaining_miss_reduction','conditions')},flush=True)
    return result


def run():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Strong-target source differs from its frozen protocol')
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute,task,method,specification)
                   for method in METHODS for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('adaptive_target',len(rows),'/ 200',flush=True)
    rows.sort(key=lambda r:(r['method'],r['target_id'],r['scenario'],r['algorithm_seed']))
    write_json(REPORT/'run_index.json',{'status':'complete','runs':rows,'protocol':specification,
                                      'new_physical_calls':0,'equivalent_queries':25600})
    return report(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','run','report'))
    {'freeze':freeze,'run':run,'report':report}[parser.parse_args().stage]()
