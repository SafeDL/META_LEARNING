"""Historical construction and development experiment for joint response transfer."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.common.response import archive_fingerprints, source_weights
from methods.gcrt.joint_response import margin_values, posterior
from methods.gcrt.joint_training import TRAINING, load_model, train, training_contract
from methods.gcrt.joint_search import select

from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import experiment_contract, method_summary
from .discovery_config import (OUTPUT, SCENARIOS, STRONG_BASELINES, development_tasks,
                               read_json, write_json)
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features, sample


REPORT = OUTPUT/'joint_response'
MODELS = ROOT/'results/gcrt/collision/discovery/joint_response/models'
PRIORS = ROOT/'results/gcrt/collision/discovery/joint_response/priors'
RUNS = ROOT/'results/gcrt/collision/discovery/joint_response/runs'
METHODS = ('joint_local','joint_global','joint_shared','joint_mean','joint_binary',
           'joint_target','joint_uncorrelated')
STRONG = (*STRONG_BASELINES,'kinematic_kernel','target_local','gp_target_local',
          'joint_target','joint_uncorrelated')


def model_path(kind,uncorrelated=False):
    return MODELS/('uncorrelated' if uncorrelated else 'correlated')/f'{kind}.pt'


def train_job(kind,uncorrelated):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history,development = load_construction(kind)
        return train(history,development,model_path(kind,uncorrelated),uncorrelated=uncorrelated)


def train_all():
    specification = json.loads(json.dumps({'config':TRAINING,'contract':training_contract(),
                                           'scenarios':SCENARIOS,'uncorrelated':[False,True]}))
    path = REPORT/'training_protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('Joint training protocol already frozen differently')
    write_json(path,specification)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(train_job,kind,uncorrelated) for uncorrelated in (False,True) for kind in SCENARIOS]
        for future in as_completed(futures):
            future.result()
    write_json(REPORT/'training_report.json',{'status':'complete','models':4,'protocol':specification,
                                             'new_physical_calls':0})


def prior_job(kind):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history,_ = load_construction(kind)
        checkpoint = model_path(kind)
        if read_json(checkpoint.with_suffix('.json'))['status'] != 'complete':
            raise ValueError('Finish complete joint training before deriving source posteriors')
        model,metadata = load_model(checkpoint)
        if metadata['specification']['history_fingerprints'] != archive_fingerprints(history):
            raise ValueError('Source posterior archives differ from joint model training')
        candidates = sample(kind,512,'development')
        x = features(candidates)
        specification = {'kind':kind,'model_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         'history_fingerprints':archive_fingerprints(history),
                         'candidate_ids':[s.scenario_id for s in candidates],'x':x.tolist()}
        destination = PRIORS/f'{kind}.npz'
        if destination.exists():
            with np.load(destination,allow_pickle=False) as saved:
                if json.loads(str(saved['specification'])) != specification:
                    raise ValueError('Existing source posterior cache belongs to different coordinates or responses')
            return read_json(destination.with_suffix('.json'))
        started = time.perf_counter()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        means,covariances = [],[]
        for archive in history:
            count = len(archive.scenarios)
            combined = np.concatenate((archive.x,x))
            size = len(combined)
            with torch.no_grad():
                _,mean,covariance = posterior(model.joint_mean(size),model.joint_covariance(combined),
                                               range(count),archive.collision,margin_values(archive.safety[:,0]))
                indices = torch.cat((torch.arange(count,size,device=device),
                                     torch.arange(size+count,2*size,device=device)))
                mean = mean[indices]
                covariance = covariance[indices][:,indices]
                covariance = (covariance+covariance.T)/2
                if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
                    raise FloatingPointError('Nonfinite source joint posterior')
                torch.linalg.cholesky(covariance+1e-7*torch.eye(2*len(x),device=device,dtype=torch.float64))
                means.append(mean.cpu().numpy())
                covariances.append(covariance.cpu().numpy())
            print('joint_prior',kind,archive.source_id,flush=True)
        PRIORS.mkdir(parents=True,exist_ok=True)
        with destination.open('wb') as stream:
            np.savez_compressed(stream,specification=np.asarray(json.dumps(specification)),x=x,
                                means=np.stack(means),covariances=np.stack(covariances),
                                meta_mean=model.joint_mean(len(x)).detach().cpu().numpy(),
                                weights=source_weights(history),names=np.asarray([a.source_id for a in history]))
        result = {'status':'complete','specification':specification,'source_count':len(history),
                  'latent_channels':2,'prediction_candidates':len(x),'psd_jitter_checked':1e-7,
                  'wall_s':time.perf_counter()-started,'new_physical_calls':0}
        write_json(destination.with_suffix('.json'),result)
        return result


def prior_all():
    for kind in SCENARIOS:
        prior_job(kind)


def protocol():
    return {'tasks':development_tasks(),'methods':list(METHODS),'main':'joint_local',
            'strong_baselines':list(STRONG),'budget':128,'initial_count':8,'candidate_count':512,
            'miss_reduction_threshold':.5,'local_bandwidth':.3,'source_prior_mass':.75,
            'target_adaptation_steps':10,'target_adaptation_rate':.003,
            'device':'cuda' if torch.cuda.is_available() else 'cpu',
            'packages':{'torch':str(torch.__version__),'numpy':np.__version__},
            'training_protocol':read_json(REPORT/'training_protocol.json'),
            'checkpoints':{f'{kind}/{variant}':hashlib.sha256(model_path(kind,variant == 'uncorrelated').read_bytes()).hexdigest()
                           for kind in SCENARIOS for variant in ('correlated','uncorrelated')},
            'source_posteriors':{kind:hashlib.sha256((PRIORS/f'{kind}.npz').read_bytes()).hexdigest() for kind in SCENARIOS},
            'prior_reports':{kind:read_json(PRIORS/f'{kind}.json') for kind in SCENARIOS},
            'existing_contract':experiment_contract(),
            'source_files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                            for p in ('benchmark/discovery_joint_response.py','methods/gcrt/joint_search.py',
                                      'methods/gcrt/joint_prediction.py')}}


def freeze():
    if read_json(REPORT/'training_report.json')['status'] != 'complete':
        raise ValueError('Finish all four joint response models first')
    for kind in SCENARIOS:
        if read_json(PRIORS/f'{kind}.json')['status'] != 'complete':
            raise ValueError('Finish both source posterior caches first')
    specification = protocol()
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('Joint response search protocol already frozen differently')
    write_json(path,specification)
    print('joint_response_protocol_frozen',flush=True)


def read_prior(kind,candidates,history,model_digest):
    with np.load(PRIORS/f'{kind}.npz',allow_pickle=False) as saved:
        specification = json.loads(str(saved['specification']))
        if (specification['candidate_ids'] != [s.scenario_id for s in candidates]
                or specification['history_fingerprints'] != archive_fingerprints(history)
                or specification['model_sha256'] != model_digest):
            raise ValueError('Joint source cache differs from candidate coordinates or allowed history')
        np.testing.assert_array_equal(saved['x'],features(candidates))
        return {key:saved[key].copy() for key in ('means','covariances','meta_mean','weights')} | {
            'names':saved['names'].tolist()}


def execute(task,method,specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history,development = load_construction(task['scenario'])
        checkpoint = model_path(task['scenario'],method == 'joint_uncorrelated')
        model,metadata = load_model(checkpoint)
        if (metadata['specification']['history_fingerprints'] != archive_fingerprints(history)
                or metadata['specification']['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('Joint model role-checked archives differ from this development experiment')
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
                raise ValueError('Saved joint-response run differs from the frozen specification')
        else:
            started = time.perf_counter()
            sources = (None if method in ('joint_target','joint_uncorrelated') else
                       read_prior(task['scenario'],bank.candidates,history,setting['checkpoint_sha256']))
            construction_s = time.perf_counter()-started
            ledger = QueryLedger(bank.candidates,128,bank)
            result = run_isolated(select,{'candidates':bank.candidates,'pretrained':model,'sources':sources,
                     'mode':method,'initial_indices':initial_indices(512,8,task['algorithm_seed']),
                     'budget':128,'device':specification['device']},ledger)
            metrics = summarize(bank.candidates,result['queries'],ledger.trajectories)
            pool = next(b for b in read_json(OUTPUT/'diagnosis/report.json')['banks']
                        if b['source_id'] == task['target_id'] and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates,pool['pool_collisions'],result['queries']))
            output = {'specification':setting,'result':result,'metrics':metrics,'query_ledger':ledger.records,
                      'cost':{'new_physical_calls':0,'equivalent_queries':128,'source_construction_s':construction_s,
                              'wall_s':time.perf_counter()-started}}
            write_json(destination,output)
    return {**task,'method':method,'path':str(destination.relative_to(ROOT)),**output['metrics']}


def gate(rows):
    expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                for t in development_tasks() for method in METHODS}
    actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Joint response gate requires the complete 280-run development matrix')
    baseline_rows = [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs'] if r['method'] in STRONG_BASELINES]
    baseline_rows += [r for r in read_json(OUTPUT/'geometry_search/run_index.json')['runs'] if r['method'] == 'kinematic_kernel']
    baseline_rows += [r for r in read_json(OUTPUT/'adaptive_target/run_index.json')['runs'] if r['method'] in ('target_local','gp_target_local')]
    expected_baselines = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                          for t in development_tasks() for method in STRONG[:-2]}
    actual_baselines = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in baseline_rows]
    if len(actual_baselines) != len(expected_baselines) or set(actual_baselines) != expected_baselines:
        raise ValueError('Joint response gate requires every existing strong baseline counterpart')
    combined = rows+baseline_rows
    denominators = {}
    for row in combined:
        if row['queries'] != 128:
            raise ValueError('Joint comparisons must retain the same complete paid budget')
        key = row['target_id'],row['scenario'],row['algorithm_seed']
        if denominators.setdefault(key,row['pool_unique_collisions']) != row['pool_unique_collisions']:
            raise ValueError('Joint comparisons have different evaluator-only denominators')
    stats = method_summary(combined)
    baseline = max(STRONG,key=lambda m:(stats[m]['collision_recall']['mean'],
                   stats[m]['unique_collisions']['mean'],stats[m]['discovery_auc']['mean'],m))
    main,ref = stats['joint_local'],stats[baseline]
    recall,reference = main['collision_recall']['mean'],ref['collision_recall']['mean']
    reduction = (recall-reference)/(1-reference) if reference < 1-1e-12 else None
    gains = {k:v-ref['collision_recall']['by_lineage'][k] for k,v in main['collision_recall']['by_lineage'].items()}
    conditions = {'remaining_misses_reduced_by_half':reduction is not None and reduction >= .5-1e-12,
                  'collision_count_not_lower':main['unique_collisions']['mean'] >= ref['unique_collisions']['mean']-1e-12,
                  'two_positive_lineages':sum(v > 1e-12 for v in gains.values()) >= 2,
                  'local_recall_exceeds_global':recall > stats['joint_global']['collision_recall']['mean']+1e-12,
                  'local_count_not_lower':main['unique_collisions']['mean'] >= stats['joint_global']['unique_collisions']['mean']-1e-12}
    mechanisms = {control:{'recall_gain':recall-stats[control]['collision_recall']['mean'],
                           'collision_count_gain':main['unique_collisions']['mean']-stats[control]['unique_collisions']['mean']}
                  for control in ('joint_shared','joint_mean','joint_binary')}
    comparisons = []
    for other in (*STRONG,'joint_global','joint_shared','joint_mean','joint_binary'):
        for metric in ('unique_collisions','collision_recall','discovery_auc'):
            paired = [r for r in combined if r['method'] in ('joint_local',other) and r[metric] is not None]
            comparisons.append(paired_lineage_bootstrap(paired,'joint_local',other,metric))
    return {'status':'passed' if all(conditions.values()) else 'not_passed','summary':stats,
            'strongest_baseline':baseline,'required_recall':.5*(1+reference),
            'remaining_miss_reduction':reduction,'conditions':conditions,'mechanism_effects':mechanisms,
            'lineage_recall_gains':gains,'comparisons':comparisons,'previous_gates_changed':False,
            'uses_feedback_matched_target_controls':True}


def report(rows=None):
    if rows is None:
        rows = read_json(REPORT/'run_index.json')['runs']
    result = gate(rows)
    write_json(REPORT/'gate.json',result)
    lines = ['# 联合碰撞与间距后验的开发结果','',
             '完整280项运行，初始化计入128次预算；真实碰撞只按布尔标记统计。',
             'joint_target和joint_uncorrelated获得相同已付费碰撞/间距反馈，并参与最强基线选择。','',
             '| 方法 | 平均碰撞数 | 平均召回率 | 发现AUC |','|---|---:|---:|---:|']
    for method,s in result['summary'].items():
        lines.append(f"| {method} | {s['unique_collisions']['mean']:.3f} | {s['collision_recall']['mean']:.5%} | {s['discovery_auc']['mean']:.3f} |")
    lines += ['',f"门槛：{result['status']}；最强基线：{result['strongest_baseline']}；要求召回率至少{result['required_recall']:.8%}。",
              '结构、在线信息与一般多任务学习的收益由各消融分别判断；未改善发现的完整协方差不保留为最终贡献。',
              '全部结果仍属于开发阶段，不能代替独立目标验证。原首轮与后续失败结果完整保留。','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({k:result[k] for k in ('status','remaining_miss_reduction','conditions')},flush=True)
    return result


def run():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Joint response source differs from its frozen protocol')
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute,task,method,specification)
                   for method in METHODS for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('joint_response',len(rows),'/ 280',flush=True)
    rows.sort(key=lambda r:(r['method'],r['target_id'],r['scenario'],r['algorithm_seed']))
    write_json(REPORT/'run_index.json',{'status':'complete','runs':rows,'protocol':specification,
                                      'new_physical_calls':0,'equivalent_queries':35840})
    return report(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('train','prior','freeze','run','report','audit'))
    stage = parser.parse_args().stage
    if stage == 'audit':
        from .discovery_joint_response_audit import audit
        audit()
    else:
        {'train':train_all,'prior':prior_all,'freeze':freeze,'run':run,'report':report}[stage]()
