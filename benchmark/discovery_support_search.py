"""Independent search experiment for the wider historical support schedule."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time

from threadpoolctl import threadpool_limits
import torch

from methods.common.response import archive_fingerprints
from methods.gcrt.consistency import MODES, load_encoder, training_contract
from methods.gcrt.discovery import discovery_posterior
from methods.gcrt.selection import select

from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import experiment_contract, method_summary
from .discovery_config import OUTPUT, development_tasks, read_json, write_json
from .discovery_support import WIDE_CONFIG, model_location_wide
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, DEFAULT, ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features


REPORT = OUTPUT / 'support/search'
RUNS = ROOT / 'results/gcrt/collision/discovery/support/runs'


def execute(task, mode, contract):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, development = load_construction(task['scenario'])
        checkpoint = model_location_wide(mode, task['scenario'])
        encoder, metadata = load_encoder(checkpoint)
        if (metadata['specification']['history_fingerprints'] != archive_fingerprints(history)
                or metadata['specification']['development_fingerprints'] != archive_fingerprints(development)
                or metadata['specification']['config']['support_counts'] != list(WIDE_CONFIG.support_counts)):
            raise ValueError('Wide-support checkpoint differs from its registered archives or support schedule')
        bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
        specification = {'task':task, 'mode':mode, 'contract':contract,
                         'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         'checkpoint_step':metadata['step'], 'bank_contract':bank.manifest['contract'],
                         'candidate_ids':[s.scenario_id for s in bank.candidates],
                         'budget':128, 'initial_count':8}
        destination = RUNS/mode/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != specification or output['result']['actual_queries'] != 128:
                raise ValueError('Saved wide-support search differs from its completed specification')
        else:
            started = time.perf_counter()
            independent, bandwidth = MODES[mode]
            model = discovery_posterior(encoder,features(bank.candidates),history,
                                        independent=independent,bandwidth=bandwidth)
            ledger = QueryLedger(bank.candidates,128,bank)
            result = run_isolated(select, {'model':model,'candidates':bank.candidates,
                                  'initial_indices':initial_indices(512,8,task['algorithm_seed']),
                                  'budget':128,'horizon':1,'greedy':True},ledger)
            metrics = summarize(bank.candidates,result['queries'],ledger.trajectories)
            diagnosis = read_json(OUTPUT/'diagnosis/report.json')
            pool = next(b for b in diagnosis['banks'] if b['source_id']==task['target_id']
                        and b['scenario']==task['scenario'])
            metrics.update(collision_recall(bank.candidates,pool['pool_collisions'],result['queries']))
            output = {'specification':specification,'result':result,'metrics':metrics,
                      'query_ledger':ledger.records,'cost':{'execution_mode':'replay',
                      'new_physical_calls':0,'equivalent_queries':128,
                      'wall_s':time.perf_counter()-started,
                      'equivalent_execution_s':metrics['equivalent_execution_s']}}
            write_json(destination,output)
    return {**task,'method':f'wide_{mode}','path':str(destination.relative_to(ROOT)),**output['metrics']}


def run():
    if read_json(OUTPUT/'support/report.json')['status'] != 'complete':
        raise ValueError('Wide-support training and paired validation must finish first')
    contract = {'pilot':experiment_contract(),'training':training_contract(),
                'search_runner':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute,task,mode,contract)
                   for mode in ('source_global','source_local') for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%8==0:
                print('wide_support_search',len(rows),'/ 80',flush=True)
    rows.sort(key=lambda r:(r['method'],r['target_id'],r['scenario'],r['algorithm_seed']))
    if len(rows)!=80 or any(r['queries']!=DEFAULT.budget for r in rows):
        raise ValueError('Wide-support search requires all 80 complete development runs')
    matched = read_json(OUTPUT/'consistency/search/report.json')['runs']
    original = [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs']
                if r['method'] in ('fsbo','pooled_ft')]
    combined = rows+matched+original
    stats = method_summary(combined)
    comparisons = []
    for mode in ('source_global','source_local'):
        for baseline in (f'matched_{mode}','fsbo','pooled_ft'):
            for metric in ('unique_collisions','collision_recall','discovery_auc'):
                paired = [r for r in combined if r['method'] in (f'wide_{mode}',baseline) and r[metric] is not None]
                comparisons.append(paired_lineage_bootstrap(paired,f'wide_{mode}',baseline,metric))
    result = {'status':'complete','runs':rows,'summary':stats,'comparisons':comparisons,
              'contract':contract,'new_physical_calls':0,'equivalent_queries':10240,
              'original_pilot_gate_changed':False,'gate_admission_claimed':False}
    write_json(REPORT/'report.json',result)
    lines = ['# 支持集规模扩展后的开发搜索诊断','',
             '80项开发回放、10,240次等效查询、新增物理调用为零。保持原候选、初始化和128次预算。',
             'wide为8/16/32/64/128支持集训练；matched为4/8/16匹配训练。原B门槛保持未通过。','',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现 AUC |','|---|---:|---:|---:|']
    for mode,s in stats.items():
        lines.append(f"| {mode} | {s['unique_collisions']['mean']:.3f} | {s['collision_recall']['mean']:.2%} | {s['discovery_auc']['mean']:.3f} |")
    lines.extend(['','全体系统、种子和配对区间保留在report.json。该诊断不自动启动C/D/E。',''])
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('wide_support_search_complete',flush=True)
    return result


if __name__ == '__main__':
    run()
