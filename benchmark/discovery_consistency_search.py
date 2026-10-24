"""Paired development discovery diagnosis for consistently trained source kernels."""

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
from .discovery_consistency import model_location
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, DEFAULT, ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features


REPORT = OUTPUT / 'consistency/search'
RUNS = ROOT / 'results/gcrt/collision/discovery/consistency/runs'


def execute(task, mode, contract):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, development = load_construction(task['scenario'])
        path = model_location(mode, task['scenario'])
        encoder, metadata = load_encoder(path)
        if (metadata['specification']['history_fingerprints'] != archive_fingerprints(history)
                or metadata['specification']['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('Matched training checkpoint belongs to different historical/development archives')
        bank = ExecutionBank(DATA_ROOT / 'physical/development' / task['scenario'] / task['target_id'])
        specification = {'task': task, 'mode': mode, 'contract': contract,
                         'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'checkpoint_step': metadata['step'], 'bank_contract': bank.manifest['contract'],
                         'candidate_ids': [s.scenario_id for s in bank.candidates],
                         'budget': DEFAULT.budget, 'initial_count': DEFAULT.initial_count}
        destination = RUNS / mode / task['target_id'] / task['scenario'] / f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != specification or output['result']['actual_queries'] != DEFAULT.budget:
                raise ValueError('Existing consistency search belongs to a different complete experiment')
        else:
            started = time.perf_counter()
            independent, bandwidth = MODES[mode]
            model = discovery_posterior(encoder, features(bank.candidates), history,
                                        independent=independent, bandwidth=bandwidth)
            ledger = QueryLedger(bank.candidates, DEFAULT.budget, bank)
            result = run_isolated(select, {'model': model, 'candidates': bank.candidates,
                                  'initial_indices': initial_indices(len(bank.candidates), DEFAULT.initial_count,
                                                                     task['algorithm_seed']),
                                  'budget': DEFAULT.budget, 'horizon': 1, 'greedy': True}, ledger)
            metrics = summarize(bank.candidates, result['queries'], ledger.trajectories)
            diagnosis = read_json(OUTPUT / 'diagnosis/report.json')
            pool = next(b for b in diagnosis['banks'] if b['source_id'] == task['target_id']
                        and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates, pool['pool_collisions'], result['queries']))
            output = {'specification': specification, 'result': result, 'metrics': metrics,
                      'query_ledger': ledger.records, 'cost': {'execution_mode': 'replay',
                      'new_physical_calls': 0, 'equivalent_queries': DEFAULT.budget,
                      'wall_s': time.perf_counter()-started,
                      'equivalent_execution_s': metrics['equivalent_execution_s']}}
            write_json(destination, output)
    return {**task, 'method': f'matched_{mode}', 'path': str(destination.relative_to(ROOT)), **output['metrics']}


def run():
    validation = read_json(OUTPUT / 'consistency/report.json')
    if validation['status'] != 'complete' or validation['protocol']['training_contract'] != training_contract():
        raise ValueError('Matched training and paired prediction validation must complete first')
    contract = {'pilot': experiment_contract(), 'training': training_contract(),
                'search_runner': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute, task, mode, contract)
                   for mode in ('source_global', 'source_local') for task in development_tasks()]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 8 == 0:
                print('consistency_search', len(rows), '/ 80', flush=True)
    rows.sort(key=lambda r: (r['method'], r['target_id'], r['scenario'], r['algorithm_seed']))
    if len(rows) != 80 or any(r['queries'] != 128 for r in rows):
        raise ValueError('The matched search diagnosis requires 80 complete paired runs')
    original = read_json(OUTPUT / 'pilot/run_index.json')['runs']
    reference = [r for r in original if r['method'] in ('fsbo', 'pooled_ft', 'source_global', 'source_local')]
    stats = method_summary(rows + reference)
    comparisons = []
    for mode in ('source_global', 'source_local'):
        for baseline in (mode, 'fsbo', 'pooled_ft'):
            for metric in ('unique_collisions', 'collision_recall', 'discovery_auc'):
                paired = [r for r in rows + reference if r['method'] in (f'matched_{mode}', baseline)
                          and r[metric] is not None]
                comparisons.append(paired_lineage_bootstrap(paired, f'matched_{mode}', baseline, metric))
    result = {'status': 'complete', 'runs': rows, 'summary': stats, 'comparisons': comparisons,
              'original_pilot_gate_changed': False, 'original_gate_status': read_json(OUTPUT/'pilot/gate.json')['status'],
              'contract': contract, 'new_physical_calls': 0, 'equivalent_queries': 80*128,
              'gate_admission_claimed': False}
    write_json(REPORT / 'report.json', result)
    lines = ['# 匹配训练后的开发碰撞搜索诊断', '',
             '80项匹配开发回放，10,240次等效查询，新增物理调用为零。',
             '使用同一候选、8次初始化、128次总预算和五个选例种子；原520项运行与失败门槛保持原样。', '',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现 AUC |', '|---|---:|---:|---:|']
    for mode, summary in stats.items():
        lines.append(f"| {mode} | {summary['unique_collisions']['mean']:.3f} | {summary['collision_recall']['mean']:.2%} | {summary['discovery_auc']['mean']:.3f} |")
    lines.extend(['', '该实验用于定位训练/推断不一致的实际影响，不改变原5个百分点门槛，不启动C/D/E。',
                  '全部谱系差值和配对分层区间见同目录report.json。', ''])
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result


if __name__ == '__main__':
    run()
