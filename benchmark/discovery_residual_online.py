"""Frozen development search using historical residual-scale estimates."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import time
import traceback

from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.response import archive_fingerprints
from methods.gcrt.joint_training import load_model
from methods.gcrt.residual_search import MODES, select

from . import discovery_joint_response as joint
from . import discovery_physical_mixture as physical
from .collision_metrics import collision_recall, paired_lineage_bootstrap, summarize
from .construction import load_construction
from .discovery import method_summary
from .discovery_config import OUTPUT, development_tasks, read_json, write_json
from .discovery_residual_scale import REPORT as LEARNING, protocol as learning_protocol
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .query import QueryLedger, run_isolated


REPORT = OUTPUT/'residual_online'
RUNS = ROOT/'results/gcrt/collision/discovery/residual_online/runs'
MAIN = 'residual_local'
METHODS = MODES
PREVIOUS = tuple(dict.fromkeys((*physical.PREVIOUS, *physical.METHODS)))
STRONG = (*PREVIOUS, 'residual_target', 'residual_dual_target')


def protocol():
    selection = read_json(LEARNING/'selection.json')
    design = read_json(LEARNING/'online_design.json')
    return {'design': design, 'tasks': development_tasks(), 'methods': list(METHODS), 'main': MAIN,
        'strong_baselines': list(STRONG), 'budget': 128, 'initial_count': 8, 'candidate_count': 512,
        'selected_beta': selection['selected_beta'], 'miss_reduction_threshold': .5, 'local_bandwidth': .3,
        'physical_mass': .1, 'learned_group_prior_mass': .9,
        'target_adaptation': {'steps': 10, 'learning_rate': .003},
        'physical_adaptation': {'steps': 10, 'learning_rate': .01},
        'foundation': physical.protocol(), 'learning_protocol': learning_protocol(),
        'learning_evidence': {p: hashlib.sha256((LEARNING/p).read_bytes()).hexdigest() for p in
                              ('selection.json', 'selection_audit.json', 'online_design.json')},
        'source_files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                         ('benchmark/discovery_residual_online.py', 'methods/gcrt/residual_search.py',
                          'benchmark/discovery_residual_selection_audit.py')},
        'online_supervision': 'paid collision Boolean only; no trajectory access',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'}


def freeze():
    selection = read_json(LEARNING/'selection.json')
    audit = read_json(LEARNING/'selection_audit.json')
    design = read_json(LEARNING/'online_design.json')
    prior_audit = read_json(physical.REPORT/'audit/report.json')
    if (selection['status'] != 'complete' or selection['protocol'] != learning_protocol()
            or audit['status'] != 'passed' or audit['unique_parameter_comparisons'] != 800
            or audit['selected_beta_recomputed'] != selection['selected_beta']
            or prior_audit['status'] != 'passed' or prior_audit['runs'] != 320
            or design['main'] != MAIN or design['methods'] != list(METHODS)
            or design['tasks'] != development_tasks() or design['selected_beta'] != selection['selected_beta']
            or any(design[k] != v for k, v in {'budget': 128, 'initial_count': 8, 'candidate_count': 512,
                                                'environment_seed': 81001, 'method_task_records': 240}.items())):
        raise ValueError('Finish and independently verify historical learning and the preceding matrix first')
    for path, identity in audit['evidence'].items():
        if hashlib.sha256((LEARNING/path).read_bytes()).hexdigest() != identity:
            raise ValueError('Historical scale-selection evidence changed after its independent audit')
    specification = protocol()
    destination = REPORT/'protocol.json'
    if destination.exists() and read_json(destination) != specification:
        raise ValueError('Residual online comparison already frozen differently')
    write_json(destination, specification)
    return specification


def source_priors(kind, candidates, history):
    digest = hashlib.sha256(physical.checkpoint_path(kind, physical.MAIN).read_bytes()).hexdigest()
    saved = joint.read_prior(kind, candidates, history, digest)
    count = len(candidates)
    return {'names': saved['names'], 'weights': saved['weights'],
            'means': saved['means'][:, :count].copy(),
            'covariances': saved['covariances'][:, :count, :count].copy()}


def execute(task, method, specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        started = time.perf_counter()
        history, development = load_construction(task['scenario'])
        pretrained, metadata = load_fsbo(physical.checkpoint_path(task['scenario'], 'physical_fsbo'))
        joint_model, joint_metadata = load_model(physical.checkpoint_path(task['scenario'], physical.MAIN))
        for construction in (metadata, joint_metadata['specification']):
            if (construction['history_fingerprints'] != archive_fingerprints(history)
                    or construction['development_fingerprints'] != archive_fingerprints(development)):
                raise ValueError('Residual-search teachers differ from the allowed construction archives')
        model_loading_s = time.perf_counter()-started
        bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
        setting = {'task': task, 'method': method, 'protocol': specification,
            'history_fingerprints': archive_fingerprints(history), 'bank_contract': bank.manifest['contract'],
            'candidate_ids': [s.scenario_id for s in bank.candidates]}
        destination = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
        if destination.exists():
            output = read_json(destination)
            if output['specification'] != setting or output['result']['actual_queries'] != 128:
                raise ValueError('Saved residual-search run differs from the frozen comparison')
        else:
            constructed = time.perf_counter()
            sources = (None if method in ('residual_target', 'residual_dual_target') else
                       source_priors(task['scenario'], bank.candidates, history))
            construction_s = time.perf_counter()-constructed
            ledger = QueryLedger(bank.candidates, 128, bank)
            public = {'candidates': bank.candidates, 'pretrained': pretrained, 'sources': sources,
                'auxiliary': joint_model if method == 'residual_dual_target' else None,
                'mode': method, 'beta': specification['selected_beta'][task['scenario']],
                'initial_indices': initial_indices(512, 8, task['algorithm_seed']), 'budget': 128,
                'device': specification['device']}
            try:
                result = run_isolated(select, public, ledger)
            except Exception:
                path = REPORT/'failed_runs'/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'specification': setting, 'status': 'failed',
                        'query_ledger': ledger.records, 'paid_attempts': len(ledger.records),
                        'error': traceback.format_exc(), 'wall_s': time.perf_counter()-started,
                        'new_physical_calls': 0}, ensure_ascii=False, allow_nan=False)+'\n')
                raise
            metrics = summarize(bank.candidates, result['queries'], ledger.trajectories)
            pool = next(b for b in read_json(OUTPUT/'diagnosis/report.json')['banks']
                        if b['source_id'] == task['target_id'] and b['scenario'] == task['scenario'])
            metrics.update(collision_recall(bank.candidates, pool['pool_collisions'], result['queries']))
            output = {'specification': setting, 'result': result, 'metrics': metrics,
                'query_ledger': ledger.records, 'cost': {'equivalent_queries': 128, 'new_physical_calls': 0,
                    'model_loading_s': model_loading_s, 'source_construction_s': construction_s,
                    'wall_s': time.perf_counter()-started, 'new_historical_training': False}}
            write_json(destination, output)
    return {**task, 'method': method, 'path': str(destination.relative_to(ROOT)), **output['metrics']}


def previous_rows():
    return physical.previous_rows()+read_json(physical.REPORT/'run_index.json')['runs']


def gate(rows, baselines=None):
    expected = {(t['target_id'], t['scenario'], t['algorithm_seed'], m) for t in development_tasks() for m in METHODS}
    actual = [(r['target_id'], r['scenario'], r['algorithm_seed'], r['method']) for r in rows]
    if len(actual) != 240 or set(actual) != expected:
        raise ValueError('Residual admission requires all 240 complete paired method/task records')
    baselines = previous_rows() if baselines is None else baselines
    required = {(t['target_id'], t['scenario'], t['algorithm_seed'], m) for t in development_tasks() for m in PREVIOUS}
    prior_keys = [(r['target_id'], r['scenario'], r['algorithm_seed'], r['method']) for r in baselines]
    if len(prior_keys) != 880 or set(prior_keys) != required:
        raise ValueError('Every one of the 22 prior references needs all 40 paired tasks')
    combined, denominators = rows+baselines, {}
    for row in combined:
        key = row['target_id'], row['scenario'], row['algorithm_seed']
        if row['queries'] != 128 or denominators.setdefault(key, row['pool_unique_collisions']) != row['pool_unique_collisions']:
            raise ValueError('Compared runs differ in budget or evaluator-only full-pool denominator')
        found, total = row['unique_collisions'], row['pool_unique_collisions']
        reported = row['collision_recall']
        valid_recall = (reported is None if not total else
                        reported is not None and math.isclose(reported, found/total, abs_tol=1e-12, rel_tol=1e-12))
        if not 0 <= found <= min(128, total) or not valid_recall:
            raise ValueError('A collision count or recall is inconsistent with the paid budget or full-pool denominator')
    stats = method_summary(combined)
    baseline = max(STRONG, key=lambda m: (stats[m]['collision_recall']['mean'] if stats[m]['collision_recall']['mean'] is not None else -1,
                   stats[m]['unique_collisions']['mean'], stats[m]['discovery_auc']['mean'], m))
    main, ref = stats[MAIN], stats[baseline]
    recall, reference = main['collision_recall']['mean'], ref['collision_recall']['mean']
    reduction = (recall-reference)/(1-reference) if recall is not None and reference is not None and reference < 1-1e-12 else None
    gains = {k: v-ref['collision_recall']['by_lineage'][k] for k, v in main['collision_recall']['by_lineage'].items()}

    def noninferior(control, metric):
        a, b = main[metric]['mean'], stats[control][metric]['mean']
        return a is not None and b is not None and a >= b-1e-12

    global_recall = stats['residual_global']['collision_recall']['mean']
    dual_recall = stats['residual_dual_target']['collision_recall']['mean']
    conditions = {'remaining_misses_reduced_by_half': reduction is not None and reduction >= .5-1e-12,
        'collision_count_not_lower_than_reference': main['unique_collisions']['mean'] >= ref['unique_collisions']['mean']-1e-12,
        'two_positive_lineages': sum(v > 1e-12 for v in gains.values()) >= 2,
        'local_recall_exceeds_global': recall is not None and global_recall is not None and recall > global_recall+1e-12,
        'local_count_not_lower_than_global': noninferior('residual_global', 'unique_collisions'),
        'recall_not_lower_than_unit_local': noninferior('residual_unit_local', 'collision_recall'),
        'count_not_lower_than_unit_local': noninferior('residual_unit_local', 'unique_collisions'),
        'recall_not_lower_than_unit_global': noninferior('residual_unit_global', 'collision_recall'),
        'count_not_lower_than_unit_global': noninferior('residual_unit_global', 'unique_collisions'),
        'recall_exceeds_same_information_dual_target': recall is not None and dual_recall is not None and recall > dual_recall+1e-12,
        'count_not_lower_than_dual_target': noninferior('residual_dual_target', 'unique_collisions')}
    comparisons = []
    for control in (*STRONG, 'residual_global', 'residual_unit_local', 'residual_unit_global'):
        for metric in ('unique_collisions', 'collision_recall', 'discovery_auc'):
            paired = [r for r in combined if r['method'] in (MAIN, control) and r[metric] is not None]
            if paired:
                comparisons.append(paired_lineage_bootstrap(paired, MAIN, control, metric))
    return {'status': 'passed' if all(conditions.values()) else 'not_passed', 'main': MAIN,
        'strongest_baseline': baseline, 'required_recall': .5*(1+reference) if reference is not None else None,
        'remaining_miss_reduction': reduction, 'conditions': conditions, 'summary': stats,
        'lineage_recall_gains': gains, 'comparisons': comparisons, 'previous_gates_changed': False,
        'scale_learning_claimed_as_standalone_originality': False, 'new_physical_calls': 0}


def smoke():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Residual-search implementation changed after freezing')
    task = development_tasks()[0]
    rows = [execute(task, method, specification) for method in METHODS]
    result = {'status': 'complete', 'runs': rows, 'valid_matrix_runs': 6, 'part_of_full_matrix': True,
              'equivalent_queries': 768, 'new_physical_calls': 0}
    write_json(REPORT/'smoke.json', result)
    return result


def run():
    specification = read_json(REPORT/'protocol.json')
    smoke_audit = read_json(REPORT/'audit/smoke.json')
    negative = read_json(REPORT/'audit/negative_controls.json')
    if (specification != protocol() or smoke_audit['status'] != 'passed' or smoke_audit['runs'] != 6
            or negative['status'] != 'passed' or negative['positive_controls'] != 6 or negative['rejected_corruptions'] < 8):
        raise ValueError('Independently audit every actual smoke run and reject corrupted records before the full matrix')
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, task, method, specification) for method in METHODS for task in development_tasks()]
        try:
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 8 == 0:
                    print('residual_online', len(rows), '/ 240', flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    rows.sort(key=lambda r: (r['method'], r['target_id'], r['scenario'], r['algorithm_seed']))
    write_json(REPORT/'run_index.json', {'status': 'complete', 'protocol': specification, 'runs': rows,
        'equivalent_queries': 30720, 'new_physical_calls': 0})
    write_json(REPORT/'gate.json', gate(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'smoke', 'run', 'audit', 'report'))
    stage = parser.parse_args().stage
    if stage == 'audit':
        from .discovery_residual_online_audit import audit
        audit()
    elif stage == 'report':
        from .discovery_residual_online_report import report
        report()
    else:
        {'freeze': freeze, 'smoke': smoke, 'run': run}[stage]()
