"""Historical forest replay with source predictions held out by coordinate."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from methods.common.response import source_weights

from . import discovery_tree_screening as original
from .collision_metrics import collision_recall, summarize
from .discovery_config import OUTPUT, read_json, write_json
from .executions import ExecutionBank
from .protocol import DATA_ROOT
from .query import QueryLedger, run_isolated
from .discovery_tree_screening import histories


REPORT = OUTPUT / 'tree_crossfit'
METHODS = original.METHODS


def coordinate_folds():
    return list(KFold(n_splits=5, shuffle=True, random_state=1301).split(np.arange(512)))


def heldout_probability(x, labels, train, test, parameters):
    forest = ExtraTreesClassifier(**parameters).fit(x[train], labels[train])
    probability = original.positive_probability(forest, x[test])
    independent = original.leaf_probability(forest, x[train], labels[train], x[test])
    np.testing.assert_allclose(probability, independent, atol=2e-12, rtol=0)
    return probability


def protocol():
    parent = original.protocol()
    return {'purpose': 'source-coordinate cross-fitting; same historical forest screening',
        'original_protocol': parent, 'methods': list(METHODS),
        'folds': [{'train': train.tolist(), 'test': test.tolist()} for train, test in coordinate_folds()],
        'fold_rule': 'five shuffled KFold splits, seed 1301; common coordinate indices for all sources',
        'source_forest': parent['source_forest'], 'target_forest': parent['target_forest'],
        'budget': 128, 'initial_count': 8, 'seed': 1301, 'historical_probability_mass': .1,
        'source_training_scope': 'each prediction excludes its entire coordinate fold; own system excluded from inputs',
        'screening': parent['screening'],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit_source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ('discovery_tree_crossfit_audit.py', 'discovery_tree_screening_audit.py')},
        'new_development_queries': 0, 'new_physical_calls': 0, 'new_neural_network_training': False}


def prepare():
    prior = read_json(original.REPORT / 'report_checks.json')
    assert prior['status'] == 'passed' and prior['screening_passed']
    assert prior['report_sha256'] == hashlib.sha256((original.REPORT / 'report.json').read_bytes()).hexdigest()
    specification = protocol()
    path = REPORT / 'protocol.json'
    if path.exists():
        assert read_json(path) == specification
    else:
        write_json(path, specification)
    costs = []
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            destination = REPORT / 'sources' / f'{kind}.npz'
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    assert json.loads(str(saved['protocol'])) == specification
                costs.extend(read_json(destination.with_suffix('.json'))['costs'])
                continue
            histories = original.histories(kind)
            probability = np.empty((8, 512))
            for source, archive in enumerate(histories):
                coverage = np.zeros(512, dtype=int)
                for fold, split in enumerate(specification['folds']):
                    train, test = np.asarray(split['train']), np.asarray(split['test'])
                    started = time.perf_counter()
                    assert not set(train) & set(test) and set(train) | set(test) == set(range(512))
                    probability[source, test] = heldout_probability(archive.x, archive.collision,
                        train, test, specification['source_forest'])
                    coverage[test] += 1
                    costs.append({'scenario': kind, 'source_id': archive.source_id, 'fold': fold,
                        'training_count': len(train), 'prediction_count': len(test),
                        'training_and_leaf_check_s': time.perf_counter() - started})
                np.testing.assert_array_equal(coverage, np.ones(512, dtype=int))
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, protocol=np.asarray(json.dumps(specification)),
                names=np.asarray([a.source_id for a in histories]), probabilities=probability)
            write_json(destination.with_suffix('.json'), {'status': 'passed', 'protocol': specification,
                'costs': [r for r in costs if r['scenario'] == kind]})
    write_json(REPORT / 'preparation.json', {'status': 'passed', 'source_fold_fits': len(costs),
        'training_and_leaf_check_s': sum(r['training_and_leaf_check_s'] for r in costs),
        'each_source_coordinate_predicted_once': True, 'train_test_coordinates_disjoint': True,
        'new_development_queries': 0, 'new_physical_calls': 0})
    return specification


def public_inputs(kind, pseudo_target, mode):
    history = original.histories(kind)
    target = next(a for a in history if a.source_id == pseudo_target)
    positions = [i for i, a in enumerate(history) if a.source_id != pseudo_target]
    sources = [history[i] for i in positions]
    with np.load(REPORT / 'sources' / f'{kind}.npz', allow_pickle=False) as saved:
        assert json.loads(str(saved['protocol'])) == read_json(REPORT / 'protocol.json')
        assert saved['names'].tolist() == [a.source_id for a in history]
        response = saved['probabilities'][positions].T
    prior = response @ source_weights(sources)
    inputs = target.x
    if mode == 'pooled_response':
        inputs = np.column_stack((inputs, prior))
    elif mode == 'source_response':
        inputs = np.column_stack((inputs, response))
    return target, inputs, prior, [a.source_id for a in sources]


def execute(kind, pseudo_target, mode):
    specification = read_json(REPORT / 'protocol.json')
    path = REPORT / 'runs' / kind / pseudo_target / f'{mode}.json'
    if path.exists():
        old = read_json(path)
        assert old['status'] == 'complete' and old['protocol'] == specification
        return old
    started = time.perf_counter()
    target, inputs, prior, names = public_inputs(kind, pseudo_target, mode)
    bank = ExecutionBank(DATA_ROOT / 'physical/history' / kind / pseudo_target)
    assert target.scenarios == bank.candidates and pseudo_target not in names and len(names) == 7
    ledger = QueryLedger(bank.candidates, 128, bank)
    try:
        result = run_isolated(original.select, {'candidates': bank.candidates, 'inputs': inputs,
            'history_probability': prior, 'initial_indices': np.random.default_rng(1301).permutation(512)[:8].tolist(),
            'mode': mode, 'specification': {'target_forest': specification['target_forest']}}, ledger)
        metrics = {**summarize(bank.candidates, result['queries'], ledger.trajectories),
            **collision_recall(bank.candidates, target.collision.tolist(), result['queries'])}
        row = {'status': 'complete', 'protocol': specification, 'scenario': kind,
            'pseudo_target': pseudo_target, 'lineage': target.lineage, 'method': mode,
            'source_ids': names, 'input_dimension': inputs.shape[1], 'result': result,
            'query_ledger': ledger.records, 'metrics': metrics, 'wall_s': time.perf_counter() - started,
            'historical_replay_queries': 128, 'new_development_queries': 0, 'new_physical_calls': 0}
        write_json(path, row)
        return row
    except Exception:
        with (REPORT / 'failed_attempts.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'scenario': kind, 'pseudo_target': pseudo_target, 'method': mode,
                'query_ledger': ledger.records, 'wall_s': time.perf_counter() - started,
                'error': traceback.format_exc(), 'new_development_queries': 0, 'new_physical_calls': 0}) + '\n')
        raise


def smoke():
    from .discovery_tree_crossfit_audit import audit_rows

    assert read_json(REPORT / 'protocol.json') == protocol()
    assert read_json(REPORT / 'audit/sources.json')['status'] == 'passed'
    rows = [execute('cut_in', original.histories('cut_in')[0].source_id, mode) for mode in METHODS]
    write_json(REPORT / 'smoke.json', {'status': 'complete', 'runs': rows})
    write_json(REPORT / 'audit/smoke.json', audit_rows(rows))


def run():
    assert read_json(REPORT / 'protocol.json') == protocol()
    checked = read_json(REPORT / 'audit/smoke.json')
    assert checked['status'] == 'passed' and checked['runs'] == 3 and checked['independent_full_pool_choices'] == 9
    tasks = [(kind, a.source_id, mode) for kind in ('cut_in', 'cut_out')
             for a in original.histories(kind) for mode in METHODS]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute, *task) for task in tasks]):
            rows.append(future.result())
            print('tree_crossfit', len(rows), '/ 48', flush=True)
    assert len(rows) == 48
    write_json(REPORT / 'run_index.json', {'status': 'complete', 'protocol': protocol(),
        'runs': [{'path': str((REPORT / 'runs' / r['scenario'] / r['pseudo_target'] / f"{r['method']}.json").relative_to(OUTPUT)),
            **{k: r[k] for k in ('scenario', 'pseudo_target', 'lineage', 'method')},
            **r['metrics']} for r in rows], 'historical_replay_queries': 6144,
        'new_development_queries': 0, 'new_physical_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'smoke', 'run'))
    globals()[parser.parse_args().stage]()
