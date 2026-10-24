"""Reconstruct held-out source folds and metered historical forest choices."""

import argparse
import hashlib
import json
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from methods.common.response import source_weights

from . import discovery_tree_crossfit as study
from . import discovery_tree_screening_audit as validation
from .discovery_config import OUTPUT, read_json, write_json


def inputs_for(row):
    kind, pseudo, mode = row['scenario'], row['pseudo_target'], row['method']
    assert mode in study.METHODS
    history = study.histories(kind)
    target = next(a for a in history if a.source_id == pseudo)
    sources = [a for a in history if a.source_id != pseudo]
    with np.load(study.REPORT / 'sources' / f'{kind}.npz', allow_pickle=False) as saved:
        assert json.loads(str(saved['protocol'])) == read_json(study.REPORT / 'protocol.json')
        assert saved['names'].tolist() == [a.source_id for a in history]
        response = saved['probabilities'][[i for i, a in enumerate(history) if a.source_id != pseudo]].T
    prior = response @ source_weights(sources)
    x = target.x
    if mode == 'pooled_response':
        x = np.column_stack((x, prior))
    elif mode == 'source_response':
        x = np.column_stack((x, response))
    assert row['source_ids'] == [a.source_id for a in sources] and pseudo not in row['source_ids']
    assert len(sources) == 7 and row['input_dimension'] == x.shape[1]
    return target, x, prior


def sources():
    started = time.perf_counter()
    specification = read_json(study.REPORT / 'protocol.json')
    assert specification == study.protocol()
    shuffled = np.arange(512)
    np.random.RandomState(1301).shuffle(shuffled)
    independently_derived = []
    offset = 0
    for size in (103, 103, 102, 102, 102):
        test = np.sort(shuffled[offset:offset + size])
        train = np.asarray(sorted(set(range(512)) - set(test)))
        independently_derived.append({'train': train.tolist(), 'test': test.tolist()})
        offset += size
    assert independently_derived == specification['folds']
    checks, maximum = [], 0.
    for kind in ('cut_in', 'cut_out'):
        history = study.histories(kind)
        with np.load(study.REPORT / 'sources' / f'{kind}.npz', allow_pickle=False) as saved:
            assert json.loads(str(saved['protocol'])) == specification
            assert saved['names'].tolist() == [a.source_id for a in history]
            for source, archive in enumerate(history):
                count = np.zeros(512, dtype=int)
                for number, split in enumerate(independently_derived):
                    train, test = np.asarray(split['train']), np.asarray(split['test'])
                    assert not set(train) & set(test)
                    forest = ExtraTreesClassifier(n_estimators=64, min_samples_leaf=4, max_features=1.,
                        bootstrap=False, n_jobs=1, random_state=1301).fit(archive.x[train], archive.collision[train])
                    predicted = np.zeros(len(test))
                    for tree in forest.estimators_:
                        leaves = tree.apply(archive.x[train])
                        hits = np.bincount(leaves, weights=archive.collision[train], minlength=tree.tree_.node_count)
                        sizes = np.bincount(leaves, minlength=tree.tree_.node_count)
                        predicted_leaves = tree.apply(archive.x[test])
                        assert np.all(sizes[predicted_leaves] > 0)
                        predicted += hits[predicted_leaves] / sizes[predicted_leaves] / 64
                    error = float(np.max(np.abs(predicted - saved['probabilities'][source, test])))
                    maximum = max(maximum, error)
                    assert error < 2e-12
                    count[test] += 1
                    checks.append({'scenario': kind, 'source_id': archive.source_id, 'fold': number,
                        'training_count': len(train), 'prediction_count': len(test), 'maximum_probability_error': error})
                np.testing.assert_array_equal(count, np.ones(512, dtype=int))
    result = {'status': 'passed', 'source_fold_fits': len(checks), 'heldout_source_predictions': 8192,
        'fold_indices_independently_reconstructed': True, 'source_coordinate_exclusion_checked': True,
        'maximum_probability_error': maximum, 'checks': checks, 'wall_s': time.perf_counter() - started,
        'new_development_queries': 0, 'new_physical_calls': 0}
    write_json(study.REPORT / 'audit/sources.json', result)
    return result


def audit_rows(rows):
    started = time.perf_counter()
    checks = [validation.check_row(row, inputs_for=inputs_for) for row in rows]
    return {'status': 'passed', 'runs': len(rows), 'raw_feedbacks_checked': len(rows) * 128,
        'independent_full_pool_choices': len(rows) * 3,
        'maximum_probability_error': max(p['probability_error'] for c in checks for p in c['prefixes']),
        'source_and_coordinate_exclusion_checked': True, 'canonical_feedback_objects_compared': True,
        'checks': checks, 'wall_s': time.perf_counter() - started,
        'new_development_queries': 0, 'new_physical_calls': 0}


def complete():
    specification = read_json(study.REPORT / 'protocol.json')
    index = read_json(study.REPORT / 'run_index.json')
    assert specification == study.protocol() == index['protocol']
    assert read_json(study.REPORT / 'audit/sources.json')['source_fold_fits'] == 80
    expected = {(kind, a.source_id, method) for kind in ('cut_in', 'cut_out')
                for a in study.histories(kind) for method in study.METHODS}
    assert len(index['runs']) == 48
    assert {(r['scenario'], r['pseudo_target'], r['method']) for r in index['runs']} == expected
    paths = [OUTPUT / row['path'] for row in index['runs']]
    identities = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    rows = [read_json(path) for path in paths]
    assert all(row['status'] == 'complete' and row['protocol'] == specification for row in rows)
    result = audit_rows(rows)
    negative = validation.negative_controls(rows[0], inputs_for=inputs_for)
    assert negative['rejected_corruptions'] == 10
    assert identities == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result.update(rejected_corruptions=10, original_run_files_unchanged=True)
    write_json(study.REPORT / 'audit/negative_controls.json', negative)
    write_json(study.REPORT / 'audit/report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('sources', 'complete'))
    result = globals()[parser.parse_args().stage]()
    print({k: v for k, v in result.items() if k != 'checks'})
