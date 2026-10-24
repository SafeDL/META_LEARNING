"""Independent forest replay audit with canonical typed query feedback."""

import copy
import hashlib
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from methods.common.response import source_weights

from . import discovery_tree_screening as study
from .collision_metrics import collision_recall
from .discovery_config import OUTPUT, read_json, write_json
from .executions import ExecutionBank
from .protocol import DATA_ROOT
from .schema import QueryFeedback


def inputs_for(row):
    kind, pseudo, mode = row['scenario'], row['pseudo_target'], row['method']
    assert mode in study.METHODS
    history = study.histories(kind)
    target = next(a for a in history if a.source_id == pseudo)
    sources = [a for a in history if a.source_id != pseudo]
    with np.load(study.REPORT / 'sources' / f'{kind}.npz', allow_pickle=False) as saved:
        assert saved['names'].tolist() == [a.source_id for a in history]
        positions = [i for i, a in enumerate(history) if a.source_id != pseudo]
        response = saved['probabilities'][positions].T
    prior = response @ source_weights(sources)
    x = target.x
    if mode == 'pooled_response':
        x = np.column_stack((x, prior))
    elif mode == 'source_response':
        x = np.column_stack((x, response))
    assert row['source_ids'] == [a.source_id for a in sources] and pseudo not in row['source_ids']
    assert row['input_dimension'] == x.shape[1] and len(sources) == 7
    return target, x, prior


def check_row(row, *, inputs_for=inputs_for):
    target, x, prior = inputs_for(row)
    queries = row['result']['queries']
    ledger = row['query_ledger']
    selected = [q['index'] for q in queries]
    assert len(queries) == len(ledger) == len(set(selected)) == 128
    assert row['result']['selected_indices'] == selected
    assert selected[:8] == np.random.default_rng(1301).permutation(512)[:8].tolist()
    bank = ExecutionBank(DATA_ROOT / 'physical/history' / row['scenario'] / row['pseudo_target'])
    assert bank.candidates == target.scenarios
    canonical_queries = []
    for number, (q, record) in enumerate(zip(queries, ledger), 1):
        assert record['call'] == number and record['status'] == 'complete' and record['index'] == q['index']
        assert record['purpose'] == ('initial' if number <= 8 else 'target') and record['account'] == 'target'
        measured = bank(bank.candidates[q['index']]).feedback
        disclosed = QueryFeedback.from_dict(q['feedback'])
        assert disclosed == QueryFeedback.from_dict(record['feedback']) == measured
        assert disclosed.collision == bool(target.collision[q['index']])
        np.testing.assert_allclose(q['prequery_history_probability'], prior[q['index']], atol=2e-12, rtol=0)
        assert 0 <= q['prequery_tree_probability'] <= 1
        np.testing.assert_allclose(q['prequery_probability'], .9 * q['prequery_tree_probability']
                                  + .1 * prior[q['index']], atol=2e-12, rtol=0)
        canonical_queries.append({**q, 'feedback': disclosed.to_dict()})
    expected_recall = collision_recall(bank.candidates, target.collision.tolist(), canonical_queries)
    assert all(row['metrics'][key] == value for key, value in expected_recall.items())
    checks = []
    for number in (9, 65, 128):
        paid = selected[:number - 1]
        y = np.asarray([q['feedback']['collision'] for q in queries[:number - 1]])
        forest = ExtraTreesClassifier(n_estimators=64, min_samples_leaf=2, max_features=1.,
            bootstrap=False, n_jobs=1, random_state=1301 + len(paid)).fit(x[paid], y)
        independent = np.zeros(512)
        for tree in forest.estimators_:
            leaves = tree.apply(x[paid])
            hits = np.bincount(leaves, weights=y, minlength=tree.tree_.node_count)
            sizes = np.bincount(leaves, minlength=tree.tree_.node_count)
            predicted_leaves = tree.apply(x)
            assert np.all(sizes[predicted_leaves] > 0)
            independent += hits[predicted_leaves] / sizes[predicted_leaves] / 64
        priority = .9 * independent + .1 * prior
        remaining = sorted(set(range(512)) - set(paid))
        maximum = max(priority[i] for i in remaining)
        chosen = min((i for i in remaining if priority[i] >= maximum - 1e-12),
                     key=lambda i: bank.candidates[i].scenario_id)
        query = queries[number - 1]
        assert chosen == query['index']
        error = abs(float(priority[chosen]) - query['prequery_probability'])
        assert error < 2e-12
        checks.append({'query': number, 'probability_error': error})
    return {'scenario': row['scenario'], 'pseudo_target': row['pseudo_target'], 'method': row['method'],
            'prefixes': checks}


def negative_controls(original, *, inputs_for=inputs_for):
    changed = []

    def edit(name, mutate):
        row = copy.deepcopy(original)
        mutate(row)
        changed.append((name, row))

    edit('collision_disclosure', lambda r: r['result']['queries'][0]['feedback'].__setitem__('collision',
        not r['result']['queries'][0]['feedback']['collision']))

    def wrong_measured_label(r):
        for feedback in (r['result']['queries'][0]['feedback'], r['query_ledger'][0]['feedback']):
            feedback['collision'] = not feedback['collision']
    edit('self_consistent_wrong_measured_label', wrong_measured_label)
    edit('source_leakage', lambda r: r['source_ids'].__setitem__(0, r['pseudo_target']))
    edit('response_dimension', lambda r: r.__setitem__('input_dimension', r['input_dimension'] + 1))
    edit('budget', lambda r: r['result']['queries'].pop())
    edit('duplicate', lambda r: r['result']['queries'][1].__setitem__('index', r['result']['queries'][0]['index']))
    edit('common_initialization', lambda r: r['result']['queries'].reverse())
    edit('history_probability', lambda r: r['result']['queries'][0].__setitem__('prequery_history_probability', .123456789))
    edit('priority_probability', lambda r: r['result']['queries'][64].__setitem__('prequery_probability', .123456789))

    def self_consistent_probability(r):
        q = r['result']['queries'][64]
        q['prequery_tree_probability'] = .123456789
        q['prequery_probability'] = .9 * q['prequery_tree_probability'] + .1 * q['prequery_history_probability']
    edit('self_consistent_wrong_model_probability', self_consistent_probability)
    rejected = []
    for name, row in changed:
        try:
            check_row(row, inputs_for=inputs_for)
        except (AssertionError, ValueError):
            rejected.append(name)
        else:
            raise AssertionError(f'Failed to reject damaged forest record: {name}')
    return {'status': 'passed', 'rejected_corruptions': len(rejected), 'cases': rejected,
            'positive_control_already_in_full_matrix': True}


def audit():
    started = time.perf_counter()
    index = read_json(study.REPORT / 'run_index.json')
    assert index['protocol'] == study.protocol() and index['status'] == 'complete'
    expected = {(kind, a.source_id, m) for kind in ('cut_in', 'cut_out')
                for a in study.histories(kind) for m in study.METHODS}
    assert len(index['runs']) == len(expected) == 48
    assert {(r['scenario'], r['pseudo_target'], r['method']) for r in index['runs']} == expected
    paths = [OUTPUT / r['path'] for r in index['runs']]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    checks, rows = [], []
    for path in paths:
        row = read_json(path)
        assert row['protocol'] == index['protocol'] and row['status'] == 'complete'
        checks.append(check_row(row))
        rows.append(row)
    damaged = negative_controls(rows[0])
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(study.REPORT / 'audit/negative_controls.json', damaged)
    result = {'status': 'passed', 'runs': 48, 'raw_feedbacks_checked': 6144,
        'independent_full_pool_choices': 144,
        'maximum_probability_error': max(p['probability_error'] for c in checks for p in c['prefixes']),
        'canonical_feedback_objects_compared': True, 'rejected_corruptions': damaged['rejected_corruptions'],
        'original_run_files_unchanged': True, 'checks': checks, 'wall_s': time.perf_counter() - started,
        'new_development_queries': 0, 'new_physical_calls': 0}
    write_json(study.REPORT / 'audit/report.json', result)
    return result


if __name__ == '__main__':
    print(audit())
