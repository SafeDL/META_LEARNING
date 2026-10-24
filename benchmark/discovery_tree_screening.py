"""Isolated historical replay with physical, pooled and per-source tree inputs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from controllers.highway.catalog import SUTS
from methods.common.response import archive_fingerprints, source_weights

from .collision_metrics import collision_recall, summarize
from .discovery_config import OUTPUT, read_json, write_json
from .executions import ExecutionBank
from .protocol import DATA_ROOT
from .query import QueryLedger, run_isolated
from .scenarios import sample
from .storage import load_archive


REPORT = OUTPUT / 'tree_screening'
METHODS = ('physical', 'pooled_response', 'source_response')


def histories(kind):
    archives = [load_archive(p) for p in sorted((DATA_ROOT / 'history' / kind).glob('*.npz'))]
    expected = {s.sut_id for s in SUTS if s.role == 'history'}
    coordinates = sample(kind, 512, 'history')
    assert len(archives) == 8 and {a.source_id for a in archives} == expected
    assert all(a.role == 'history' and a.scenarios == coordinates and a.provenance['qualified'] for a in archives)
    return archives


def protocol():
    return {'purpose': 'historical expression screening; no development or final targets',
        'methods': list(METHODS), 'candidate_count': 512, 'budget': 128, 'initial_count': 8,
        'seed': 1301, 'source_forest': {'n_estimators': 64, 'min_samples_leaf': 4,
                                     'max_features': 1., 'bootstrap': False, 'n_jobs': 1, 'random_state': 1301},
        'target_forest': {'n_estimators': 64, 'min_samples_leaf': 2, 'max_features': 1.,
                         'bootstrap': False, 'n_jobs': 1, 'random_state': '1301 + paid_count'},
        'historical_probability_mass': .1, 'feedback': 'queried collision Boolean only',
        'teacher_scope': 'each source forest uses that source only; own source excluded from replay inputs',
        'screening': 'source response R128 strictly higher, N128 and discovery AUC no lower than both controls; '
                     'at least two positive historical lineages versus each control',
        'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'history_fingerprints': {kind: archive_fingerprints(histories(kind)) for kind in ('cut_in', 'cut_out')},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_physical_calls': 0, 'new_development_queries': 0, 'new_neural_network_training': False}


def positive_probability(model, x):
    if True not in model.classes_:
        return np.zeros(len(x))
    return model.predict_proba(x)[:, int(np.flatnonzero(model.classes_ == True)[0])]


def leaf_probability(model, training_x, training_y, x):
    predictions = []
    for estimator in model.estimators_:
        leaves = estimator.apply(training_x)
        counts = np.bincount(leaves, minlength=estimator.tree_.node_count)
        hits = np.bincount(leaves, weights=training_y, minlength=estimator.tree_.node_count)
        test_leaves = estimator.apply(x)
        assert np.all(counts[test_leaves] > 0)
        predictions.append(hits[test_leaves] / counts[test_leaves])
    return np.mean(predictions, axis=0)


def prepare():
    assert read_json(OUTPUT / 'converged_search/report_checks.json')['status'] == 'passed'
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
            archives = histories(kind)
            predictions = []
            for archive in archives:
                started = time.perf_counter()
                model = ExtraTreesClassifier(**specification['source_forest']).fit(archive.x, archive.collision)
                probability = positive_probability(model, archive.x)
                independent = leaf_probability(model, archive.x, archive.collision, archive.x)
                np.testing.assert_allclose(probability, independent, atol=2e-12)
                predictions.append(probability)
                costs.append({'scenario': kind, 'source_id': archive.source_id,
                    'training_and_leaf_check_s': time.perf_counter() - started,
                    'leaf_probability_maximum_error': float(np.max(np.abs(probability - independent)))})
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, protocol=np.asarray(json.dumps(specification)),
                names=np.asarray([a.source_id for a in archives]), probabilities=np.asarray(predictions))
            write_json(destination.with_suffix('.json'), {'status': 'passed', 'protocol': specification,
                'costs': [c for c in costs if c['scenario'] == kind]})
    write_json(REPORT / 'preparation.json', {'status': 'passed', 'sources': 16,
        'source_training_and_check_s': sum(c['training_and_leaf_check_s'] for c in costs),
        'maximum_leaf_probability_error': max(c['leaf_probability_maximum_error'] for c in costs),
        'new_physical_calls': 0, 'new_development_queries': 0})
    return specification


def public_inputs(kind, pseudo_target, mode):
    archives = histories(kind)
    target = next(a for a in archives if a.source_id == pseudo_target)
    positions = [i for i, a in enumerate(archives) if a.source_id != pseudo_target]
    sources = [archives[i] for i in positions]
    with np.load(REPORT / 'sources' / f'{kind}.npz', allow_pickle=False) as saved:
        assert json.loads(str(saved['protocol'])) == read_json(REPORT / 'protocol.json')
        assert saved['names'].tolist() == [a.source_id for a in archives]
        response = saved['probabilities'][positions].T
    prior = response @ source_weights(sources)
    inputs = target.x
    if mode == 'pooled_response':
        inputs = np.column_stack((inputs, prior))
    elif mode == 'source_response':
        inputs = np.column_stack((inputs, response))
    return target, inputs, prior, [a.source_id for a in sources]


def select(candidates, oracle, *, inputs, history_probability, initial_indices, mode, specification):
    selected, labels, queries = [], [], []
    remaining = set(range(len(candidates)))
    for step in range(128):
        started = time.perf_counter()
        if step:
            options = {**specification['target_forest'], 'random_state': 1301 + step}
            model = ExtraTreesClassifier(**options).fit(inputs[selected], labels)
            target_probability = positive_probability(model, inputs)
        else:
            target_probability = np.full(len(candidates), .5)
        adaptation_s = time.perf_counter() - started
        started = time.perf_counter()
        probability = .9 * target_probability + .1 * history_probability
        if step < 8:
            index = int(initial_indices[step])
        else:
            maximum = max(probability[i] for i in remaining)
            index = min((i for i in remaining if probability[i] >= maximum - 1e-12),
                        key=lambda i: candidates[i].scenario_id)
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose='initial' if step < 8 else 'target')
        assert feedback.scenario_id == candidates[index].scenario_id
        queries.append({'query': step + 1, 'index': index, 'feedback': feedback.to_dict(),
            'prequery_probability': float(probability[index]),
            'prequery_tree_probability': float(target_probability[index]),
            'prequery_history_probability': float(history_probability[index]),
            'selection_s': selection_s, 'adaptation_s': adaptation_s})
        selected.append(index)
        labels.append(feedback.collision)
        remaining.remove(index)
    return {'method': mode, 'queries': queries, 'selected_indices': selected}


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
    initial = np.random.default_rng(1301).permutation(512)[:8].tolist()
    try:
        result = run_isolated(select, {'candidates': bank.candidates, 'inputs': inputs,
            'history_probability': prior, 'initial_indices': initial, 'mode': mode,
            'specification': {k: specification[k] for k in ('target_forest',)}}, ledger)
        metrics = {**summarize(bank.candidates, result['queries'], ledger.trajectories),
            **collision_recall(bank.candidates, target.collision.tolist(), result['queries'])}
        row = {'status': 'complete', 'protocol': specification, 'scenario': kind,
            'pseudo_target': pseudo_target, 'lineage': target.lineage, 'method': mode,
            'source_ids': names, 'input_dimension': inputs.shape[1], 'result': result,
            'query_ledger': ledger.records, 'metrics': metrics, 'wall_s': time.perf_counter() - started,
            'historical_replay_queries': 128, 'new_physical_calls': 0, 'new_development_queries': 0}
        write_json(path, row)
        return row
    except Exception:
        with (REPORT / 'failed_attempts.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'scenario': kind, 'pseudo_target': pseudo_target, 'method': mode,
                'wall_s': time.perf_counter() - started, 'query_ledger': ledger.records,
                'error': traceback.format_exc(), 'new_physical_calls': 0, 'new_development_queries': 0}) + '\n')
        raise


def audit_records(rows):
    checked, maximum_error = [], 0.
    specification = read_json(REPORT / 'protocol.json')
    for row in rows:
        kind, pseudo, mode = row['scenario'], row['pseudo_target'], row['method']
        target, inputs, prior, names = public_inputs(kind, pseudo, mode)
        assert row['source_ids'] == names and pseudo not in names and len(names) == 7
        bank = ExecutionBank(DATA_ROOT / 'physical/history' / kind / pseudo)
        queries = row['result']['queries']
        selected = [q['index'] for q in queries]
        assert row['protocol'] == specification and row['result']['selected_indices'] == selected
        assert len(selected) == len(set(selected)) == len(row['query_ledger']) == 128
        assert selected[:8] == np.random.default_rng(1301).permutation(512)[:8].tolist()
        for query, ledger in zip(queries, row['query_ledger']):
            assert ledger['status'] == 'complete' and ledger['index'] == query['index']
            assert query['feedback'] == ledger['feedback'] == bank(bank.candidates[query['index']]).feedback.to_dict()
        recall = collision_recall(bank.candidates, target.collision.tolist(), queries)
        for key, value in recall.items():
            assert row['metrics'][key] == value
        prefixes = []
        for number in (9, 65, 128):
            paid = selected[:number - 1]
            labels = np.asarray([q['feedback']['collision'] for q in queries[:number - 1]])
            model = ExtraTreesClassifier(n_estimators=64, min_samples_leaf=2, max_features=1.,
                bootstrap=False, n_jobs=1, random_state=1301 + len(paid)).fit(inputs[paid], labels)
            independent = leaf_probability(model, inputs[paid], labels, inputs)
            np.testing.assert_allclose(independent, positive_probability(model, inputs), atol=2e-12)
            priority = .9 * independent + .1 * prior
            remaining = sorted(set(range(512)) - set(paid))
            maximum = max(priority[i] for i in remaining)
            index = min((i for i in remaining if priority[i] >= maximum - 1e-12),
                        key=lambda i: bank.candidates[i].scenario_id)
            query = queries[number - 1]
            error = abs(priority[index] - query['prequery_probability'])
            maximum_error = max(maximum_error, error)
            assert index == query['index'] and error < 2e-12
            prefixes.append({'query': number, 'probability_error': error})
        checked.append({'scenario': kind, 'pseudo_target': pseudo, 'method': mode, 'prefixes': prefixes})
    return {'status': 'passed', 'runs': len(rows), 'raw_feedbacks_checked': len(rows) * 128,
        'independent_full_pool_choices': len(rows) * 3, 'maximum_probability_error': maximum_error,
        'checks': checked, 'new_development_queries': 0, 'new_physical_calls': 0}


def smoke():
    assert read_json(REPORT / 'protocol.json') == protocol()
    rows = [execute('cut_in', histories('cut_in')[0].source_id, mode) for mode in METHODS]
    write_json(REPORT / 'smoke.json', {'status': 'complete', 'runs': rows})
    write_json(REPORT / 'audit/smoke.json', audit_records(rows))


def run():
    assert read_json(REPORT / 'protocol.json') == protocol()
    checked = read_json(REPORT / 'audit/smoke.json')
    assert checked['status'] == 'passed' and checked['runs'] == 3 and checked['independent_full_pool_choices'] == 9
    tasks = [(kind, a.source_id, m) for kind in ('cut_in', 'cut_out') for a in histories(kind) for m in METHODS]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute, *task) for task in tasks]):
            rows.append(future.result())
            print('tree_screening', len(rows), '/ 48', flush=True)
    assert len(rows) == 48
    write_json(REPORT / 'run_index.json', {'status': 'complete', 'protocol': protocol(),
        'runs': [{'path': str((REPORT / 'runs' / r['scenario'] / r['pseudo_target'] / f"{r['method']}.json").relative_to(OUTPUT)),
            **{k: r[k] for k in ('scenario', 'pseudo_target', 'lineage', 'method')},
            **r['metrics']} for r in rows], 'historical_replay_queries': 6144,
        'new_development_queries': 0, 'new_physical_calls': 0})


def audit():
    index = read_json(REPORT / 'run_index.json')
    assert index['status'] == 'complete' and len(index['runs']) == 48 and index['protocol'] == protocol()
    result = audit_records([read_json(OUTPUT / r['path']) for r in index['runs']])
    write_json(REPORT / 'audit/report.json', result)
    return result


def aggregate(rows, metric):
    groups = {}
    for row in rows:
        if row[metric] is not None:
            groups.setdefault(row['lineage'], []).append(row[metric])
    return sum(np.mean(v) for v in groups.values()) / len(groups) if groups else None


def report():
    index = read_json(REPORT / 'run_index.json')
    checked = read_json(REPORT / 'audit/report.json')
    assert index['protocol'] == protocol() and checked['status'] == 'passed' and checked['runs'] == 48
    rows = index['runs']
    summary = {m: {metric: aggregate([r for r in rows if r['method'] == m], metric)
        for metric in ('unique_collisions', 'collision_recall', 'discovery_auc')} for m in METHODS}
    comparisons = []
    for control in ('physical', 'pooled_response'):
        gains = {lineage: aggregate([r for r in rows if r['lineage'] == lineage and r['method'] == 'source_response'],
            'collision_recall') - aggregate([r for r in rows if r['lineage'] == lineage and r['method'] == control],
            'collision_recall') for lineage in sorted({r['lineage'] for r in rows})}
        a, b = summary['source_response'], summary[control]
        conditions = {'recall_higher': a['collision_recall'] > b['collision_recall'] + 1e-12,
            'count_not_lower': a['unique_collisions'] >= b['unique_collisions'] - 1e-12,
            'auc_not_lower': a['discovery_auc'] >= b['discovery_auc'] - 1e-12,
            'two_positive_lineages': sum(v > 1e-12 for v in gains.values()) >= 2}
        comparisons.append({'control': control, 'conditions': conditions, 'lineage_recall_gains': gains})
    result = {'status': 'complete_audited_historical_screening', 'summary': summary,
        'comparisons': comparisons, 'screening_passed': all(all(c['conditions'].values()) for c in comparisons),
        'historical_runs': 48, 'historical_replay_queries': 6144, 'new_development_queries': 0,
        'new_physical_calls': 0, 'development_admission_claimed': False}
    write_json(REPORT / 'report.json', result)
    lines = ['# 逐来源响应的历史树模型筛选', '',
        '完整48项历史隔离回放、6,144条反馈及144组独立全候选选点核对通过。',
        '每项排除自身来源，只使用其余7个历史森林，不调用旧深核教师或开发数据。', '',
        '| 输入 | 平均去重碰撞数 | 平均召回 | 发现曲线AUC |', '|---|---:|---:|---:|']
    for mode, values in summary.items():
        lines.append(f"| {mode} | {values['unique_collisions']:.6f} | {values['collision_recall']:.6%} | "
                     f"{values['discovery_auc']:.6f} |")
    lines += ['', f"预登记历史筛选：{'通过' if result['screening_passed'] else '未通过'}。",
        '一个选例种子、两个场景的历史筛选不等于开发准入或最终独立验证。',
        '树模型、逐来源特征与随机分区核均不单独宣称原创性；完整对照及成本见各运行与preparation.json。',
        '新增神经网络训练、开发选例及物理调用均为零；未自动进入C/D/E。', '']
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'smoke', 'run', 'audit', 'report'))
    globals()[parser.parse_args().stage]()
