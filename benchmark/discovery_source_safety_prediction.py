"""Source-only held-out safety prediction; no target model or query execution."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from . import discovery_tree_crossfit as context
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_tree_screening_report import mean_by_lineage


REPORT = OUTPUT / 'source_safety_prediction'
VIEWS = ('collision_probability', 'negative_clearance', 'negative_ttc')
PARAMETERS = {'n_estimators': 64, 'min_samples_leaf': 4, 'max_features': 1.,
              'bootstrap': False, 'n_jobs': 1, 'random_state': 1301, 'criterion': 'squared_error'}


def predict_fold(x, auxiliary, train, test):
    forest = ExtraTreesRegressor(**PARAMETERS).fit(x[train], auxiliary[train])
    predicted = forest.predict(x[test])
    independent = np.zeros((len(test), 2))
    for tree in forest.estimators_:
        leaves = tree.apply(x[train])
        sizes = np.bincount(leaves, minlength=tree.tree_.node_count)
        mapped = tree.apply(x[test])
        assert np.all(sizes[mapped] > 0)
        for channel in range(2):
            totals = np.bincount(leaves, weights=auxiliary[train, channel], minlength=tree.tree_.node_count)
            independent[:, channel] += totals[mapped] / sizes[mapped] / 64
    error = float(np.max(np.abs(predicted - independent)))
    assert error < 2e-12
    return predicted, error


def ranking(scores, labels):
    ap = float(average_precision_score(labels, scores)) if labels.any() else None
    auc = float(roc_auc_score(labels, scores)) if labels.any() and not labels.all() else None
    if ap is not None:
        hits, independent = 0, 0.
        for threshold in sorted(set(scores), reverse=True):
            tied = scores == threshold
            hits += int(labels[tied].sum())
            independent += float(labels[tied].sum()) / int(labels.sum()) * hits / int(np.sum(scores >= threshold))
        np.testing.assert_allclose(ap, independent, atol=2e-12, rtol=0)
    if auc is not None:
        a, b = scores[labels], scores[~labels]
        independent = float(np.mean((a[:, None] > b[None, :]) + .5 * (a[:, None] == b[None, :])))
        np.testing.assert_allclose(auc, independent, atol=2e-12, rtol=0)
    return {'average_precision': ap, 'roc_auc': auc}


def run():
    checked = read_json(context.REPORT / 'report_checks.json')
    assert checked['status'] == 'passed' and not checked['screening_passed']
    specification = {'purpose': 'held-out source safety signal diagnostic; not target transfer or online search',
        'source_parameters': PARAMETERS, 'context': context.protocol(), 'source_tasks': 16, 'source_fold_fits': 80,
        'normalization': 'unchanged ResponseArchive.auxiliary clearance and TTC transforms',
        'views': list(VIEWS), 'primary_view': 'negative_clearance',
        'screening': 'primary AP higher and AUC no lower than collision predictor in both scenarios; '
                     'two positive source AP lineages in each scenario',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_development_queries': 0, 'new_physical_calls': 0, 'new_historical_search_queries': 0}
    path = REPORT / 'protocol.json'
    if path.exists():
        assert read_json(path) == specification
    else:
        write_json(path, specification)
    assert not (REPORT / 'report.json').exists()
    started = time.perf_counter()
    rows, folds, errors = [], [], []
    poison_passed = False
    for kind in ('cut_in', 'cut_out'):
        histories = context.histories(kind)
        predictions = np.empty((8, 512, 2))
        with np.load(context.REPORT / 'sources' / f'{kind}.npz', allow_pickle=False) as baseline:
            assert baseline['names'].tolist() == [a.source_id for a in histories]
            for source, archive in enumerate(histories):
                for number, split in enumerate(specification['context']['folds']):
                    train, test = np.asarray(split['train']), np.asarray(split['test'])
                    stamp = time.perf_counter()
                    assert not set(train) & set(test)
                    predicted, error = predict_fold(archive.x, archive.auxiliary, train, test)
                    predictions[source, test] = predicted
                    errors.append(error)
                    if not poison_passed:
                        changed = archive.auxiliary.copy()
                        changed[test] += 17.
                        poisoned, _ = predict_fold(archive.x, changed, train, test)
                        np.testing.assert_array_equal(poisoned, predicted)
                        poison_passed = True
                    folds.append({'scenario': kind, 'source_id': archive.source_id, 'fold': number,
                        'training_count': len(train), 'prediction_count': len(test),
                        'maximum_leaf_prediction_error': error, 'wall_s': time.perf_counter() - stamp})
                scores = (baseline['probabilities'][source], -predictions[source, :, 0], -predictions[source, :, 1])
                for view, score in zip(VIEWS, scores):
                    rows.append({'scenario': kind, 'source_id': archive.source_id, 'lineage': archive.lineage,
                        'view': view, 'collision_coordinates': int(archive.collision.sum()),
                        **ranking(score, archive.collision)})
        destination = REPORT / 'sources' / f'{kind}.npz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, protocol=np.asarray(json.dumps(specification)),
            names=np.asarray([a.source_id for a in histories]), auxiliary_predictions=predictions)
    assert len(folds) == 80 and len(rows) == 48 and poison_passed
    summary, screening = [], []
    for kind in ('cut_in', 'cut_out'):
        for view in VIEWS:
            group = [r for r in rows if r['scenario'] == kind and r['view'] == view]
            summary.append({'scenario': kind, 'view': view,
                **{metric: mean_by_lineage(group, metric) for metric in ('average_precision', 'roc_auc')},
                'valid_source_lineages': sum(r['average_precision'] is not None for r in group)})
        old = next(r for r in summary if (r['scenario'], r['view']) == (kind, 'collision_probability'))
        new = next(r for r in summary if (r['scenario'], r['view']) == (kind, 'negative_clearance'))
        gains = {}
        for row in rows:
            if row['scenario'] != kind or row['view'] != 'negative_clearance' or row['average_precision'] is None:
                continue
            baseline = next(r for r in rows if (r['scenario'], r['source_id'], r['view'])
                            == (kind, row['source_id'], 'collision_probability'))
            gains[row['lineage']] = row['average_precision'] - baseline['average_precision']
        screening.append({'scenario': kind, 'primary_ap_gain': new['average_precision'] - old['average_precision'],
            'primary_auc_gain': new['roc_auc'] - old['roc_auc'], 'lineage_ap_gains': gains,
            'passed': new['average_precision'] > old['average_precision'] + 1e-12
                      and new['roc_auc'] >= old['roc_auc'] - 1e-12 and sum(v > 1e-12 for v in gains.values()) >= 2})
    result = {'status': 'complete_independently_checked_source_prediction', 'protocol': specification,
        'rows': rows, 'summary': summary, 'screening': screening,
        'screening_passed': all(r['passed'] for r in screening), 'source_fold_fits': 80,
        'heldout_source_predictions': 8192, 'maximum_leaf_prediction_error': max(errors),
        'heldout_auxiliary_label_poison_invariance_passed': poison_passed, 'folds': folds,
        'wall_s': time.perf_counter() - started, 'new_historical_search_queries': 0,
        'new_development_queries': 0, 'new_physical_calls': 0, 'development_admission_claimed': False}
    write_json(REPORT / 'report.json', result)
    lines = ['# 历史辅助响应的折外来源预测', '',
        '80个多输出来源森林只用训练折的历史安全量，碰撞Boolean仅在评估端使用。',
        '所有预测均与训练叶均值独立核对，未查询折的安全量污染不改变该折预测。', '',
        '| 场景 | 排序分数 | 来源平均精度 | 来源ROC AUC |', '|---|---|---:|---:|']
    for row in summary:
        lines.append(f"| {row['scenario']} | {row['view']} | {row['average_precision']:.6f} | {row['roc_auc']:.6f} |")
    lines += ['', f"主要间距通道的预登记诊断筛选：{'通过' if result['screening_passed'] else '未通过'}。",
        'TTC仅作辅助分析，不按结果晋升为主通道；安全量未用于重新定义碰撞标签。',
        '来源排序不代表对新系统的迁移、在线发现或正式独立验证；未执行新的搜索矩阵。',
        f"计算经过{result['wall_s']:.2f}秒，新增历史搜索、开发选例、神经网络训练和物理调用均为零。", '']
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({k: result[k] for k in ('status', 'screening_passed', 'summary', 'wall_s')})
    return result


if __name__ == '__main__':
    run()
