"""Compare probability and percentile fusion on unchanged historical states."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from . import discovery_geometry_retention as retention
from .discovery_config import OUTPUT, read_json, write_json
from .scenarios import dedup_key


REPORT = OUTPUT / 'rank_diagnosis'
METHODS = ('local_probability', 'global_probability', 'local_percentile',
           'global_percentile', 'target_only', 'physical_only')
METRICS = ('top8_unique_collisions', 'top8_recall', 'average_precision', 'roc_auc')


def prepare():
    old = read_json(retention.REPORT / 'report.json')
    assert old['protocol'] == retention.protocol() and not old['screening_passed']
    assert read_json(retention.REPORT / 'summary_audit.json')['status'] == 'passed'
    specification = {'purpose': 'probability scale and ranking headroom; fixed historical states only',
        'historical_tasks': 16, 'states': 80, 'support_counts': [8, 16, 32, 64, 128],
        'methods': list(METHODS), 'query_count': 32, 'top_count': 8,
        'rank_pool': 'all 160 public coordinates except the current paid prefix',
        'percentile': '(average ascending rank - .5) / number of unpaid coordinates; exact-score ties',
        'reliability': 'unchanged stored causal local/global scores from adapted collision kernels',
        'independent_target_prior_mass': .25, 'physical_mass': .1,
        'scope': 'not active-query trajectories; teachers include all historical systems',
        'oracle_scope': 'best single component on query labels; not a bound on every possible mixture',
        'screening': 'local percentile top8 count higher and average precision no lower in both scenarios; '
                     'two positive-count historical lineages in each scenario versus local probability',
        'input_report_sha256': hashlib.sha256((retention.REPORT / 'report.json').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_training': False, 'new_selector_queries': 0, 'new_physical_calls': 0}
    path = REPORT / 'protocol.json'
    if path.exists():
        assert read_json(path) == specification
    else:
        write_json(path, specification)
    return specification


def aggregate(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            groups[row['lineage']].append(row[metric])
    return (sum(sum(v) / len(v) for v in groups.values()) / len(groups)) if groups else None


def evaluate(scores, truth, group_keys, identifiers):
    order = np.lexsort((identifiers, -scores))
    positives = {key for key, label in zip(group_keys, truth) if label}
    found = {group_keys[i] for i in order[:8] if truth[i]}
    count = len(found)
    # A separate sequential pass checks both the tie rule and grouped discoveries.
    checked_order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), identifiers[i]))
    np.testing.assert_array_equal(order, checked_order)
    checked_found = set()
    for position in checked_order[:8]:
        if truth[position]:
            checked_found.add(group_keys[position])
    assert checked_found == found
    average_precision = auc = None
    if truth.any():
        average_precision = float(average_precision_score(truth, scores))
        independent_ap, hits = 0., 0
        for threshold in sorted(set(scores), reverse=True):
            tied = scores == threshold
            hits += int(truth[tied].sum())
            independent_ap += float(truth[tied].sum()) / truth.sum() * hits / np.count_nonzero(scores >= threshold)
        np.testing.assert_allclose(average_precision, independent_ap, atol=2e-12)
    if truth.any() and not truth.all():
        auc = float(roc_auc_score(truth, scores))
        positive, negative = scores[truth], scores[~truth]
        independent_auc = np.mean((positive[:, None] > negative[None, :])
                                   + .5 * (positive[:, None] == negative[None, :]))
        np.testing.assert_allclose(auc, independent_auc, atol=2e-12)
    return {'top8_unique_collisions': count, 'top8_recall': count / len(positives) if positives else None,
        'query_unique_collisions': len(positives), 'query_positive_coordinates': int(truth.sum()),
        'average_precision': average_precision, 'roc_auc': auc,
        'top8_query_positions': order[:8].tolist()}


def run():
    specification = read_json(REPORT / 'protocol.json')
    assert specification == prepare() and not (REPORT / 'report.json').exists()
    started = time.perf_counter()
    rows, states, percentile_error, probability_error, weight_error = [], [], 0., 0., 0.
    for kind in ('cut_in', 'cut_out'):
        history, _, _ = retention.correction.history_inputs(kind)
        for archive in history:
            path = retention.REPORT / 'histories' / kind / f'{archive.source_id}.json'
            task = read_json(path)
            assert task['protocol'] == retention.protocol() and task['status'] == 'complete'
            saved = retention.correction.cache_inputs(kind, archive)
            query_scenarios = [archive.scenarios[i] for i in saved['ordered'][128:]]
            keys = [dedup_key(s) for s in query_scenarios]
            identifiers = np.asarray([s.scenario_id for s in query_scenarios])
            truth = saved['labels'][128:].astype(bool)
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['labels'], saved['labels'])
                np.testing.assert_array_equal(arrays['ordered_indices'], saved['ordered'])
                for checkpoint, count in enumerate((8, 16, 32, 64, 128)):
                    probability = arrays['checkpoint_components'][checkpoint, 0]
                    components = np.concatenate((probability, arrays[f'raw_{count}'][None]))
                    unpaid = components[:, count:]
                    percentiles = (rankdata(unpaid, method='average', axis=1) - .5) / unpaid.shape[1]
                    independent = ((unpaid[:, :, None] > unpaid[:, None, :]).sum(-1)
                        + .5 * (unpaid[:, :, None] == unpaid[:, None, :]).sum(-1)) / unpaid.shape[1]
                    percentile_error = max(percentile_error, float(np.max(np.abs(percentiles - independent))))
                    np.testing.assert_allclose(percentiles, independent, atol=2e-15)
                    priorities = {'target_only': probability[-1, 128:], 'physical_only': components[-1, 128:]}
                    distance = np.square(arrays['x'][:count, None] - arrays['x'][None]).mean(-1)
                    for mode in ('local', 'global'):
                        weight = arrays[f'adapted_{mode}_{count}_weights']
                        influence = np.exp(-distance / .18) if mode == 'local' else np.ones((count, 160))
                        log_weight = (np.log(arrays['source_prior'])[:, None]
                            + arrays['adapted_prequery_log_likelihoods'][:, :count] @ influence)
                        expected_weight = np.exp(log_weight - logsumexp(log_weight, axis=0))
                        weight_error = max(weight_error, float(np.max(np.abs(weight - expected_weight))))
                        np.testing.assert_allclose(weight, expected_weight, atol=2e-12)
                        full_weight = np.concatenate((.9 * weight[:, count:], np.full((1, 160 - count), .1)))
                        scores = (full_weight * unpaid).sum(0)
                        expected = arrays[f'adapted_{mode}_{count}_probability'][count:]
                        probability_error = max(probability_error, float(np.max(np.abs(scores - expected))))
                        np.testing.assert_allclose(scores, expected, atol=2e-12)
                        priorities[f'{mode}_probability'] = scores[128 - count:]
                        priorities[f'{mode}_percentile'] = (full_weight * percentiles).sum(0)[128 - count:]
                    identity = {'scenario': kind, 'pseudo_target': archive.source_id,
                                'lineage': archive.lineage, 'support_count': count}
                    for method in METHODS:
                        rows.append({**identity, 'method': method, **evaluate(priorities[method], truth, keys, identifiers)})
                    individual = [evaluate(p[128:], truth, keys, identifiers)['top8_unique_collisions'] for p in components]
                    states.append({**identity, 'query_unique_collisions': len({k for k, y in zip(keys, truth) if y}),
                        'best_single_component_top8': max(individual), 'component_top8': individual,
                        'component_names': [*task['source_ids'], 'target_fsbo', 'physical']})
            print('rank_diagnosis', kind, archive.source_id, flush=True)
    assert len(rows) == 480 and len(states) == 80
    summary, screening, pairs = [], [], []
    for kind in ('cut_in', 'cut_out'):
        for method in METHODS:
            group = [r for r in rows if r['scenario'] == kind and r['method'] == method]
            summary.append({'scenario': kind, 'method': method, **{m: aggregate(group, m) for m in METRICS},
                'valid_average_precision_states': sum(r['average_precision'] is not None for r in group),
                'valid_auc_states': sum(r['roc_auc'] is not None for r in group)})
        for mode in ('local', 'global'):
            gains = {}
            for lineage in sorted({r['lineage'] for r in rows if r['scenario'] == kind}):
                group = [r for r in rows if r['scenario'] == kind and r['lineage'] == lineage]
                gains[lineage] = aggregate([r for r in group if r['method'] == f'{mode}_percentile'], 'top8_unique_collisions') - aggregate(
                    [r for r in group if r['method'] == f'{mode}_probability'], 'top8_unique_collisions')
            new = next(r for r in summary if (r['scenario'], r['method']) == (kind, f'{mode}_percentile'))
            old = next(r for r in summary if (r['scenario'], r['method']) == (kind, f'{mode}_probability'))
            paired = {'scenario': kind, 'reliability': mode,
                'top8_count_gain': new['top8_unique_collisions'] - old['top8_unique_collisions'],
                'average_precision_gain': new['average_precision'] - old['average_precision'],
                'lineage_top8_count_gains': gains}
            pairs.append(paired)
            if mode == 'local':
                screening.append({'scenario': kind, 'passed': paired['top8_count_gain'] > 1e-12
                    and paired['average_precision_gain'] >= -1e-12 and sum(g > 1e-12 for g in gains.values()) >= 2})
    result = {'status': 'complete_independently_checked_fixed_state_diagnosis', 'protocol': specification,
        'historical_tasks': 16, 'states': 80, 'rows': rows, 'summary': summary, 'paired_comparisons': pairs,
        'component_diagnostics': states, 'screening': screening, 'screening_passed': all(s['passed'] for s in screening),
        'maximum_percentile_error': percentile_error, 'maximum_control_probability_error': probability_error,
        'maximum_causal_weight_error': weight_error, 'wall_s': time.perf_counter() - started,
        'new_selector_queries': 0, 'new_physical_calls': 0, 'development_admission_claimed': False}
    write_json(REPORT / 'report.json', result)
    lines = ['# 固定历史状态的概率与百分位融合', '',
        '16个历史任务、80个相同支持状态，全部未查询公开坐标用于百分位，32点查询标签仅作评估。',
        '前8个发现数是固定状态排序指标，不是新的在线查询或128次发现曲线。', '',
        '| 场景 | 融合/分量 | 前8个去重碰撞数 | 平均精度 | ROC AUC |', '|---|---|---:|---:|---:|']
    for value in summary:
        def display(x):
            return '不适用' if x is None else f'{x:.6f}'
        lines.append(f"| {value['scenario']} | {value['method']} | {display(value['top8_unique_collisions'])} | "
                     f"{display(value['average_precision'])} | {display(value['roc_auc'])} |")
    lines += ['', f"预登记诊断筛选：{'通过' if result['screening_passed'] else '未通过'}。",
        '来源、物理分量及目标分量的最佳单分量结果使用评估标签，仅是单分量选择类的诊断上界。',
        '百分位、ROC AUC、平均精度、去重发现数和原因果权重/概率已交叉核对，完整配对结果见report.json。',
        '教师包含全部历史对象；自身显式来源排除，这项诊断不是独立留系统泛化证据。',
        f"共享机器经过{result['wall_s']:.2f}秒，新增训练、开发选例与物理调用均为零。", '']
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    globals()[parser.parse_args().stage]()
