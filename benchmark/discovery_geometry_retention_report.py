"""Independently aggregate saved historical geometry-retention predictions."""

from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

from . import discovery_geometry_retention as study
from .discovery_config import read_json, write_json


def average(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[row['lineage']].append(row[metric])
    return sum(sum(values) / len(values) for values in groups.values()) / len(groups)


def report():
    result = read_json(study.REPORT / 'report.json')
    assert result['status'] == 'complete_historical_screening' and result['protocol'] == study.protocol()
    rows, maximum_weight_error, maximum_probability_error = [], 0., 0.
    for kind in ('cut_in', 'cut_out'):
        histories, _, _ = study.correction.history_inputs(kind)
        for archive in histories:
            path = study.REPORT / 'histories' / kind / f'{archive.source_id}.json'
            task = read_json(path)
            assert task['status'] == 'complete' and task['protocol'] == result['protocol']
            assert task['source_ids'] == [a.source_id for a in histories if a.source_id != archive.source_id]
            cached = study.correction.cache_inputs(kind, archive)
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['labels'], cached['labels'])
                np.testing.assert_array_equal(saved['ordered_indices'], cached['ordered'])
                np.testing.assert_array_equal(saved['x'], cached['x'])
                prior = saved['source_prior']
                assert len(prior) == 8 and np.all(prior > 0)
                np.testing.assert_allclose([prior[:-1].sum(), prior[-1]], [.75, .25], atol=1e-15)
                distance = np.square(saved['x'][:, None] - saved['x'][None]).mean(-1)
                truth = saved['labels'][128:]
                for index, count in enumerate((8, 16, 32, 64, 128)):
                    adaptive, frozen = saved['checkpoint_components'][index]
                    components = {'adapted': adaptive, 'frozen': frozen,
                        'fixed_half': .5 * (adaptive + frozen), 'learned_views': np.concatenate((adaptive, frozen))}
                    for variant, probability in components.items():
                        mass = np.r_[prior / 2, prior / 2] if variant == 'learned_views' else prior
                        ll = saved[f'{variant}_prequery_log_likelihoods'][:, :count]
                        for mode in ('local', 'global'):
                            influence = (np.exp(-distance[:count] / .18) if mode == 'local'
                                         else np.ones((count, 160)))
                            score = np.log(mass)[:, None] + ll @ influence
                            weight = np.exp(score - logsumexp(score, axis=0))
                            actual_weight = saved[f'{variant}_{mode}_{count}_weights']
                            maximum_weight_error = max(maximum_weight_error, float(np.max(np.abs(weight - actual_weight))))
                            np.testing.assert_allclose(weight, actual_weight, atol=2e-12, rtol=2e-12)
                            prediction = np.asarray([.9 * sum(float(weight[k, j] * probability[k, j])
                                for k in range(len(mass))) + .1 * float(saved[f'raw_{count}'][j]) for j in range(160)])
                            maximum_probability_error = max(maximum_probability_error,
                                float(np.max(np.abs(prediction - saved[f'{variant}_{mode}_{count}_probability']))))
                            np.testing.assert_allclose(prediction, saved[f'{variant}_{mode}_{count}_probability'],
                                                       atol=2e-12, rtol=2e-12)
                            p = np.clip(prediction[128:], 1e-12, 1 - 1e-12)
                            row = {'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
                                'variant': variant, 'reliability': mode, 'support_count': count,
                                'query_nll': float(np.mean(np.where(truth, -np.log(p), -np.log1p(-p)))),
                                'query_brier': float(np.mean((p - truth)**2))}
                            original = next(r for r in task['rows'] if (r['variant'], r['reliability'], r['support_count'])
                                            == (variant, mode, count))
                            for metric in ('query_nll', 'query_brier'):
                                np.testing.assert_allclose(row[metric], original[metric], atol=2e-12, rtol=2e-12)
                            rows.append(row)
    assert len(rows) == len(result['rows']) == 640
    summaries, screens, maximum_summary_error = [], [], 0.
    for kind in ('cut_in', 'cut_out'):
        group = [r for r in rows if r['scenario'] == kind]
        for variant in study.VARIANTS:
            for mode in study.MODES:
                matched = [r for r in group if r['variant'] == variant and r['reliability'] == mode]
                entry = {'scenario': kind, 'variant': variant, 'reliability': mode,
                    **{metric: average(matched, metric) for metric in ('query_nll', 'query_brier')}}
                old = next(r for r in result['summary'] if (r['scenario'], r['variant'], r['reliability'])
                           == (kind, variant, mode))
                for metric in ('query_nll', 'query_brier'):
                    maximum_summary_error = max(maximum_summary_error, abs(entry[metric] - old[metric]))
                    np.testing.assert_allclose(entry[metric], old[metric], atol=2e-12, rtol=2e-12)
                summaries.append(entry)
        local = [r for r in summaries if r['scenario'] == kind and r['reliability'] == 'local']
        candidate = next(r for r in local if r['variant'] == 'learned_views')
        controls = [r for r in local if r['variant'] != 'learned_views']
        best = min(controls, key=lambda r: (r['query_nll'], r['query_brier'], r['variant']))
        global_control = next(r for r in summaries if (r['scenario'], r['variant'], r['reliability'])
                              == (kind, 'learned_views', 'global'))
        gains = {lineage: average([r for r in group if r['lineage'] == lineage and r['variant'] == best['variant']
                                  and r['reliability'] == 'local'], 'query_nll')
                         - average([r for r in group if r['lineage'] == lineage and r['variant'] == 'learned_views'
                                    and r['reliability'] == 'local'], 'query_nll')
                 for lineage in sorted({r['lineage'] for r in group})}
        passed = (all(candidate['query_nll'] < r['query_nll'] - 1e-12
                      and candidate['query_brier'] <= r['query_brier'] + 1e-12 for r in (*controls, global_control))
                  and sum(g > 1e-12 for g in gains.values()) >= 2)
        original = next(r for r in result['screening'] if r['scenario'] == kind)
        assert original['passed'] == passed and original['strongest_local_control'] == best['variant']
        for lineage, gain in gains.items():
            np.testing.assert_allclose(gain, original['lineage_nll_gains'][lineage], atol=2e-12)
        screens.append({'scenario': kind, 'passed': passed, 'strongest_local_control': best['variant'],
                        'lineage_nll_gains': gains})
    assert result['screening_passed'] == all(r['passed'] for r in screens)
    audit = {'status': 'passed', 'historical_tasks': 16, 'score_rows': 640,
        'maximum_weight_error': maximum_weight_error, 'maximum_probability_error': maximum_probability_error,
        'maximum_summary_error': maximum_summary_error, 'screening': screens,
        'screening_passed': result['screening_passed'], 'development_admission_claimed': False,
        'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(study.REPORT / 'summary_audit.json', audit)
    lines = ['# 历史碰撞几何的保留与适配', '',
        '原16个历史任务、128点有序支持与不相交32点查询；不新增训练、开发选例或仿真。',
        '两种核保持均值及平均边际方差一致，每次只累计查询前似然。教师使用全部历史对象，'
        '自身显式后验排除，因此不是严格留系统泛化证据。', '',
        '| 场景 | 核构造 | 可靠性 | 查询NLL | Brier |', '|---|---|---|---:|---:|']
    for entry in summaries:
        lines.append(f"| {entry['scenario']} | {entry['variant']} | {entry['reliability']} | "
                     f"{entry['query_nll']:.8f} | {entry['query_brier']:.8f} |")
    lines += ['', f"预登记历史筛选：{'通过' if result['screening_passed'] else '未通过'}。",
        '全部640项评分、128组独立分量预测及160项原适配控制复现已核查；保存数组另经独立汇总。',
        '局部学习视图必须在两个场景均优于三种局部控制及全局学习视图，且至少两个谱系改善。',
        '这项筛选不等同128次在线发现。未通过时不扩大此分支，不改写既有在线结果或准入门槛。',
        f"共享机器历史计算经过{result['wall_s']:.2f}秒；新增开发查询和物理调用均为零。", '']
    (study.REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return audit


if __name__ == '__main__':
    print(report())
