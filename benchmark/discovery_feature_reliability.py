"""Historical-only test of fixed and target-adapted reliability feature distances."""

import argparse
import hashlib
import json
import time
import traceback

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.response import archive_fingerprints, source_weights
from methods.gcrt.converged_response import component_probabilities
from methods.gcrt.feature_reliability import log_weights, squared_distance
from methods.gcrt.residual_scale import cached_covariance, residual_prior

from . import discovery_converged_response as correction
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_physical_mixture import checkpoint_path
from .discovery_residual_scale import COUNTS
from .protocol import ROOT


REPORT = OUTPUT / 'feature_reliability'
SETTINGS = ({'name': 'global', 'geometry': 'physical', 'bandwidth': None}, *(
    {'name': f'{geometry}_{width:g}', 'geometry': geometry, 'bandwidth': width}
    for geometry in ('physical', 'fixed_feature', 'adapted_feature') for width in (.15, .3, .6)))


def protocol():
    return {'purpose': 'historical reliability geometry diagnosis; not a new online method registration',
        'scenarios': ['cut_in', 'cut_out'], 'historical_tasks': 16, 'settings': list(SETTINGS),
        'counts': list(COUNTS), 'support_count': 128, 'query_count': 32, 'permutation_seed': 1301,
        'selected_beta': read_json(correction.REPORT / 'history.json')['selected_beta'],
        'feature_scale': 'fixed RMS deviation of base FSBO features on the complete 512 public historical coordinates',
        'distance': 'mean squared feature difference divided by fixed feature scale squared',
        'dynamic_rule': 'recompute locality from current target features; never recompute past prequery log likelihoods',
        'source_mass': .75, 'target_mass': .25, 'physical_mass': .1,
        'source_conditioning': 'converged CPU float64; gradient at most 1e-10',
        'target_adaptation': 'reuse original 129 CUDA states; no new training',
        'raw_probability': 'unchanged cached physical component; this test isolates reliability geometry',
        'teacher_scope': 'same existing teachers; own explicit source posterior excluded; not strict held-system validation',
        'selection': 'leave one historical lineage out for bandwidth; NLL, Brier, larger bandwidth',
        'screening': 'adapted feature NLL lower and Brier no higher than global and selected fixed feature and physical distances; positive NLL gain on at least two historical lineages in each scenario',
        'historical_evidence': {p: hashlib.sha256((correction.REPORT / p).read_bytes()).hexdigest()
            for p in ('protocol.json', 'history.json', 'summary_audit.json')},
        'checkpoints': {kind: hashlib.sha256(checkpoint_path(kind, 'physical_fsbo').read_bytes()).hexdigest()
            for kind in ('cut_in', 'cut_out')},
        'source_files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
            'benchmark/discovery_feature_reliability.py', 'methods/gcrt/feature_reliability.py',
            'methods/gcrt/converged_response.py')},
        'new_selector_queries': 0, 'new_physical_calls': 0}


def prepare():
    historical = read_json(correction.REPORT / 'history.json')
    audit = read_json(correction.REPORT / 'summary_audit.json')
    if (historical['protocol'] != correction.protocol() or audit['status'] != 'passed'
            or audit['score_rows'] != 2400 or audit['nonempty_mode_solves'] != 81920
            or audit['selected_beta_independently_recomputed'] != historical['selected_beta']):
        raise ValueError('Complete and independently check converged historical inference first')
    specification = protocol()
    path = REPORT / 'protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('The feature-reliability diagnosis was already registered differently')
    write_json(path, specification)
    return specification


def score_task(kind, archive, history, all_means, all_covariances, base_features, scale, specification):
    destination = REPORT / 'histories' / kind / f'{archive.source_id}.json'
    if destination.exists():
        old = read_json(destination)
        if old['protocol'] != specification or old['status'] != 'complete':
            raise ValueError('A saved feature-reliability task differs from the registration')
        return old
    started = time.perf_counter()
    saved = correction.cache_inputs(kind, archive)
    ordered, labels = saved['ordered'], saved['labels']
    np.testing.assert_allclose(saved['encoded'][0], base_features[ordered], atol=2e-9, rtol=2e-8)
    positions = [i for i, a in enumerate(history) if a.source_id != archive.source_id]
    sources = [history[i] for i in positions]
    source_m = torch.tensor(all_means[positions][:, ordered])
    source_c = torch.tensor(all_covariances[positions][:, ordered, :][:, :, ordered])
    prior = torch.tensor(np.r_[.75*source_weights(sources), .25])
    likelihoods = torch.empty((8, 128), dtype=torch.float64)
    static = {'physical': squared_distance(saved['x']),
              'fixed_feature': squared_distance(saved['encoded'][0], scale)}
    rows, probability_checks, weight_checks, convergence, probabilities_saved, weights_saved = [], [], [], [], [], []
    beta = specification['selected_beta'][kind]
    for count in range(129):
        parameters = saved['parameters'][count]
        covariance = cached_covariance(saved['encoded'][count], parameters[1], parameters[2])
        means, covariances = residual_prior(parameters[0], covariance, source_m, source_c,
                                           saved['parameters'][0, 0], beta)
        with torch.no_grad():
            probabilities, info = component_probabilities(means, covariances, list(range(count)), labels[:count])
        convergence.extend(info)
        if count in (0, 8, 64, 128):
            independent = correction.independent_probabilities(means.numpy(), covariances.numpy(),
                                                                list(range(count)), labels[:count])
            np.testing.assert_allclose(probabilities, independent, atol=2e-7, rtol=2e-6)
            probability_checks.append({'support_count': count,
                'probability_maximum_error': float(np.max(np.abs(probabilities.numpy()-independent)))})
        if count in COUNTS:
            distances = {**static, 'adapted_feature': squared_distance(saved['encoded'][count], scale)}
            all_weights = []
            for setting in SETTINGS:
                log_weight = log_weights(prior, likelihoods[:, :count], list(range(count)),
                                         distances[setting['geometry']], setting['bandwidth'])
                weights = log_weight.exp()
                mixed = .9*(weights*probabilities).sum(0).numpy()+.1*saved['raw_probability'][count]
                geometry = setting['geometry']
                encoded = (saved['x'].numpy() if geometry == 'physical' else
                           saved['encoded'][count if geometry == 'adapted_feature' else 0].numpy())
                distance = np.square(encoded[:, None]-encoded[None]).mean(-1)/(1 if geometry == 'physical' else scale**2)
                influence = (np.ones((count, 160)) if setting['bandwidth'] is None else
                             np.exp(-distance[:count]/(2*setting['bandwidth']**2)))
                score = np.log(prior.numpy())[:, None]+likelihoods[:, :count].numpy()@influence
                expected = np.exp(score-logsumexp(score, axis=0))
                np.testing.assert_allclose(weights, expected, atol=2e-12, rtol=2e-12)
                expected_p = .9*(expected*probabilities.numpy()).sum(0)+.1*saved['raw_probability'][count]
                np.testing.assert_allclose(mixed, expected_p, atol=2e-12, rtol=2e-12)
                rows.append({'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
                    'source_ids': [a.source_id for a in sources], 'setting': setting['name'], 'support_count': count,
                    **correction.query_metrics(mixed[128:], labels[128:])})
                weight_checks.append({'setting': setting['name'], 'support_count': count,
                    'weight_maximum_error': float(np.max(np.abs(weights.numpy()-expected))),
                    'mixture_maximum_error': float(np.max(np.abs(mixed-expected_p)))})
                all_weights.append(log_weight.numpy())
            probabilities_saved.append(probabilities.numpy())
            weights_saved.append(np.asarray(all_weights))
        if count < 128:
            likelihoods[:, count] = (probabilities[:, count] if labels[count] else 1-probabilities[:, count]).clamp_min(1e-15).log()
    result = {'status': 'complete', 'protocol': specification, 'scenario': kind, 'pseudo_target': archive.source_id,
        'lineage': archive.lineage, 'source_ids': [a.source_id for a in sources], 'feature_scale': scale,
        'rows': rows, 'probability_checks': probability_checks, 'weight_checks': weight_checks,
        'nonempty_mode_solves': 128*8, 'maximum_final_gradient': max(r['gradient_max'] for r in convergence),
        'maximum_mode_iterations': max(r['iterations'] for r in convergence),
        'backtracks': sum(r['backtracks'] for r in convergence), 'wall_s': time.perf_counter()-started,
        'new_selector_queries': 0, 'new_physical_calls': 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination.with_suffix('.npz'), prequery_log_likelihoods=likelihoods.numpy(),
                        checkpoint_probabilities=np.asarray(probabilities_saved), checkpoint_log_weights=np.asarray(weights_saved),
                        ordered_indices=ordered, feature_scale=np.asarray(scale))
    write_json(destination, result)
    print('feature_reliability', kind, archive.source_id, flush=True)
    return result


def choose(rows, geometry):
    return min(({'setting': s['name'], **correction.aggregate([r for r in rows if r['setting'] == s['name']])}
                for s in SETTINGS if s['geometry'] == geometry and s['bandwidth'] is not None),
               key=lambda r: (r['query_nll'], r['query_brier'], -float(r['setting'].rsplit('_', 1)[-1])))


def run():
    specification = read_json(REPORT / 'protocol.json')
    if specification != protocol() or (REPORT / 'report.json').exists():
        raise ValueError('Run only the unchanged and not already completed historical geometry diagnosis')
    started = time.perf_counter()
    torch.set_num_threads(1)
    records = []
    with threadpool_limits(1):
        for kind in specification['scenarios']:
            history, means, covariances = correction.history_inputs(kind)
            model, metadata = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
            if metadata['history_fingerprints'] != archive_fingerprints(history):
                raise ValueError('Feature scaling must use the same historical representation')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.no_grad():
                encoded = model.to(device).features(history[0].x).cpu().numpy()
            scale = float(np.sqrt(np.square(encoded-encoded.mean(0)).mean()))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('A collapsed historical feature map cannot define this geometry')
            path = REPORT / 'normalization' / f'{kind}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, base_features=encoded, scale=np.asarray(scale))
            for archive in history:
                try:
                    records.append(score_task(kind, archive, history, means, covariances, encoded, scale, specification))
                except Exception:
                    with (REPORT / 'failed_attempts.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'scenario': kind, 'pseudo_target': archive.source_id,
                            'error': traceback.format_exc(), 'wall_s': time.perf_counter()-started,
                            'new_selector_queries': 0, 'new_physical_calls': 0})+'\n')
                    raise
    rows = [r for record in records for r in record['rows']]
    if len(records) != 16 or len(rows) != 800 or protocol() != specification:
        raise ValueError('Complete every registered historical geometry and prefix before interpretation')
    old = read_json(correction.REPORT / 'history.json')['rows']
    control_errors = []
    for row in rows:
        if row['setting'] not in ('global', 'physical_0.3'):
            continue
        mode = 'global' if row['setting'] == 'global' else 'local'
        match = next(r for r in old if (r['scenario'], r['pseudo_target'], r['support_count'], r['beta'],
            r['reliability'], r['variant']) == (row['scenario'], row['pseudo_target'], row['support_count'],
            specification['selected_beta'][row['scenario']], mode, 'converged_prequential'))
        error = max(abs(row[m]-match[m]) for m in ('query_nll', 'query_brier'))
        np.testing.assert_allclose(error, 0, atol=2e-8, rtol=0)
        control_errors.append(error)
    summary, selected, held = [], {}, []
    for kind in specification['scenarios']:
        group = [r for r in rows if r['scenario'] == kind]
        summary += [{'scenario': kind, 'setting': s['name'],
                     **correction.aggregate([r for r in group if r['setting'] == s['name']])} for s in SETTINGS]
        selected[kind] = {g: choose(group, g)['setting'] for g in ('physical', 'fixed_feature', 'adapted_feature')}
        for lineage in sorted({r['lineage'] for r in group}):
            for geometry in ('physical', 'fixed_feature', 'adapted_feature'):
                chosen = choose([r for r in group if r['lineage'] != lineage], geometry)['setting']
                held.append({'scenario': kind, 'lineage': lineage, 'geometry': geometry, 'selected_setting': chosen,
                             **correction.aggregate([r for r in group if r['lineage'] == lineage and r['setting'] == chosen])})
            held.append({'scenario': kind, 'lineage': lineage, 'geometry': 'global', 'selected_setting': 'global',
                         **correction.aggregate([r for r in group if r['lineage'] == lineage and r['setting'] == 'global'])})
    cross_summary = {kind: {g: correction.aggregate([r for r in held if r['scenario'] == kind and r['geometry'] == g])
        for g in ('global', 'physical', 'fixed_feature', 'adapted_feature')} for kind in specification['scenarios']}
    screening = []
    for kind, values in cross_summary.items():
        candidate = values['adapted_feature']
        lookup = {(r['geometry'], r['lineage']): r for r in held if r['scenario'] == kind}
        controls = []
        for control in ('global', 'physical', 'fixed_feature'):
            positive = [lineage for geometry, lineage in lookup if geometry == 'adapted_feature'
                        and lookup[geometry, lineage]['query_nll'] < lookup[control, lineage]['query_nll']-1e-12]
            controls.append({'control': control, 'positive_nll_lineages': positive,
                'nll_difference': candidate['query_nll']-values[control]['query_nll'],
                'brier_difference': candidate['query_brier']-values[control]['query_brier'],
                'condition': candidate['query_nll'] < values[control]['query_nll']-1e-12
                    and candidate['query_brier'] <= values[control]['query_brier']+1e-12 and len(positive) >= 2})
        screening.append({'scenario': kind, 'supports_further_online_registration': all(c['condition'] for c in controls),
                          'controls': controls})
    result = {'status': 'complete_numerically_checked_historical_diagnosis', 'protocol': specification,
        'rows': rows, 'summary': summary, 'selected_by_geometry': selected, 'held_lineage_rows': held,
        'held_lineage_summary': cross_summary, 'screening': screening,
        'numerical': {'independent_probability_groups': 64, 'independent_weight_mixture_groups': 800,
            'unchanged_control_scores': len(control_errors), 'maximum_control_score_error': max(control_errors),
            'maximum_probability_error': max(c['probability_maximum_error'] for r in records for c in r['probability_checks']),
            'maximum_weight_error': max(c['weight_maximum_error'] for r in records for c in r['weight_checks']),
            'maximum_mixture_error': max(c['mixture_maximum_error'] for r in records for c in r['weight_checks']),
            'nonempty_modes': sum(r['nonempty_mode_solves'] for r in records),
            'maximum_final_gradient': max(r['maximum_final_gradient'] for r in records)},
        'wall_s': time.perf_counter()-started, 'new_selector_queries': 0, 'new_physical_calls': 0,
        'discovery_admission_claimed': False, 'online_design_changed': False, 'formal_phases_entered': False}
    write_json(REPORT / 'report.json', result)
    lines = ['# 目标适配特征与来源可靠性：历史诊断', '',
        '16个历史任务、800项查询集评分。固定已存查询前对数似然，只改变来源可靠性的传播距离。', '',
        '| 场景 | 距离 | 留出谱系NLL | 留出谱系Brier |', '|---|---|---:|---:|']
    for kind, values in cross_summary.items():
        for geometry, value in values.items():
            lines.append(f"| {kind} | {geometry} | {value['query_nll']:.6f} | {value['query_brier']:.6f} |")
    lines += ['', f"后续诊断筛选：{[(r['scenario'], r['supports_further_online_registration']) for r in screening]}。", '',
        '只在带宽选择中留出历史谱系；基础教师仍见过全部历史对象，本项不是独立系统或严格留家族验证。',
        '预测评分不能替代128次碰撞发现门槛；本项不改变已登记160项搜索，不启动新开发查询或C/D/E。',
        f"64组独立分量概率、800组权重/混合概率及160项旧收敛控制评分通过。耗时{result['wall_s']:.2f}秒。", '']
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    globals()[parser.parse_args().stage]()
