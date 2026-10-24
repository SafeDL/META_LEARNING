"""Historical-only comparison of parameter and response reliability distances."""

import argparse
import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from controllers.highway.catalog import SUTS
from methods.common.classification import logistic_moments, torch_moments
from methods.common.response import archive_fingerprints, source_weights
from methods.gcrt.conditional_adaptation import component_probabilities
from methods.gcrt.residual_scale import cached_covariance, residual_prior
from methods.gcrt.response_locality import influence_matrix, squared_distance

from .discovery_conditional_adaptation import probabilities_numpy
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_residual_scale import ARTIFACTS, COUNTS, REPORT as LEARNING, cache_path
from .discovery_residual_scale import protocol as learning_protocol
from .protocol import DATA_ROOT, ROOT
from .scenarios import sample
from .storage import load_archive


REPORT = OUTPUT/'response_locality'
SETTINGS = ({'name': 'global', 'fraction': 0., 'bandwidth': None}, *(
    {'name': f'{name}_{bandwidth:g}', 'fraction': fraction, 'bandwidth': bandwidth}
    for name, fraction in (('physical', 0.), ('hybrid', .5), ('response', 1.))
    for bandwidth in (.15, .3, .6)))


def protocol():
    selection = read_json(LEARNING/'selection.json')
    return {'purpose': 'historical locality diagnosis; no new target queries or discovery admission',
        'scenarios': ['cut_in', 'cut_out'], 'pseudo_targets': 16, 'settings': list(SETTINGS),
        'counts': list(COUNTS), 'ordered_seed': 1301, 'support_count': 128, 'query_count': 32,
        'selected_beta': selection['selected_beta'], 'physical_mass': .1,
        'source_prior_mass': .75, 'target_prior_mass': .25, 'device': 'cpu',
        'geometry': 'fixed marginal collision probabilities of the seven other historical source posteriors',
        'selection': 'per-distance NLL, Brier, larger bandwidth; equal lineage averages',
        'cross_check': 'leave one historical lineage out of bandwidth selection only',
        'teacher_scope': 'all historical systems; not strict held-system or held-family validation',
        'historical_protocol': learning_protocol(),
        'historical_selection_audit': hashlib.sha256((LEARNING/'selection_audit.json').read_bytes()).hexdigest(),
        'source_files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'benchmark/discovery_response_locality.py', 'methods/gcrt/response_locality.py')},
        'new_selector_queries': 0, 'new_physical_calls': 0}


def prepare():
    audit = read_json(LEARNING/'selection_audit.json')
    selection = read_json(LEARNING/'selection.json')
    if (audit['status'] != 'passed' or selection['protocol'] != learning_protocol()
            or audit['selected_beta_recomputed'] != selection['selected_beta']):
        raise ValueError('Preserve and audit historical adaptation and beta selection first')
    current = protocol()
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != current:
        raise ValueError('The locality diagnosis is already registered differently')
    write_json(path, current)
    return current


def history_inputs(kind):
    history = [load_archive(p) for p in sorted((DATA_ROOT/'history'/kind).glob('*.npz'))]
    expected = {s.sut_id: s for s in SUTS if s.role == 'history'}
    if len(history) != 8 or {a.source_id for a in history} != set(expected):
        raise ValueError('Locality diagnosis requires exactly the eight historical objects')
    coordinates = sample(kind, 512, 'history')
    for archive in history:
        sut = expected[archive.source_id]
        if ((archive.role, archive.family, archive.lineage) != ('history', sut.family, sut.lineage)
                or archive.scenarios != coordinates or not archive.provenance.get('qualified')):
            raise ValueError('Nonhistorical or changed coordinates cannot enter locality diagnosis')
    with np.load(ARTIFACTS/'priors'/f'{kind}.npz', allow_pickle=False) as saved:
        identity = json.loads(str(saved['specification']))
        expected_identity = {'protocol': learning_protocol(), 'source_ids': [a.source_id for a in history],
                             'history_fingerprints': archive_fingerprints(history), 'x': history[0].x.tolist()}
        if identity != expected_identity:
            raise ValueError('Historical response priors differ from the independently checked archives')
        return history, saved['means'].copy(), saved['covariances'].copy()


def aggregate(rows):
    lineages = sorted({r['lineage'] for r in rows})
    if not lineages:
        raise ValueError('A historical scoring group is empty')
    return {metric: float(np.mean([np.mean([r[metric] for r in rows if r['lineage'] == lineage])
                                  for lineage in lineages])) for metric in ('nll', 'brier')}


def choose(rows, geometry):
    eligible = []
    for setting in SETTINGS:
        if setting['name'].split('_')[0] == geometry:
            group = [r for r in rows if r['setting'] == setting['name']]
            eligible.append({**setting, **aggregate(group)})
    return min(eligible, key=lambda r: (r['nll'], r['brier'], -(r['bandwidth'] or 0)))


def score_history(kind, archive, history, source_means, source_covariances, specification):
    started = time.perf_counter()
    sources = [a for a in history if a.source_id != archive.source_id]
    positions = [i for i, a in enumerate(history) if a.source_id != archive.source_id]
    with np.load(cache_path(kind, archive.source_id), allow_pickle=False) as saved:
        cached_spec = json.loads(str(saved['specification']))
        expected = {'protocol': learning_protocol(), 'kind': kind, 'source_id': archive.source_id,
            'role': 'history', 'history_fingerprints': archive_fingerprints(history),
            'ordered_indices': np.random.default_rng(1301).permutation(512)[:160].tolist()}
        if cached_spec != expected:
            raise ValueError('Historical adaptation cache differs from its ordered support/query split')
        ordered, labels = saved['ordered_indices'].copy(), saved['labels'].copy()
        np.testing.assert_array_equal(saved['labels'], archive.collision[ordered])
        np.testing.assert_array_equal(saved['x'], archive.x[ordered])
        encoded = torch.tensor(saved['encoded'], dtype=torch.float64)
        parameters = torch.tensor(saved['parameters'], dtype=torch.float64)
        public_x = torch.tensor(saved['x'], dtype=torch.float64)
        raw_probability = saved['raw_probabilities'].copy()
    means = torch.tensor(source_means[positions][:, ordered], dtype=torch.float64)
    covariances = torch.tensor(source_covariances[positions][:, ordered, :][:, :, ordered], dtype=torch.float64)
    with torch.no_grad():
        response = torch_moments(means, covariances.diagonal(dim1=-2, dim2=-1).clamp_min(1e-10))[0]
    expected_response = logistic_moments(means.numpy(), covariances.diagonal(dim1=-2, dim2=-1).numpy())[0]
    np.testing.assert_allclose(response, expected_response, atol=2e-12, rtol=2e-12)
    distances = {f: squared_distance(public_x, response, f) for f in (0., .5, 1.)}
    kernels = torch.stack([influence_matrix(distances[s['fraction']], s['bandwidth']) for s in SETTINGS])
    initial = np.r_[.75*source_weights(sources), .25]
    scores = torch.tensor(initial).log()[None, :, None].expand(len(SETTINGS), -1, 160).clone()
    likelihoods = np.empty((8, 128))
    independent_kernels = []
    x, q = public_x.numpy(), response.numpy().T
    physical_distance = np.square(x[:, None]-x[None]).mean(-1)
    response_distance = np.square(q[:, None]-q[None]).mean(-1)
    for setting in SETTINGS:
        distance = (1-setting['fraction'])*physical_distance+setting['fraction']*response_distance
        kernel = np.ones_like(distance) if setting['bandwidth'] is None else np.exp(-distance/(2*setting['bandwidth']**2))
        independent_kernels.append(kernel)
    np.testing.assert_allclose(kernels, independent_kernels, atol=2e-14, rtol=2e-14)
    rows, probability_checks, weight_checks = [], [], []
    beta = specification['selected_beta'][kind]
    for count in range(129):
        covariance = cached_covariance(encoded[count], parameters[count, 1], parameters[count, 2])
        mean, covariance = residual_prior(parameters[count, 0], covariance, means, covariances, parameters[0, 0], beta)
        with torch.no_grad():
            probabilities = component_probabilities(mean, covariance, list(range(count)), labels[:count])
        if count in (0, 8, 64, 128):
            independent = probabilities_numpy(mean.numpy(), covariance.numpy(), list(range(count)), labels[:count])
            np.testing.assert_allclose(probabilities, independent, atol=2e-7, rtol=2e-6)
            probability_checks.append({'support_count': count,
                'maximum_error': float(np.max(np.abs(probabilities.numpy()-independent)))})
        if count in COUNTS:
            weights = scores.softmax(1)
            mixture = (.9*(weights[:, :, 128:]*probabilities[None, :, 128:]).sum(1)).numpy()+.1*raw_probability[count, 128:]
            for j, setting in enumerate(SETTINGS):
                expected_score = np.log(initial)[:, None]+likelihoods[:, :count] @ independent_kernels[j][:count]
                expected_weight = np.exp(expected_score-expected_score.max(0))
                expected_weight /= expected_weight.sum(0)
                expected_probability = .9*(expected_weight[:, 128:]*probabilities.numpy()[:, 128:]).sum(0)+.1*raw_probability[count, 128:]
                np.testing.assert_allclose(weights[j], expected_weight, atol=2e-12, rtol=2e-12)
                np.testing.assert_allclose(mixture[j], expected_probability, atol=2e-12, rtol=2e-12)
                p = np.clip(mixture[j], 1e-12, 1-1e-12)
                y = labels[128:]
                rows.append({'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
                    'source_ids': [a.source_id for a in sources], 'setting': setting['name'], 'support_count': count,
                    'nll': float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))), 'brier': float(np.square(p-y).mean())})
                weight_checks.append({'support_count': count, 'setting': setting['name'],
                    'weight_maximum_error': float(np.max(np.abs(weights[j].numpy()-expected_weight))),
                    'mixture_maximum_error': float(np.max(np.abs(mixture[j]-expected_probability)))})
        if count < 128:
            likelihood = (probabilities[:, count] if labels[count] else 1-probabilities[:, count]).clamp_min(1e-15).log()
            likelihoods[:, count] = likelihood.numpy()
            scores += likelihood[None, :, None]*kernels[:, None, count]
            scores -= torch.logsumexp(scores, 1, keepdim=True)
    result = {'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
        'source_ids': [a.source_id for a in sources], 'rows': rows, 'probability_checks': probability_checks,
        'weight_checks': weight_checks, 'wall_s': time.perf_counter()-started}
    write_json(REPORT/'histories'/kind/f'{archive.source_id}.json', result)
    print('response_locality', kind, archive.source_id, flush=True)
    return result


def run():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Locality diagnosis changed after registration')
    if (REPORT/'report.json').exists():
        raise ValueError('The complete historical locality diagnosis is already saved')
    started = time.perf_counter()
    torch.set_num_threads(1)
    records = []
    with threadpool_limits(1):
        for kind in specification['scenarios']:
            history, means, covariances = history_inputs(kind)
            for archive in history:
                records.append(score_history(kind, archive, history, means, covariances, specification))
    rows = [r for record in records for r in record['rows']]
    if len(rows) != 800 or specification != protocol():
        raise ValueError('Complete all historical tasks without changing the registered diagnosis')
    original = read_json(LEARNING/'selection.json')['rows']
    registered_endpoint_checks = []
    for row in rows:
        if row['setting'] not in ('physical_0.3', 'global'):
            continue
        mode = 'local' if row['setting'] == 'physical_0.3' else 'global'
        matched = [r for r in original if r['scenario'] == row['scenario']
                   and r['pseudo_target'] == row['pseudo_target'] and r['support_count'] == row['support_count']
                   and r['beta'] == specification['selected_beta'][row['scenario']] and r['reliability'] == mode]
        if len(matched) != 1:
            raise ValueError('Every unchanged physical/global score needs one registered historical counterpart')
        errors = [abs(row[metric]-matched[0][f'query_{metric}']) for metric in ('nll', 'brier')]
        np.testing.assert_allclose(errors, 0, atol=2e-8, rtol=0)
        registered_endpoint_checks.append(max(errors))
    summary, selected, cross_checked = [], {}, []
    for kind in specification['scenarios']:
        group = [r for r in rows if r['scenario'] == kind]
        summary += [{'scenario': kind, 'setting': s['name'], **aggregate([r for r in group if r['setting'] == s['name']])}
                    for s in SETTINGS]
        selected[kind] = {name: choose(group, name)['name'] for name in ('physical', 'hybrid', 'response')}
        for lineage in sorted({r['lineage'] for r in group}):
            for geometry in ('physical', 'hybrid', 'response'):
                chosen = choose([r for r in group if r['lineage'] != lineage], geometry)
                held = [r for r in group if r['lineage'] == lineage and r['setting'] == chosen['name']]
                cross_checked.append({'scenario': kind, 'lineage': lineage, 'geometry': geometry,
                                      'selected_setting': chosen['name'], **aggregate(held)})
    cross_summary = {kind: {geometry: aggregate([r for r in cross_checked if r['scenario'] == kind and r['geometry'] == geometry])
                           for geometry in ('physical', 'hybrid', 'response')} for kind in specification['scenarios']}
    screening = []
    for kind in specification['scenarios']:
        reference = cross_summary[kind]['physical']
        lookup = {(r['geometry'], r['lineage']): r for r in cross_checked if r['scenario'] == kind}
        for geometry in ('hybrid', 'response'):
            candidate = cross_summary[kind][geometry]
            positive = [lineage for name, lineage in lookup if name == geometry
                        and lookup[geometry, lineage]['nll'] < lookup['physical', lineage]['nll']]
            screening.append({'scenario': kind, 'geometry': geometry,
                'nll_difference': candidate['nll']-reference['nll'], 'brier_difference': candidate['brier']-reference['brier'],
                'positive_nll_lineages': positive,
                'supports_further_diagnosis': candidate['nll'] < reference['nll'] and candidate['brier'] <= reference['brier'] and len(positive) >= 2})
    numerical = {'status': 'passed', 'independent_component_predictions': 64,
        'independent_weight_and_mixture_predictions': 800,
        'component_maximum_error': max(c['maximum_error'] for r in records for c in r['probability_checks']),
        'weight_maximum_error': max(c['weight_maximum_error'] for r in records for c in r['weight_checks']),
        'mixture_maximum_error': max(c['mixture_maximum_error'] for r in records for c in r['weight_checks']),
        'unchanged_registered_scores_checked': len(registered_endpoint_checks),
        'unchanged_registered_score_maximum_error': max(registered_endpoint_checks)}
    result = {'status': 'complete_numerically_checked_historical_diagnosis', 'protocol': specification,
        'summary': summary, 'selected_by_geometry': selected, 'cross_checked_rows': cross_checked,
        'cross_checked_summary': cross_summary, 'screening': screening, 'rows': rows, 'numerical': numerical,
        'wall_s': time.perf_counter()-started, 'new_selector_queries': 0, 'new_physical_calls': 0,
        'discovery_admission_claimed': False, 'formal_phases_entered': False}
    write_json(REPORT/'report.json', result)
    lines = ['# 历史响应距离：历史诊断结果', '',
        '16个历史伪目标、800项查询集评分；没有新增开发选例或物理调用。',
        '带宽选择按谱系留出，但基础教师见过全部历史对象；本项不作为严格留家族验证。', '',
        '| 场景 | 距离 | 留出谱系NLL | 留出谱系Brier |', '|---|---|---:|---:|']
    for kind, group in cross_summary.items():
        for geometry, metrics in group.items():
            lines.append(f"| {kind} | {geometry} | {metrics['nll']:.6f} | {metrics['brier']:.6f} |")
    lines += ['', '## 后续筛查', '']
    for row in screening:
        lines.append(f"- {row['scenario']} / {row['geometry']}：NLL差{row['nll_difference']:+.6f}，"
                     f"Brier差{row['brier_difference']:+.6f}；正收益谱系{len(row['positive_nll_lineages'])}个；"
                     f"满足进一步诊断条件：{row['supports_further_diagnosis']}。")
    lines += ['', f"64组独立分量概率、800组权重及混合概率核对通过，最大分量差{numerical['component_maximum_error']:.3g}。",
        '完整10项配置、所有支持规模及带宽选择结果见report.json。预测损失不代表碰撞发现收益，',
        '本项没有登记或启动新的开发搜索，也没有改变既有主方法、失败结论或C/D/E状态。',
        f"历史诊断耗时{result['wall_s']:.2f}秒；与在运行的开发回放共享CPU资源。"]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    {'prepare': prepare, 'run': run}[parser.parse_args().stage]()
