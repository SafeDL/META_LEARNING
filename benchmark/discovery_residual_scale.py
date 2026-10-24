"""Historical-only residual-amplitude learning with causal adaptation prefixes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.common.response import archive_fingerprints, source_weights
from methods.gcrt.conditional_adaptation import component_probabilities
from methods.gcrt.joint_response import margin_values, posterior
from methods.gcrt.joint_training import load_model
from methods.gcrt.residual_scale import cached_covariance, residual_prior, update_reliability

from .construction import load_construction
from .discovery_conditional_adaptation import probabilities_numpy
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_joint_response_audit import predict_numpy, prepare_numpy
from .discovery_physical_mixture import MAIN, REPORT as FOUNDATION, checkpoint_path, protocol as foundation_protocol
from .discovery_physical_mixture_audit import classifier_probabilities_numpy
from .protocol import ROOT


REPORT = OUTPUT/'residual_scale'
ARTIFACTS = ROOT/'results/gcrt/collision/discovery/residual_scale'
COUNTS = (8, 16, 32, 64, 128)
BETAS = (0., .1, .3, 1., 3.)


def protocol():
    return {'purpose': 'historical parameter learning, not independent validation or new search runs',
        'scenarios': ['cut_in', 'cut_out'], 'pseudo_targets_per_scenario': 8,
        'support_count': 128, 'query_count': 32, 'counts': list(COUNTS), 'permutation_seed': 1301,
        'beta_grid': list(BETAS), 'main_bandwidth': .3,
        'target_adaptation': {'steps': 10, 'learning_rate': .003},
        'raw_adaptation': {'steps': 10, 'learning_rate': .01},
        'source_prior_mass': .75, 'target_prior_mass': .25, 'physical_mass': .1,
        'selection': 'per-scenario local-weight query NLL; Brier; distance to beta=1; beta',
        'foundation': foundation_protocol(),
        'source_files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'benchmark/discovery_residual_scale.py', 'methods/gcrt/residual_scale.py',
            'methods/gcrt/conditional_adaptation.py', 'methods/baselines/fsbo/method.py')},
        'new_selector_queries': 0, 'new_physical_calls': 0}


def cache_path(kind, source_id):
    return ARTIFACTS/'adaptation'/kind/f'{source_id}.npz'


def prepare():
    audit = read_json(FOUNDATION/'audit/report.json')
    if audit['status'] != 'passed' or audit['runs'] != 320:
        raise ValueError('Finish the preceding frozen online comparison first')
    specification = protocol()
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('Historical residual learning already registered differently')
    write_json(path, specification)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        for kind in specification['scenarios']:
            history, _ = load_construction(kind)
            if any(a.role != 'history' or not np.array_equal(a.x, history[0].x) for a in history):
                raise ValueError('Historical source priors require the registered shared historical pool')
            model, _ = load_model(checkpoint_path(kind, MAIN))
            model = model.to(specification['foundation']['device'])
            source_spec = {'protocol': specification, 'history_fingerprints': archive_fingerprints(history),
                           'source_ids': [a.source_id for a in history], 'x': history[0].x.tolist()}
            destination = ARTIFACTS/'priors'/f'{kind}.npz'
            if destination.exists():
                with np.load(destination, allow_pickle=False) as old:
                    if json.loads(str(old['specification'])) != source_spec:
                        raise ValueError('Historical-coordinate source priors changed')
                continue
            started = time.perf_counter()
            means, covariances, checks = [], [], []
            for archive in history:
                count = len(archive.x)
                with torch.no_grad():
                    m, c = model.joint_mean(count), model.joint_covariance(archive.x)
                    _, result_m, result_c = posterior(m, c, list(range(count)), archive.collision,
                                                       margin_values(archive.safety[:, 0]))
                    cm = result_m[:count].cpu().numpy()
                    cc = result_c[:count, :count].cpu().numpy()
                    cc = (cc+cc.T)/2
                    np.linalg.cholesky(cc+1e-7*np.eye(count))
                    m_np, c_np = m.cpu().numpy()[None], c.cpu().numpy()[None]
                state = prepare_numpy(m_np, c_np, range(count), archive.collision.astype(float), margin_values(archive.safety[:, 0]))
                check_m, check_c = predict_numpy(m_np, c_np, state, range(count), full_covariance=True)
                np.testing.assert_allclose(cm, check_m[0], atol=2e-7, rtol=2e-6)
                np.testing.assert_allclose(cc, check_c[0], atol=2e-7, rtol=2e-6)
                means.append(cm)
                covariances.append(cc)
                checks.append({'source_id': archive.source_id,
                    'mean_max_error': float(np.max(np.abs(cm-check_m[0]))),
                    'covariance_max_error': float(np.max(np.abs(cc-check_c[0])))})
                print('residual_history_prior', kind, archive.source_id, flush=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, specification=np.asarray(json.dumps(source_spec)),
                                means=np.array(means), covariances=np.array(covariances))
            write_json(destination.with_suffix('.json'), {'status': 'complete', 'specification': source_spec,
                'source_count': len(history), 'checks': checks, 'wall_s': time.perf_counter()-started,
                'new_physical_calls': 0})
    return specification


def cache_job(kind, source_id, specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, _ = load_construction(kind)
        archive = next(a for a in history if a.source_id == source_id)
        if archive.role != 'history':
            raise ValueError('Residual parameter learning cannot adapt on development targets')
        ordered = np.random.default_rng(1301).permutation(512)[:160]
        source_spec = {'protocol': specification, 'kind': kind, 'source_id': source_id,
            'role': archive.role, 'history_fingerprints': archive_fingerprints(history),
            'ordered_indices': ordered.tolist()}
        destination = cache_path(kind, source_id)
        if destination.exists():
            with np.load(destination, allow_pickle=False) as old:
                if json.loads(str(old['specification'])) != source_spec or old['encoded'].shape != (129, 160, 32):
                    raise ValueError('Adaptation cache differs from the registered historical prefix')
            report = read_json(destination.with_suffix('.json'))
            if report['status'] != 'complete':
                raise ValueError('Historical cache is incomplete')
            return report
        base, _ = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
        base = base.to(specification['foundation']['device'])
        raw_base = ClassificationKernel(archive.x.shape[1]).to(base.mean.device)
        model, raw = copy.deepcopy(base), copy.deepcopy(raw_base)
        encoded, parameters, raw_probabilities, checks = [], [], [], []
        started = time.perf_counter()
        for count in range(129):
            paid = ordered[:count].tolist()
            labels = archive.collision[paid]
            with torch.no_grad():
                z = model.features(archive.x)[ordered]
                parameters.append([float(model.mean), float(model.log_lengthscale), float(model.log_scale)])
                encoded.append(z.cpu().numpy())
                raw_p = collision_probabilities(raw, archive.x[paid], labels, archive.x[ordered]).cpu().numpy()
                raw_probabilities.append(raw_p)
                if count in (0, 8, 64, 128):
                    expected = model.covariance(archive.x)[ordered][:, ordered]
                    rebuilt = cached_covariance(z, model.log_lengthscale, model.log_scale)
                    kernel_error = float((expected-rebuilt).abs().max())
                    torch.testing.assert_close(rebuilt, expected, atol=2e-9, rtol=2e-8)
            if count in (0, 8, 64, 128):
                independent = classifier_probabilities_numpy(raw, archive.x[ordered], list(range(count)), labels)
                np.testing.assert_allclose(raw_p, independent, atol=2e-9, rtol=2e-9)
                checks.append({'support_count': count, 'kernel_max_error': kernel_error,
                               'raw_probability_max_error': float(np.max(np.abs(raw_p-independent)))})
            if count < 128:
                next_paid = ordered[:count+1]
                model, raw = copy.deepcopy(base), copy.deepcopy(raw_base)
                fit_target(model, archive.x[next_paid], archive.collision[next_paid], steps=10, learning_rate=.003)
                fit_target(raw, archive.x[next_paid], archive.collision[next_paid], steps=10, learning_rate=.01)
            if count % 16 == 0:
                print('residual_adaptation_cache', kind, source_id, count, '/ 128', flush=True)
        if model.mean.is_cuda:
            torch.cuda.synchronize()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, specification=np.asarray(json.dumps(source_spec)),
                encoded=np.array(encoded), parameters=np.array(parameters), raw_probabilities=np.array(raw_probabilities),
                x=archive.x[ordered], labels=archive.collision[ordered], ordered_indices=ordered)
        temporary.replace(destination)
        result = {'status': 'complete', 'specification': source_spec, 'states': 129,
            'support_feedback_count': 128, 'query_count': 32, 'checks': checks,
            'wall_s': time.perf_counter()-started, 'new_selector_queries': 0, 'new_physical_calls': 0}
        write_json(destination.with_suffix('.json'), result)
        return result


def cache():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol():
        raise ValueError('Historical cache implementation changed after registration')
    tasks = [(kind, a.source_id) for kind in specification['scenarios'] for a in load_construction(kind)[0]]
    completed = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cache_job, kind, name, specification) for kind, name in tasks]
        try:
            for future in as_completed(futures):
                completed.append(future.result())
                print('residual_cache_complete', len(completed), '/ 16', flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    write_json(REPORT/'cache_index.json', {'status': 'complete', 'protocol': specification,
        'records': sorted(completed, key=lambda r: (r['specification']['kind'], r['specification']['source_id'])),
        'historical_sequences': 16, 'new_selector_queries': 0, 'new_physical_calls': 0})


def audit():
    specification = read_json(REPORT/'protocol.json')
    index = read_json(REPORT/'cache_index.json')
    if specification != protocol() or index['status'] != 'complete' or len(index['records']) != 16:
        raise ValueError('Complete every historical cache under the same registered implementation')
    torch.set_num_threads(1)
    checks = []
    with threadpool_limits(1):
        for record in index['records']:
            setting = record['specification']
            kind, name = setting['kind'], setting['source_id']
            history, _ = load_construction(kind)
            archive = next(a for a in history if a.source_id == name)
            with np.load(cache_path(kind, name), allow_pickle=False) as saved:
                if json.loads(str(saved['specification'])) != setting:
                    raise ValueError('Saved cache and completed index disagree')
                ordered = saved['ordered_indices']
                np.testing.assert_array_equal(ordered, np.random.default_rng(1301).permutation(512)[:160])
                np.testing.assert_array_equal(saved['x'], archive.x[ordered])
                np.testing.assert_array_equal(saved['labels'], archive.collision[ordered])
                if len(set(ordered[:128]) & set(ordered[128:])):
                    raise ValueError('Historical query labels entered the support sequence')
                base, _ = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
                base = base.to(specification['foundation']['device'])
                for count in (0, 8, 64, 128):
                    model = copy.deepcopy(base)
                    if count:
                        fit_target(model, archive.x[ordered[:count]], archive.collision[ordered[:count]], steps=10, learning_rate=.003)
                    with torch.no_grad():
                        features = model.features(archive.x)[ordered].cpu().numpy()
                    np.testing.assert_allclose(saved['encoded'][count], features, atol=2e-7, rtol=2e-6)
                    np.testing.assert_allclose(saved['parameters'][count],
                        [float(model.mean), float(model.log_lengthscale), float(model.log_scale)], atol=2e-8, rtol=2e-7)
                    checks.append({'kind': kind, 'pseudo_target': name, 'support_count': count,
                                   'feature_max_error': float(np.max(np.abs(saved['encoded'][count]-features)))})
    result = {'status': 'passed', 'historical_sequences': 16, 'independent_prefix_refits': len(checks),
              'checks': checks, 'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(REPORT/'cache_audit.json', result)
    return result


def fit():
    specification = read_json(REPORT/'protocol.json')
    if (specification != protocol() or read_json(REPORT/'cache_audit.json')['status'] != 'passed'):
        raise ValueError('Independently audit all historical prefix caches before choosing beta')
    torch.set_num_threads(1)
    device = specification['foundation']['device']
    rows, checks = [], []
    started = time.perf_counter()
    with threadpool_limits(1):
        for kind in specification['scenarios']:
            history, _ = load_construction(kind)
            with np.load(ARTIFACTS/'priors'/f'{kind}.npz', allow_pickle=False) as prior:
                expected = {'protocol': specification, 'history_fingerprints': archive_fingerprints(history),
                            'source_ids': [a.source_id for a in history], 'x': history[0].x.tolist()}
                if json.loads(str(prior['specification'])) != expected:
                    raise ValueError('Historical source prior identity changed')
                means, covariances = prior['means'].copy(), prior['covariances'].copy()
            for archive in history:
                sources = [a for a in history if a.source_id != archive.source_id]
                source_positions = [i for i, a in enumerate(history) if a.source_id != archive.source_id]
                if any(a.source_id == archive.source_id for a in sources) or len(sources) != 7:
                    raise ValueError('The pseudo target leaked into its own source context')
                with np.load(cache_path(kind, archive.source_id), allow_pickle=False) as saved:
                    ordered, labels = saved['ordered_indices'], saved['labels']
                    z = torch.as_tensor(saved['encoded'], dtype=torch.float64, device=device)
                    parameters = torch.as_tensor(saved['parameters'], dtype=torch.float64, device=device)
                    public_x = torch.as_tensor(saved['x'], dtype=torch.float64, device=device)
                    raw_p = saved['raw_probabilities']
                sm = torch.as_tensor(means[source_positions][:, ordered], dtype=torch.float64, device=device)
                sc = torch.as_tensor(covariances[source_positions][:, ordered, :][:, :, ordered], dtype=torch.float64, device=device)
                initial = np.r_[.75*source_weights(sources), .25]
                for beta in BETAS:
                    scores = {mode: torch.as_tensor(initial, dtype=torch.float64, device=device).log()[:, None].expand(-1, 160).clone()
                              for mode in ('local', 'global')}
                    for count in range(129):
                        mean, length, scale = parameters[count]
                        covariance = cached_covariance(z[count], length, scale)
                        m, c = residual_prior(mean, covariance, sm, sc, parameters[0, 0], beta)
                        with torch.no_grad():
                            p = component_probabilities(m, c, list(range(count)), labels[:count]).cpu().numpy()
                        if count in (0, 8, 64, 128):
                            independent = probabilities_numpy(m.cpu().numpy(), c.cpu().numpy(), list(range(count)), labels[:count])
                            np.testing.assert_allclose(p, independent, atol=2e-7, rtol=2e-6)
                            checks.append({'scenario': kind, 'pseudo_target': archive.source_id, 'beta': beta,
                                'support_count': count, 'probability_max_error': float(np.max(np.abs(p-independent)))})
                        p_tensor = torch.as_tensor(p, dtype=torch.float64, device=device)
                        if count in COUNTS:
                            for mode, log_weight in scores.items():
                                q = (.9*(log_weight.softmax(0)[:, 128:]*p_tensor[:, 128:]).sum(0)).cpu().numpy()+.1*raw_p[count, 128:]
                                q = np.clip(q, 1e-12, 1-1e-12)
                                truth = labels[128:]
                                rows.append({'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
                                    'beta': beta, 'reliability': mode, 'support_count': count,
                                    'source_context_ids': [a.source_id for a in sources],
                                    'query_nll': float(-np.mean(truth*np.log(q)+(1-truth)*np.log1p(-q))),
                                    'query_brier': float(np.square(q-truth).mean())})
                        if count < 128:
                            for mode in scores:
                                scores[mode] = update_reliability(scores[mode], p_tensor, count, bool(labels[count]),
                                                                  public_x, .3 if mode == 'local' else None)
                    print('residual_beta_fit', kind, archive.source_id, beta, flush=True)
    if len(rows) != 800 or specification != protocol():
        raise ValueError('Residual selection requires all 16 histories, 5 amplitudes and 5 support budgets')
    choices, summary = {}, []
    for kind in specification['scenarios']:
        for beta in BETAS:
            for mode in ('local', 'global'):
                group = [r for r in rows if (r['scenario'], r['beta'], r['reliability']) == (kind, beta, mode)]
                lineages = sorted({r['lineage'] for r in group})
                summary.append({'scenario': kind, 'beta': beta, 'reliability': mode,
                    **{metric: float(np.mean([np.mean([r[metric] for r in group if r['lineage'] == lineage])
                                             for lineage in lineages])) for metric in ('query_nll', 'query_brier')}})
        eligible = [r for r in summary if r['scenario'] == kind and r['reliability'] == 'local']
        choices[kind] = min(eligible, key=lambda r: (r['query_nll'], r['query_brier'], abs(r['beta']-1), r['beta']))['beta']
    result = {'status': 'complete', 'protocol': specification, 'selected_beta': choices, 'rows': rows,
        'summary': summary, 'wall_s': time.perf_counter()-started, 'new_selector_queries': 0, 'new_physical_calls': 0,
        'scope': 'historical parameter learning; not a discovery gate or held-system evaluation'}
    write_json(REPORT/'selection.json', result)
    write_json(REPORT/'prediction_checks.json', {'status': 'passed', 'prefix_comparisons': len(checks), 'checks': checks})
    lines = ['# 历史系统间残差幅度学习', '',
        '16条历史伪目标序列，每条128个随机有序支持点及不相交的32点查询集。',
        '自身来源从上下文排除；基础教师曾使用全部历史对象，因此不是严格留系统验证。',
        '模型适配和可靠性按真实顺序重建；后32个查询标签只用于历史参数学习损失。', '',
        '| 场景 | β | 可靠性 | 查询NLL | Brier |', '|---|---:|---|---:|---:|']
    for row in summary:
        lines.append(f"| {row['scenario']} | {row['beta']} | {row['reliability']} | {row['query_nll']:.6f} | {row['query_brier']:.6f} |")
    lines += ['', f'预定局部配置选定幅度：{choices}。',
              '选择只使用历史响应；表中误差不是128次碰撞发现结果。新在线矩阵须另行登记，并保留β=1及相同β的全局控制。',
              '旧失败结论、22种强参考和阶段C/D/E状态保持。']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'cache', 'audit', 'fit'))
    {'prepare': prepare, 'cache': cache, 'audit': audit, 'fit': fit}[parser.parse_args().stage]()
