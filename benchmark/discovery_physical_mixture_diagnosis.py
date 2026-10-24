"""Retrospective model and query-path diagnosis on already paid PPO prefixes."""

import argparse
import copy
import time

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.classification import logistic_moments, torch_moments
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.gcrt.joint_prediction import marginal_posterior
from methods.gcrt.joint_response import fit_joint
from methods.gcrt.joint_search import transfer_prior
from methods.gcrt.joint_training import load_model

from . import discovery_joint_response as joint
from .construction import load_construction
from .discovery_config import development_tasks, read_json, write_json
from .discovery_joint_response_audit import predict_numpy, prepare_numpy, prior_numpy
from .discovery_physical_mixture import MAIN, REPORT, RUNS, checkpoint_path, protocol
from .discovery_physical_mixture_audit import classifier_probabilities_numpy
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT
from .scenarios import features
from .schema import QueryFeedback


DESTINATION = REPORT/'paid_prefix_diagnosis'
PREFIXES = (64, 96, 127)
POINTS = (90, 386)
PATHS = (MAIN, 'physical_fsbo')


def assess(probability, available, truth, candidates, points):
    p = np.clip(probability[available], 1e-12, 1-1e-12)
    y = truth[available]
    ranking = sorted(available, key=lambda i: (-probability[i], candidates[i].scenario_id))
    positions = {index: rank for rank, index in enumerate(ranking, 1)}
    return {
        'collision_nll': float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
        'brier': float(np.square(p-y).mean()),
        'remaining_collision_count': int(y.sum()),
        'focus': {str(i): {'probability': float(probability[i]), 'rank': positions.get(i),
                          'already_queried': i not in positions,
                          'scenario_id': candidates[i].scenario_id} for i in points},
    }


def diagnose():
    frozen = read_json(REPORT/'protocol.json')
    if frozen != protocol():
        raise ValueError('Paid-prefix diagnosis requires the unchanged physical-mixture protocol')
    endpoint = read_json(REPORT/'main_endpoint_comparison.json')
    if endpoint['status'] != 'main_40_runs_complete_and_independently_audited':
        raise ValueError('Finish the complete independently checked main arm first')
    tasks = [t for t in development_tasks() if t['target_id'] == 'development_ppo' and t['scenario'] == 'cut_out']
    specification = {
        'tasks': tasks, 'query_paths': list(PATHS), 'paid_prefixes': list(PREFIXES),
        'focus_indices_from_completed_development_diagnosis': list(POINTS),
        'adaptation': frozen['joint_adaptation'], 'raw_adaptation': frozen['physical_adaptation'],
        'physical_mass': .1, 'online_fit_feedback': 'paid collision labels only',
        'additional_selector_queries': 0, 'new_physical_calls': 0,
        'purpose': 'retrospective prediction and ranking diagnosis, not additional discoveries',
    }
    path = DESTINATION/'protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('Paid-prefix diagnosis was registered differently')
    write_json(path, specification)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, _ = load_construction('cut_out')
        bank = ExecutionBank(DATA_ROOT/'physical/development/cut_out/development_ppo')
        feedback = [bank(s).feedback.to_dict() for s in bank.candidates]
        truth = np.array([f['collision'] for f in feedback], dtype=bool)
        x = features(bank.candidates)
        base, _ = load_model(checkpoint_path('cut_out', MAIN))
        fsbo_base, _ = load_fsbo(checkpoint_path('cut_out', 'physical_fsbo'))
        device = frozen['device']
        base, fsbo_base = base.to(device), fsbo_base.to(device)
        first = read_json(RUNS/MAIN/tasks[0]['target_id']/'cut_out'/f"{tasks[0]['algorithm_seed']}.json")
        sources = joint.read_prior('cut_out', bank.candidates, history, first['specification']['checkpoint_sha256'])
        tensor_sources = {k: torch.as_tensor(v, dtype=torch.float64, device=device)
                          for k, v in sources.items() if k != 'names'}
        prior = np.r_[.75*sources['weights'], .25]
        rows, checks, overrides, predictions = [], [], [], {}
        started = time.perf_counter()
        for method in PATHS:
            for task in tasks:
                seed = task['algorithm_seed']
                saved = read_json(RUNS/method/task['target_id']/'cut_out'/f'{seed}.json')
                audited = read_json(REPORT/'audit/runs'/method/task['target_id']/'cut_out'/f'{seed}.json')
                queries = saved['result']['queries']
                if (audited['status'] != 'passed' or saved['specification']['protocol'] != frozen
                        or len(queries) != 128 or len({q['index'] for q in queries}) != 128
                        or [q['index'] for q in queries[:8]] != initial_indices(512, 8, seed)
                        or any(QueryFeedback.from_dict(q['feedback']).to_dict() != feedback[q['index']] for q in queries)):
                    raise ValueError('Diagnosis inputs differ from the audited paid trajectory')
                for prefix in PREFIXES:
                    paid = [q['index'] for q in queries[:prefix]]
                    labels = np.array([q['feedback']['collision'] for q in queries[:prefix]], dtype=bool)
                    available = sorted(set(range(512))-set(paid))
                    next_query = queries[prefix]
                    index = next_query['index']
                    raw = ClassificationKernel(x.shape[1]).to(device)
                    fit_target(raw, x[paid], labels, steps=10, learning_rate=.01)
                    fsbo = copy.deepcopy(fsbo_base)
                    fit_target(fsbo, x[paid], labels, steps=10, learning_rate=.003)
                    with torch.no_grad():
                        raw_p = collision_probabilities(raw, x[paid], labels, x).cpu().numpy()
                        fsbo_p = collision_probabilities(fsbo, x[paid], labels, x).cpu().numpy()
                    raw_np = classifier_probabilities_numpy(raw, x, paid, labels)
                    fsbo_np = classifier_probabilities_numpy(fsbo, x, paid, labels)
                    np.testing.assert_allclose(raw_p, raw_np, atol=2e-9, rtol=2e-9)
                    np.testing.assert_allclose(fsbo_p, fsbo_np, atol=2e-7, rtol=2e-6)
                    values = {'physical': raw_p, 'fsbo': fsbo_p, 'fsbo_mixture': .9*fsbo_p+.1*raw_p}
                    errors = {'physical': float(np.max(np.abs(raw_p-raw_np))),
                              'fsbo': float(np.max(np.abs(fsbo_p-fsbo_np)))}
                    if method == MAIN:
                        model = copy.deepcopy(base)
                        clearance = [q['feedback']['safety']['min_clearance_m'] for q in queries[:prefix]]
                        fit_joint(model, x[paid], labels, clearance, steps=10, learning_rate=.003, use_margin=False)
                        with torch.no_grad():
                            mean, covariance = transfer_prior(model, x, tensor_sources, 'joint_local')
                            mean, variance, _ = marginal_posterior(mean, covariance, paid, labels, None)
                            component_p = torch_moments(mean[:, :512], variance[:, :512])[0].cpu().numpy()
                        numpy_mean, numpy_covariance = prior_numpy(model, x, sources, 'joint_local')
                        state = prepare_numpy(numpy_mean, numpy_covariance, paid, labels.astype(float), None)
                        mean_np, variance_np = predict_numpy(numpy_mean, numpy_covariance, state, range(512))
                        component_np = logistic_moments(mean_np, variance_np)[0]
                        np.testing.assert_allclose(component_p, component_np, atol=2e-7, rtol=2e-6)
                        score = np.log(prior)+np.array([q['prequery_joint_log_likelihood'] for q in queries[:prefix]]).sum(0)
                        weight = np.exp(score-logsumexp(score))
                        np.testing.assert_allclose(weight, next_query['prequery_joint_weights'], atol=2e-12)
                        group = weight @ component_p
                        values.update(group=group, main_mixture=.9*group+.1*raw_p)
                        errors['source_components'] = float(np.max(np.abs(component_p-component_np)))
                        actual = values['main_mixture']
                        for point in POINTS:
                            if point in available:
                                overrides.append({'path': method, 'seed': seed, 'prefix': prefix, 'index': point,
                                    'selected_next': index, 'selected_collision': bool(truth[index]),
                                    'raw_prefers_focus_to_selected': bool(raw_p[point] > raw_p[index]+1e-12),
                                    'weighted_group_gap': float(.9*(group[index]-group[point])),
                                    'weighted_physical_gap': float(.1*(raw_p[index]-raw_p[point])),
                                    'mixture_gap': float(actual[index]-actual[point]),
                                    'source_names': saved['result']['component_names'][:-1],
                                    'source_weights': weight.tolist(),
                                    'source_probabilities_at_focus': component_p[:, point].tolist()})
                    else:
                        actual = values['fsbo_mixture']
                    np.testing.assert_allclose(raw_p[index], next_query['prequery_physical_probability'], atol=2e-9, rtol=2e-9)
                    np.testing.assert_allclose(actual[index], next_query['prequery_probability'], atol=2e-7, rtol=2e-6)
                    maximum = max(actual[i] for i in available)
                    if actual[index] < maximum-2e-7:
                        raise ValueError('Diagnosis fails to reconstruct the actual next selection')
                    checks.append({'path': method, 'seed': seed, 'prefix': prefix,
                                   'independent_probability_max_errors': errors,
                                   'actual_next_probability_error': float(abs(actual[index]-next_query['prequery_probability'])),
                                   'actual_next_acquisition_gap': float(maximum-actual[index])})
                    for name, probability in values.items():
                        rows.append({'path': method, 'seed': seed, 'prefix': prefix, 'predictor': name,
                                     'evaluation_count': len(available),
                                     **assess(probability, available, truth, bank.candidates, POINTS)})
                        predictions[f'{method}/{seed}/{prefix}/{name}'] = probability
                    print('physical_prefix_diagnosis', len(checks), '/ 30', method, seed, prefix, flush=True)
        if protocol() != frozen:
            raise ValueError('Frozen experiment changed during retrospective diagnosis')
        np.savez_compressed(DESTINATION/'probabilities.npz', **predictions)
        result = {'status': 'complete', 'specification': specification, 'paid_prefixes': len(checks),
                  'prediction_comparisons': len(rows), 'rows': rows, 'ranking_gap_diagnosis': overrides,
                  'additional_selector_queries': 0, 'new_physical_calls': 0,
                  'wall_s': time.perf_counter()-started,
                  'scope': 'one development lineage and scenario; selected failure coordinates; descriptive diagnosis only'}
        write_json(DESTINATION/'numerical_checks.json', {'status': 'passed', 'prefixes': 30,
                   'full_pool_probability_comparisons': 75, 'actual_next_selections_checked': 30, 'checks': checks})
        write_json(DESTINATION/'report.json', result)
        report()
        return result


def report():
    data = read_json(DESTINATION/'report.json')
    numerical = read_json(DESTINATION/'numerical_checks.json')
    if data['status'] != 'complete' or numerical['status'] != 'passed' or len(data['rows']) != 120:
        raise ValueError('Finish the matched predictions and independent numerical checks first')
    rows = data['rows']
    bounds, bound_checks = [], 0
    with np.load(DESTINATION/'probabilities.npz', allow_pickle=False) as predictions:
        for task in data['specification']['tasks']:
            seed = task['algorithm_seed']
            saved = read_json(RUNS/MAIN/task['target_id']/'cut_out'/f'{seed}.json')
            for prefix in PREFIXES:
                paid = {q['index'] for q in saved['result']['queries'][:prefix]}
                available = sorted(set(range(512))-paid)
                group = predictions[f'{MAIN}/{seed}/{prefix}/group']
                raw = predictions[f'{MAIN}/{seed}/{prefix}/physical']
                for point in POINTS:
                    if point in paid:
                        continue
                    dominating = (group[available] > group[point]+1e-12) & (raw[available] > raw[point]+1e-12)
                    lower_bound = 1+int(dominating.sum())
                    for mass in (0., .1, .5, .9, 1.):
                        mixed = (1-mass)*group+mass*raw
                        if 1+int((mixed[available] > mixed[point]).sum()) < lower_bound:
                            raise ValueError('Static convex-mixture rank violates the dominance bound')
                        bound_checks += 1
                    bounds.append({'seed': seed, 'prefix': prefix, 'index': point,
                                   'strictly_dominating_candidates': int(dominating.sum()),
                                   'rank_lower_bound': lower_bound,
                                   'remaining_budget': 128-prefix})
    write_json(DESTINATION/'convex_mixture_bounds.json', {
        'scope': 'fixed-prefix static ranking; not a bound on future adaptive query trajectories',
        'rule': 'Any candidate strictly higher in both group and physical probabilities remains higher for every nonnegative convex mixing weight.',
        'rows': bounds, 'formula_sanity_checks': bound_checks,
        'additional_selector_queries': 0, 'mixture_weights_selected': False})
    lines = ['# PPO切出剩余漏检：相同付费前缀诊断', '',
             '使用主方法和FSBO物理混合各五个已完成轨迹，在64、96、127次付费查询后重建预测。',
             '共30个前缀、120项预测比较；拟合只用该前缀已付费的碰撞标签，未新增选例或物理调用。',
             '完整标签仅用于评估剩余候选。FSBO在主方法前缀上的预测是同反馈比较，不是新的在线发现结果。', '',
             '| 轨迹来源 | 前缀 | 预测器 | 剩余库平均NLL | Brier |', '|---|---:|---|---:|---:|']
    for method in PATHS:
        for prefix in PREFIXES:
            for predictor in sorted({r['predictor'] for r in rows if r['path'] == method}):
                group = [r for r in rows if (r['path'], r['prefix'], r['predictor']) == (method, prefix, predictor)]
                lines.append(f"| {method} | {prefix} | {predictor} | {np.mean([r['collision_nll'] for r in group]):.6f} | {np.mean([r['brier'] for r in group]):.6f} |")
    lines += ['', '| 主方法127次前缀的点 | 预测器 | 尚未查询的种子数 | 概率范围 | 剩余排名范围 |',
              '|---:|---|---:|---:|---:|']
    for point in POINTS:
        for name in ('group', 'physical', 'main_mixture', 'fsbo_mixture'):
            values = [r['focus'][str(point)] for r in rows if (r['path'], r['prefix'], r['predictor']) == (MAIN, 127, name)
                      and not r['focus'][str(point)]['already_queried']]
            probability = f"{min(v['probability'] for v in values):.6g}–{max(v['probability'] for v in values):.6g}" if values else '已全部查询'
            rank = f"{min(v['rank'] for v in values)}–{max(v['rank'] for v in values)}" if values else '不适用'
            lines.append(f'| {point} | {name} | {len(values)} | {probability} | {rank} |')
    gaps = data['ranking_gap_diagnosis']
    count = sum(g['raw_prefers_focus_to_selected'] and not g['selected_collision'] for g in gaps)
    lines += ['', f'在{len(gaps)}个尚未查询的关注点/前缀比较中，有{count}次物理GP更偏好关注点，但混合后的实际下一点为安全场景。',
              '具体两部分排序差、来源权重及各来源概率见report.json；该比较解释当前排序，不等同于重跑一种选例策略的收益。', '',
              '| 索引90的付费前缀 | 任意非负凸混合的排名下界范围 | 剩余预算 |', '|---:|---:|---:|']
    for prefix in PREFIXES:
        values = [b['rank_lower_bound'] for b in bounds if (b['prefix'], b['index']) == (prefix, 90)]
        lines.append(f'| {prefix} | {min(values)}–{max(values)} | {128-prefix} |')
    lines += ['', '下界由同时在两类模型上严格高于关注点的未查询候选数计算，不搜索混合权重。',
              '在这15个索引90的固定前缀状态下，仅改变两组非负凸混合权重，静态排名仍超出剩余预算。',
              '该结论只约束当前固定预测；后续查询会改变反馈和模型，所以它不是任何自适应策略必然失败的证明。',
              '下一步应检查能提供不同失效邻域的目标表示，以及查询路径怎样改变模型；不把物理质量调大视作充分修复。', '',
              '75项全512点概率经独立NumPy Gaussian/Laplace推断核对，30项真实下一点概率和选点均重建通过。',
              '两类轨迹上的物理GP使用相同初始化、核和适配设置；它们的差异来自已获得反馈的位置和标签。',
              '分析只覆盖已确认的PPO切出开发漏检，不能外推为全部谱系或独立目标收益。', '',
              '![相同64次付费反馈下的漏检点排名](ranks.png)', '']
    (DESTINATION/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = {'group': 'Source group', 'physical': 'Physical GP',
              'main_mixture': 'Main mixture', 'fsbo_mixture': 'FSBO mixture (same feedback)'}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for point, axis in zip(POINTS, axes):
        for name, label in labels.items():
            group = sorted([r for r in rows if (r['path'], r['prefix'], r['predictor']) == (MAIN, 64, name)],
                           key=lambda r: r['seed'])
            axis.plot([str(r['seed']) for r in group], [r['focus'][str(point)]['rank'] for r in group],
                      marker='o', label=label)
        axis.set(title=f'PPO cut-out, index {point}', xlabel='Selection seed',
                 ylim=(1, 448), yticks=(1, 64, 128, 256, 384, 448))
        axis.grid(alpha=.2)
    axes[0].set_ylabel('Unqueried rank (lower is better)')
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc='lower center', ncol=2, fontsize=9)
    figure.tight_layout(rect=(0, .14, 1, 1))
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(DESTINATION/f'ranks.{extension}', dpi=180, bbox_inches='tight')
    plt.close(figure)


def diagnose_influence():
    """Delete one paid collision from fitting/conditioning while holding weights fixed."""
    destination = DESTINATION/'collision_influence'
    original = read_json(DESTINATION/'report.json')
    frozen = read_json(REPORT/'protocol.json')
    if original['status'] != 'complete' or frozen != protocol():
        raise ValueError('Complete the matched prefix diagnosis under the unchanged experiment first')
    specification = {
        'tasks': original['specification']['tasks'], 'path': MAIN, 'prefix': 127,
        'removed_paid_index': 386, 'focus_index': 90,
        'conditions': ['full_fit_full_support', 'full_fit_without_point', 'refit_without_point'],
        'source_group_weights': 'held at actual full-prefix prequery weights in all conditions',
        'candidate_ranking_pool': 'actual full-prefix remaining candidates in every condition',
        'additional_selector_queries': 0, 'new_physical_calls': 0,
    }
    registered = destination/'protocol.json'
    if registered.exists() and read_json(registered) != specification:
        raise ValueError('Paid-collision influence analysis was registered differently')
    write_json(registered, specification)
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows, checks, decompositions = [], [], []
    predictions = {}
    with threadpool_limits(1), np.load(DESTINATION/'probabilities.npz', allow_pickle=False) as expected:
        history, _ = load_construction('cut_out')
        bank = ExecutionBank(DATA_ROOT/'physical/development/cut_out/development_ppo')
        x = features(bank.candidates)
        joint_base, _ = load_model(checkpoint_path('cut_out', MAIN))
        fsbo_base, _ = load_fsbo(checkpoint_path('cut_out', 'physical_fsbo'))
        device = frozen['device']
        joint_base, fsbo_base = joint_base.to(device), fsbo_base.to(device)
        raw_base = ClassificationKernel(x.shape[1]).to(device)
        first = read_json(RUNS/MAIN/'development_ppo/cut_out/11.json')
        sources = joint.read_prior('cut_out', bank.candidates, history, first['specification']['checkpoint_sha256'])
        tensor_sources = {k: torch.as_tensor(v, dtype=torch.float64, device=device)
                          for k, v in sources.items() if k != 'names'}
        for task in specification['tasks']:
            seed = task['algorithm_seed']
            saved = read_json(RUNS/MAIN/task['target_id']/'cut_out'/f'{seed}.json')
            if saved['specification']['protocol'] != frozen:
                raise ValueError('Influence analysis trajectory differs from the frozen experiment')
            queries = saved['result']['queries'][:127]
            paid = [q['index'] for q in queries]
            if 90 in paid or (386 in paid and not next(q['feedback']['collision'] for q in queries if q['index'] == 386)):
                raise ValueError('Influence analysis requires an unqueried focus and a paid collision to remove')
            subsets = {'full': queries, 'without': [q for q in queries if q['index'] != 386]}
            available = sorted(set(range(512))-set(paid))
            weight = np.asarray(saved['result']['queries'][127]['prequery_joint_weights'])
            for name, base in (('group', joint_base), ('fsbo', fsbo_base), ('physical', raw_base)):
                fitted = {}
                for subset, feedback in subsets.items():
                    support = [q['index'] for q in feedback]
                    labels = [q['feedback']['collision'] for q in feedback]
                    model = copy.deepcopy(base)
                    if name == 'group':
                        clearance = [q['feedback']['safety']['min_clearance_m'] for q in feedback]
                        fit_joint(model, x[support], labels, clearance, steps=10, learning_rate=.003, use_margin=False)
                    else:
                        fit_target(model, x[support], labels, steps=10, learning_rate=.01 if name == 'physical' else .003)
                    fitted[subset] = model
                values = {}
                for condition, model_key, support_key in (
                        ('full_fit_full_support', 'full', 'full'),
                        ('full_fit_without_point', 'full', 'without'),
                        ('refit_without_point', 'without', 'without')):
                    model = fitted[model_key]
                    support = [q['index'] for q in subsets[support_key]]
                    labels = np.array([q['feedback']['collision'] for q in subsets[support_key]], dtype=bool)
                    with torch.no_grad():
                        if name == 'group':
                            mean, covariance = transfer_prior(model, x, tensor_sources, 'joint_local')
                            mean, variance, _ = marginal_posterior(mean, covariance, support, labels, None)
                            components = torch_moments(mean[:, :512], variance[:, :512])[0].cpu().numpy()
                            p = weight @ components
                        else:
                            p = collision_probabilities(model, x[support], labels, x).cpu().numpy()
                    if name == 'group':
                        np_mean, np_covariance = prior_numpy(model, x, sources, 'joint_local')
                        state = prepare_numpy(np_mean, np_covariance, support, labels.astype(float), None)
                        np_mean, np_variance = predict_numpy(np_mean, np_covariance, state, range(512))
                        np_components = logistic_moments(np_mean, np_variance)[0]
                        np.testing.assert_allclose(components, np_components, atol=2e-7, rtol=2e-6)
                        p_np = weight @ np_components
                    else:
                        p_np = classifier_probabilities_numpy(model, x, support, labels)
                    np.testing.assert_allclose(p, p_np, atol=2e-7, rtol=2e-6)
                    if condition == 'full_fit_full_support':
                        np.testing.assert_allclose(p, expected[f'{MAIN}/{seed}/127/{name}'], atol=2e-7, rtol=2e-6)
                    ranking = sorted(available, key=lambda i: (-p[i], bank.candidates[i].scenario_id))
                    rows.append({'seed': seed, 'predictor': name, 'condition': condition,
                                 'removed': 386 in paid, 'support_count': len(support),
                                 'probability_at_90': float(p[90]), 'rank_at_90': ranking.index(90)+1,
                                 'source_weights_held_fixed': weight.tolist() if name == 'group' else None})
                    checks.append({'seed': seed, 'predictor': name, 'condition': condition,
                                   'independent_probability_max_error': float(np.max(np.abs(p-p_np)))})
                    values[condition] = p
                    predictions[f'{seed}/{name}/{condition}'] = p
                full, fixed, refitted = (values[c][90] for c in specification['conditions'])
                if 386 not in paid:
                    for p in values.values():
                        np.testing.assert_allclose(p, values['full_fit_full_support'], atol=2e-12, rtol=2e-12)
                decompositions.append({'seed': seed, 'predictor': name, 'removed': 386 in paid,
                                       'posterior_conditioning_effect': float(full-fixed),
                                       'refitting_effect': float(fixed-refitted),
                                       'total_effect': float(full-refitted)})
            print('paid_collision_influence', seed, 'complete', flush=True)
    if protocol() != frozen:
        raise ValueError('Frozen experiment changed during influence analysis')
    np.savez_compressed(destination/'probabilities.npz', **predictions)
    result = {'status': 'complete', 'specification': specification, 'rows': rows,
              'effect_decomposition': decompositions, 'prediction_comparisons': len(rows),
              'wall_s': time.perf_counter()-started, 'additional_selector_queries': 0,
              'new_physical_calls': 0, 'scope': 'predictor influence, not an alternative online query trajectory'}
    write_json(destination/'report.json', result)
    write_json(destination/'numerical_checks.json', {'status': 'passed', 'full_pool_probability_checks': 45,
               'original_full_prefix_predictions_checked': 15, 'unchanged_support_controls': 3, 'checks': checks})
    lines = ['# 已付费碰撞对模型的影响', '',
             '对主方法五个127次付费前缀，比较完整拟合、固定完整参数后移除索引386的后验条件、移除后重新适配。',
             '四个种子实际移除一个已付费碰撞，种子11原本未查询该点，作为支持集不变对照。',
             '所有排名都在原始前缀的同一剩余候选集计算；没有新增查询或仿真。', '',
             '| 预测器 | 种子 | 实际移除 | 完整拟合：概率/排名 | 固定参数移除：概率/排名 | 移除后重适配：概率/排名 |',
             '|---|---:|---|---:|---:|---:|']
    for name in ('group', 'fsbo', 'physical'):
        for seed in sorted({r['seed'] for r in rows}):
            group = [r for r in rows if (r['predictor'], r['seed']) == (name, seed)]
            values = ['{:.6f}/{}'.format(r['probability_at_90'], r['rank_at_90']) for r in group]
            lines.append(f"| {name} | {seed} | {'是' if group[0]['removed'] else '否'} | {' | '.join(values)} |")
    lines += ['',
              '固定参数移除仍保留该标签对参数适配的影响，只隔离后验条件化的影响；重新适配同时改变均值、核参数和网络参数。',
              '两项差值按这一顺序相加得到总预测差，不能把重新适配差单独解释成神经表征的因果贡献。',
              '历史模型组的来源权重固定为实际完整前缀的权重，不重建一个未曾查询该点的选例过程。',
              '45项全512点概率经独立NumPy推断核对，15项完整预测与先前保存结果一致，3项支持集不变对照通过。',
              '该分析解释已经获得的反馈如何影响预测，不能计作新的失效发现或独立目标验证。', '']
    (destination/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scope', nargs='?', default='prefixes', choices=('prefixes', 'influence'))
    {'prefixes': diagnose, 'influence': diagnose_influence}[parser.parse_args().scope]()
