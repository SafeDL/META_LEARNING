"""Matched paid-prefix prediction diagnosis of source-conditional adaptation."""

import copy
import hashlib
import time

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.classification import logistic_moments
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.gcrt.conditional_adaptation import component_probabilities, conditional_prior, fit_conditional
from methods.gcrt.joint_training import load_model

from . import discovery_joint_response as joint
from .construction import load_construction
from .discovery_config import OUTPUT, development_tasks, read_json, write_json
from .discovery_physical_mixture import MAIN, REPORT, RUNS, checkpoint_path, protocol
from .discovery_physical_mixture_audit import classifier_probabilities_numpy
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .scenarios import features
from .schema import QueryFeedback


DESTINATION = OUTPUT/'conditional_adaptation'
CONFIGURATIONS = ('joint_target_fit', 'joint_conditional_fit', 'fsbo_target_fit', 'fsbo_conditional_fit')


def probabilities_numpy(means, covariances, paid, labels):
    """Rebuild Gaussian/Laplace inference with NumPy/SciPy, without Torch solvers."""
    probabilities = []
    labels = np.asarray(labels, dtype=float)
    for mean, covariance in zip(means, covariances):
        gram = covariance[np.ix_(paid, paid)]+1e-5*np.eye(len(paid))
        center, f = mean[paid], mean[paid].copy()
        for _ in range(10):
            p = expit(f)
            weight = np.maximum(p*(1-p), 1e-9)
            root = np.sqrt(weight)
            factor = cholesky(np.eye(len(paid))+root[:, None]*gram*root[None], lower=True)
            b = weight*(f-center)+labels-p
            alpha = b-root*cho_solve((factor, True), root*(gram @ b))
            f = center+gram @ alpha
        p = expit(f)
        root = np.sqrt(np.maximum(p*(1-p), 1e-9))
        factor = cholesky(np.eye(len(paid))+root[:, None]*gram*root[None], lower=True)
        alpha = cho_solve((cholesky(gram, lower=True), True), f-center)
        cross = covariance[:, paid]
        predicted = mean+cross @ alpha
        solved = solve_triangular(factor, root[:, None]*cross.T, lower=True)
        variance = np.maximum(np.diag(covariance)-np.square(solved).sum(0), 1e-10)
        probabilities.append(logistic_moments(predicted, variance)[0])
    return np.asarray(probabilities)


def assess(probability, paid, truth, candidates):
    available = sorted(set(range(len(candidates)))-set(paid))
    ranking = sorted(available, key=lambda i: (-probability[i], candidates[i].scenario_id))
    positions = {index: rank for rank, index in enumerate(ranking, 1)}
    ranks = [positions[i] for i in available if truth[i]]
    p, y = np.clip(probability[available], 1e-12, 1-1e-12), truth[available]
    return {'evaluation_count': len(available), 'remaining_collision_count': len(ranks),
            'collision_nll': float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'brier': float(np.square(p-y).mean()),
            'mean_collision_rank': float(np.mean(ranks)) if ranks else None,
            'best_collision_rank': min(ranks) if ranks else None,
            'all_remaining_collision_ranks': [{'index': i, 'rank': positions[i], 'probability': float(probability[i])}
                                              for i in available if truth[i]]}


def diagnose():
    frozen = read_json(REPORT/'protocol.json')
    if protocol() != frozen:
        raise ValueError('The source experiment must remain frozen during prediction diagnosis')
    if read_json(REPORT/'main_endpoint_comparison.json')['status'] != 'main_40_runs_complete_and_independently_audited':
        raise ValueError('Diagnose only the complete independently checked main arm')
    tasks = development_tasks()
    specification = {'tasks': tasks, 'paid_prefix': 127, 'configurations': list(CONFIGURATIONS),
        'source_protocol': frozen, 'source_query_path': MAIN,
        'weights': 'actual source-path prequery weights at query 128, held fixed',
        'conditional_objective': 'fixed weighted sum of component Laplace negative log evidences, divided by 127',
        'source_means': 'mu_source + (target_mean - target_initial_mean)',
        'source_covariances': 'source posterior covariance + target covariance',
        'physical_mass': .1, 'steps': 10, 'learning_rate': .003, 'gradient_clip': 5,
        'training_feedback': '127 paid collision labels only',
        'evaluation': 'all 385 remaining candidates, including zero-collision tasks',
        'additional_selector_queries': 0, 'new_physical_calls': 0,
        'source_files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                         ('methods/gcrt/conditional_adaptation.py', 'benchmark/discovery_conditional_adaptation.py')}}
    path = DESTINATION/'protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('This prediction diagnosis was registered differently')
    write_json(path, specification)
    torch.set_num_threads(1)
    rows, checks, predictions = [], [], {}
    started = time.perf_counter()
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            history, _ = load_construction(kind)
            joint_base, _ = load_model(checkpoint_path(kind, MAIN))
            fsbo_base, _ = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
            joint_base, fsbo_base = joint_base.to(frozen['device']), fsbo_base.to(frozen['device'])
            for task in (t for t in tasks if t['scenario'] == kind):
                target, seed = task['target_id'], task['algorithm_seed']
                bank = ExecutionBank(DATA_ROOT/'physical/development'/kind/target)
                saved = read_json(RUNS/MAIN/target/kind/f'{seed}.json')
                audit = read_json(REPORT/'audit/runs'/MAIN/target/kind/f'{seed}.json')
                queries = saved['result']['queries']
                feedback = [bank(s).feedback.to_dict() for s in bank.candidates]
                if (audit['status'] != 'passed' or saved['specification']['protocol'] != frozen
                        or len(queries) != 128 or len({q['index'] for q in queries}) != 128
                        or [q['index'] for q in queries[:8]] != initial_indices(512, 8, seed)
                        or any(QueryFeedback.from_dict(q['feedback']).to_dict() != feedback[q['index']] for q in queries)):
                    raise ValueError('The prediction diagnosis differs from the audited paid prefix')
                paid = [q['index'] for q in queries[:127]]
                labels = np.array([q['feedback']['collision'] for q in queries[:127]], dtype=bool)
                truth = np.array([f['collision'] for f in feedback], dtype=bool)
                x = features(bank.candidates)
                sources = joint.read_prior(kind, bank.candidates, history, saved['specification']['checkpoint_sha256'])
                means = torch.as_tensor(sources['means'][:, :512], dtype=torch.float64, device=frozen['device'])
                covariances = torch.as_tensor(sources['covariances'][:, :512, :512], dtype=torch.float64, device=frozen['device'])
                score = np.log(np.r_[.75*sources['weights'], .25])+np.array(
                    [q['prequery_joint_log_likelihood'] for q in queries[:127]]).sum(0)
                weights = np.exp(score-logsumexp(score))
                np.testing.assert_allclose(weights, queries[127]['prequery_joint_weights'], atol=2e-12)
                raw = ClassificationKernel(x.shape[1]).to(frozen['device'])
                fit_target(raw, x[paid], labels, steps=10, learning_rate=.01)
                with torch.no_grad():
                    raw_p = collision_probabilities(raw, x[paid], labels, x).cpu().numpy()
                raw_np = classifier_probabilities_numpy(raw, x, paid, labels)
                np.testing.assert_allclose(raw_p, raw_np, atol=2e-9, rtol=2e-9)
                for configuration in CONFIGURATIONS:
                    base = joint_base if configuration.startswith('joint') else fsbo_base
                    model = copy.deepcopy(base)
                    reference = float(base.mean.detach())
                    adapted = time.perf_counter()
                    if configuration.endswith('conditional_fit'):
                        losses = fit_conditional(model, x[paid], labels, weights, source_means=means[:, paid],
                            source_covariances=covariances[:, paid, :][:, :, paid], reference_mean=reference)
                    else:
                        fit_target(model, x[paid], labels, steps=10, learning_rate=.003)
                        losses = None
                    if model.mean.is_cuda:
                        torch.cuda.synchronize()
                    adaptation_s = time.perf_counter()-adapted
                    with torch.no_grad():
                        m, c = conditional_prior(model, x, means, covariances, reference)
                        component_p = component_probabilities(m, c, paid, labels).cpu().numpy()
                        m_np, c_np = m.cpu().numpy(), c.cpu().numpy()
                    independent = probabilities_numpy(m_np, c_np, paid, labels)
                    np.testing.assert_allclose(component_p, independent, atol=2e-7, rtol=2e-6)
                    probability = .9*(weights @ component_p)+.1*raw_p
                    record = {**task, 'configuration': configuration, 'adaptation_s': adaptation_s,
                              'conditional_fit_losses': losses,
                              **assess(probability, paid, truth, bank.candidates)}
                    if target == 'development_ppo' and kind == 'cut_out':
                        for point in (90, 386):
                            record[f'focus_{point}'] = next((r for r in record['all_remaining_collision_ranks']
                                                           if r['index'] == point), None)
                    baseline_error = None
                    if configuration == 'joint_target_fit':
                        selected = queries[127]['index']
                        baseline_error = abs(float(probability[selected])-queries[127]['prequery_probability'])
                        np.testing.assert_allclose(probability[selected], queries[127]['prequery_probability'], atol=2e-7, rtol=2e-6)
                        maximum = max(probability[i] for i in set(range(512))-set(paid))
                        if probability[selected] < maximum-2e-7:
                            raise ValueError('Target-fit control fails to reproduce the actual next acquisition')
                        previous = REPORT/'paid_prefix_diagnosis/probabilities.npz'
                        if target == 'development_ppo' and kind == 'cut_out':
                            with np.load(previous, allow_pickle=False) as values:
                                np.testing.assert_allclose(probability, values[f'{MAIN}/{seed}/127/main_mixture'], atol=2e-7, rtol=2e-6)
                    rows.append(record)
                    key = f'{target}/{kind}/{seed}/{configuration}'
                    predictions[key] = probability
                    checks.append({'key': key, 'full_pool_component_checks': len(component_p),
                                   'max_probability_error': float(np.max(np.abs(component_p-independent))),
                                   'original_next_probability_error': baseline_error,
                                   'raw_probability_max_error': float(np.max(np.abs(raw_p-raw_np)))})
                print('conditional_adaptation_diagnosis', len(rows), '/ 160', target, kind, seed, flush=True)
    if protocol() != frozen:
        raise ValueError('The frozen source experiment changed during prediction diagnosis')
    np.savez_compressed(DESTINATION/'probabilities.npz', **predictions)
    result = {'status': 'complete', 'specification': specification, 'rows': rows,
              'prediction_configurations': len(rows), 'wall_s': time.perf_counter()-started,
              'additional_selector_queries': 0, 'new_physical_calls': 0}
    write_json(DESTINATION/'report.json', result)
    write_json(DESTINATION/'numerical_checks.json', {'status': 'passed', 'configurations': len(checks),
        'full_pool_component_checks': sum(c['full_pool_component_checks'] for c in checks),
        'original_next_selections_checked': sum(c['original_next_probability_error'] is not None for c in checks),
        'original_full_pool_predictions_checked': 5, 'checks': checks})
    report()
    return result


def report():
    result = read_json(DESTINATION/'report.json')
    numerical = read_json(DESTINATION/'numerical_checks.json')
    if result['prediction_configurations'] != 160 or numerical['status'] != 'passed':
        raise ValueError('Complete and numerically check all prediction configurations first')
    rows = result['rows']
    lines = ['# 来源条件适配：相同付费反馈的预测诊断', '',
        '在原主方法40条轨迹各127次已付费前缀上，对照目标初始化与适配目标，共160项配置。',
        '拟合仅使用这些前缀的碰撞标签。来源可靠性固定为原轨迹查询128之前的值，物理分量固定0.10。',
        '剩余385点标签只用于下表的评估；本诊断没有新增选例或物理调用，不是新的发现数量或召回率。', '',
        '| 配置 | 全部剩余候选NLL | Brier | 含剩余碰撞任务的平均碰撞排名 |', '|---|---:|---:|---:|']
    aggregate = {}
    lineages = sorted({r['lineage'] for r in rows})
    for configuration in CONFIGURATIONS:
        group = [r for r in rows if r['configuration'] == configuration]
        aggregate[configuration] = {}
        for metric in ('collision_nll', 'brier', 'mean_collision_rank'):
            by_lineage = {}
            for lineage in lineages:
                values = [r[metric] for r in group if r['lineage'] == lineage and r[metric] is not None]
                by_lineage[lineage] = float(np.mean(values)) if values else None
            values = [v for v in by_lineage.values() if v is not None]
            aggregate[configuration][metric] = {'mean': float(np.mean(values)) if values else None,
                                                 'by_lineage': by_lineage}
        summary = aggregate[configuration]
        rank = summary['mean_collision_rank']['mean']
        lines.append(f"| {configuration} | {summary['collision_nll']['mean']:.6f} | {summary['brier']['mean']:.6f} | {rank:.3f} |")
    lines += ['', '先在谱系内汇总，再对有效谱系等权汇总。零碰撞任务仍参与NLL和Brier；没有剩余碰撞的任务排名记为不适用。',
              '排名汇总的有效任务集合与预测误差不同，不把较低排名解释为在线发现保证。所有剩余碰撞的逐项排名保存在report.json。', '',
              '| 初始化 | 谱系 | 条件适配减独立适配：NLL | Brier |', '|---|---|---:|---:|']
    for initialization in ('joint', 'fsbo'):
        for lineage in lineages:
            conditional = aggregate[f'{initialization}_conditional_fit']
            target = aggregate[f'{initialization}_target_fit']
            differences = [conditional[m]['by_lineage'][lineage]-target[m]['by_lineage'][lineage]
                           for m in ('collision_nll', 'brier')]
            lines.append(f'| {initialization} | {lineage} | {differences[0]:.6f} | {differences[1]:.6f} |')
    lines += ['', '| PPO切出索引90的配置 | 尚未查询的种子数 | 概率范围 | 排名范围 |', '|---|---:|---:|---:|']
    for configuration in CONFIGURATIONS:
        focus = [r['focus_90'] for r in rows if r['configuration'] == configuration and r.get('focus_90') is not None]
        lines.append(f"| {configuration} | {len(focus)} | {min(v['probability'] for v in focus):.6g}–{max(v['probability'] for v in focus):.6g} | {min(v['rank'] for v in focus)}–{max(v['rank'] for v in focus)} |")
    lines += ['', f"160项预测配置、{numerical['full_pool_component_checks']}组全512点分量概率经独立NumPy/SciPy Gaussian/Laplace推断核对；",
              '40次原下一选点和5组原完整预测重建通过。无来源退化、梯度、127支持点、全零/全一及来源排列另由针对性测试核对。',
              f"离线重建总耗时{result['wall_s']:.2f}秒，与在线成本分开记录。", '',
              '条件适配优化的是固定可靠性加权的分量Laplace负对数证据之和，不是完整混合模型的精确证据。',
              '这里冻结了另一条真实查询轨迹的权重和反馈，不能代表新策略运行后的可靠性和后续预测。',
              '当前主方法失败结论及阶段C/D/E状态保持。新的在线实验须先登记主配置与机制对照，强参考包括当前整组已完成方法。']
    write_json(DESTINATION/'summary.json', {'aggregate': aggregate, 'scope': 'matched prediction diagnosis only'})
    (DESTINATION/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    diagnose()
