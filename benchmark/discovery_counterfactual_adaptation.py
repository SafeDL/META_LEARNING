"""Historical diagnosis of fixed-parameter versus refitted virtual feedback."""

import argparse
import copy
import hashlib
import json
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.common.response import source_weights
from methods.gcrt.conditional_adaptation import component_probabilities
from methods.gcrt.counterfactual_adaptation import MODES, batch_utility, virtual_update
from methods.gcrt.residual_scale import cached_covariance, residual_prior
from methods.gcrt.residual_search import learned_probabilities

from .discovery_conditional_adaptation import probabilities_numpy
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_physical_mixture import checkpoint_path
from .discovery_physical_mixture_audit import classifier_probabilities_numpy
from .discovery_residual_scale import REPORT as LEARNING, cache_path
from .discovery_residual_scale import protocol as learning_protocol
from .discovery_response_locality import history_inputs
from .protocol import ROOT


REPORT = OUTPUT/'counterfactual_adaptation'


def protocol():
    return {'purpose': 'historical prediction and two-candidate utility diagnosis; no development search',
        'scenarios': ['cut_in', 'cut_out'], 'historical_tasks': 16, 'support_count': 64,
        'ordered_seed': 1301, 'candidate_count': 512, 'candidate_rules': ['greedy', 'entropy_excluding_greedy'],
        'virtual_labels': [False, True], 'modes': list(MODES), 'horizons': [5, 64],
        'target_adaptation': {'steps': 10, 'learning_rate': .003},
        'physical_adaptation': {'steps': 10, 'learning_rate': .01},
        'local_bandwidth': .3, 'physical_mass': .1, 'source_prior_mass': .75, 'target_prior_mass': .25,
        'selected_beta': read_json(LEARNING/'selection.json')['selected_beta'],
        'historical_protocol': learning_protocol(), 'device': learning_protocol()['foundation']['device'],
        'failed_cpu_prefix_protocol_sha256': hashlib.sha256((REPORT/'failed_cpu_prefix/protocol.json').read_bytes()).hexdigest(),
        'teacher_scope': 'all historical systems; exclude pseudo-target source posterior only',
        'source_files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'benchmark/discovery_counterfactual_adaptation.py', 'methods/gcrt/counterfactual_adaptation.py',
            'benchmark/discovery_response_locality.py')},
        'new_selector_queries': 0, 'new_physical_calls': 0, 'formal_phases_entered': False}


def prepare():
    audit = read_json(LEARNING/'selection_audit.json')
    if audit['status'] != 'passed' or read_json(LEARNING/'protocol.json') != learning_protocol():
        raise ValueError('Preserve the independently checked historical priors and adaptation caches')
    current = protocol()
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != current:
        raise ValueError('The virtual-adaptation diagnosis is already registered differently')
    write_json(path, current)
    return current


def state_copy(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def check_unchanged(model, saved):
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key], atol=0, rtol=0)


def reliability_at_64(archive, source_positions, source_means, source_covariances, initial, beta):
    with np.load(cache_path(archive.scenarios[0].kind, archive.source_id), allow_pickle=False) as saved:
        expected = np.random.default_rng(1301).permutation(512)[:160]
        np.testing.assert_array_equal(saved['ordered_indices'], expected)
        np.testing.assert_array_equal(saved['labels'], archive.collision[expected])
        np.testing.assert_array_equal(saved['x'], archive.x[expected])
        setting = json.loads(str(saved['specification']))
        if (setting['protocol'] != learning_protocol() or setting['role'] != 'history'
                or setting['source_id'] != archive.source_id):
            raise ValueError('Virtual adaptation requires the original historical prefix cache')
        z = torch.tensor(saved['encoded'][:65], dtype=torch.float64)
        parameters = torch.tensor(saved['parameters'][:65], dtype=torch.float64)
        raw_probability = saved['raw_probabilities'][64].copy()
    sm = torch.tensor(source_means[source_positions][:, expected], dtype=torch.float64)
    sc = torch.tensor(source_covariances[source_positions][:, expected, :][:, :, expected], dtype=torch.float64)
    labels = archive.collision[expected[:64]]
    likelihoods = []
    with torch.no_grad():
        for count in range(64):
            covariance = cached_covariance(z[count], parameters[count, 1], parameters[count, 2])
            mean, covariance = residual_prior(parameters[count, 0], covariance, sm, sc, parameters[0, 0], beta)
            p = component_probabilities(mean, covariance, list(range(count)), labels[:count])[:, count]
            likelihoods.append((p if labels[count] else 1-p).clamp_min(1e-15).log())
    likelihoods = torch.stack(likelihoods, 1)
    x = torch.tensor(archive.x, dtype=torch.float64)
    influence = (-torch.square(x[expected[:64], None]-x[None]).mean(-1)/(2*.3**2)).exp()
    scores = torch.tensor(initial, dtype=torch.float64).log()[:, None]+likelihoods @ influence
    return expected, labels.tolist(), scores, z[64], parameters[64], raw_probability


def check_branch(branch, current, physical_current, sources, reference, beta, x, paid, labels,
                 scores, before, index, label):
    started = time.perf_counter()
    extended, feedback = [*paid, index], [*labels, label]
    component, physical = {}, {}
    component_errors, physical_errors = [], []
    for name, model in (('fixed', current), ('refit', branch['target'])):
        with torch.no_grad():
            mean, covariance = residual_prior(model.mean, model.covariance(x), sources['means'],
                                               sources['covariances'], reference, beta)
        expected = probabilities_numpy(mean.cpu().numpy(), covariance.cpu().numpy(), extended, feedback)
        np.testing.assert_allclose(branch['components'][name].cpu(), expected, atol=2e-7, rtol=2e-6)
        component[name] = expected
        component_errors.append(float(np.max(np.abs(branch['components'][name].cpu().numpy()-expected))))
    for name, model in (('fixed', physical_current), ('refit', branch['physical'])):
        expected = classifier_probabilities_numpy(model, x, extended, feedback)
        np.testing.assert_allclose(branch['physical_probabilities'][name].cpu(), expected, atol=2e-9, rtol=2e-9)
        physical[name] = expected
        physical_errors.append(float(np.max(np.abs(branch['physical_probabilities'][name].cpu().numpy()-expected))))
    likelihood = np.log(np.maximum(before[:, index] if label else 1-before[:, index], 1e-15))
    influence = np.exp(-np.square(x-x[index]).mean(-1)/(2*.3**2))
    weight = scores.cpu().numpy()+likelihood[:, None]*influence
    weight = np.exp(weight-weight.max(0))
    weight /= weight.sum(0)
    np.testing.assert_allclose(branch['log_weights'].softmax(0).cpu(), weight, atol=2e-12, rtol=2e-12)
    errors = []
    for mode in MODES:
        learned = component['fixed' if mode == 'fixed' else 'refit']
        raw = physical['refit' if mode == 'refit_both' else 'fixed']
        expected = .9*(weight*learned).sum(0)+.1*raw
        np.testing.assert_allclose(branch['probabilities'][mode].cpu(), expected, atol=2e-7, rtol=2e-6)
        errors.append(float(np.max(np.abs(branch['probabilities'][mode].cpu().numpy()-expected))))
    return {'component_max_error': max(component_errors), 'physical_max_error': max(physical_errors),
        'mixture_max_error': max(errors), 'independent_check_s': time.perf_counter()-started}


def choose(probability, available, candidates):
    maximum = max(probability[i] for i in available)
    return min((i for i in available if probability[i] >= maximum-1e-12), key=lambda i: candidates[i].scenario_id)


def diagnose_history(kind, archive, history, source_means, source_covariances, specification):
    started = time.perf_counter()
    sources = [a for a in history if a.source_id != archive.source_id]
    positions = [i for i, a in enumerate(history) if a.source_id != archive.source_id]
    beta, x = specification['selected_beta'][kind], archive.x
    initial = np.r_[.75*source_weights(sources), .25]
    ordered, labels, scores, encoded, parameters, cached_raw = reliability_at_64(
        archive, positions, source_means, source_covariances, initial, beta)
    paid = ordered[:64].tolist()
    pretrained, _ = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
    device = specification['device']
    pretrained = pretrained.to(device)
    physical_base = ClassificationKernel(x.shape[1]).to(device)
    scores = scores.to(device)
    current, physical_current = copy.deepcopy(pretrained), copy.deepcopy(physical_base)
    fit_target(current, x[paid], labels, steps=10, learning_rate=.003)
    fit_target(physical_current, x[paid], labels, steps=10, learning_rate=.01)
    with torch.no_grad():
        np.testing.assert_allclose(current.features(x)[ordered].cpu(), encoded, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose([float(current.mean), float(current.log_lengthscale), float(current.log_scale)],
                                  parameters, atol=2e-8, rtol=2e-7)
        cached_kernel = cached_covariance(encoded, parameters[1], parameters[2])
        rebuilt_kernel = current.covariance(x)[ordered][:, ordered].cpu()
        np.testing.assert_allclose(rebuilt_kernel, cached_kernel, atol=2e-7, rtol=2e-6)
        prefix_kernel_max_error = float((rebuilt_kernel-cached_kernel).abs().max())
    source = {'means': torch.tensor(source_means[positions], dtype=torch.float64, device=device),
              'covariances': torch.tensor(source_covariances[positions], dtype=torch.float64, device=device)}
    reference = float(pretrained.mean.detach())
    with torch.no_grad():
        before = learned_probabilities(current, None, x, paid, labels, source, reference, beta)
        raw = collision_probabilities(physical_current, x[paid], labels, x)
        probability = (.9*(scores.softmax(0)*before).sum(0)+.1*raw).cpu().numpy()
    np.testing.assert_allclose(raw[ordered].cpu(), cached_raw, atol=2e-9, rtol=2e-9)
    held = ordered[128:]
    p, truth = np.clip(probability[held], 1e-12, 1-1e-12), archive.collision[held]
    metrics = [float(-np.mean(truth*np.log(p)+(1-truth)*np.log1p(-p))), float(np.square(p-truth).mean())]
    registered = next(r for r in read_json(LEARNING/'selection.json')['rows'] if
        (r['scenario'], r['pseudo_target'], r['support_count'], r['beta'], r['reliability']) ==
        (kind, archive.source_id, 64, beta, 'local'))
    np.testing.assert_allclose(metrics, [registered['query_nll'], registered['query_brier']], atol=2e-8, rtol=0)
    available = sorted(set(range(512))-set(paid))
    greedy = choose(probability, available, archive.scenarios)
    clipped = np.clip(probability, 1e-12, 1-1e-12)
    entropy = -clipped*np.log(clipped)-(1-clipped)*np.log1p(-clipped)
    candidates = {'greedy': greedy, 'entropy_excluding_greedy': choose(entropy, [i for i in available if i != greedy], archive.scenarios)}
    models = (pretrained, physical_base, current, physical_current)
    states = [state_copy(model) for model in models]
    saved_scores, saved_sources = scores.clone(), {k: v.clone() for k, v in source.items()}
    rows, checks, costs = [], [], []
    prefix_reconstruction_s = time.perf_counter()-started
    predictions = np.empty((2, 2, 3, 512))
    for case, (rule, index) in enumerate(candidates.items()):
        branches = {}
        for label in (False, True):
            branch = virtual_update(pretrained, physical_base, current, physical_current, x, paid, labels,
                                    source, reference, beta, scores, before, index, label)
            checks.append(check_branch(branch, current, physical_current, source, reference, beta, x, paid,
                                       labels, scores, before.cpu().numpy(), index, label))
            costs.append(branch['cost'])
            branches[label] = {mode: value.cpu().numpy() for mode, value in branch['probabilities'].items()}
            for mode_index, mode in enumerate(MODES):
                predictions[case, int(label), mode_index] = branches[label][mode]
        actual = bool(archive.collision[index])
        future = [i for i in available if i != index]
        for mode in MODES:
            after = branches[actual][mode]
            p, truth = np.clip(after[future], 1e-12, 1-1e-12), archive.collision[future]
            remaining_nll = float(-np.mean(truth*np.log(p)+(1-truth)*np.log1p(-p)))
            remaining_brier = float(np.square(p-truth).mean())
            ranking = sorted(future, key=lambda i: (-after[i], archive.scenarios[i].scenario_id))
            for horizon in (5, 64):
                utility = batch_utility(probability, index, {y: branches[y][mode] for y in (False, True)}, available, horizon)
                rows.append({'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
                    'candidate_rule': rule, 'candidate_index': index, 'actual_label': actual, 'mode': mode,
                    'horizon': horizon, 'expected_batch_utility': utility,
                    'actual_branch_batch_collisions': int(actual)+int(archive.collision[ranking[:horizon-1]].sum()),
                    'remaining_nll': remaining_nll, 'remaining_brier': remaining_brier})
    for model, state in zip(models, states):
        check_unchanged(model, state)
    torch.testing.assert_close(scores, saved_scores, atol=0, rtol=0)
    for key, value in source.items():
        torch.testing.assert_close(value, saved_sources[key], atol=0, rtol=0)
    result = {'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
        'source_ids': [a.source_id for a in sources], 'candidates': candidates, 'rows': rows,
        'numerical_checks': checks, 'shared_branch_costs': costs, 'real_state_unchanged': True,
        'prefix_reconstruction_s': prefix_reconstruction_s,
        'prefix_kernel_max_error': prefix_kernel_max_error,
        'registered_prefix_loss_max_error': float(np.max(np.abs(np.array(metrics)-[registered['query_nll'], registered['query_brier']]))),
        'wall_s': time.perf_counter()-started}
    destination = REPORT/'histories'/kind/f'{archive.source_id}.json'
    write_json(destination, result)
    np.savez_compressed(destination.with_suffix('.npz'), predictions=predictions,
                        prequery_probability=probability, prequery_components=before.cpu().numpy(),
                        prequery_log_weights=scores.cpu().numpy(), paid=np.array(paid),
                        candidate_indices=np.array(list(candidates.values())))
    print('virtual_adaptation', kind, archive.source_id, flush=True)
    return result


def aggregate(rows):
    return {metric: float(np.mean([np.mean([r[metric] for r in rows if r['lineage'] == lineage])
                                  for lineage in sorted({r['lineage'] for r in rows})]))
            for metric in ('expected_batch_utility', 'actual_branch_batch_collisions', 'remaining_nll', 'remaining_brier')}


def run():
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol() or (REPORT/'report.json').exists():
        raise ValueError('Run the registered, not already completed historical diagnosis')
    torch.set_num_threads(1)
    started = time.perf_counter()
    records = []
    with threadpool_limits(1):
        for kind in specification['scenarios']:
            history, means, covariances = history_inputs(kind)
            for archive in history:
                try:
                    records.append(diagnose_history(kind, archive, history, means, covariances, specification))
                except Exception:
                    failure = {'scenario': kind, 'pseudo_target': archive.source_id,
                        'completed_historical_tasks': len(records), 'wall_s': time.perf_counter()-started,
                        'error': traceback.format_exc(), 'new_selector_queries': 0, 'new_physical_calls': 0}
                    with (REPORT/'failed_attempts.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(failure, ensure_ascii=False)+'\n')
                    raise
    rows = [row for record in records for row in record['rows']]
    if len(records) != 16 or len(rows) != 192 or specification != protocol():
        raise ValueError('Complete every historical task under the unchanged diagnosis')
    summary = []
    for kind in specification['scenarios']:
        for rule in specification['candidate_rules']:
            for horizon in (5, 64):
                for mode in MODES:
                    group = [r for r in rows if (r['scenario'], r['candidate_rule'], r['horizon'], r['mode']) == (kind, rule, horizon, mode)]
                    summary.append({'scenario': kind, 'candidate_rule': rule, 'horizon': horizon, 'mode': mode, **aggregate(group)})
    choices = []
    for record in records:
        for horizon in (5, 64):
            gaps = {mode: (next(r['expected_batch_utility'] for r in record['rows'] if
                                r['mode'] == mode and r['horizon'] == horizon and r['candidate_rule'] == 'entropy_excluding_greedy')
                          - next(r['expected_batch_utility'] for r in record['rows'] if
                                 r['mode'] == mode and r['horizon'] == horizon and r['candidate_rule'] == 'greedy'))
                    for mode in MODES}
            preferred = {mode: 'entropy_excluding_greedy' if gap > 1e-8 else 'greedy' for mode, gap in gaps.items()}
            choices.append({'scenario': record['scenario'], 'pseudo_target': record['pseudo_target'],
                            'horizon': horizon, 'preferred_between_two_candidates': preferred,
                            'utility_gap_entropy_minus_greedy': gaps})
    checks = [check for record in records for check in record['numerical_checks']]
    numerical = {'status': 'passed', 'component_prediction_groups': 128, 'physical_prediction_groups': 128,
        'mixture_prediction_groups': 192, 'actual_states_unchanged': 16,
        **{key: max(c[key] for c in checks) for key in ('component_max_error', 'physical_max_error', 'mixture_max_error')},
        'prefix_kernel_max_error': max(r['prefix_kernel_max_error'] for r in records),
        'registered_prefix_loss_max_error': max(r['registered_prefix_loss_max_error'] for r in records)}
    result = {'status': 'complete_numerically_checked_historical_diagnosis', 'protocol': specification,
        'summary': summary, 'candidate_choices': choices, 'rows': rows, 'numerical': numerical,
        'shared_costs': {key: sum(c[key] for r in records for c in r['shared_branch_costs'])
                         for key in ('target_refit_s', 'physical_refit_s', 'conditional_prediction_s')},
        'prefix_reconstruction_s': sum(r['prefix_reconstruction_s'] for r in records),
        'independent_check_s': sum(c['independent_check_s'] for c in checks),
        'device_equivalence_probe_wall_s': read_json(REPORT/'device_equivalence_probe.json')['wall_s'],
        'failed_attempts': [json.loads(line) for line in (REPORT/'failed_attempts.jsonl').read_text(encoding='utf-8').splitlines()],
        'wall_s': time.perf_counter()-started, 'new_selector_queries': 0, 'new_physical_calls': 0,
        'formal_phases_entered': False, 'development_admission_claimed': False}
    write_json(REPORT/'report.json', result)
    lines = ['# 虚拟反馈中的目标核适配：历史诊断', '',
        '16个历史任务，64点相同反馈，各比较两个预定候选的两个虚拟标签。',
        '192组全512点预测完成独立核对；没有新增开发搜索或物理调用。', '',
        '| 场景 | 候选 | 窗口 | 更新 | 预测批量收益 | 实际分支批量碰撞潜力 | 剩余NLL |',
        '|---|---|---:|---|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['scenario']} | {row['candidate_rule']} | {row['horizon']} | {row['mode']} | "
                     f"{row['expected_batch_utility']:.3f} | {row['actual_branch_batch_collisions']:.3f} | {row['remaining_nll']:.6f} |")
    changed = sum(c['preferred_between_two_candidates']['fixed'] != c['preferred_between_two_candidates']['refit_both']
                  and min(abs(c['utility_gap_entropy_minus_greedy'][m]) for m in ('fixed', 'refit_both')) > 1e-8
                  for c in choices)
    lines += ['', f'完整重拟合相对固定参数，在32项两个候选/窗口比较中改变{changed}项偏好。',
        '实际分支批量潜力使用历史评估标签，一次查询后固定批量，不是128次自适应搜索的发现数。',
        '本诊断只比较两个预定候选，没有对448个可查询候选执行完整前瞻选例。',
        '基础教师见过全部历史对象，自身仅从来源后验排除；结果不作为严格留系统验证。',
        f"总耗时{result['wall_s']:.2f}秒，与基线回放共享机器资源；重拟合及独立核对分项见report.json。",
        '正式阶段、原失败结论与当前开发矩阵保持原状；没有从本项自动启动新的选例实验。']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    {'prepare': prepare, 'run': run}[parser.parse_args().stage]()
