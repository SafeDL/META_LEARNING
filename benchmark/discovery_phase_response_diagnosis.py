"""Prediction-only diagnosis of censored historical interaction-stage responses."""

import copy
import hashlib
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.common.response import source_weights
from methods.gcrt.conditional_adaptation import component_probabilities, conditional_prior
from methods.gcrt.joint_training import load_model
from methods.gcrt.phase_response import (phase_bases, phase_labels, phase_support,
                                         source_phase_priors, total_collision_probability)

from .construction import load_construction
from .discovery_conditional_adaptation import assess, probabilities_numpy
from .discovery_config import OUTPUT, development_tasks, read_json, write_json
from .discovery_physical_mixture import MAIN, REPORT, RUNS, checkpoint_path, protocol
from .discovery_physical_mixture_audit import classifier_probabilities_numpy
from .protocol import ROOT
from .scenarios import features


DESTINATION = OUTPUT/'phase_response'
MODES = ('phase_transfer', 'phase_target', 'phase_zero_fill')
INITIALIZATIONS = ('joint', 'fsbo')


def recorded_observations(records, kind, archives):
    observations = {}
    for archive in archives:
        rows = [r for r in records if r['source_id'] == archive.source_id and r['scenario'] == kind]
        rows.sort(key=lambda r: r['index'])
        if (len(rows) != 512 or [r['index'] for r in rows] != list(range(512))
                or any(r['role'] != archive.role or r['scenario_id'] != s.scenario_id
                       or r['collision'] != bool(y)
                       for r, s, y in zip(rows, archive.scenarios, archive.collision))):
            raise ValueError('Stage observations differ from their role-checked archive')
        labels = np.array([phase_labels(r['events'], r['first_collision_time_s'], r['episode_end_time_s'])
                           for r in rows], dtype=float)
        if (not np.array_equal(np.nansum(labels, axis=1).astype(bool), archive.collision)
                or any(np.nansum(row) > 1 or (np.isfinite(row[1]) and not np.isfinite(row[0]))
                       or (np.isfinite(row[2]) and not np.isfinite(row[1])) for row in labels)):
            raise ValueError('Stage censoring differs from the true first collision')
        observations[archive.source_id] = labels
    return observations


def diagnose():
    frozen = read_json(REPORT/'protocol.json')
    if frozen != protocol() or read_json(REPORT/'audit/report.json')['runs'] != 320:
        raise ValueError('Finish and independently audit the unchanged preceding study first')
    stage_report = read_json(REPORT/'phase_diagnosis/report.json')
    tasks = development_tasks()
    specification = {'tasks': tasks, 'source_protocol': frozen, 'source_path': MAIN, 'paid_prefix': 127,
        'initializations': list(INITIALIZATIONS), 'modes': list(MODES),
        'hazard_stages': ['before_motion', 'during_maneuver', 'after_maneuver_until_10s'],
        'source_weights': 'fixed family-balanced .75 historical and .25 independent target',
        'base_mean': 'logit of family-weighted observed historical hazard frequency, smoothed (a+.5)/(n+1)',
        'source_and_target_fit': {'steps': 10, 'learning_rate': .003, 'gradient_clip': 5},
        'physical_mass': .1, 'physical_fit': {'steps': 10, 'learning_rate': .01},
        'history_feedback': 'historical stage labels only; zero-fill ablation refits its own source priors',
        'target_fit_feedback': 'stage labels from 127 paid queries only',
        'unqueried_feedback': 'evaluator only', 'additional_selector_queries': 0, 'new_physical_calls': 0,
        'source_files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'methods/gcrt/phase_response.py', 'methods/gcrt/conditional_adaptation.py',
            'benchmark/discovery_phase_response_diagnosis.py', 'benchmark/discovery_conditional_adaptation.py')},
        'stage_report_sha256': hashlib.sha256((REPORT/'phase_diagnosis/report.json').read_bytes()).hexdigest()}
    path = DESTINATION/'protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('This stage-response diagnosis was registered differently')
    write_json(path, specification)
    torch.set_num_threads(1)
    rows, checks, predictions, constructions = [], [], {}, []
    checked_records, wall_start = 0, time.perf_counter()
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            history, development = load_construction(kind)
            observations = recorded_observations(stage_report['records'], kind, history+development)
            checked_records += sum(len(v) for v in observations.values())
            candidates, x = development[0].scenarios, development[0].x
            if any(not np.array_equal(a.x, x) for a in development):
                raise ValueError('Shared historical-prior construction requires identical development coordinates')
            prefix_inputs = []
            for task in (t for t in tasks if t['scenario'] == kind):
                saved = read_json(RUNS/MAIN/task['target_id']/kind/f"{task['algorithm_seed']}.json")
                audit = read_json(REPORT/'audit/runs'/MAIN/task['target_id']/kind/f"{task['algorithm_seed']}.json")
                queries = saved['result']['queries']
                archive = next(a for a in development if a.source_id == task['target_id'])
                if (audit['status'] != 'passed' or saved['specification']['protocol'] != frozen
                        or len(queries) != 128 or len({q['index'] for q in queries}) != 128
                        or any(q['feedback']['scenario_id'] != archive.scenarios[q['index']].scenario_id
                               or q['feedback']['collision'] != bool(archive.collision[q['index']]) for q in queries)):
                    raise ValueError('Stage prediction inputs differ from the audited paid trajectory')
                paid = [q['index'] for q in queries[:127]]
                labels = np.array([q['feedback']['collision'] for q in queries[:127]], dtype=bool)
                raw = ClassificationKernel(x.shape[1]).to(frozen['device'])
                fit_target(raw, x[paid], labels, steps=10, learning_rate=.01)
                with torch.no_grad():
                    raw_p = collision_probabilities(raw, x[paid], labels, x).cpu().numpy()
                np.testing.assert_allclose(raw_p, classifier_probabilities_numpy(raw, x, paid, labels), atol=2e-9, rtol=2e-9)
                prefix_inputs.append((task, archive, paid, raw_p))
            for initialization in INITIALIZATIONS:
                base = (load_model(checkpoint_path(kind, MAIN))[0] if initialization == 'joint' else
                        load_fsbo(checkpoint_path(kind, 'physical_fsbo'))[0]).to(frozen['device'])
                bases = phase_bases(base, history, observations)
                began = time.perf_counter()
                censored, censored_supports = source_phase_priors(bases, history, observations, x)
                filled, filled_supports = source_phase_priors(bases, history, observations, x, zero_fill=True)
                if base.mean.is_cuda:
                    torch.cuda.synchronize()
                constructions.append({'scenario': kind, 'initialization': initialization,
                    'base_means': [float(b.mean.detach()) for b in bases],
                    'censored_source_supports': censored_supports, 'zero_fill_source_supports': filled_supports,
                    'historical_construction_wall_s': time.perf_counter()-began})
                for task, archive, paid, raw_p in prefix_inputs:
                    for mode in MODES:
                        configuration = f'{initialization}_{mode}'
                        weight = np.ones(1) if mode == 'phase_target' else np.r_[.75*source_weights(history), .25]
                        priors = filled if mode == 'phase_zero_fill' else censored
                        hazards, local_checks, sizes = [], [], []
                        began = time.perf_counter()
                        for phase, phase_base in enumerate(bases):
                            indices, labels = phase_support(paid, observations[archive.source_id], phase,
                                                             zero_fill=mode == 'phase_zero_fill')
                            model = copy.deepcopy(phase_base)
                            if len(indices):
                                fit_target(model, x[indices], labels, steps=10, learning_rate=.003)
                            with torch.no_grad():
                                m, c = (conditional_prior(model, x) if mode == 'phase_target' else
                                        conditional_prior(model, x, *priors[phase], float(phase_base.mean.detach())))
                                p = component_probabilities(m, c, indices, labels).cpu().numpy()
                                m_np, c_np = m.cpu().numpy(), c.cpu().numpy()
                            independent = probabilities_numpy(m_np, c_np, indices, labels)
                            np.testing.assert_allclose(p, independent, atol=2e-7, rtol=2e-6)
                            hazards.append(weight @ p)
                            local_checks.append({'phase': phase, 'components': len(p),
                                'probability_max_error': float(np.max(np.abs(p-independent))),
                                'minimum_source_prior_variance': float(np.min(np.diagonal(c_np, axis1=-2, axis2=-1)))})
                            sizes.append({'phase': phase, 'observed': len(indices), 'collisions': int(labels.sum())})
                        probability = 1-np.prod(1-np.array(hazards), axis=0)
                        torch_total = total_collision_probability(torch.as_tensor(np.array(hazards))).numpy()
                        np.testing.assert_allclose(probability, torch_total, atol=2e-15)
                        probability = .9*probability+.1*raw_p
                        record = {**task, 'configuration': configuration, 'paid_phase_supports': sizes,
                            'target_reconstruction_wall_s': time.perf_counter()-began,
                            **assess(probability, paid, archive.collision, archive.scenarios)}
                        for phase in range(3):
                            phase_truth = observations[archive.source_id][:, phase] == 1
                            phase_assessment = assess(probability, paid, phase_truth, archive.scenarios)
                            record[f'phase_{phase}_collision_ranks'] = phase_assessment['all_remaining_collision_ranks']
                        if task['target_id'] == 'development_ppo' and kind == 'cut_out':
                            for point in (90, 386):
                                record[f'focus_{point}'] = next((r for r in record['all_remaining_collision_ranks']
                                                               if r['index'] == point), None)
                        key = f"{task['target_id']}/{kind}/{task['algorithm_seed']}/{configuration}"
                        predictions[key], predictions[key+'/hazards'] = probability, np.array(hazards)
                        rows.append(record)
                        checks.append({'key': key, 'phases': local_checks})
                    print('phase_response_diagnosis', len(rows), '/ 240', initialization,
                          task['target_id'], kind, task['algorithm_seed'], flush=True)
                del bases, censored, filled
    if protocol() != frozen or checked_records != 12288:
        raise ValueError('Stage diagnosis incomplete or the original frozen experiment changed')
    np.savez_compressed(DESTINATION/'probabilities.npz', **predictions)
    result = {'status': 'complete', 'specification': specification, 'rows': rows,
        'prediction_configurations': len(rows), 'stage_records_checked': checked_records,
        'historical_constructions': constructions, 'wall_s': time.perf_counter()-wall_start,
        'additional_selector_queries': 0, 'new_physical_calls': 0}
    write_json(DESTINATION/'report.json', result)
    write_json(DESTINATION/'numerical_checks.json', {'status': 'passed', 'configurations': len(checks),
        'full_pool_component_checks': sum(p['components'] for r in checks for p in r['phases']),
        'stage_records_checked': checked_records, 'checks': checks})
    report()
    return result


def report():
    data, numerical = read_json(DESTINATION/'report.json'), read_json(DESTINATION/'numerical_checks.json')
    if data['prediction_configurations'] != 240 or numerical['status'] != 'passed':
        raise ValueError('Finish all matched stage-response predictions and numerical checks first')
    rows = data['rows']
    configurations = [f'{i}_{m}' for i in INITIALIZATIONS for m in MODES]
    aggregate = {}
    lineages = sorted({r['lineage'] for r in rows})
    lines = ['# 阶段响应与删失：已付费前缀的预测诊断', '',
        '40条既有轨迹各使用127次已付费查询，比较2个冻结初始化和3项阶段信息配置，共240项预测。',
        '历史阶段标签只来自8个历史系统；目标阶段标签只用于已付费前缀。未查询标签只参与评估。',
        '全部候选及碰撞标签保持；未新增选例、仿真或目标系统训练。本报告不是在线发现数量与召回结果。', '',
        '| 配置 | 全部剩余候选NLL | Brier | 含剩余碰撞任务的平均碰撞排名 |', '|---|---:|---:|---:|']
    for configuration in configurations:
        aggregate[configuration] = {}
        for metric in ('collision_nll', 'brier', 'mean_collision_rank'):
            by_lineage = {}
            for lineage in lineages:
                values = [r[metric] for r in rows if r['configuration'] == configuration
                          and r['lineage'] == lineage and r[metric] is not None]
                by_lineage[lineage] = float(np.mean(values)) if values else None
            valid = [v for v in by_lineage.values() if v is not None]
            aggregate[configuration][metric] = {'mean': float(np.mean(valid)) if valid else None,
                                                 'by_lineage': by_lineage}
        a = aggregate[configuration]
        lines.append(f"| {configuration} | {a['collision_nll']['mean']:.6f} | {a['brier']['mean']:.6f} | {a['mean_collision_rank']['mean']:.3f} |")
    lines += ['', '先在谱系内汇总，再对有效谱系等权。零碰撞任务参与误差评估，没有剩余碰撞的排名为不适用。',
              '完整分谱系误差和全部剩余碰撞逐点排名见summary.json、report.json。', '',
              '| PPO切出索引90的配置 | 尚未查询的种子数 | 概率范围 | 排名范围 |', '|---|---:|---:|---:|']
    for configuration in configurations:
        focus = [r['focus_90'] for r in rows if r['configuration'] == configuration and r.get('focus_90') is not None]
        lines.append(f"| {configuration} | {len(focus)} | {min(v['probability'] for v in focus):.6g}–{max(v['probability'] for v in focus):.6g} | {min(v['rank'] for v in focus)}–{max(v['rank'] for v in focus)} |")
    costs = {'historical_stage_construction_wall_s': sum(c['historical_construction_wall_s'] for c in data['historical_constructions']),
             'total_diagnosis_wall_s': data['wall_s'], 'additional_selector_queries': 0, 'new_physical_calls': 0}
    lines += ['', f"12,288条阶段标签与原档案对应关系通过核对；{numerical['full_pool_component_checks']}组全512点分量概率通过独立NumPy/SciPy推断核对。",
              f"历史阶段后验构建{costs['historical_stage_construction_wall_s']:.2f}秒，诊断总耗时{data['wall_s']:.2f}秒；均为离线成本。", '',
              '两个迁移版本均使用固定来源家族权重，没有在本诊断中验证阶段可靠性更新。',
              '零填充消融在历史与目标适配中均把未观察阶段作为0，其他基础均值与拟合规则相同。',
              '条件阶段模型获得了额外阶段信息；收益必须与获得相同信息的phase_target比较，不能只与二值原型比较。',
              '阶段概率乘积采用独立潜在GP近似，不保证完整联合后验与概率校准。',
              '预测误差与排名用于判断是否值得进入在线试验，不能替代128预算下的真实发现门槛。',
              '当前全部旧失败结论与阶段C/D/E状态保持。新在线矩阵须另行登记唯一主配置、同信息与删失对照，并纳入全部22种既有强参考。']
    write_json(DESTINATION/'summary.json', {'aggregate': aggregate, 'costs': costs,
                                         'scope': 'matched prediction diagnosis with additional paid stage information'})
    (DESTINATION/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    diagnose()
