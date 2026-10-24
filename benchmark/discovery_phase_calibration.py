"""Controlled repetition of stage predictions with marginally matched prior means."""

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
from methods.gcrt.phase_calibration import calibrated_phase_bases
from methods.gcrt.phase_response import phase_support, source_phase_priors

from .construction import load_construction
from .discovery_conditional_adaptation import assess, probabilities_numpy
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_phase_response_diagnosis import recorded_observations
from .discovery_physical_mixture import MAIN, REPORT, RUNS, checkpoint_path, protocol
from .discovery_physical_mixture_audit import classifier_probabilities_numpy
from .protocol import ROOT


DESTINATION = OUTPUT/'phase_calibration'


def diagnose():
    design = read_json(DESTINATION/'design.json')
    source_protocol = read_json(OUTPUT/'phase_response/protocol.json')
    numerical = read_json(OUTPUT/'phase_response/numerical_checks.json')
    initial_check = read_json(DESTINATION/'prior_check.json')
    if (design['all_other_fit_and_inference_rules'] != source_protocol
            or source_protocol['source_protocol'] != protocol()
            or numerical['status'] != 'passed' or numerical['configurations'] != 240
            or initial_check['status'] != 'passed' or initial_check['initializations_checked'] != 12):
        raise ValueError('Complete and check the registered original stage predictions and initialization diagnosis first')
    for path, identity in source_protocol['source_files'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != identity:
            raise ValueError('An original stage-diagnosis dependency changed')
    specification = {'design': design, 'source_files': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ('benchmark/discovery_phase_calibration.py', 'methods/gcrt/phase_calibration.py')},
        'prior_check_sha256': hashlib.sha256((DESTINATION/'prior_check.json').read_bytes()).hexdigest()}
    registered = DESTINATION/'protocol.json'
    if registered.exists() and read_json(registered) != specification:
        raise ValueError('The mean-matching repetition was frozen differently')
    write_json(registered, specification)
    records = read_json(REPORT/'phase_diagnosis/report.json')['records']
    previous_rows = read_json(OUTPUT/'phase_response/report.json')['rows']
    tasks = design['tasks']
    rows, checks, predictions, constructions = [], [], {}, []
    started = time.perf_counter()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            history, development = load_construction(kind)
            observations = recorded_observations(records, kind, history+development)
            x = development[0].x
            if any(not np.array_equal(a.x, x) for a in development):
                raise ValueError('Stage-prior sharing requires the same public candidate coordinates')
            prefixes = []
            for task in (t for t in tasks if t['scenario'] == kind):
                archive = next(a for a in development if a.source_id == task['target_id'])
                saved = read_json(RUNS/MAIN/task['target_id']/kind/f"{task['algorithm_seed']}.json")
                audit = read_json(REPORT/'audit/runs'/MAIN/task['target_id']/kind/f"{task['algorithm_seed']}.json")
                queries = saved['result']['queries']
                if (audit['status'] != 'passed' or saved['specification']['protocol'] != source_protocol['source_protocol']
                        or len(queries) != 128 or len({q['index'] for q in queries}) != 128
                        or any(q['feedback']['scenario_id'] != archive.scenarios[q['index']].scenario_id
                               or q['feedback']['collision'] != bool(archive.collision[q['index']]) for q in queries)):
                    raise ValueError('Mean matching must use the same audited paid query prefixes')
                paid = [q['index'] for q in queries[:127]]
                labels = [q['feedback']['collision'] for q in queries[:127]]
                raw = ClassificationKernel(x.shape[1]).to(source_protocol['source_protocol']['device'])
                fit_target(raw, x[paid], labels, steps=10, learning_rate=.01)
                with torch.no_grad():
                    raw_p = collision_probabilities(raw, x[paid], labels, x).cpu().numpy()
                np.testing.assert_allclose(raw_p, classifier_probabilities_numpy(raw, x, paid, labels), atol=2e-9, rtol=2e-9)
                prefixes.append((task, archive, paid, raw_p))
            for initialization in design['initializations']:
                base = (load_model(checkpoint_path(kind, MAIN))[0] if initialization == 'joint' else
                        load_fsbo(checkpoint_path(kind, 'physical_fsbo'))[0]).to(source_protocol['source_protocol']['device'])
                bases, mean_checks = calibrated_phase_bases(base, history, observations, x)
                for row in mean_checks:
                    earlier = next(r for r in initial_check['rows'] if (r['scenario'], r['initialization'], r['phase'])
                                   == (kind, initialization, row['phase']))
                    np.testing.assert_allclose(row['marginally_matched_latent_mean'], earlier['marginally_matched_latent_mean'], atol=1e-9)
                before = time.perf_counter()
                censored, support = source_phase_priors(bases, history, observations, x)
                filled, filled_support = source_phase_priors(bases, history, observations, x, zero_fill=True)
                if base.mean.is_cuda:
                    torch.cuda.synchronize()
                constructions.append({'scenario': kind, 'initialization': initialization, 'mean_checks': mean_checks,
                    'censored_source_supports': support, 'zero_fill_source_supports': filled_support,
                    'historical_construction_wall_s': time.perf_counter()-before})
                for task, archive, paid, raw_p in prefixes:
                    for mode in design['modes']:
                        configuration = f'{initialization}_{mode}'
                        weights = np.ones(1) if mode == 'phase_target' else np.r_[.75*source_weights(history), .25]
                        priors = filled if mode == 'phase_zero_fill' else censored
                        hazards, phase_checks, sizes = [], [], []
                        before = time.perf_counter()
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
                                independent = probabilities_numpy(m.cpu().numpy(), c.cpu().numpy(), indices, labels)
                            np.testing.assert_allclose(p, independent, atol=2e-7, rtol=2e-6)
                            hazards.append(weights @ p)
                            phase_checks.append({'phase': phase, 'components': len(p),
                                                 'probability_max_error': float(np.max(np.abs(p-independent)))})
                            sizes.append({'phase': phase, 'observed': len(indices), 'collisions': int(labels.sum())})
                        probability = .9*(1-np.prod(1-np.array(hazards), axis=0))+.1*raw_p
                        row = {**task, 'configuration': configuration, 'paid_phase_supports': sizes,
                            'target_reconstruction_wall_s': time.perf_counter()-before,
                            **assess(probability, paid, archive.collision, archive.scenarios)}
                        for phase in range(3):
                            phase_truth = observations[archive.source_id][:, phase] == 1
                            row[f'phase_{phase}_collision_ranks'] = assess(probability, paid, phase_truth, archive.scenarios)['all_remaining_collision_ranks']
                        if task['target_id'] == 'development_ppo' and kind == 'cut_out':
                            for point in (90, 386):
                                row[f'focus_{point}'] = next((r for r in row['all_remaining_collision_ranks'] if r['index'] == point), None)
                        old = next(r for r in previous_rows if all(r[k] == row[k] for k in
                            ('configuration', 'target_id', 'scenario', 'algorithm_seed')))
                        if old['paid_phase_supports'] != sizes:
                            raise ValueError('Mean matching changed a paid observation or censoring rule')
                        row['difference_from_original'] = {m: row[m]-old[m] if row[m] is not None else None
                                                          for m in ('collision_nll', 'brier', 'mean_collision_rank')}
                        key = f"{task['target_id']}/{kind}/{task['algorithm_seed']}/{configuration}"
                        predictions[key], predictions[key+'/hazards'] = probability, np.array(hazards)
                        rows.append(row)
                        checks.append({'key': key, 'phases': phase_checks})
                    print('phase_calibration_diagnosis', len(rows), '/ 240', initialization,
                          task['target_id'], kind, task['algorithm_seed'], flush=True)
    if protocol() != source_protocol['source_protocol']:
        raise ValueError('The preceding frozen online protocol changed')
    np.savez_compressed(DESTINATION/'probabilities.npz', **predictions)
    result = {'status': 'complete', 'specification': specification, 'rows': rows,
        'prediction_configurations': len(rows), 'historical_constructions': constructions,
        'wall_s': time.perf_counter()-started, 'additional_selector_queries': 0, 'new_physical_calls': 0}
    write_json(DESTINATION/'report.json', result)
    write_json(DESTINATION/'numerical_checks.json', {'status': 'passed', 'configurations': len(checks),
        'full_pool_component_checks': sum(p['components'] for r in checks for p in r['phases']),
        'same_paid_supports_checked': len(rows), 'checks': checks})
    report()
    return result


def report():
    data = read_json(DESTINATION/'report.json')
    if data['prediction_configurations'] != 240:
        raise ValueError('Finish the complete registered mean-matching repetition first')
    rows, aggregate = data['rows'], {}
    lineages = sorted({r['lineage'] for r in rows})
    lines = ['# 初始边际概率匹配：阶段预测对照', '',
        '仅改变阶段基础均值，保持相同40条付费前缀、两个冻结初始化和三种阶段配置。',
        '频率与匹配均值只由历史风险集和公开特征计算；不使用开发目标结果选取均值。',
        '没有新增选例或物理调用，此前240项预测保持原样。', '',
        '| 配置 | NLL | Brier | 含剩余碰撞任务的平均排名 |', '|---|---:|---:|---:|']
    for configuration in sorted({r['configuration'] for r in rows}):
        aggregate[configuration] = {}
        for metric in ('collision_nll', 'brier', 'mean_collision_rank'):
            by_lineage = {}
            for lineage in lineages:
                values = [r[metric] for r in rows if r['configuration'] == configuration
                          and r['lineage'] == lineage and r[metric] is not None]
                by_lineage[lineage] = float(np.mean(values)) if values else None
            valid = [v for v in by_lineage.values() if v is not None]
            aggregate[configuration][metric] = {'mean': float(np.mean(valid)) if valid else None, 'by_lineage': by_lineage}
        a = aggregate[configuration]
        lines.append(f"| {configuration} | {a['collision_nll']['mean']:.6f} | {a['brier']['mean']:.6f} | {a['mean_collision_rank']['mean']:.3f} |")
    lines += ['', '先在谱系内再等权汇总；零碰撞任务参与误差评估，没有剩余碰撞的排名为不适用。',
              '逐任务与原初始化的成对差异、全部剩余碰撞和分阶段排名在report.json中。', '',
              '| PPO切出索引90的配置 | 概率范围 | 排名范围 |', '|---|---:|---:|']
    for configuration in aggregate:
        values = [r['focus_90'] for r in rows if r['configuration'] == configuration and r.get('focus_90') is not None]
        lines.append(f"| {configuration} | {min(v['probability'] for v in values):.6g}–{max(v['probability'] for v in values):.6g} | {min(v['rank'] for v in values)}–{max(v['rank'] for v in values)} |")
    lines += ['', '匹配只约束初始独立目标分量，不保证来源协方差叠加或适配后的概率校准。',
              '这是同付费信息的预测诊断，不是新的碰撞发现数或召回率；不改变旧失败结论与阶段C/D/E状态。',
              '阶段目标模型仍使用旧冻结表示；若进入在线验证，须补齐相同历史阶段训练的Pooled-FT/FSBO强控制。',
              f"诊断总耗时{data['wall_s']:.2f}秒，历史阶段构建与目标重建分别保存在report.json中。"]
    write_json(DESTINATION/'summary.json', {'aggregate': aggregate, 'scope': 'matched paid-prefix predictions only'})
    (DESTINATION/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    diagnose()
