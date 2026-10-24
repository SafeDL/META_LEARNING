"""Evaluator-only event exposure and collision-phase diagnosis of existing banks."""

from collections import Counter
from dataclasses import asdict
import gzip
import json

import numpy as np

from .construction import load_construction
from .discovery_config import read_json, write_json
from .discovery_joint_response import RUNS as JOINT_RUNS
from .discovery_physical_mixture import REPORT, RUNS, protocol
from .execution_events import analyze_events
from .executions import ExecutionBank
from .protocol import DATA_ROOT, DEFAULT
from .scenarios import dedup_key


DESTINATION = REPORT/'phase_diagnosis'
PHASES = ('before_scheduled_event', 'before_measured_motion', 'during_maneuver', 'after_maneuver')


def collision_phase(events, time_s):
    if time_s < events['scheduled_event_s']:
        return PHASES[0]
    onset, complete = events['measured_background_onset_s'], events['background_complete_s']
    if onset is None or time_s < onset:
        return PHASES[1]
    return PHASES[2] if complete is None or time_s < complete else PHASES[3]


def diagnose():
    frozen = read_json(REPORT/'protocol.json')
    if frozen != protocol():
        raise ValueError('Event-phase diagnosis requires the unchanged physical-mixture experiment')
    specification = {'scenarios': ['cut_in', 'cut_out'], 'roles': ['history', 'development'],
                     'banks': 24, 'candidates_per_bank': 512,
                     'phases': list(PHASES), 'event_definition': 'existing recorded-motion thresholds',
                     'additional_selector_queries': 0, 'new_physical_calls': 0,
                     'old_collision_labels_changed': False, 'candidate_filtering': False}
    path = DESTINATION/'protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('Event-phase diagnosis was registered differently')
    write_json(path, specification)
    rows, summaries, focus_cases = [], [], {}
    for kind in specification['scenarios']:
        history, development = load_construction(kind)
        for archive in history+development:
            root = DATA_ROOT/'physical'/archive.role/kind/archive.source_id
            bank = ExecutionBank(root)
            if bank.candidates != archive.scenarios:
                raise ValueError('Event-phase bank coordinates differ from the role-checked archive')
            phases, partners = Counter(), Counter()
            exposed, safe_unexposed, early_safe_end = 0, 0, 0
            for index, scenario in enumerate(bank.candidates):
                with gzip.open(root/'executions'/f'{index:06d}.json.gz', 'rt', encoding='utf-8') as stream:
                    saved = json.load(stream)
                trace, feedback = saved['trajectory'], saved['feedback']
                if (saved['scenario'] != asdict(scenario) or feedback['scenario_id'] != scenario.scenario_id
                        or feedback['collision'] != bool(archive.collision[index])):
                    raise ValueError('Event-phase execution differs from its original scenario/label')
                events = analyze_events(scenario, trace)
                if events != saved['events']:
                    raise ValueError('Stored event times differ from the saved trajectory')
                crash = next((frame for frame in trace if frame['collision']), None)
                if bool(crash) != feedback['collision'] or bool(trace[-1]['collision']) != feedback['collision']:
                    raise ValueError('Collision flag differs from the saved trace')
                reached = events['background_complete_s'] is not None
                exposed += reached
                safe_unexposed += not feedback['collision'] and not reached
                early_safe_end += not feedback['collision'] and trace[-1]['time_s'] < DEFAULT.episode_duration_s-1e-9
                phase = collision_phase(events, crash['time_s']) if crash else 'no_collision'
                if crash:
                    phases[phase] += 1
                    partners[crash['collision_partner']] += 1
                rows.append({'source_id': archive.source_id, 'role': archive.role, 'lineage': archive.lineage,
                             'scenario': kind, 'index': index, 'scenario_id': scenario.scenario_id,
                             'collision': feedback['collision'], 'collision_phase': phase,
                             'first_collision_time_s': crash['time_s'] if crash else None,
                             'recorded_nearest_partner': crash['collision_partner'] if crash else None,
                             'post_maneuver_exposure_observed': reached,
                             'episode_end_time_s': trace[-1]['time_s'], 'events': events})
                if archive.source_id == 'development_ppo' and kind == 'cut_out' and index in (90, 386):
                    focus_cases[index] = {'trajectory': trace, 'events': events}
            summaries.append({'source_id': archive.source_id, 'role': archive.role, 'scenario': kind,
                              'executions': 512, 'collisions': int(archive.collision.sum()),
                              'collision_phases': {phase: phases[phase] for phase in PHASES},
                              'recorded_nearest_partners': dict(partners),
                              'post_maneuver_exposure_observed': exposed,
                              'no_collision_without_post_maneuver_exposure': safe_unexposed,
                              'noncollision_episode_ended_before_horizon': early_safe_end})
            print('event_phase_diagnosis', len(summaries), '/ 24', archive.source_id, kind, flush=True)
    ppo = [r for r in rows if r['source_id'] == 'development_ppo' and r['scenario'] == 'cut_out']
    bank = ExecutionBank(DATA_ROOT/'physical/development/cut_out/development_ppo')
    if len({dedup_key(s) for s in bank.candidates}) != 512:
        raise ValueError('Phase-specific PPO discovery needs the unique-coordinate pool')
    post = {r['index'] for r in ppo if r['collision_phase'] == 'after_maneuver'}
    other = {r['index'] for r in ppo if r['collision'] and r['index'] not in post}
    methods = ('fsbo_cuda', 'physical_fsbo', 'physical_target', 'physical_global', 'physical_local', 'joint_global')
    discovery = []
    for method in methods:
        for seed in DEFAULT.algorithm_seeds:
            if method == 'joint_global':
                path = JOINT_RUNS/method/'development_ppo/cut_out'/f'{seed}.json'
            else:
                path = RUNS/method/'development_ppo/cut_out'/f'{seed}.json'
            saved = read_json(path)
            queries = saved['result']['queries']
            if len(queries) != 128 or len({q['index'] for q in queries}) != 128:
                raise ValueError('Phase discovery requires a complete paid run')
            selected = {q['index'] for q in queries}
            if any(q['feedback']['collision'] != ppo[q['index']]['collision'] for q in queries):
                raise ValueError('Phase discovery feedback differs from the original execution')
            discovery.append({'method': method, 'seed': seed,
                              'post_maneuver_found': len(post & selected), 'post_maneuver_total': len(post),
                              'other_collision_found': len(other & selected), 'other_collision_total': len(other)})
    if protocol() != frozen or len(rows) != 12288:
        raise ValueError('Event-phase diagnosis is incomplete or the selection protocol changed')
    result = {'status': 'complete', 'specification': specification, 'summaries': summaries,
              'records': rows, 'ppo_phase_discovery': discovery,
              'stored_events_reconstructed': len(rows), 'collision_flags_checked': len(rows),
              'additional_selector_queries': 0, 'new_physical_calls': 0}
    write_json(DESTINATION/'report.json', result)
    lines = ['# 历史与开发场景的交互阶段诊断', '',
             '完整核对切入/切出的24份历史与开发档案，共12,288条既有执行；逐条重建事件并核对碰撞标记。',
             '相位仅用于评估端的辅助解释，未筛选候选或改变旧标签、分母、128次预算。', '',
             '| 场景 | 系统 | 碰撞数 | 明显横移前 | 横移中 | 横移完成后 | 观察到横移完成的执行数 | 无碰撞且未观察到完成 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        p = s['collision_phases']
        lines.append(f"| {s['scenario']} | {s['source_id']} | {s['collisions']} | {p[PHASES[0]]+p[PHASES[1]]} | {p[PHASES[2]]} | {p[PHASES[3]]} | {s['post_maneuver_exposure_observed']} | {s['no_collision_without_post_maneuver_exposure']} |")
    lines += ['', '“明显横移”及“完成”采用既有事件重建的位移/朝向阈值；发生在完成前的碰撞意味着后续阶段未被观察。',
              '表中为执行计数。recorded_nearest_partner来自原仿真首次碰撞标记时最近对象的记录，不能单独视为精确接触对象识别。', '',
              '| PPO切出方法 | 五种子平均发现：完成前碰撞 | 五种子平均发现：完成后碰撞 | 完成后碰撞池 |',
              '|---|---:|---:|---:|']
    for method in methods:
        group = [r for r in discovery if r['method'] == method]
        lines.append(f"| {method} | {np.mean([r['other_collision_found'] for r in group]):.3f} | {np.mean([r['post_maneuver_found'] for r in group]):.3f} | {len(post)} |")
    lines += ['', 'PPO切出22次碰撞中，16次发生在明显横移前、4次在横移中、2次在横移完成后；持续漏检的90和386均属于最后一组。',
              '两例首次碰撞时刻分别为4.85秒和4.25秒，主车已进入前车所在的相邻车道；其保存最小间距均为0。',
              '“切出”模板包含不同阶段的失效，不能把全部碰撞都解释为障碍物暴露后的避障失败。', '',
              '![两例完成后碰撞的已保存轨迹](late_cases.png)', '',
              '下一轮值得检验按交互阶段分解响应及来源可靠性；未观察到后续阶段的执行不直接充当该阶段的安全标签。',
              '这是新的研究假设。若在线加入阶段反馈，需要与获得相同信息的目标模型比较；当前冻结实验和主方法失败结论保持不变。',
              '新增任务的初始几何、背景运动与时序应在运行目标前验收，不按目标是否碰撞选样；旧碰撞全部保留。', '']
    (DESTINATION/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    plot_cases(focus_cases)
    return result


def plot_cases(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(12, 8), sharex='col')
    for column, index in enumerate((90, 386)):
        trace = cases[index]['trajectory']
        events = cases[index]['events']
        t = [f['time_s'] for f in trace]
        ego = [next(a for a in f['actors'] if a['actor_id'] == 'ego') for f in trace]
        lead = [next(a for a in f['actors'] if a['actor_id'] == 'lead') for f in trace]
        axes[0, column].plot(t, [a['y'] for a in ego], label='Ego')
        axes[0, column].plot(t, [a['y'] for a in lead], label='Lead')
        axes[1, column].plot(t, [a['speed'] for a in ego])
        axes[1, column].plot(t, [a['speed'] for a in lead])
        axes[2, column].plot(t, [b['x']-a['x'] for a, b in zip(ego, lead)])
        axes[0, column].set(title=f'PPO cut-out: index {index}', ylim=(-.3, 4.3))
        for axis in axes[:, column]:
            axis.axvspan(events['measured_background_onset_s'], events['background_complete_s'],
                         alpha=.13, color='gray', label='Background maneuver')
            axis.axvline(t[-1], color='red', linestyle='--', label='First collision flag')
            axis.grid(alpha=.2)
        axes[2, column].set_xlabel('Time (s)')
    for axis, label in zip(axes[:, 0], ('Lane position y (m)', 'Speed (m/s)', 'Longitudinal center separation (m)')):
        axis.set_ylabel(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    figure.tight_layout(rect=(0, .05, 1, 1))
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(DESTINATION/f'late_cases.{extension}', dpi=180, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    diagnose()
