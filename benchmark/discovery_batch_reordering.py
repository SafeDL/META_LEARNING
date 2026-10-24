"""Historical decision-score comparison reusing every virtual probability field."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.gcrt.batch_reordering import reordering_values
from methods.gcrt.converged_lookahead import choose

from . import discovery_budget_lookahead as context
from .discovery_budget_lookahead_report import mean_lineages
from .discovery_config import read_json, write_json
from .protocol import ROOT


REPORT = context.REPORT / 'reordering'
RULES = ('greedy', 'ens', 'reordering')


def prepare():
    assert read_json(context.REPORT / 'summary_audit.json')['status'] == 'passed'
    specification = {'purpose': 'exploratory fixed-state batch-score comparison; no new search trajectory',
        'context': context.protocol(), 'historical_states': 16, 'rules': list(RULES),
        'horizons': [5, 64], 'primary_horizon': 64,
        'formula': 'original ENS minus expected drift of its pre-observation best fixed future batch',
        'screening': 'local 64 actual batch potential higher than both greedy and ENS; neither scene lower; '
                     'two positive lineages against each control',
        'new_virtual_enumerations': 0, 'new_development_queries': 0, 'new_physical_calls': 0,
        'original_state_index_sha256': hashlib.sha256((context.REPORT / 'all_states.json').read_bytes()).hexdigest(),
        'source_files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'benchmark/discovery_batch_reordering.py', 'methods/gcrt/batch_reordering.py')}}
    path = REPORT / 'protocol.json'
    if path.exists():
        assert read_json(path) == specification
    else:
        write_json(path, specification)
    return specification


def run():
    specification = read_json(REPORT / 'protocol.json')
    assert specification == prepare() and not (REPORT / 'report.json').exists()
    index = read_json(context.REPORT / 'all_states.json')
    started = time.perf_counter()
    torch.set_num_threads(1)
    records, checks, maximum_error, reconstruction_s = [], [], 0., 0.
    with threadpool_limits(1):
        for state in index['states']:
            kind = state['scenario']
            history, means, covariances = context.correction.history_inputs(kind)
            archive = next(a for a in history if a.source_id == state['pseudo_target'])
            identifiers = [s.scenario_id for s in archive.scenarios]
            path = context.REPORT / 'histories' / kind / f'{archive.source_id}.npz'
            with np.load(path, allow_pickle=False) as packed:
                saved = {key: packed[key] for key in packed.files}
            available = saved['available']
            chosen_to_check = set()
            arrays = {}
            for mode, name in enumerate(context.MODES):
                p = saved['probability'][mode]
                conditional = saved['conditional'][mode]
                greedy = choose(p[available], available, identifiers)
                for horizon in context.HORIZONS:
                    values = reordering_values(p, conditional, available, identifiers, horizon)
                    # Verify the direct decision-gain identity independently of subtracting drift.
                    for position, candidate in enumerate(available):
                        future = [i for i in available if i != candidate]
                        fixed = sorted(future, key=lambda i: (-p[i], identifiers[i]))[:horizon - 1]
                        gains = []
                        for label in (0, 1):
                            optimal = float(np.sort(conditional[position, label, future])[-horizon + 1:].sum())
                            gains.append(optimal - float(np.sum(conditional[position, label, fixed])))
                        direct = p[candidate] + np.sum(p[fixed]) + (1 - p[candidate]) * gains[0] + p[candidate] * gains[1]
                        error = abs(float(direct - values['adjusted'][position]))
                        maximum_error = max(maximum_error, error)
                        assert error < 1e-10
                    candidates = {'greedy': greedy, 'ens': choose(values['original'], available, identifiers),
                                  'reordering': choose(values['adjusted'], available, identifiers)}
                    chosen_to_check.add(candidates['reordering'])
                    for rule, candidate in candidates.items():
                        position = int(np.flatnonzero(available == candidate)[0])
                        if rule in ('greedy', 'ens'):
                            old = next(r for r in state['records'] if (r['reliability'], r['horizon'], r['rule'])
                                == (name, horizon, 'greedy' if rule == 'greedy' else 'lookahead'))
                            assert old['candidate'] == candidate
                        # Historical ground truth is used only after choices are fixed.
                        label = bool(archive.collision[candidate])
                        after = conditional[position, int(label)]
                        future = sorted((i for i in available if i != candidate),
                                        key=lambda i: (-after[i], identifiers[i]))[:horizon - 1]
                        hits = int(label) + int(archive.collision[future].sum())
                        if rule in ('greedy', 'ens'):
                            assert old['actual_batch_collisions'] == hits
                        records.append({'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage,
                            'reliability': name, 'horizon': horizon, 'rule': rule, 'candidate': candidate,
                            'actual_batch_collisions': hits,
                            'original_ens_score': float(values['original'][position]),
                            'adjusted_score': float(values['adjusted'][position]),
                            'fixed_batch_drift': float(values['fixed_batch_drift'][position])})
                    for key, array in values.items():
                        arrays[f'{name}_{horizon}_{key}'] = array
            current = context.state(kind, archive, history, means, covariances)
            reconstruction_s += current['preparation_s']
            np.testing.assert_allclose(current['before'], saved['before'], atol=2e-7, rtol=2e-6)
            for candidate in sorted(chosen_to_check):
                position = int(np.flatnonzero(available == candidate)[0])
                for label in (False, True):
                    error = context.independent_branch(current, candidate, label, saved['conditional'][:, position, int(label)])
                    checks.append({'scenario': kind, 'pseudo_target': archive.source_id,
                        'candidate': candidate, 'virtual_label': label, 'maximum_probability_error': error})
            target = REPORT / 'histories' / kind / f'{archive.source_id}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, available=available, **arrays)
            print('batch_reordering', kind, archive.source_id, flush=True)
    assert len(records) == 192
    comparisons, summary = [], []
    for mode in context.MODES:
        for horizon in context.HORIZONS:
            for control in ('greedy', 'ens'):
                paired = []
                for kind in ('cut_in', 'cut_out'):
                    for state in (s for s in index['states'] if s['scenario'] == kind):
                        values = {rule: next(r['actual_batch_collisions'] for r in records if
                            (r['pseudo_target'], r['scenario'], r['reliability'], r['horizon'], r['rule'])
                            == (state['pseudo_target'], kind, mode, horizon, rule)) for rule in ('reordering', control)}
                        paired.append({'scenario': kind, 'lineage': state['lineage'],
                                       'gain': values['reordering'] - values[control]})
                scenes = {kind: mean_lineages([r for r in paired if r['scenario'] == kind], 'gain')
                          for kind in ('cut_in', 'cut_out')}
                by = {lineage: mean_lineages([r for r in paired if r['lineage'] == lineage], 'gain')
                      for lineage in sorted({r['lineage'] for r in paired})}
                gain = mean_lineages(paired, 'gain')
                comparisons.append({'reliability': mode, 'horizon': horizon, 'control': control,
                    'mean_actual_batch_gain': gain, 'scene_gains': scenes, 'lineage_gains': by,
                    'passed': gain > 1e-12 and all(v >= -1e-12 for v in scenes.values())
                              and sum(v > 1e-12 for v in by.values()) >= 2})
            for rule in RULES:
                group = [r for r in records if (r['reliability'], r['horizon'], r['rule']) == (mode, horizon, rule)]
                summary.append({'reliability': mode, 'horizon': horizon, 'rule': rule,
                    'mean_actual_batch_collisions': mean_lineages(group, 'actual_batch_collisions')})
    primary = [r for r in comparisons if (r['reliability'], r['horizon']) == ('local', 64)]
    result = {'status': 'complete_independently_checked_reordering_diagnosis', 'protocol': specification,
        'historical_states': 16, 'records': records, 'summary': summary, 'comparisons': comparisons,
        'screening_passed': all(r['passed'] for r in primary),
        'maximum_score_identity_error': maximum_error, 'independent_new_choice_branch_groups': len(checks),
        'maximum_independent_probability_error': max(r['maximum_probability_error'] for r in checks),
        'independent_branch_checks': checks, 'state_reconstruction_s': reconstruction_s,
        'wall_s': time.perf_counter() - started, 'new_virtual_enumerations': 0,
        'new_development_queries': 0, 'new_physical_calls': 0, 'formal_stage_C_started': False}
    write_json(REPORT / 'report.json', result)
    lines = ['# 保留批量重排收益的历史计分诊断', '',
        '复用原16个状态的全部虚拟概率，不重新枚举条件分支；新选点另做独立NumPy/SciPy核对。',
        '本项是观察原状态后的探索性分析，不是新的独立验证。', '',
        '| 窗口 | 可靠性 | 对照 | 切入实际批量差 | 切出实际批量差 | 平均差 |', '|---:|---|---|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['horizon']} | {row['reliability']} | {row['control']} | "
            f"{row['scene_gains']['cut_in']:.6f} | {row['scene_gains']['cut_out']:.6f} | {row['mean_actual_batch_gain']:.6f} |")
    lines += ['', f"预定主要局部64步条件：{'通过' if result['screening_passed'] else '未通过'}。",
        '未改写原ENS结果，不将5步辅助窗口改为主要窗口。',
        '该计分未校准真实碰撞概率或改变后验，不继承知识梯度或ENS的理论保证。',
        '没有新增开发查询或仿真，未自动进入阶段C。', '']
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({k: result[k] for k in ('status', 'screening_passed', 'comparisons', 'wall_s')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    globals()[parser.parse_args().stage]()
