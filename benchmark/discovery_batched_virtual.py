"""Validate faster fixed-prior virtual inference using historical data only."""

import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.gcrt.batched_virtual import virtual_probabilities
from methods.gcrt.converged_lookahead import acquisitions, choose, mix_virtual
from methods.gcrt.converged_response import component_probabilities

from . import discovery_budget_lookahead as history
from .discovery_config import read_json, write_json
from .protocol import ROOT


REPORT = history.REPORT / 'batched_compute'
SUPPORT_COUNTS = (8, 64, 127)


def gpu_predictions(current, paid, labels, candidates):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    probability, diagnostics = virtual_probabilities(
        current['means'].to('cuda'), current['covariances'].to('cuda'), paid, labels, candidates)
    probability = probability.cpu().numpy()
    diagnostics = {key: value.cpu().numpy() for key, value in diagnostics.items()}
    torch.cuda.synchronize()
    return probability, diagnostics, time.perf_counter() - started, torch.cuda.max_memory_allocated()


def cpu_predictions(current, paid, labels, candidates):
    started = time.perf_counter()
    probability = []
    maximum_gradient = 0.
    for candidate in candidates:
        branches = []
        for label in (False, True):
            with torch.no_grad():
                values, information = component_probabilities(current['means'], current['covariances'],
                    [*paid, candidate], [*labels, label])
            branches.append(values.numpy())
            maximum_gradient = max(maximum_gradient, max(r['gradient_max'] for r in information))
        probability.append(branches)
    return np.asarray(probability), time.perf_counter() - started, maximum_gradient


def compare_mixtures(current, archive, probability):
    path = history.REPORT / 'histories' / archive.scenarios[0].kind / f'{archive.source_id}.npz'
    with np.load(path, allow_pickle=False) as packed:
        expected = packed['conditional'].copy()
    actual = np.empty_like(expected)
    available = current['available']
    for position, candidate in enumerate(available):
        influence = np.exp(-np.square(current['x'] - current['x'][candidate]).mean(-1) / .18)
        for label in (False, True):
            for mode, name in enumerate(history.MODES):
                actual[mode, position, int(label)] = mix_virtual(probability[position, int(label)],
                    current['before'], current['scores'][name],
                    influence if name == 'local' else np.ones(512), candidate, label)
    maximum_probability_error = float(np.max(np.abs(actual - expected)))
    assert maximum_probability_error < 2e-9
    identifiers = [s.scenario_id for s in archive.scenarios]
    maximum_acquisition_error = 0.
    decisions = []
    for mode, name in enumerate(history.MODES):
        old = acquisitions(current['probability'][name], expected[mode], available, (5, 64))
        new = acquisitions(current['probability'][name], actual[mode], available, (5, 64))
        for horizon in (5, 64):
            error = float(np.max(np.abs(old[horizon] - new[horizon])))
            maximum_acquisition_error = max(maximum_acquisition_error, error)
            assert error < 2e-8
            before = choose(old[horizon], available, identifiers)
            after = choose(new[horizon], available, identifiers)
            assert before == after
            decisions.append({'reliability': name, 'horizon': horizon, 'chosen_candidate': after})
    return maximum_probability_error, maximum_acquisition_error, decisions


def run():
    assert not (REPORT / 'report.json').exists()
    assert read_json(history.REPORT / 'summary_audit.json')['status'] == 'passed'
    torch.set_num_threads(1)
    protocol = {'purpose': 'historical computation readiness, no new development experiment',
        'historical_states': 16, 'full_saved_state_support_count': 64,
        'additional_support_counts': list(SUPPORT_COUNTS),
        'additional_support_scope': 'first historical object of each scene; same fixed 64-support prior',
        'full_saved_comparison': 'both local/global virtual mixtures; both horizons; exact tie policy',
        'cpu_reference_scope': 'all candidates, both labels and all nine components at all three support sizes in two scenes',
        'mode_gradient_tolerance': 1e-10, 'component_probability_tolerance': 2e-9,
        'mixture_probability_tolerance': 2e-9, 'acquisition_tolerance': 2e-8,
        'chunk_size': 32, 'minimum_support64_speedup': 5.,
        'frozen_means_covariances_and_prequery_scores': True,
        'original_history_protocol': history.protocol(),
        'source_files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'benchmark/discovery_batched_virtual.py', 'methods/gcrt/batched_virtual.py')},
        'new_development_queries': 0, 'new_physical_calls': 0, 'formal_phase_C_started': False}
    write_json(REPORT / 'protocol.json', protocol)
    started = time.perf_counter()
    states, references = [], []
    warmup_s = 0.
    try:
        with threadpool_limits(1):
            for kind in ('cut_in', 'cut_out'):
                archives, means, covariances = history.correction.history_inputs(kind)
                for position, archive in enumerate(archives):
                    current = history.state(kind, archive, archives, means, covariances)
                    original = {key: current[key].clone() for key in ('means', 'covariances')}
                    original_scores = {key: value.copy() for key, value in current['scores'].items()}
                    candidates = current['available'].tolist()
                    if not states:
                        _, _, warmup_s, _ = gpu_predictions(current, current['paid'], current['labels'], candidates[:8])
                    probability, diagnostics, elapsed, peak = gpu_predictions(
                        current, current['paid'], current['labels'], candidates)
                    error, acquisition_error, decisions = compare_mixtures(current, archive, probability)
                    assert float(diagnostics['gradients'].max()) <= 1e-10
                    path = REPORT / 'histories' / kind / f'{archive.source_id}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **diagnostics)
                    entry = {'scenario': kind, 'pseudo_target': archive.source_id,
                        'component_modes': int(diagnostics['gradients'].size),
                        'maximum_gradient': float(diagnostics['gradients'].max()),
                        'maximum_iterations': int(diagnostics['iterations'].max()),
                        'maximum_mixed_probability_error': error, 'maximum_acquisition_error': acquisition_error,
                        'decisions': decisions, 'gpu_including_transfers_s': elapsed,
                        'peak_allocated_bytes': peak, 'state_preparation_s': current['preparation_s']}
                    states.append(entry)
                    print('batched_virtual', kind, archive.source_id, elapsed, error, flush=True)
                    if position == 0:
                        order = np.random.default_rng(1301).permutation(512)
                        for count in SUPPORT_COUNTS:
                            paid = order[:count].tolist()
                            labels = archive.collision[paid].tolist()
                            available = sorted(set(range(512)) - set(paid))
                            if count == 64:
                                actual, info, gpu_s, memory = probability, diagnostics, elapsed, peak
                            else:
                                actual, info, gpu_s, memory = gpu_predictions(current, paid, labels, available)
                            expected, cpu_s, cpu_gradient = cpu_predictions(current, paid, labels, available)
                            difference = float(np.max(np.abs(actual - expected)))
                            assert difference < 2e-9 and float(info['gradients'].max()) <= 1e-10
                            reference = {'scenario': kind, 'pseudo_target': archive.source_id,
                                'support_count': count, 'candidate_count': len(available),
                                'component_modes': int(info['gradients'].size),
                                'maximum_component_probability_error': difference,
                                'maximum_gpu_gradient': float(info['gradients'].max()),
                                'maximum_cpu_gradient': cpu_gradient, 'cpu_reference_s': cpu_s,
                                'gpu_including_transfers_s': gpu_s, 'speedup': cpu_s / gpu_s,
                                'peak_allocated_bytes': memory}
                            references.append(reference)
                            write_json(REPORT / 'reference_progress.json', {'references': references, 'states_complete': len(states)})
                            print('batched_reference', kind, count, cpu_s / gpu_s, difference, flush=True)
                    for key, saved in original.items():
                        torch.testing.assert_close(current[key], saved, atol=0, rtol=0)
                    for key, saved in original_scores.items():
                        np.testing.assert_array_equal(current['scores'][key], saved)
        speedups = [r['speedup'] for r in references if r['support_count'] == 64]
        assert len(states) == 16 and len(references) == 6
        result = {'status': 'complete_historical_compute_validation', 'protocol': protocol,
            'states': states, 'cpu_references': references,
            'full_saved_state_gpu_component_modes': sum(r['component_modes'] for r in states),
            'cpu_reference_component_modes': sum(r['component_modes'] for r in references),
            'additional_gpu_component_modes': sum(r['component_modes'] for r in references if r['support_count'] != 64),
            'maximum_mixed_probability_error': max(r['maximum_mixed_probability_error'] for r in states),
            'maximum_acquisition_error': max(r['maximum_acquisition_error'] for r in states),
            'maximum_component_probability_error': max(r['maximum_component_probability_error'] for r in references),
            'maximum_gradient': max([r['maximum_gradient'] for r in states]
                                    + [r['maximum_gpu_gradient'] for r in references]),
            'full_state_selected_candidates_unchanged': sum(len(r['decisions']) for r in states),
            'minimum_support64_speedup': min(speedups), 'compute_readiness_passed': min(speedups) >= 5.,
            'gpu_full_state_sum_s': sum(r['gpu_including_transfers_s'] for r in states),
            'cpu_reference_sum_s': sum(r['cpu_reference_s'] for r in references),
            'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in states + references),
            'warmup8_s': warmup_s, 'warmup_modes_separate': 144,
            'state_preparation_sum_s': sum(r['state_preparation_s'] for r in states),
            'wall_s': time.perf_counter() - started, 'actual_states_unchanged': True,
            'new_development_queries': 0, 'new_physical_calls': 0, 'formal_phase_C_started': False}
        write_json(REPORT / 'report.json', result)
        lines = ['# 批量虚拟推断的历史计算验收', '',
            '数值计算优化；不改变模型、可靠性或ENS计分，不启动新的开发或正式实验。', '',
            '| 场景 | 支持数 | 分量模式数 | CPU秒 | GPU含传输秒 | 加速倍数 | 最大概率差 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in references:
            lines.append(f"| {row['scenario']} | {row['support_count']} | {row['component_modes']} | "
                f"{row['cpu_reference_s']:.3f} | {row['gpu_including_transfers_s']:.3f} | "
                f"{row['speedup']:.2f} | {row['maximum_component_probability_error']:.3g} |")
        lines += ['', f"历史64点状态的64次选点全部一致；最低64点加速{min(speedups):.2f}倍。",
            f"计算准备条件：{'通过' if result['compute_readiness_passed'] else '未通过'}。",
            '支持数8/127只改变同一固定先验的条件化规模，不代表这些规模的在线模型适配或发现收益。',
            '全部16个历史64点状态的局部/全局、5/64步分数均与旧CPU数组核对。',
            '本报告不解除阶段顺序调整所需确认，不证明完整开发矩阵通过。', '']
        (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
        print(json.dumps({key: value for key, value in result.items()
                         if not isinstance(value, (dict, list))}), flush=True)
        return result
    except Exception as error:
        write_json(REPORT / 'failure.json', {'error': repr(error), 'states': states, 'cpu_references': references,
            'wall_s': time.perf_counter() - started, 'new_development_queries': 0, 'new_physical_calls': 0})
        raise


if __name__ == '__main__':
    run()
