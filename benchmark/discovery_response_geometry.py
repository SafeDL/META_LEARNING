"""Fixed-support response-kernel geometry checks; no selector or physical calls."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.gp_classifier import fit_target
from methods.gcrt.joint_training import load_model

from . import discovery_converged_search as study
from . import discovery_physical_mixture as physical
from .construction import load_construction
from .discovery_config import read_json, write_json
from .discovery_converged_search_audit import probabilities_numpy
from .discovery_residual_online import source_priors
from .executions import ExecutionBank
from .protocol import DATA_ROOT, ROOT
from .scenarios import features


REPORT = study.REPORT / 'diagnosis/geometry'
VIEWS = ('adapted_collision', 'frozen_collision', 'frozen_response', 'half_response')
PREFIXES = (9, 65, 128)


def geometry_kernels(adapted, frozen_collision, frozen_response):
    """Keep average marginal variance fixed while changing response geometry."""
    scale = float(np.trace(adapted) / len(adapted))
    normalize = lambda matrix: matrix * (scale / (np.trace(matrix) / len(matrix)))
    collision, response = normalize(frozen_collision), normalize(frozen_response)
    return np.stack((adapted, collision, response, .5 * (adapted + response)))


def prepare():
    diagnosis = read_json(study.REPORT / 'diagnosis/report.json')
    assert diagnosis['status'] == 'complete_independently_reviewed_paid_prefix_diagnosis'
    assert diagnosis['paired_runs'] == 40 and diagnosis['prefixes_checked'] == 120
    assert read_json(study.REPORT / 'protocol.json') == study.protocol()
    specification = {'purpose': 'controlled fixed-support response geometry diagnostic',
        'views': list(VIEWS), 'prefixes': list(PREFIXES), 'runs': 40,
        'mean': 'unchanged adapted collision teacher mean and original source offsets',
        'variance': 'each view has the adapted collision kernel mean diagonal',
        'reliability': 'unchanged original prequery scores; no label rescoring',
        'teachers': 'existing frozen collision and joint collision-clearance teachers',
        'response_teacher_target_adaptation': False,
        'registered_protocol_sha256': hashlib.sha256((study.REPORT / 'protocol.json').read_bytes()).hexdigest(),
        'sets_sha256': hashlib.sha256((study.REPORT / 'diagnosis/sets.json').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_selector_queries': 0, 'new_physical_calls': 0,
        'counterfactuals_are_online_results': False}
    path = REPORT / 'protocol.json'
    if path.exists():
        assert read_json(path) == specification
    else:
        write_json(path, specification)
    return specification


def task(row, specification):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        started = time.perf_counter()
        output = read_json(ROOT / row['main_path'])
        assert output['specification']['protocol'] == study.protocol()
        path = REPORT / 'prefixes' / row['target_id'] / row['scenario'] / f"{row['algorithm_seed']}.npz"
        identity = {'protocol': specification,
                    'run': hashlib.sha256((ROOT / row['main_path']).read_bytes()).hexdigest()}
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                assert json.loads(str(saved['identity'])) == identity
                return json.loads(str(saved['summary']))
        history, _ = load_construction(row['scenario'])
        candidates = ExecutionBank(DATA_ROOT / 'physical/development' / row['scenario'] / row['target_id']).candidates
        x = features(candidates)
        base, _ = load_fsbo(physical.checkpoint_path(row['scenario'], 'physical_fsbo'))
        response, _ = load_model(physical.checkpoint_path(row['scenario'], physical.MAIN))
        device = output['specification']['protocol']['kernel_and_adaptation_device']
        base, response = base.to(device), response.to(device)
        with torch.no_grad():
            frozen_collision = base.covariance(x).cpu().numpy()
            frozen_response = response.covariance(x).cpu().numpy()
        sources = source_priors(row['scenario'], candidates, history)
        old_path = study.REPORT / 'diagnosis/prefixes' / row['target_id'] / row['scenario'] / f"{row['algorithm_seed']}.npz"
        records, arrays, original_error = [], {}, 0.
        with np.load(old_path, allow_pickle=False) as previous:
            for number in PREFIXES:
                paid_queries = output['result']['queries'][:number - 1]
                paid = [q['index'] for q in paid_queries]
                labels = np.asarray([q['feedback']['collision'] for q in paid_queries], dtype=bool)
                model = copy.deepcopy(base)
                fit_target(model, x[paid], labels, steps=10, learning_rate=.003)
                with torch.no_grad():
                    covariance = model.covariance(x).cpu().numpy()
                    center = float(model.mean.detach())
                kernels = geometry_kernels(covariance, frozen_collision, frozen_response)
                expected_variance = float(np.trace(covariance) / len(x))
                np.testing.assert_allclose(kernels.diagonal(axis1=-2, axis2=-1).mean(1),
                                           expected_variance, atol=1e-12)
                means = np.concatenate((sources['means'] + (center - float(base.mean.detach())),
                                        np.full((1, len(x)), center)))
                all_components = []
                for kernel in kernels:
                    covariances = np.concatenate((sources['covariances'] + output['result']['beta']**2 * kernel[None],
                                                  kernel[None]))
                    all_components.append(probabilities_numpy(means, covariances, paid, labels))
                components = np.asarray(all_components)
                np.testing.assert_allclose(components[0], previous[f'q{number}_components'][:-1],
                                           atol=2e-7, rtol=2e-6)
                weights = previous[f'q{number}_weights']
                physical_p = previous[f'q{number}_components'][-1]
                scores = .9 * np.sum(components * weights[None], axis=1) + .1 * physical_p
                original_error = max(original_error, float(np.max(np.abs(scores[0] - previous[f'q{number}_scores'][0]))))
                np.testing.assert_allclose(scores[0], previous[f'q{number}_scores'][0], atol=2e-7, rtol=2e-6)
                remaining = previous[f'q{number}_remaining']
                for case in row['cases']:
                    if case['category'] not in ('reference_only', 'both_missed'):
                        continue
                    positive = np.asarray(case['positive_indices'])
                    assert set(positive) <= set(remaining)
                    records.append({'query': number, 'group': case['group'], 'category': case['category'],
                        'ranks': {name: 1 + int(np.count_nonzero(scores[i, remaining]
                                  > scores[i, positive].max() + 1e-12)) for i, name in enumerate(VIEWS)}})
                arrays[f'q{number}_scores'] = scores
                arrays[f'q{number}_components'] = components
                arrays[f'q{number}_kernel_mean_diagonal'] = kernels.diagonal(axis1=-2, axis2=-1).mean(1)
        summary = {k: row[k] for k in ('target_id', 'lineage', 'scenario', 'algorithm_seed')}
        summary.update(prefixes=3, original_score_max_error=original_error,
                       records=records, wall_s=time.perf_counter() - started)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, identity=np.asarray(json.dumps(identity)),
                            summary=np.asarray(json.dumps(summary)), **arrays)
        return summary


def run(smoke=False):
    specification = read_json(REPORT / 'protocol.json')
    assert specification == prepare()
    rows = read_json(study.REPORT / 'diagnosis/sets.json')['records']
    rows = rows[:1] if smoke else rows
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(task, row, specification) for row in rows]):
            results.append(future.result())
            print('response_geometry', len(results), '/', len(rows), flush=True)
    grouped = defaultdict(list)
    for row in results:
        for record in row['records']:
            grouped[(row['lineage'], row['scenario'], record['query'], record['category'])].append(record)
    aggregates = [{'lineage': k[0], 'scenario': k[1], 'query': k[2], 'category': k[3],
                   'paired_case_records': len(values),
                   'median_rank': {name: float(np.median([r['ranks'][name] for r in values])) for name in VIEWS}}
                  for k, values in sorted(grouped.items())]
    result = {'status': 'complete_fixed_support_geometry_diagnostic', 'scope': 'smoke' if smoke else 'complete',
              'runs': len(results), 'prefixes': len(results) * 3, 'views': list(VIEWS),
              'maximum_original_score_error': max(r['original_score_max_error'] for r in results),
              'records': results, 'aggregates': aggregates, 'new_selector_queries': 0,
              'new_physical_calls': 0, 'counterfactuals_are_online_results': False}
    write_json(REPORT / ('smoke.json' if smoke else 'report.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'smoke', 'run'))
    stage = parser.parse_args().stage
    prepare() if stage == 'prepare' else run(smoke=stage == 'smoke')
