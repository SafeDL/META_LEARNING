"""Recompute historical scale selection and verify exact context/coverage sets."""

import hashlib
import time

import numpy as np

from .construction import load_construction
from .discovery_config import read_json, write_json
from .discovery_residual_scale import ARTIFACTS, BETAS, COUNTS, REPORT, protocol


def audit():
    started = time.perf_counter()
    selection = read_json(REPORT/'selection.json')
    prefixes = read_json(REPORT/'cache_audit.json')
    predictions = read_json(REPORT/'prediction_checks.json')
    index = read_json(REPORT/'cache_index.json')
    if (selection['status'] != 'complete' or selection['protocol'] != protocol()
            or prefixes['status'] != 'passed' or predictions['status'] != 'passed'):
        raise ValueError('Complete the unchanged historical learning and independent prefix checks first')
    histories = {kind: load_construction(kind)[0] for kind in ('cut_in', 'cut_out')}
    expected_tasks = {(kind, a.source_id) for kind, history in histories.items() for a in history}
    actual_tasks = [(r['specification']['kind'], r['specification']['source_id']) for r in index['records']]
    if len(actual_tasks) != 16 or len(set(actual_tasks)) != 16 or set(actual_tasks) != expected_tasks:
        raise ValueError('Historical adaptation cache coverage differs from the exact registered set')
    expected_rows = {(kind, name, beta, mode, count) for kind, name in expected_tasks
                     for beta in BETAS for mode in ('local', 'global') for count in COUNTS}
    actual_rows = [(r['scenario'], r['pseudo_target'], r['beta'], r['reliability'], r['support_count'])
                   for r in selection['rows']]
    if len(actual_rows) != 800 or len(set(actual_rows)) != 800 or set(actual_rows) != expected_rows:
        raise ValueError('Historical parameter comparisons are duplicated or incomplete')
    for row in selection['rows']:
        history = histories[row['scenario']]
        target = next(a for a in history if a.source_id == row['pseudo_target'])
        allowed = [a.source_id for a in history if a.source_id != target.source_id]
        if (row['source_context_ids'] != allowed or row['lineage'] != target.lineage
                or target.role != 'history' or any(a.role != 'history' for a in history)):
            raise ValueError('A pseudo target or nonhistorical object entered the historical context')
    for kind, history in histories.items():
        summaries = []
        for beta in BETAS:
            for mode in ('local', 'global'):
                group = [r for r in selection['rows'] if (r['scenario'], r['beta'], r['reliability']) == (kind, beta, mode)]
                values = {metric: float(np.mean([np.mean([r[metric] for r in group if r['lineage'] == a.lineage])
                                                for a in history])) for metric in ('query_nll', 'query_brier')}
                recorded = next(r for r in selection['summary'] if (r['scenario'], r['beta'], r['reliability']) == (kind, beta, mode))
                for metric in values:
                    np.testing.assert_allclose(values[metric], recorded[metric], atol=1e-12, rtol=1e-12)
                if mode == 'local':
                    summaries.append((values['query_nll'], values['query_brier'], abs(beta-1), beta))
        if min(summaries)[-1] != selection['selected_beta'][kind]:
            raise ValueError('Selected scale differs from the preregistered historical selection rule')
    expected_checks = {(kind, name, beta, count) for kind, name in expected_tasks for beta in BETAS for count in (0, 8, 64, 128)}
    actual_checks = [(r['scenario'], r['pseudo_target'], r['beta'], r['support_count']) for r in predictions['checks']]
    if len(actual_checks) != 320 or len(set(actual_checks)) != 320 or set(actual_checks) != expected_checks:
        raise ValueError('Independent probability checks do not cover their exact registered set')
    selected_checks = [r for r in predictions['checks'] if r['beta'] == selection['selected_beta'][r['scenario']]]
    result = {'status': 'passed', 'unique_historical_cache_tasks': 16, 'unique_parameter_comparisons': 800,
        'own_source_exclusion_rows_checked': 800, 'independent_prefix_refits': prefixes['independent_prefix_refits'],
        'independent_probability_prefixes': 320,
        'maximum_independent_probability_error': max(r['probability_max_error'] for r in predictions['checks']),
        'selected_beta_probability_max_error': max(r['probability_max_error'] for r in selected_checks),
        'selected_beta_recomputed': selection['selected_beta'],
        'historical_cache_sum_worker_wall_s': sum(r['wall_s'] for r in index['records']),
        'parameter_fit_wall_s': selection['wall_s'], 'selection_audit_wall_s': time.perf_counter()-started,
        'evidence': {p: hashlib.sha256((REPORT/p).read_bytes()).hexdigest() for p in
                     ('selection.json', 'cache_index.json', 'cache_audit.json', 'prediction_checks.json')},
        'new_selector_queries': 0, 'new_physical_calls': 0}
    write_json(REPORT/'selection_audit.json', result)
    return result


if __name__ == '__main__':
    print(audit())
