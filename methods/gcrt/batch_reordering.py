"""Separate future batch reallocation from fixed-batch probability drift."""

import numpy as np

from .converged_lookahead import acquisitions


def reordering_values(probability, conditional, available, scenario_ids, horizon):
    available = np.asarray(available, dtype=int)
    original = acquisitions(probability, conditional, available, (horizon,))[horizon]
    drift = np.zeros(len(available))
    baseline = np.asarray(probability)[available].copy()
    if horizon > 1:
        for position, candidate in enumerate(available):
            fixed = sorted((j for j in available if j != candidate),
                           key=lambda j: (-probability[j], scenario_ids[j]))[:horizon - 1]
            before = float(np.sum(probability[fixed]))
            expected = ((1 - probability[candidate]) * np.sum(conditional[position, 0, fixed])
                        + probability[candidate] * np.sum(conditional[position, 1, fixed]))
            drift[position] = expected - before
            baseline[position] += before
    adjusted = original - drift
    if np.min(adjusted - baseline) < -1e-10:
        raise FloatingPointError('Reallocating a fixed batch cannot have negative optimality gain')
    return {'adjusted': adjusted, 'original': original, 'fixed_batch_drift': drift,
            'passive_batch': baseline}
