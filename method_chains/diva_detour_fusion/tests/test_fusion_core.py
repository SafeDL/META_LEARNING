from __future__ import annotations

import numpy as np

from method_chains.diva_detour_fusion.fusion import (
    detour_fused_diagnostic_mining,
    hierarchy_aware_support_indices,
    hierarchy_prior,
)
from diva_highway_env.diva.low_rank_prior import LowRankPrior


def test_detour_hierarchy_uses_source_labels_and_returns_complete_ranking():
    history_features = np.array([[0.0, 0.0], [0.1, 0.0], [0.9, 0.9], [1.0, 1.0]])
    candidates = np.array([[0.05, 0.0], [0.95, 1.0], [0.5, 0.5]])
    hierarchy = hierarchy_prior(history_features,
                                candidates,
                                np.array([True, True, False, False]),
                                seed=7)
    assert sorted(hierarchy.order.tolist()) == [0, 1, 2]
    assert hierarchy.rank_score[hierarchy.order[0]] == 1.0
    assert hierarchy.rank_score[hierarchy.order[-1]] == 0.0


def test_detour_fusion_counts_support_without_unselected_target_lookahead():
    prior = LowRankPrior.fit(np.array([[0.1, 0.7, 0.2, 0.9, 0.3], [0.2, 0.8, 0.1, 0.8, 0.4]]),
                             rank=2)
    hierarchy = hierarchy_prior(
        np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]]),
        np.array([[0.0, 0.0], [0.2, 0.0], [0.8, 1.0], [1.0, 1.0], [0.5, 0.5]]),
        np.array([True, True, False, False]),
        seed=3,
    )
    vulnerability = np.array([0.8, 0.7, 0.1, 0.9, 0.2])
    collisions = np.array([True, False, False, True, False])
    near_misses = np.array([False, True, False, False, False])
    trace = detour_fused_diagnostic_mining(prior, hierarchy, vulnerability, collisions,
                                           near_misses, 2, 4)
    assert len(trace.queried_indices) == len(np.unique(trace.queried_indices)) == 4
    assert trace.method == "DIVA Diagnostic + DETOUR Hierarchy"
    support = hierarchy_aware_support_indices(prior, hierarchy, vulnerability, 2)
    changed_unrevealed = vulnerability.copy()
    changed_unrevealed[[index for index in range(len(vulnerability))
                        if index not in support]] = 99.0
    assert np.array_equal(
        support,
        hierarchy_aware_support_indices(prior, hierarchy, changed_unrevealed, 2),
    )
