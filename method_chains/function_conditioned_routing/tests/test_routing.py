from __future__ import annotations

import numpy as np

from highway_env_benchmark.mining.low_rank_prior import LowRankPrior
from method_chains.function_conditioned_routing.config import RoutingExperimentConfig
from method_chains.function_conditioned_routing.routing import (
    adate_support,
    coverage_adate_support,
    functional_routed_mining,
    routing_prediction,
)


def _config() -> RoutingExperimentConfig:
    return RoutingExperimentConfig(
        num_anchors=12,
        support_budget=4,
        total_budget=8,
        random_repeats=1,
    )


def test_functional_routing_recovers_distinct_source_by_mode() -> None:
    sources = np.asarray([
        [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95, 0.10, 0.20, 0.30, 0.40],
        [0.80, 0.85, 0.90, 0.95, 0.05, 0.10, 0.15, 0.20, 0.70, 0.75, 0.80, 0.85],
        [0.35, 0.40, 0.45, 0.50, 0.35, 0.40, 0.45, 0.50, 0.40, 0.45, 0.50, 0.55],
    ])
    modes = np.asarray(["cutin"] * 4 + ["follow"] * 4 + ["brake"] * 4)
    target = np.concatenate((sources[0, :4], sources[1, 4:8], sources[2, 8:]))
    selected = np.asarray([0, 2, 4, 6, 8, 10])
    prior = LowRankPrior.fit(sources, rank=2)
    state = routing_prediction(
        sources,
        modes,
        prior,
        selected,
        target[selected],
        _config(),
    )
    assert np.mean((state.routed_prediction - target) ** 2) < 3e-3
    assert np.argmax(state.mode_alphas["cutin"]) == 0
    assert np.argmax(state.mode_alphas["follow"]) == 1
    assert np.argmax(state.mode_alphas["brake"]) == 2


def test_trust_gate_rejects_incompatible_history() -> None:
    sources = np.asarray([
        np.linspace(0.0, 0.3, 12),
        np.linspace(0.2, 0.5, 12),
        np.linspace(0.4, 0.7, 12),
    ])
    modes = np.asarray(["cutin"] * 6 + ["follow"] * 6)
    target = np.ones(12)
    selected = np.asarray([0, 2, 4, 6, 8, 10])
    prior = LowRankPrior.fit(sources, rank=2)
    state = routing_prediction(
        sources,
        modes,
        prior,
        selected,
        target[selected],
        _config(),
    )
    assert max(state.mode_gates.values()) < 0.25


def test_mining_trace_is_unique_and_respects_budget() -> None:
    sources = np.asarray([
        np.linspace(0.0, 0.8, 12),
        np.linspace(0.8, 0.0, 12),
        np.full(12, 0.4),
    ])
    modes = np.asarray(["cutin"] * 4 + ["follow"] * 4 + ["brake"] * 4)
    target = sources[0]
    prior = LowRankPrior.fit(sources, rank=2)
    critical = target > 0.6
    result = functional_routed_mining(
        "proposed",
        sources,
        modes,
        prior,
        target,
        critical,
        np.zeros(12, dtype=bool),
        _config(),
    )
    queried = result.trace.queried_indices
    assert len(queried) == 8
    assert len(np.unique(queried)) == 8
    assert set(result.support_indices).issubset(set(queried))


def test_risk_preserving_support_is_exactly_the_validated_a0_rule() -> None:
    sources = np.asarray([
        np.linspace(0.1, 0.9, 12),
        np.linspace(0.9, 0.1, 12),
        np.linspace(0.3, 0.7, 12),
    ])
    target = sources[1]
    support = adate_support(sources, target, budget=4)
    prior = LowRankPrior.fit(sources, rank=2)
    modes = np.asarray(["cutin"] * 4 + ["follow"] * 4 + ["brake"] * 4)
    result = functional_routed_mining(
        "proposed",
        sources,
        modes,
        prior,
        target,
        target > 0.7,
        np.zeros(12, dtype=bool),
        _config(),
        support_policy="adate",
    )
    assert np.array_equal(result.support_indices, support)


def test_coverage_support_preserves_risk_and_probes_every_mode() -> None:
    sources = np.asarray([
        np.linspace(0.1, 0.9, 12),
        np.linspace(0.9, 0.1, 12),
        np.linspace(0.3, 0.7, 12),
    ])
    modes = np.asarray(["cutin"] * 4 + ["follow"] * 4 + ["brake"] * 4)
    target = sources[1]
    support = coverage_adate_support(sources, modes, target, budget=4)
    assert len(np.unique(support)) == 4
    assert set(modes[support]) == {"cutin", "follow", "brake"}
