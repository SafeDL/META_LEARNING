from __future__ import annotations

import numpy as np

from method_chains.diva_detour_fusion.config import FusionExperimentConfig
from diva_highway_env.data.generate_anchor_bank import generate_multifunction_anchor_bank
from diva_highway_env.data.response_bank import ResponseBank
from method_chains.diva_detour_fusion.experiment import (
    METHOD_ORDER,
    mode_discovery_rows,
    run_loso_multifunction,
    summarize,
)


def test_multifunction_protocol_keeps_target_outcomes_query_bounded():
    anchors, modes = generate_multifunction_anchor_bank(10, seed=7)
    vulnerability = np.array([
        [0.1, 0.2, 0.9, 0.8, 0.3, 0.2, 0.7, 0.1, 0.8, 0.4],
        [0.2, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.2, 0.7, 0.3],
        [0.3, 0.2, 0.7, 0.9, 0.1, 0.4, 0.8, 0.1, 0.6, 0.2],
    ])
    collisions = vulnerability >= 0.8
    near_misses = (vulnerability >= 0.6) & ~collisions
    bank = ResponseBank(
        anchors=anchors,
        sut_names=("A", "B", "C"),
        vulnerability=vulnerability,
        collisions=collisions,
        near_misses=near_misses,
        min_ttc=np.ones_like(vulnerability),
        min_distance=np.ones_like(vulnerability),
        completed=~collisions,
        modes=modes,
    )
    config = FusionExperimentConfig(
        num_anchors=10, prior_rank=2, support_budget=2, total_budget=5,
        random_support_repeats=2, seed=7,
    )
    rows = run_loso_multifunction(bank, config)
    assert set(row["method"] for row in rows) == set(METHOD_ORDER)
    for row in rows:
        queried = [int(value) for value in str(row["queried_indices"]).split(";")]
        assert len(queried) == len(set(queried)) == config.total_budget
        assert 0.0 <= float(row["critical_recall_at_50"]) <= 1.0
    mode_rows = mode_discovery_rows(bank, rows)
    summary = summarize(bank, rows, mode_rows, config)
    assert summary["protocol"]["total_budget"] == 5
    assert summary["protocol"]["excluded_methods"] == ["RSS", "Shared-Prior", "Shared-Risk"]
