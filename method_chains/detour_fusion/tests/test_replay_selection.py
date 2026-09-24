from __future__ import annotations

import numpy as np

from highway_env_benchmark.data.generate_anchor_bank import generate_multifunction_anchor_bank
from highway_env_benchmark.data.response_bank import ResponseBank
from method_chains.detour_fusion.replay import (
    METHOD,
    select_replay_cases,
)


def test_replay_selection_uses_only_fusion_selected_cases_and_covers_every_mode():
    anchors, modes = generate_multifunction_anchor_bank(10, seed=3)
    vulnerability = np.array([
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ])
    bank = ResponseBank(
        anchors=anchors,
        sut_names=("A", "B"),
        vulnerability=vulnerability,
        collisions=vulnerability >= 0.8,
        near_misses=(vulnerability >= 0.6) & (vulnerability < 0.8),
        min_ttc=np.ones_like(vulnerability),
        min_distance=np.ones_like(vulnerability),
        completed=np.ones_like(vulnerability, dtype=bool),
        modes=modes,
    )
    rows = [
        {"method": METHOD, "target_sut": "A", "queried_indices": "0;2;4;6;8"},
        {"method": METHOD, "target_sut": "B", "queried_indices": "1;3;5;7;9"},
    ]
    cases = select_replay_cases(bank, rows)
    assert [case.mode for case in cases] == [
        "fast_intrusion", "cutin_braking", "lead_braking", "stop_and_go", "slow_lead_following",
    ]
    assert {case.anchor_index for case in cases} <= set(range(10))
    assert all(case.target_sut in {"A", "B"} for case in cases)
