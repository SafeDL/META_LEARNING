from __future__ import annotations

import pytest

from metadrive_benchmark.mining.response import VulnerabilityResponseConfig, compute_vulnerability_response
from metadrive_benchmark.mining.types import CutInDesign, MiningObservation

CONFIG = VulnerabilityResponseConfig(5.0, 10.0, 0.74, 0.75, 0.75)


def test_response_preserves_oracle_event_separation_and_monotonicity() -> None:
    collision = compute_vulnerability_response({"valid_target_collision": True}, "valid_event",
                                               CONFIG)
    near_miss = compute_vulnerability_response({"valid_critical_near_miss": True}, "valid_event",
                                               CONFIG)
    noncritical = compute_vulnerability_response(
        {
            "challenge_min_ttc": 5.0,
            "challenge_min_distance": 10.0
        }, "completed_noncritical", CONFIG)
    tighter = compute_vulnerability_response(
        {
            "challenge_min_ttc": 1.0,
            "challenge_min_distance": 2.0
        }, "completed_noncritical", CONFIG)
    assert collision == 1.0
    assert near_miss >= 0.75
    assert 0.0 <= noncritical <= 0.74
    assert tighter >= noncritical
    assert compute_vulnerability_response({}, "invalid", CONFIG) is None


def test_observation_requires_response_exactly_when_eligible() -> None:
    common = dict(
        design=CutInDesign(0, (0.0, ) * 5),
        task_id="x",
        sut_ref="a",
        geometry_id="cutin-g01",
        logical_domain_id="mining_source_interaction_v2",
        episode_seed=1,
        score=0.0,
        is_valid_episode=True,
        outcome={},
        concrete_scenario={},
        behavior_contract_hash="x",
        elapsed_seconds=1.0,
    )
    with pytest.raises(ValueError):
        MiningObservation(status="completed_noncritical",
                        posterior_eligible=True,
                        vulnerability_response=None,
                        **common)
    with pytest.raises(ValueError):
        MiningObservation(status="censored",
                        posterior_eligible=False,
                        vulnerability_response=0.2,
                        **common)
    with pytest.raises(ValueError):
        MiningObservation(status="completed_noncritical",
                        posterior_eligible=True,
                        vulnerability_response=1.1,
                        **common)
