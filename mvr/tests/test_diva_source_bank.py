from __future__ import annotations

import numpy as np
import pytest

from mvr.diva.source_bank import SourceBank
from mvr.diva.types import DivaCutInDesign, DivaObservation


def _observation(source: str, design: DivaCutInDesign, score: float, eligible: bool) -> DivaObservation:
    status = "valid_event" if score else "completed_noncritical" if eligible else "censored"
    return DivaObservation(
        design=design,
        task_id=f"task-{source}",
        sut_ref=source,
        geometry_id="cutin-g01",
        logical_domain_id="cutin_interaction_core",
        episode_seed=11,
        score=score,
        is_valid_episode=True,
        status=status,
        posterior_eligible=eligible,
        outcome={},
        concrete_scenario={},
        behavior_contract_hash="hash",
        elapsed_seconds=1.0,
    )


def test_source_bank_keeps_censored_scores_out_of_the_response_matrix() -> None:
    designs = tuple(DivaCutInDesign(candidate, (0.0, 0.0, 0.0, 0.0, -0.1 + index * 0.1)) for candidate in (0, 1) for index in range(2))
    rows = [_observation(source, design, 0.5 if source == "a" else 0.0, not (source == "b" and design == designs[-1])) for source in ("a", "b") for design in designs]
    bank = SourceBank.from_observations(rows, ("a", "b"))
    censored_index = bank.design_ids.index(designs[-1].design_id)
    assert bank.scores.shape == (2, 4)
    assert not bank.eligible[1, censored_index]
    assert np.isnan(bank.scores[1, censored_index])
    assert bank.eligible.all(axis=0).sum() == 3
    with pytest.raises(ValueError, match="eligible zero and positive"):
        bank.require_response_boundary()
