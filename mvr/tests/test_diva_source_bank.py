from __future__ import annotations

import numpy as np
import pytest

from mvr.diva.source_bank import SourceBank, observation_from_dict
from mvr.diva.types import DIVA_SCHEMA, DivaCutInDesign, DivaObservation


def _observation(
    source: str,
    design: DivaCutInDesign,
    score: float,
    response: float | None,
    eligible: bool,
) -> DivaObservation:
    status = "valid_event" if score else "completed_noncritical" if eligible else "censored"
    return DivaObservation(
        design=design,
        task_id=f"task-{source}",
        sut_ref=source,
        geometry_id="cutin-g01",
        logical_domain_id="diva_source_interaction_v2",
        episode_seed=11,
        score=score,
        is_valid_episode=True,
        status=status,
        posterior_eligible=eligible,
        vulnerability_response=response,
        outcome={},
        concrete_scenario={},
        behavior_contract_hash="hash",
        elapsed_seconds=1.0,
    )


def test_source_bank_keeps_formal_scores_separate_from_responses() -> None:
    designs = tuple(
        DivaCutInDesign(candidate, (0.0, 0.0, 0.0, 0.0, -0.1 + index * 0.1))
        for candidate in (0, 1)
        for index in range(2)
    )
    rows = [
        _observation(
            source,
            design,
            0.5 if source == "a" else 0.0,
            (0.8 if source == "a" else 0.2)
            if not (source == "b" and design == designs[-1]) else None,
            not (source == "b" and design == designs[-1]),
        )
        for source in ("a", "b")
        for design in designs
    ]
    bank = SourceBank.from_observations(rows, ("a", "b"))
    censored_index = bank.design_ids.index(designs[-1].design_id)
    assert bank.formal_scores.shape == bank.responses.shape == (2, 4)
    assert bank.formal_scores[0, 0] != bank.responses[0, 0]
    assert not bank.eligible[1, censored_index]
    assert np.isnan(bank.responses[1, censored_index])
    assert bank.eligible.all(axis=0).sum() == 3
    assert bank.source_summary()["formal_event_rate"] == pytest.approx(0.5)


def test_v2_reader_rejects_old_observation_schema() -> None:
    design = DivaCutInDesign(0, (0.0,) * 5)
    row = _observation("a", design, 0.0, 0.2, True).to_dict()
    row["schema"] = "diva_mine_cutin_constant_speed_physical"
    with pytest.raises(ValueError, match="incompatible observation schema"):
        observation_from_dict(row)
    row["schema"] = DIVA_SCHEMA
    assert observation_from_dict(row).vulnerability_response == pytest.approx(0.2)
