from __future__ import annotations

import numpy as np
import pytest

from diva_metadrive.diva.source_bank import SourceBank
from diva_metadrive.diva.types import DivaCutInDesign
from diva_metadrive.scripts.evaluate_diva_cutin import _require_g1_pass, _source_logical_domain
from diva_metadrive.scripts.fit_diva_prior import _select_rank


def _bank() -> SourceBank:
    designs = tuple(
        DivaCutInDesign(candidate, (float(index) / 10.0, ) * 5) for candidate in (0, 1)
        for index in range(6))
    responses = np.asarray(
        [
            [0.1, 0.3, 0.8, 0.2, 0.5, 0.9] * 2,
            [0.4, 0.2, 0.7, 0.5, 0.1, 0.6] * 2,
            [0.9, 0.5, 0.2, 0.8, 0.4, 0.1] * 2,
            [0.2, 0.8, 0.6, 0.1, 0.9, 0.5] * 2,
        ],
        dtype=np.float64,
    )
    formal = np.where(responses >= 0.75, 0.5, 0.0)
    return SourceBank(("a", "b", "c", "d"), designs, formal, responses,
                      np.ones_like(responses, dtype=bool), ())


def test_rank_selection_reports_deterministic_loso_metrics() -> None:
    bank = _bank()
    rank, metrics = _select_rank(bank, (0, 1, 2), noise_var=0.02, seeds=(1701, 1702))
    reordered = SourceBank(bank.source_refs, tuple(reversed(bank.designs)),
                           bank.formal_scores[:, ::-1], bank.responses[:, ::-1],
                           bank.eligible[:, ::-1], ())
    reordered_rank, _ = _select_rank(reordered, (0, 1, 2), noise_var=0.02, seeds=(1701, 1702))
    assert rank in {0, 1, 2}
    assert reordered_rank in {0, 1, 2}
    assert set(metrics) == {"0", "1", "2"}
    assert all(np.isfinite(item["nll"]) and np.isfinite(item["rmse"]) for item in metrics.values())


def test_target_evaluation_domain_is_taken_from_frozen_source_bank() -> None:
    class Observation:
        logical_domain_id = "diva_source_interaction_v2"

    assert _source_logical_domain([Observation()]) == "diva_source_interaction_v2"
    with pytest.raises(ValueError, match="exactly one"):
        _source_logical_domain(
            [Observation(), type("Other", (), {"logical_domain_id": "other"})()])


def test_target_evaluation_requires_matching_passing_g1_report(tmp_path) -> None:
    prior_path = tmp_path / "prior_v2.pt"
    prior_path.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="G1 report is required"):
        _require_g1_pass(str(prior_path), "diva_source_interaction_v2")
    (tmp_path / "source_loso_g1_v2.json").write_text(
        '{"schema":"diva_source_loso_g1_v2","logical_domain_id":"diva_source_interaction_v2","g1_source_only_diagnostic":{"pass":true}}',
        encoding="utf-8",
    )
    _require_g1_pass(str(prior_path), "diva_source_interaction_v2")
