from __future__ import annotations

import numpy as np
import pytest

from mvr.diva.source_bank import SourceBank
from mvr.diva.types import DivaCutInDesign
from mvr.scripts.evaluate_diva_cutin import _source_logical_domain
from mvr.scripts.fit_diva_prior import _select_rank


def _bank() -> SourceBank:
    designs = tuple(
        DivaCutInDesign(candidate, (float(index) / 10.0,) * 5)
        for candidate in (0, 1)
        for index in range(6)
    )
    scores = np.asarray(
        [
            [0.0, 0.5, 1.0, 0.0, 0.5, 1.0] * 2,
            [0.5, 0.0, 1.0, 0.5, 0.0, 1.0] * 2,
            [1.0, 0.5, 0.0, 1.0, 0.5, 0.0] * 2,
            [0.0, 1.0, 0.5, 0.0, 1.0, 0.5] * 2,
        ],
        dtype=np.float64,
    )
    return SourceBank(("a", "b", "c", "d"), designs, scores, np.ones_like(scores, dtype=bool), ())


def test_rank_selection_reports_loso_working_nll() -> None:
    rank, losses = _select_rank(_bank(), (1, 2), noise_var=0.02)

    assert rank in {1, 2}
    assert set(losses) == {"1", "2"}
    assert all(np.isfinite(value) for value in losses.values())


def test_target_evaluation_domain_is_taken_from_frozen_source_bank() -> None:
    class Observation:
        logical_domain_id = "cutin_tight_gap"

    assert _source_logical_domain([Observation()]) == "cutin_tight_gap"
    with pytest.raises(ValueError, match="exactly one"):
        _source_logical_domain([Observation(), type("Other", (), {"logical_domain_id": "cutin_late_fast"})()])
