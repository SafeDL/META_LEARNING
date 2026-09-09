from __future__ import annotations

import pytest

from mvr.evaluation.diva_protocol import DivaBudgetLedger


def test_diva_budget_counts_censored_and_invalid_attempts() -> None:
    ledger = DivaBudgetLedger(total_budget=2)
    first = ledger.reserve(phase="support", design_id="a", episode_seed=11)
    second = ledger.reserve(phase="mining", design_id="b", episode_seed=12)
    ledger.complete(first, status="censored", score=0.0)
    ledger.complete(second, status="invalid", score=0.0)
    assert ledger.consumed == 2
    assert ledger.cumulative_score == 0.0
    with pytest.raises(RuntimeError):
        ledger.reserve(phase="mining", design_id="c", episode_seed=13)
