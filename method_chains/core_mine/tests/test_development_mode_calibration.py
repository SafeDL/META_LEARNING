"""Mechanism checks for the frozen mode-only calibration development."""

import numpy as np

from method_chains.core_mine.development_mode_calibration import _candidate_scores


def test_calibration_preserves_source_order_within_each_mode() -> None:
    z = np.asarray([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    percentile = np.asarray([0.0, .5, 1.0] * 2)
    modes = np.asarray([0, 0, 0, 1, 1, 1])
    for method in ("MarginModeCal", "MarginModeRate"):
        scores = _candidate_scores(method, z, percentile, modes, [0, 3],
                                   [True, False])
        assert scores[0] < scores[1] < scores[2]
        assert scores[3] < scores[4] < scores[5]
        assert scores[1] > scores[4]


def test_calibration_uses_only_revealed_feedback() -> None:
    z = np.asarray([0.0, 0.0, 0.0, 0.0])
    percentile = np.asarray([.5] * 4)
    modes = np.asarray([0, 0, 1, 1])
    initial = _candidate_scores("MarginModeCal", z, percentile, modes, [], [])
    updated = _candidate_scores("MarginModeCal", z, percentile, modes,
                                [0, 2], [True, False])
    assert np.all(initial == .5)
    assert updated[1] > .5 > updated[3]
