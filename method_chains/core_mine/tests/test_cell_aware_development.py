from types import SimpleNamespace

import numpy as np

from method_chains.core_mine.cell_aware_development import (
    PENALTY, cell_aware_scores,
)
from method_chains.core_mine.simple_residual_ablation import corrected_scores


def _task():
    return SimpleNamespace(
        count=4,
        modes=np.asarray(["fast_intrusion"] * 4),
        anchors=np.asarray([[6.5, -7.0], [8.0, -6.0],
                            [12.0, -7.0], [16.0, -7.0]]),
        features=np.zeros((4, 2)),
        source_y=np.asarray([[.2, .3, .4, .5]]),
    )


def test_only_observed_hazard_cell_receives_fixed_penalty():
    task = _task()
    base = corrected_scores(task, [0], [1.1],
                            "HistoryMargin-ModeShift")
    scores = cell_aware_scores(task, [0], [1.1], [1.0])
    assert np.allclose(scores, base - np.asarray([PENALTY, PENALTY, 0, 0]))
    assert np.allclose(cell_aware_scores(task, [0], [.1], [0.0]),
                       corrected_scores(task, [0], [.1],
                                        "HistoryMargin-ModeShift"))


def test_feedback_ledgers_must_align():
    try:
        cell_aware_scores(_task(), [0], [], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned target feedback was accepted")
