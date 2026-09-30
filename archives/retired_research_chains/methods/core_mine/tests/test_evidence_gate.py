from types import SimpleNamespace

import numpy as np

from methods.core_mine.evidence_gate import (
    gated_scores, log_mode_heterogeneity_bayes_factor,
    static_mode_quantile_scores,
)
from methods.core_mine.simple_residual_ablation import corrected_scores


def _task():
    return SimpleNamespace(
        count=20,
        modes=np.asarray(["a"] * 10 + ["b"] * 10),
        source_y=np.asarray([np.linspace(.01, .2, 20)]),
        features=np.zeros((20, 2)),
    )


def test_bayes_factor_uses_only_revealed_event_modes():
    modes = np.asarray(["a", "a", "b", "b"])
    assert log_mode_heterogeneity_bayes_factor(modes, [], []) == 0.0
    assert log_mode_heterogeneity_bayes_factor(
        modes, [0, 1, 2, 3], [True, True, False, False]) > 0
    assert log_mode_heterogeneity_bayes_factor(
        modes, [0, 1, 2, 3], [True, True, True, True]) < 0


def test_gate_uses_static_prefix_and_can_switch():
    task = _task()
    selected = list(range(5)) + list(range(10, 15))
    events = [True] * 5 + [False] * 5
    responses = [1.1] * 5 + [.1] * 5
    scores, expert, log_bf = gated_scores(
        task, selected[:9], responses[:9], events[:9])
    assert expert == "static"
    assert np.allclose(scores, static_mode_quantile_scores(task))
    scores, expert, log_bf = gated_scores(task, selected, responses, events)
    assert expert == "adaptive" and log_bf > 0
    assert np.allclose(scores, corrected_scores(
        task, selected, responses, "HistoryMargin-ModeShift"))


def test_static_percentiles_rank_only_source_safe_eligible_candidates():
    task = SimpleNamespace(
        count=4, modes=np.asarray(["a"] * 4),
        source_y=np.asarray([[.01, .10, .20, .30]]),
    )
    scores = static_mode_quantile_scores(
        task, np.asarray([0, 1, 3], dtype=int))
    assert np.allclose(scores[[0, 1, 3]], [0.0, .5, 1.0])
    assert np.isneginf(scores[2])
