from types import SimpleNamespace

import numpy as np

from method_chains.core_mine.simple_residual_ablation import corrected_scores


def _task():
    return SimpleNamespace(
        source_y=np.asarray([[0.1, 0.2, 0.1, 0.2],
                             [0.3, 0.4, 0.3, 0.4]]),
        modes=np.asarray(["a", "a", "b", "b"]),
        features=np.asarray([[0.0], [0.1], [0.0], [0.1]]),
    )


def test_simple_residuals_use_only_revealed_cases_within_mode():
    task = _task()
    initial = task.source_y.mean(axis=0)
    for method in ("HistoryMargin-ModeShift", "HistoryMargin-KernelShift"):
        np.testing.assert_allclose(corrected_scores(task, [], [], method), initial)
        changed = corrected_scores(task, [0], [0.8], method)
        assert changed[0] > initial[0]
        assert changed[1] > initial[1]
        np.testing.assert_allclose(changed[2:], initial[2:])


def test_kernel_shift_is_local_and_shrunk():
    task = _task()
    shifted = corrected_scores(task, [0], [0.8], "HistoryMargin-KernelShift")
    source = task.source_y.mean(axis=0)
    assert 0 < shifted[1] - source[1] < shifted[0] - source[0] < 0.6
