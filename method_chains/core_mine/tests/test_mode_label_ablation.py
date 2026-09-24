from types import SimpleNamespace

import numpy as np

from method_chains.core_mine.mode_label_ablation import label_shift_scores
from method_chains.core_mine.simple_residual_ablation import corrected_scores


def test_label_shift_changes_only_mode_offsets():
    task = SimpleNamespace(
        source_y=np.asarray([[0.1, 0.2, 0.3, 0.4],
                             [0.2, 0.3, 0.4, 0.5]]),
        modes=np.asarray(["a", "a", "b", "b"]),
    )
    baseline = task.source_y.mean(axis=0)
    np.testing.assert_allclose(label_shift_scores(task, [], []), baseline)
    scores = label_shift_scores(task, [0, 2], [1.0, 0.5])
    np.testing.assert_allclose(scores[:2] - baseline[:2], [0.85, 0.85])
    np.testing.assert_allclose(scores[2:] - baseline[2:], [0.15, 0.15])
    np.testing.assert_allclose(
        scores,
        corrected_scores(task, [0, 2], [1.0, 0.5],
                         "HistoryMargin-ModeShift"),
    )
