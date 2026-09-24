"""Source-only ranking contract for the frozen physical replication."""

from types import SimpleNamespace

import numpy as np

from method_chains.core_mine.heterogeneous20_replication import _mode_quantile


def test_mode_quantile_uses_only_eligible_within_mode_ranks() -> None:
    task = SimpleNamespace(
        source_y=np.asarray([[.1, .4, .9, .3, .5, .8],
                             [.1, .4, .9, .3, .5, .8]]),
        modes=np.asarray(["a", "a", "a", "b", "b", "b"]),
        count=6,
    )
    scores = _mode_quantile(task, np.asarray([0, 1, 3, 4]))
    np.testing.assert_array_equal(scores[[0, 1, 3, 4]], [0, 1, 0, 1])
    assert np.isneginf(scores[[2, 5]]).all()
