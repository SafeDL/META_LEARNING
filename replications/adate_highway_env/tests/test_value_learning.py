from __future__ import annotations

import numpy as np

from replications.adate_highway_env.adate.adaptive_policy import gap_ucb_scores, surrogate_gap
from replications.adate_highway_env.adate.dense_value import SparseExpectedTD


def test_expected_policy_td_uses_phi_not_max_and_masks_terminal_bootstrap():
    td = SparseExpectedTD(2, np.array([0.25, 0.75]), learning_rate=1.0, gamma=1.0)
    td.action_values((1, ))[...] = [1.0, 3.0]
    td.update((0, ), 0, reward=0.0, next_state=(1, ), terminal=False, critical=True)
    assert td.action_values((0, ))[0] == 2.5
    td.update((0, ), 1, reward=1.0, next_state=(1, ), terminal=True, critical=True)
    assert td.action_values((0, ))[1] == 1.0


def test_gap_bonus_matches_paper_divisor_position():
    scores = gap_ucb_scores(np.zeros(2),
                            np.zeros(2),
                            np.ones(2),
                            np.array([3.0, 0.0]),
                            exploration=2.0)
    assert np.allclose(scores, [np.sqrt(3) / 2, 2 * np.sqrt(3)])
    assert surrogate_gap(0.2, 0.0) > 1e5

