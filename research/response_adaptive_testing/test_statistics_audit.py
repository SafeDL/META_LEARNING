"""Compare the half-sum randomization calculation against exhaustive signs."""
import itertools

import numpy as np
import pytest

from .audit_statistics import exact_p_meet_in_middle


@pytest.mark.parametrize("mode", ["zero", "positive", "mixed"])
def test_half_sum_exact_p_matches_explicit_sign_combinations(mode):
    values = np.random.default_rng(11).normal(size=(8, 3))
    if mode == "zero":
        values[:] = 0
    elif mode == "positive":
        values = np.abs(values) + 0.1
    signs = np.array(list(itertools.product((-1, 1), repeat=len(values))))
    null = np.abs(signs @ values / len(values))
    observed = np.abs(values.mean(0))
    expected = (null >= observed - 1e-12).mean(0)
    np.testing.assert_array_equal(exact_p_meet_in_middle(values), expected)
