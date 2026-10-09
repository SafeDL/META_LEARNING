"""Compare bounded-memory exact inference against full sign enumeration."""
import itertools

import numpy as np

from .statistics import exact_sign_flip, holm_adjust


def test_half_sum_search_matches_full_enumeration_with_zeros_and_ties():
    differences = np.array([[0., 1., 0.], [1., 1., 0.], [-1., 1., 0.],
                            [2., 0., 0.], [1., 0., 0.], [0., 0., 0.],
                            [-0.25, 0., 0.], [0.5, 0., 0.]])
    signs = np.array(list(itertools.product((-1, 1), repeat=len(differences))))
    reference = (abs(signs @ differences)
                 >= abs(differences.sum(0)) - len(differences) * 1e-12).mean(0)
    np.testing.assert_array_equal(exact_sign_flip(differences), reference)
    np.testing.assert_allclose(holm_adjust(np.array([0.01, 0.03, 0.02, 0.4])),
                               [0.04, 0.06, 0.06, 0.4],
                               atol=1e-15)
