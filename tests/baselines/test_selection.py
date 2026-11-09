"""Shared paid-initialization contracts."""

import pytest

from testing_methods.selection import validate_initial_indices


@pytest.mark.parametrize(
    "initial,budget",
    [([], 3), ([1, 1], 3), ([-1], 3), ([4], 3), ([True], 3), ([1.0], 3), ([0, 1], 1), ([0], 5)],
)
def test_invalid_initialization_is_rejected(initial, budget):
    with pytest.raises(ValueError, match="initialization"):
        validate_initial_indices(4, initial, budget)


def test_initial_queries_count_toward_the_budget():
    validate_initial_indices(4, [2, 0], 2)
    validate_initial_indices(4, [2, 0], 4)


@pytest.mark.parametrize("minimum", [2, 3])
def test_method_specific_initialization_minimum_is_preserved(minimum):
    with pytest.raises(ValueError, match="initialization"):
        validate_initial_indices(4, list(range(minimum - 1)), 4, minimum=minimum)
    validate_initial_indices(4, list(range(minimum)), 4, minimum=minimum)
