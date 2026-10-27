"""Shared initialization and deterministic greedy selection for fixed pools."""


def validate_initial_indices(size, initial_indices, budget, *, minimum=1):
    if (
        not minimum <= len(initial_indices) <= budget <= size
        or len(set(initial_indices)) != len(initial_indices)
        or any(type(index) is not int or not 0 <= index < size for index in initial_indices)
    ):
        raise ValueError("Common initialization must be unique and inside the paid budget")


def greedy_index(scores, available, candidates):
    """Resolve scores within 1e-12 of the maximum by public scenario ID."""
    maximum = max(scores[index] for index in available)
    return min(
        (index for index in available if scores[index] >= maximum - 1e-12),
        key=lambda index: candidates[index].scenario_id,
    )
