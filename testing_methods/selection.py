"""Shared paid-initialization validation for fixed candidate pools."""


def validate_initial_indices(size, initial_indices, budget, *, minimum=1):
    if (
        not minimum <= len(initial_indices) <= budget <= size
        or len(set(initial_indices)) != len(initial_indices)
        or any(type(index) is not int or not 0 <= index < size for index in initial_indices)
    ):
        raise ValueError("Common initialization must be unique and inside the paid budget")
