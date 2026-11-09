"""Random fixed-pool baseline under the common paid-query budget."""

import time

import numpy as np
import torch

from experiments.schema import QueryFeedback
from testing_methods.selection import validate_initial_indices


def run(candidates, oracle, *, method="random", initial_indices, budget=128, seed=11):
    if method != "random":
        raise ValueError(f"Unknown collision baseline: {method}")
    size = len(candidates)
    initial = list(initial_indices)
    validate_initial_indices(size, initial, budget)
    if len({c.scenario_id for c in candidates}) != size:
        raise ValueError("Candidate identifiers must be unique")
    torch.set_num_threads(1)
    random_order = np.random.default_rng(seed).permutation(size).tolist()
    remaining = set(range(size))
    selected, queries = [], []
    for step in range(budget):
        started = time.perf_counter()
        index = (
            initial[step]
            if step < len(initial)
            else next(i for i in random_order if i in remaining)
        )
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if (
            not isinstance(feedback, QueryFeedback)
            or feedback.scenario_id != candidates[index].scenario_id
        ):
            raise ValueError("Baseline feedback does not match the paid query")
        selected.append(index)
        remaining.remove(index)
        queries.append(
            {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                "prequery_probability": None,
                "selection_s": selection_s,
                "adaptation_s": 0.0,
                "reason": "common_initialization" if step < len(initial) else "random",
            }
        )
    return {
        "method": method,
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "feedback_used": "paid collision labels",
    }
