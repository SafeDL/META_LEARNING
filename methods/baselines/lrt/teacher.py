"""Historical-only forced-action branches followed by unchanged CG-BRT."""

import copy

import numpy as np

from methods.baselines.cg_brt.reference import ReferencePolicy


def replay_reference(common, queries, budget, initial_queries):
    reference = ReferencePolicy(
        common["covariance"],
        common["priors"],
        common["probability"],
        common["fixed_prior"],
        common["seed"],
        budget,
        initial_queries,
        common.get("initial_indices"),
    )
    remaining = np.ones(len(common["x"]), bool)
    for step, observation in enumerate(queries):
        reference.recommend(remaining, step)
        reference.observe(
            observation["index"], observation["risk"], observation["risk"] > 0.5
        )
        remaining[observation["index"]] = False
    return reference, remaining


def branch_return(reference, remaining, candidate, risk):
    """Risk is supplied only by the historical/evaluation teacher, never selection."""
    anchor = copy.deepcopy(
        reference, {id(reference.model.covariance): reference.model.covariance}
    )
    available = remaining.copy()
    start = len(anchor.model.indices)
    if not available[candidate]:
        raise ValueError("A branch action must be unqueried")
    selected = []
    for step in range(start, anchor.budget):
        advised, _ = anchor.recommend(available, step)
        index = candidate if step == start else advised
        value = float(risk[index])
        anchor.observe(index, value, value > 0.5)
        available[index] = False
        selected.append(int(index))
    return int(np.count_nonzero(risk[selected] > 0.5)), selected


def training_candidates(
    models, remaining, reference_probability, reference_index, seed
):
    indices = np.flatnonzero(remaining)
    probability = models[1].probabilities()
    generator = np.random.default_rng(seed)
    scores = (
        probability,
        abs(probability - reference_probability),
        probability * (1 - probability),
    )
    heads = [
        indices[np.argsort(-score[indices], kind="stable")[:2]] for score in scores
    ]
    return np.unique(
        np.concatenate(
            (
                [reference_index],
                *heads,
                generator.choice(indices, min(2, len(indices)), replace=False),
            )
        )
    )
