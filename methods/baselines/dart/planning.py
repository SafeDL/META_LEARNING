"""Finite-budget discovery value and loss-constrained advice replacement."""

import numpy as np
import torch
from scipy.stats import gamma

from methods.baselines.cg_brt.ranking import (
    bivariate_normal_cdf,
)

from .config import CANDIDATES_PER_RULE, FANTASIES, VALUE_TOLERANCE
from .fantasy import fantasy_probabilities


def shortlist(remaining, probability, reference_probability, reference_index):
    indices = np.flatnonzero(remaining)
    uncertainty = probability * (1 - probability)
    scores = (probability, np.abs(probability - reference_probability), uncertainty)
    heads = [
        indices[np.argsort(-score[indices], kind="stable")[:CANDIDATES_PER_RULE]]
        for score in scores
    ]
    return np.unique(np.concatenate(([reference_index], *heads)))


def top_sum(probability, count):
    if count == 0:
        return np.zeros(probability.shape[:-1])
    if count >= probability.shape[-1]:
        return probability.sum(axis=-1)
    return np.partition(probability, probability.shape[-1] - count, axis=-1)[
        ..., -count:
    ].sum(-1)


def acquisition_values(
    model,
    candidates,
    remaining,
    horizon,
    generator,
    fantasies=FANTASIES,
    spatial=True,
    draws=None,
):
    probability = np.exp(model.log_probabilities()[0][0])
    if horizon == 1:
        return (
            probability[candidates],
            probability[candidates].copy(),
            np.zeros(len(candidates)),
        )
    if draws is None:
        draws = model.draws(candidates, fantasies, generator, 1.0)
    values, fixed_values, information = [], [], []
    available = np.flatnonzero(remaining)
    continuation_gpu = None
    if torch.cuda.is_available():
        from .acceleration import continuation_values

        continuation_gpu = continuation_values(
            model, candidates, draws, available, horizon - 1, spatial
        )
    for column, candidate in enumerate(candidates):
        future = available[available != candidate]
        fixed = float(top_sum(probability[future], horizon - 1))
        if continuation_gpu is None:
            conditional = fantasy_probabilities(
                model, candidate, draws[:, column], future, spatial
            )
            continuation = float(top_sum(conditional, horizon - 1).mean())
        else:
            continuation = float(continuation_gpu[column])
        values.append(probability[candidate] + continuation)
        fixed_values.append(probability[candidate] + fixed)
        information.append(continuation - fixed)
    return np.asarray(values), np.asarray(fixed_values), np.asarray(information)


def replacement_losses(model, candidates, reference_index):
    losses = np.zeros(len(candidates))
    positions = np.flatnonzero(candidates != reference_index)
    if not len(positions):
        return losses
    indices = np.r_[candidates[positions], reference_index]
    block = model.covariance_block(indices)
    sd = np.sqrt(np.diag(block))
    correlation = block[:-1, -1] / (sd[:-1] * sd[-1])
    precision = gamma.ppf(model.scale_quantiles, a=model.alpha)
    scales = np.sqrt(model.beta[:, None] / precision)
    za = (model.source_mean[indices[:-1]] - 0.5) / sd[:-1, None]
    zb = (model.source_mean[reference_index] - 0.5) / sd[-1]
    cdf = bivariate_normal_cdf(
        -za[..., None] / scales, zb[None, :, None] / scales, -correlation[:, None, None]
    )
    losses[positions] = (cdf @ model.scale_weights) @ model.weights
    return losses


def choose(
    model,
    remaining,
    reference_probability,
    reference_index,
    horizon,
    loss_remaining,
    seed,
    *,
    lookahead=True,
    protection=True,
    spatial=True
):
    probability = np.exp(model.log_probabilities()[0][0])
    candidates = shortlist(
        remaining, probability, reference_probability, reference_index
    )
    losses = replacement_losses(model, candidates, reference_index)
    feasible = (
        losses <= loss_remaining if protection else np.ones(len(candidates), dtype=bool)
    )
    reference_position = int(np.flatnonzero(candidates == reference_index)[0])
    feasible[reference_position] = True
    if lookahead:
        generator = np.random.default_rng(seed)
        draws = (
            model.draws(candidates, FANTASIES, generator, 1.0) if horizon > 1 else None
        )
        computed, fixed_computed, info_computed = acquisition_values(
            model,
            candidates[feasible],
            remaining,
            horizon,
            generator,
            spatial=spatial,
            draws=draws[:, feasible] if draws is not None else None,
        )
        values = np.full(len(candidates), -np.inf)
        fixed = np.zeros(len(candidates))
        information = np.zeros(len(candidates))
        values[feasible], fixed[feasible], information[feasible] = (
            computed,
            fixed_computed,
            info_computed,
        )
    else:
        values = probability[candidates]
        fixed, information = values.copy(), np.zeros(len(candidates))
    proposed_position = int(np.argmax(np.where(feasible, values, -np.inf)))
    best = int(np.argmax(np.where(feasible, values, -np.inf)))
    if values[best] <= values[reference_position] + VALUE_TOLERANCE:
        best = reference_position
    selected = int(candidates[best])
    charged = float(losses[best]) if selected != reference_index and protection else 0.0
    return (
        selected,
        charged,
        {
            "reference_index": int(reference_index),
            "proposed_index": int(candidates[proposed_position]),
            "candidate_indices": candidates.tolist(),
            "candidate_values": [float(v) if np.isfinite(v) else None for v in values],
            "value_evaluated": np.isfinite(values).tolist(),
            "candidate_losses": losses.tolist(),
            "candidate_feasible": feasible.tolist(),
            "reference_value": float(values[reference_position]),
            "selected_value": float(values[best]),
            "selected_fixed_value": float(fixed[best]),
            "information_value_mc": float(information[best]),
            "posterior_replacement_loss": float(losses[best]),
            "loss_charged": charged,
            "loss_remaining_before": float(loss_remaining),
            "loss_remaining_after": float(loss_remaining - charged),
            "rejected_candidates": int((~feasible).sum()),
            "horizon": int(horizon),
            "reason": (
                "enhanced_replacement"
                if selected != reference_index
                else "reference_retained"
            ),
        },
    )
