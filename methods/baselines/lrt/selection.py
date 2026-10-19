"""One paid risk per step, robust continuation advantage and loss accounting."""

import numpy as np
import torch

from methods.baselines.dart.planning import shortlist
from methods.baselines.cg_brt.reference import ReferencePolicy
from methods.baselines.dart.selection import select as previous_select

from .config import (
    BUDGET,
    FANTASIES,
    INITIAL,
    LOSS_FRACTION,
    NOMINAL,
    VALUE_TOLERANCE,
    options,
)
from .posterior import model_set
from .value import make_features


def choose(
    models,
    remaining,
    anchor,
    reference_index,
    horizon,
    loss_remaining,
    step,
    seed,
    value_model,
    variant="full",
):
    reference_probability = np.exp(anchor.gated[0])
    candidates = shortlist(
        remaining,
        models[NOMINAL].probabilities(),
        reference_probability,
        reference_index,
    )
    losses = np.stack([m.losses(candidates, reference_index) for m in models])
    conservative_loss = losses[NOMINAL] if variant == "nominal_guard" else losses.max(0)
    protection = variant != "unprotected"
    feasible = (
        conservative_loss <= loss_remaining
        if protection
        else np.ones(len(candidates), bool)
    )
    reference_position = int(np.flatnonzero(candidates == reference_index)[0])
    if protection and loss_remaining <= 0:
        feasible[:] = False
    feasible[reference_position] = True
    values = np.full((len(models), len(candidates)), np.nan)
    if feasible.sum() > 1:
        # Draw before screening, so a budget change does not change feasible fantasies.
        draws = (
            [
                m.draws(candidates, FANTASIES, np.random.default_rng(seed))[:, feasible]
                for m in models
            ]
            if horizon > 1
            else None
        )
        if horizon == 1:
            values[:, feasible] = np.stack(
                [m.probabilities()[candidates[feasible]] for m in models]
            )
        else:
            features = make_features(
                models,
                remaining,
                candidates[feasible],
                reference_probability,
                anchor.gate.trust,
                horizon,
                step,
                losses[:, feasible],
                seed,
                draws=draws,
            )
            if variant == "no_value_learning":
                values[:, feasible] = features["lookahead"]
            else:
                with torch.no_grad():
                    values[:, feasible] = value_model(features).cpu().numpy()
        gains = np.full(len(candidates), -np.inf)
        gains[feasible] = (
            values[:, feasible] - values[:, reference_position, None]
        ).min(0)
        best = int(np.argmax(gains))
        if gains[best] <= VALUE_TOLERANCE:
            best = reference_position
    else:
        best = reference_position
        gains = np.full(len(candidates), -np.inf)
        gains[best] = 0.0
    index = int(candidates[best])
    charged = (
        float(conservative_loss[best])
        if protection and index != reference_index
        else 0.0
    )
    return (
        index,
        charged,
        {
            "reference_index": int(reference_index),
            "selected_index": index,
            "candidate_indices": candidates.tolist(),
            "candidate_values": [
                [None if not np.isfinite(v) else float(v) for v in row]
                for row in values
            ],
            "candidate_losses": losses.tolist(),
            "candidate_feasible": feasible.tolist(),
            "conservative_gain": float(gains[best]),
            "posterior_replacement_loss": float(conservative_loss[best]),
            "loss_charged": charged,
            "loss_remaining_before": float(loss_remaining),
            "loss_remaining_after": float(loss_remaining - charged),
            "rejected_candidates": int((~feasible).sum()),
            "horizon": horizon,
            "source_weights": [m.weights() for m in models],
            "reason": (
                "reference_retained"
                if index == reference_index
                else "selective_replacement"
            ),
        },
    )


def select(
    x,
    covariance,
    priors,
    probability,
    fixed_prior,
    oracle,
    seed=11,
    *,
    transfer_prior,
    value_model,
    budget=BUDGET,
    initial_queries=INITIAL,
    variant="full",
    enabled=True,
    loss_fraction=LOSS_FRACTION,
    on_state=None,
    initial_indices=None,
    clock=None,
    stopper=None,
):
    if not enabled:
        return previous_select(
            x,
            covariance,
            priors,
            probability,
            fixed_prior,
            oracle,
            seed,
            proposal_covariance=transfer_prior.enhanced_covariance,
            budget=budget,
            initial_queries=initial_queries,
            initial_indices=initial_indices,
            clock=clock,
        )
    if not 0 <= initial_queries <= budget <= len(x):
        raise ValueError("Budget must include initialization and fit the pool")
    if variant != "no_value_learning" and value_model is None:
        raise ValueError("Continuation selection requires a trained value network")
    models = model_set(transfer_prior, **options(variant))
    anchor = ReferencePolicy(
        covariance,
        priors,
        probability,
        fixed_prior,
        seed,
        budget,
        initial_queries,
        initial_indices,
    )
    remaining = np.ones(len(x), bool)
    loss_remaining = loss_fraction * budget
    queries, evidence, decisions = [], [], []
    stopping = []
    if clock is not None:
        clock.start()
    for step in range(budget):
        reference_index, reference_decision = anchor.recommend(remaining, step)
        if on_state is not None:
            on_state(step, models, anchor, remaining, reference_index)
        if stopper is not None and step >= initial_queries:
            check = stopper.evaluate(
                models, remaining, anchor, queries, evidence, budget
            )
            stopping.append(check)
            if check["stop"]:
                if clock is not None:
                    clock.finish_decision()
                break
        if step < initial_queries:
            index, charged = reference_index, 0.0
            decision = {
                "reference_index": index,
                "selected_index": index,
                "reason": "common_initialization",
                "loss_charged": 0.0,
                "posterior_replacement_loss": 0.0,
                "loss_remaining_before": loss_remaining,
                "loss_remaining_after": loss_remaining,
            }
        else:
            index, charged, decision = choose(
                models,
                remaining,
                anchor,
                reference_index,
                budget - step,
                loss_remaining,
                step,
                29011 + seed + step,
                value_model,
                variant,
            )
        prequery = float(models[NOMINAL].probabilities()[index])
        risk = float(oracle.query(index).risk)
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Paid risk must be finite and in [0,1]")
        anchor_evidence = anchor.observe(index, risk, risk > 0.5)
        for model in models:
            model.observe(index, risk)
        remaining[index] = False
        loss_remaining -= charged
        queries.append({"index": index, "risk": risk, "high_risk": risk > 0.5})
        evidence.append(
            {"reference": anchor_evidence, "prequery_probability": prequery}
        )
        decisions.append(
            {"query": step + 1, "reference_decision": reference_decision, **decision}
        )
        if clock is not None:
            clock.step()
    result = {
        "selected_indices": [q["index"] for q in queries],
        "queries": queries,
        "evidence": evidence,
        "decisions": decisions,
        "loss_budget_initial": loss_fraction * budget,
        "loss_budget_remaining": loss_remaining,
        "variant": variant,
        "source_names": [
            *transfer_prior.names,
            *(["target_only"] if models[0].target_probability else []),
        ],
        "model_interpretations": [
            {"kappa": model.kappa, "target_prior_probability": model.target_probability}
            for model in models
        ],
        "feedback_used": "R only; H=1[R>0.5]; collision unavailable to selector",
    }
    if stopper is not None:
        result.update(
            actual_queries=len(queries),
            maximum_queries=budget,
            stop_reason=(
                "quality_retained" if len(queries) < budget else "budget_exhausted"
            ),
            stopping=stopping,
        )
    return result
