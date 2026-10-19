"""Risk-only execution, same-history reference and a posterior loss ledger."""

import numpy as np

from methods.baselines.cg_brt.posterior import RiskPosterior

from .config import BUDGET, INITIAL_QUERIES, LOSS_FRACTION, THRESHOLD
from .planning import choose
from methods.baselines.cg_brt.reference import ReferencePolicy


def select(
    x,
    covariance,
    priors,
    probability,
    fixed_prior,
    oracle,
    seed=11,
    *,
    proposal_covariance,
    budget=BUDGET,
    initial_queries=INITIAL_QUERIES,
    lookahead=True,
    protection=True,
    spatial=True,
    enabled=True,
    loss_fraction=LOSS_FRACTION,
    initial_indices=None,
    clock=None,
):
    if not 0 <= initial_queries <= budget <= len(x):
        raise ValueError(
            "Budget must fit the candidate pool and include initialization"
        )
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
    proposal = RiskPosterior(proposal_covariance, priors, probability, fixed_prior)
    remaining = np.ones(len(x), dtype=bool)
    loss_remaining = loss_fraction * budget
    queries, evidence, decisions = [], [], []
    if clock is not None:
        clock.start()
    for step in range(budget):
        reference_index, reference_decision = anchor.recommend(remaining, step)
        transfer, _ = proposal.log_probabilities()
        if step < initial_queries or not enabled:
            index, charged = reference_index, 0.0
            decision = {
                "reference_index": index,
                "proposed_index": index,
                "reason": (
                    "common_initialization"
                    if step < initial_queries
                    else "enhancement_disabled"
                ),
                "loss_charged": 0.0,
                "posterior_replacement_loss": 0.0,
                "loss_remaining_before": loss_remaining,
                "loss_remaining_after": loss_remaining,
            }
        else:
            index, charged, decision = choose(
                proposal,
                remaining,
                np.exp(anchor.gated[0]),
                reference_index,
                budget - step,
                loss_remaining,
                29011 + seed + step,
                lookahead=lookahead,
                protection=protection,
                spatial=spatial,
            )
        risk = float(oracle.query(index).risk)
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("The paid oracle must disclose a finite risk in [0,1]")
        high_risk = risk > THRESHOLD
        prequery_probability = float(np.exp(transfer[0, index]))
        anchor_evidence = anchor.observe(index, risk, high_risk)
        proposal.observe(index, risk)
        remaining[index] = False
        loss_remaining -= charged
        queries.append({"index": index, "risk": risk, "high_risk": high_risk})
        evidence.append(
            {
                "reference": anchor_evidence,
                "proposal": proposal.evidence[-1],
                "prequery_probability": prequery_probability,
            }
        )
        decisions.append(
            {
                "query": step + 1,
                "selected_index": index,
                "reference_decision": reference_decision,
                **decision,
            }
        )
        if clock is not None:
            clock.step()
    return {
        "selected_indices": [row["index"] for row in queries],
        "queries": queries,
        "evidence": evidence,
        "decisions": decisions,
        "loss_budget_initial": loss_fraction * budget,
        "loss_budget_remaining": loss_remaining,
        "feedback_used": "R only; H=1[R>0.5]; collision unavailable to selector",
    }
