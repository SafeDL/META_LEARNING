"""Risk-only selection; no collision labels or conditional-event models."""
import numpy as np

from methods.history_guided_testing.gp import clipped_risk_ei

from .config import BUDGET, INITIAL_QUERIES, THRESHOLD
from .decision import choose_candidate
from .gate import RankingGate, gated_log_probabilities
from .posterior import RiskPosterior


def select(x, covariance, priors, probability, fixed_prior, oracle, seed=11,
           *, budget=BUDGET, initial_queries=INITIAL_QUERIES):
    model = RiskPosterior(covariance, priors, probability, fixed_prior)
    gate = RankingGate()
    remaining = np.ones(len(x), dtype=bool)
    queries, evidence, decisions = [], [], []
    for step in range(budget):
        transfer, reference = model.log_probabilities()
        gated = gated_log_probabilities(transfer, reference, gate.trust)
        if step < initial_queries:
            score = clipped_risk_ei(model.mean, model.variance, 0.)
            score[~remaining] = -np.inf
            index = int(np.argmax(score))
            decision = {"reason": "common_initialization"}
        else:
            index, decision = choose_candidate(
                model, remaining, gated, transfer, gate.trust,
                budget - step, 19011 + seed + step, budget,
            )
        observation = oracle.query(index)
        risk = float(observation.risk)
        high_risk = risk > THRESHOLD
        gate_record = gate.observe(index, high_risk, transfer, reference)
        model.observe(index, risk)
        remaining[index] = False
        queries.append({"index": index, "risk": risk, "high_risk": high_risk})
        evidence.append({**model.evidence[-1], **gate_record})
        decisions.append({"query": step + 1, "selected_index": index, **decision})
    return {
        "selected_indices": [row["index"] for row in queries], "queries": queries,
        "evidence": evidence, "decisions": decisions,
        "feedback_used": "R only; H=1[R>0.5]; collision unavailable to selector",
    }
