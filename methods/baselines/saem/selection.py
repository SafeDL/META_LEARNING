"""One paid feedback updates component evidence, residuals and scale."""
import numpy as np

from methods.baselines.hem.config import BUDGET, INITIAL_QUERIES
from methods.baselines.hgt.search import TestingSession

from .posterior import ScaleMixture


def select(x, covariance, priors, probability, fixed_prior, oracle, seed=11):
    initial = TestingSession(x, seed, mode="global_feedback", prior=fixed_prior)
    model = ScaleMixture(covariance, priors, probability)
    available = np.ones(len(x), dtype=bool)
    records = []
    for query in range(BUDGET):
        if query < INITIAL_QUERIES:
            index = initial.next_index()
        else:
            scores = model.acquisition()
            scores[~available] = -np.inf
            index = int(np.argmax(scores))
        observation = oracle.query(index)
        model.observe(index, observation.risk)
        if query < INITIAL_QUERIES:
            initial.observe(observation.risk)
        available[index] = False
        records.append({"index": index, "risk": float(observation.risk),
                        "collision": bool(observation.collision), "query_number": query + 1})
    return {"selected_indices": [r["index"] for r in records], "queries": records,
            "evidence": model.evidence, "updates_per_query": 1,
            "feedback_used": "risk only; collision recorded for evaluation"}
