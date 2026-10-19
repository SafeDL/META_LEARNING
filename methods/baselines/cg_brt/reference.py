"""Original CG-BRT advice, updated exclusively with the executed queries."""

import numpy as np

from methods.baselines.cg_brt.decision import choose_candidate
from methods.baselines.cg_brt.gate import (
    RankingGate,
    gated_log_probabilities,
)
from methods.baselines.cg_brt.posterior import RiskPosterior
from methods.baselines.hgt.gp import clipped_risk_ei
from benchmark.initialization import validate_initial


class ReferencePolicy:
    def __init__(
        self,
        covariance,
        priors,
        probability,
        fixed_prior,
        seed,
        budget,
        initial_queries,
        initial_indices=None,
    ):
        validate_initial(initial_indices, len(covariance), initial_queries)
        self.initial_indices = initial_indices
        self.model = RiskPosterior(covariance, priors, probability, fixed_prior)
        self.gate = RankingGate()
        self.seed, self.budget, self.initial_queries = seed, budget, initial_queries

    def recommend(self, remaining, step):
        self.transfer, self.reference = self.model.log_probabilities()
        self.gated = gated_log_probabilities(
            self.transfer, self.reference, self.gate.trust
        )
        if step < self.initial_queries:
            if self.initial_indices is not None:
                return int(self.initial_indices[step]), {
                    "reason": "common_initialization"
                }
            score = clipped_risk_ei(self.model.mean, self.model.variance, 0.0)
            score[~remaining] = -np.inf
            return int(np.argmax(score)), {"reason": "common_initialization"}
        return choose_candidate(
            self.model,
            remaining,
            self.gated,
            self.transfer,
            self.gate.trust,
            self.budget - step,
            19011 + self.seed + step,
            self.budget,
        )

    def observe(self, index, risk, high_risk):
        record = self.gate.observe(index, high_risk, self.transfer, self.reference)
        self.model.observe(index, risk)
        return {**self.model.evidence[-1], **record}
