"""Generalized Bayesian trust from paid high-risk ranking outcomes."""
import numpy as np
from scipy.special import logsumexp


class RankingGate:
    def __init__(self):
        self.log_weight = np.full(2, -np.log(2))
        self.records = []

    @property
    def trust(self):
        return max(0., 2 * float(np.exp(self.log_weight[0])) - 1)

    def observe(self, index, high_risk, transfer, reference):
        opposite = [score for _, outcome, score in self.records if outcome != high_risk]
        current = np.asarray([
            transfer[0, index] - transfer[1, index],
            reference[0, index] - reference[1, index],
        ])
        before = self.trust
        scores = np.zeros(2)
        if opposite:
            difference = current[:, None] - np.asarray(opposite).T
            difference *= 1 if high_risk else -1
            scores = -np.logaddexp(0, -difference).mean(axis=1)
            self.log_weight += scores
            self.log_weight -= logsumexp(self.log_weight)
        self.records.append((int(index), bool(high_risk), current.copy()))
        return {
            "opposite_pairs": len(opposite), "ranking_log_score": scores.tolist(),
            "prequery_log_odds": current.tolist(),
            "trust_before": before, "trust_after": self.trust,
        }


def gated_log_probabilities(transfer, reference, trust):
    if trust == 0:
        return reference
    if trust == 1:
        return transfer
    return np.logaddexp(np.log(trust) + transfer, np.log1p(-trust) + reference)
