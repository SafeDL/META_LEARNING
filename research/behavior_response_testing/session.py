"""Infer a behavioral hypothesis from risk using a shared discrepancy GP."""
import math

import numpy as np
import torch

from methods.history_guided_testing.kernel import matern52

BUDGET = 200
DISCREPANCY_LENGTH = 0.2


def rmse_discrepancy(variance):
    return {
        "gp_variance": [0.5 * value for value in variance],
        "noise_variance": [0.5 * value for value in variance],
        "length": [DISCREPANCY_LENGTH, DISCREPANCY_LENGTH]
    }


class BehaviorTestingSession:

    def __init__(self,
                 x,
                 risk_predictions,
                 collision_predictions,
                 discrepancy,
                 budget=BUDGET,
                 device="cuda"):
        self.risk_mean = torch.as_tensor(risk_predictions,
                                         dtype=torch.float64,
                                         device=device).clone()
        self.collision = torch.as_tensor(collision_predictions,
                                         dtype=torch.float64,
                                         device=device)
        x = torch.as_tensor(x, dtype=torch.float64, device=device)
        family = x[:, 4].long()
        amplitude = x.new_tensor(discrepancy["gp_variance"])[family].sqrt()
        length = x.new_tensor(discrepancy["length"])[family]
        self.rr = amplitude[:, None] * amplitude[None, :] * matern52(
            torch.cdist(x[:, :4], x[:, :4]),
            length[:, None]) * (family[:, None] == family[None, :])
        self.noise = x.new_tensor(discrepancy["noise_variance"])[family]
        self.log_weights = x.new_full((len(risk_predictions), ),
                                      -math.log(len(risk_predictions)))
        self.remaining = torch.ones(len(x), dtype=torch.bool, device=device)
        self.budget, self.count, self.pending = budget, 0, None
        self.records = []

    def probabilities(self):
        return self.log_weights.softmax(0) @ self.collision

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("Observe the pending query first")
        if self.count == self.budget:
            return None
        q = self.probabilities()
        self.pending = int(q.masked_fill(~self.remaining, -torch.inf).argmax())
        self.records.append({
            "index": self.pending,
            "query_number": self.count + 1,
            "collision_probability": float(q[self.pending])
        })
        return self.pending

    def observe(self, risk):
        if self.pending is None:
            raise RuntimeError("Request a query before feedback")
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid continuous risk")
        index = self.pending
        denominator = self.rr[index, index] + self.noise[index]
        residual = float(risk) - self.risk_mean[:, index].clone()
        self.log_weights += -0.5 * (residual.square() / denominator +
                                    torch.log(2 * math.pi * denominator))
        self.log_weights -= torch.logsumexp(self.log_weights, 0)
        cross = self.rr[:, index].clone()
        self.risk_mean += residual[:, None] * cross[None, :] / denominator
        self.rr -= cross[:, None] * cross[None, :] / denominator
        weights = self.log_weights.exp()
        self.records[-1].update({
            "continuous_risk":
            float(risk),
            "posterior_entropy":
            float(-(weights * self.log_weights).sum()),
            "largest_hypothesis_mass":
            float(weights.max())
        })
        self.remaining[index] = False
        self.count += 1
        self.pending = None
