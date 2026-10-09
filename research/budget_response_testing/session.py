"""Greedy failure selection with one disclosed risk per physical query."""
import numpy as np
import torch

from .model import joint_prior, risk_coordinate

BUDGET = 200


class BudgetTestingSession:

    def __init__(self,
                 x,
                 risks,
                 collisions,
                 state,
                 budget=BUDGET,
                 device="cuda"):
        (self.mean_r, self.mean_c, self.rr, self.cr, self.cc,
         self.source_r) = joint_prior(x, risks, collisions, state, device)
        self.coefficients = self.mean_r.new_zeros(self.source_r.shape[0])
        self.noise = state["options"]["noise"]
        self.remaining = torch.ones(len(x), dtype=torch.bool, device=device)
        self.budget, self.count, self.pending = budget, 0, None
        self.records = []

    def probabilities(self):
        return torch.special.ndtr(self.mean_c /
                                  (1 + self.cc.clamp(min=0)).sqrt())

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("Observe the pending query first")
        if self.count == self.budget:
            return None
        probabilities = self.probabilities()
        self.pending = int(
            probabilities.masked_fill(~self.remaining, -torch.inf).argmax())
        self.records.append({
            "index":
            self.pending,
            "query_number":
            self.count + 1,
            "collision_probability":
            float(probabilities[self.pending])
        })
        return self.pending

    def observe(self, risk):
        if self.pending is None:
            raise RuntimeError("Request a query before feedback")
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid continuous risk")
        index = self.pending
        residual = float(risk_coordinate(risk)) - self.mean_r[index].clone()
        denominator = self.rr[index, index] + self.noise
        risk_cross = self.rr[:, index].clone()
        collision_cross = self.cr[:, index].clone()
        source_cross = self.source_r[:, index].clone()
        self.mean_r += risk_cross / denominator * residual
        self.mean_c += collision_cross / denominator * residual
        self.coefficients += source_cross / denominator * residual
        self.rr -= risk_cross[:, None] * risk_cross[None, :] / denominator
        self.cr -= collision_cross[:, None] * risk_cross[None, :] / denominator
        self.cc -= collision_cross.square() / denominator
        self.source_r -= source_cross[:,
                                      None] * risk_cross[None, :] / denominator
        self.records[-1].update({
            "continuous_risk":
            float(risk),
            "response_coefficients":
            self.coefficients.tolist()
        })
        self.remaining[index] = False
        self.count += 1
        self.pending = None
