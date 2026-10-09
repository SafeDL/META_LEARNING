"""One risk observation updates the joint risk and collision-score posterior."""
import math

import numpy as np
import torch

from .config import BUDGET, QUADRATURE, SHORTLIST
from .model import clipped_expectation, joint_prior, logistic_expectation, risk_coordinate


class AdaptiveTestingSession:

    def __init__(self,
                 x,
                 risks,
                 collisions,
                 options,
                 budget=BUDGET,
                 device="cuda"):
        full_collision = options.get("ranking_information", False)
        (self.mean_r, self.mean_c, self.rr, self.cr, cc,
         self.coefficient_r) = joint_prior(x,
                                           risks,
                                           collisions,
                                           options,
                                           device,
                                           full_collision=full_collision)
        self.cc_matrix = cc if full_collision else None
        self.cc = cc.diag().clone() if full_collision else cc
        self.coefficients = self.mean_r.new_zeros(self.coefficient_r.shape[0])
        self.options, self.budget = options, budget
        self.remaining = torch.ones(len(x), dtype=torch.bool, device=device)
        self.count, self.pending = 0, None
        self.records = []
        nodes, masses = np.polynomial.hermite.hermgauss(QUADRATURE)
        self.nodes = self.mean_r.new_tensor(nodes * math.sqrt(2))
        self.masses = self.mean_r.new_tensor(masses / math.sqrt(math.pi))

    def probabilities(self):
        return self.readout(self.mean_c, self.cc)

    def readout(self, mean, variance):
        if self.options.get("collision_transform", "raw") == "probit":
            return torch.special.ndtr(mean /
                                      (1 + variance.clamp(min=0)).sqrt())
        if self.options.get("collision_transform", "raw") == "logit":
            return logistic_expectation(mean, variance)
        return clipped_expectation(mean, variance)

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("Observe the pending query first")
        if self.count == self.budget:
            return None
        q = self.probabilities()
        score = q.masked_fill(~self.remaining, -torch.inf)
        index = int(score.argmax())
        bonus = 0.0
        if self.options.get("lookahead",
                            False) and self.count < self.budget - 1:
            size = min(SHORTLIST // 2, int(self.remaining.sum()))
            uncertain = (self.ranking_information(q)
                         if self.cc_matrix is not None else self.cc.sqrt())
            uncertain = uncertain.masked_fill(~self.remaining, -torch.inf)
            proposals = torch.unique(
                torch.cat(
                    (score.topk(size).indices, uncertain.topk(size).indices)))
            values, bonuses = self.future_values(proposals, q)
            best = int(values.argmax())
            index, bonus = int(proposals[best]), float(bonuses[best])
        self.pending = index
        self.records.append({
            "index": index,
            "query_number": self.count + 1,
            "collision_score": float(q[index]),
            "decision_value": bonus
        })
        return index

    def ranking_information(self, q):
        count = min(self.budget - self.count, int(self.remaining.sum()) - 1)
        order = q.masked_fill(~self.remaining,
                              -torch.inf).topk(count + 1).indices
        first, second = order[:-1], order[1:]
        variance = (self.cc[first] + self.cc[second] -
                    2 * self.cc_matrix[first, second]).clamp(min=1e-12)
        margin = self.mean_c[first] - self.mean_c[second]
        ambiguity = torch.exp(-0.5 * margin.square() / variance) / variance
        weights = q.new_full((count, ), 0.5 / self.budget)
        weights[-1] = 0.5 + 0.5 * (self.budget - self.count -
                                   count) / self.budget
        cross_difference = self.cr[first] - self.cr[second]
        denominator = self.rr.diag() + self.options["noise"]
        return (cross_difference.square() *
                (weights * ambiguity)[:, None]).sum(0) / denominator

    def future_values(self, proposals, q):
        remaining_budget = self.budget - self.count
        horizon = min(remaining_budget - 1,
                      self.options.get("horizon", remaining_budget - 1))
        order = torch.arange(1, horizon + 1, device=q.device, dtype=q.dtype)
        # Equal endpoint and discovery-curve contributions; no late-query discount to zero.
        future_weights = 0.5 + 0.5 * (remaining_budget - order) / self.budget
        immediate_weight = 0.5 + 0.5 * remaining_budget / self.budget
        values, bonuses = [], []
        for index in proposals:
            denominator = self.rr[index, index] + self.options["noise"]
            cross = self.cr[:, index]
            future_mean = (
                self.mean_c[None, :] +
                self.nodes[:, None] * cross[None, :] / denominator.sqrt())
            future_variance = (self.cc -
                               cross.square() / denominator).clamp(min=0)
            future_q = self.readout(future_mean, future_variance[None, :])
            future_q[:, ~self.remaining] = -torch.inf
            future_q[:, index] = -torch.inf
            values_future = (future_q.topk(horizon, dim=1).values *
                             future_weights).sum(1)
            baseline = q.masked_fill(~self.remaining, -torch.inf).clone()
            baseline[index] = -torch.inf
            committed = (baseline.topk(horizon).values * future_weights).sum()
            value_future = values_future @ self.masses
            values.append(immediate_weight * q[index] + value_future)
            bonuses.append(value_future - committed)
        return torch.stack(values), torch.stack(bonuses)

    def observe(self, risk):
        if self.pending is None:
            raise RuntimeError("Request a query before feedback")
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid continuous risk")
        index = self.pending
        value = float(risk_coordinate(risk, self.options["transform"]))
        residual = value - self.mean_r[index].clone()
        denominator = self.rr[index, index] + self.options["noise"]
        cross_r, cross_c = self.rr[:, index].clone(), self.cr[:, index].clone()
        cross_coefficient = self.coefficient_r[:, index].clone()
        self.mean_r += cross_r / denominator * residual
        self.mean_c += cross_c / denominator * residual
        self.coefficients += cross_coefficient / denominator * residual
        self.rr -= cross_r[:, None] * cross_r[None, :] / denominator
        self.cr -= cross_c[:, None] * cross_r[None, :] / denominator
        self.cc -= cross_c.square() / denominator
        if self.cc_matrix is not None:
            self.cc_matrix -= cross_c[:, None] * cross_c[None, :] / denominator
        self.coefficient_r -= cross_coefficient[:, None] * cross_r[
            None, :] / denominator
        self.records[-1].update({
            "continuous_risk":
            float(risk),
            "response_coefficients":
            self.coefficients.tolist()
        })
        self.remaining[index] = False
        self.count += 1
        self.pending = None
