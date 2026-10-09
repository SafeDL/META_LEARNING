"""Two GP regression outputs updated by one continuous risk observation."""
import math

import numpy as np
import torch
from scipy.special import expit

from .config import BUDGET, REGULARIZATION_VARIANCE


class RiskTestingSession:

    def __init__(self,
                 x,
                 covariance,
                 risk_means,
                 collision_means,
                 weights,
                 calibration,
                 budget=BUDGET,
                 device="cuda",
                 mode="dual"):
        self.families = np.asarray(x)[:, 4].astype(int)
        self.covariance = torch.as_tensor(covariance,
                                          dtype=torch.float64,
                                          device=device).clone()
        means = np.column_stack(
            (risk_means @ weights, collision_means @ weights))
        self.mean = torch.as_tensor(means, dtype=torch.float64,
                                    device=device).clone()
        self.calibration = calibration
        self.budget = budget
        self.mode = mode
        self.count, self.pending = 0, None
        self.remaining = torch.ones(len(x), dtype=torch.bool, device=device)
        self.records = []

    def expected_scores(self):
        deviation = self.covariance.diag().clamp(min=1e-14).sqrt()[:, None]

        def positive(bound):
            z = (self.mean - bound) / deviation
            return ((self.mean - bound) * torch.special.ndtr(z) + deviation *
                    torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi))

        return positive(0.0) - positive(1.0)

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("Observe the pending query first")
        if self.count >= self.budget or not self.remaining.any():
            return None
        scores = self.expected_scores()
        alpha = 1 - self.count / self.budget
        if self.mode == "risk":
            alpha = 0.0
        elif self.mode == "calibrated":
            alpha = 1.0
        acquisition = ((1 - alpha) * scores[:, 0] +
                       alpha * scores[:, 1]).masked_fill(
                           ~self.remaining, -torch.inf)
        self.pending = int(acquisition.argmax())
        self.records.append({
            "index": self.pending,
            "query_number": self.count + 1,
            "calibrated_weight": alpha,
            "score": float(acquisition[self.pending]),
        })
        return self.pending

    def observe(self, risk):
        if self.pending is None:
            raise RuntimeError("Request a query before feedback")
        index = self.pending
        proxy = None
        if risk is not None:
            if not np.isfinite(risk) or not 0 <= risk <= 1:
                raise ValueError("Invalid continuous risk")
            options = self.calibration[self.families[index]]
            proxy = float(expit(options["slope"] * risk +
                                options["intercept"]))
            observation = self.mean.new_tensor([risk, proxy])
            residual = observation - self.mean[index].clone()
            cross = self.covariance[:, index].clone()
            denominator = self.covariance[index,
                                          index] + REGULARIZATION_VARIANCE
            self.mean += cross[:, None] / denominator * residual
            self.covariance -= cross[:, None] * cross[None, :] / denominator
        self.records[-1].update({
            "continuous_risk": risk,
            "calibrated_feedback": proxy
        })
        self.remaining[index] = False
        self.pending = None
        self.count += 1
