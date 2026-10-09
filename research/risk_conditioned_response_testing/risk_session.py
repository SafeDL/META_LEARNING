"""Select collision tests using risk-residual-conditioned predictions."""
import math

import torch

from research.behavior_response_testing.session import (
    BUDGET, BehaviorTestingSession)


class RiskConditionedTestingSession(BehaviorTestingSession):
    """Update behavior from risk, then decode its residual into collision risk.

    The collision decoder is fitted offline on revealed development responses.
    At test time, the selector sees only the conditional-model predictions,
    scenario coordinates and one queried continuous-risk value at a time.
    """

    def __init__(self,
                 x,
                 risk_predictions,
                 collision_logits,
                 discrepancy,
                 calibrators,
                 collision_only=False,
                 budget=BUDGET,
                 device="cuda"):
        logits = torch.as_tensor(collision_logits,
                                 dtype=torch.float64,
                                 device=device)
        super().__init__(x, risk_predictions, logits.sigmoid(), discrepancy,
                         budget, device)
        self.baseline_risk = self.risk_mean.clone()
        self.collision_logits = logits
        families = torch.as_tensor(x, dtype=torch.float64,
                                   device=device)[:, 4].long()
        self.family = families
        self.calibrators = {}
        for family in (0, 1):
            fitted = calibrators[str(family)]
            decoder = fitted["collision_only"] if collision_only else fitted
            parameters = {
                key: float(fitted[key])
                for key in ("risk_residual_mean", "risk_residual_scale")
            }
            parameters.update({
                "intercept": float(decoder["intercept"]),
                "collision_logit_scale": float(
                    decoder["collision_logit_scale"]),
                "standardized_risk_residual_scale": 0.0
                if collision_only else float(
                    decoder["standardized_risk_residual_scale"]),
            })
            self.calibrators[family] = parameters
            if self.calibrators[family]["risk_residual_scale"] <= 0:
                raise ValueError("Risk residual scale must be positive")

    def probabilities(self):
        residual_mean = self.risk_mean - self.baseline_risk
        residual_variance = self.rr.diagonal().clamp_min(0)
        probabilities = torch.empty_like(self.collision_logits)
        for family, fitted in self.calibrators.items():
            mask = self.family == family
            eta = (
                fitted["intercept"]
                + fitted["collision_logit_scale"]
                * self.collision_logits[:, mask]
                + fitted["standardized_risk_residual_scale"]
                * (residual_mean[:, mask] - fitted["risk_residual_mean"])
                / fitted["risk_residual_scale"]
            )
            slope = (fitted["standardized_risk_residual_scale"]
                     / fitted["risk_residual_scale"])
            logistic_normal_scale = torch.sqrt(
                1 + (math.pi / 8) * slope * slope
                * residual_variance[mask]
            )
            probabilities[:, mask] = torch.sigmoid(
                eta / logistic_normal_scale[None, :])
        return self.log_weights.softmax(0) @ probabilities

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
