"""Risk-only online adaptation with a censored observation model."""
import torch

from .config import BUDGET
from .model import collision_probabilities, mixture_log_probabilities
from .risk_state import CensoredRiskState


class MetaTestingSession:

    def __init__(self, x, risk_predictions, collision_logits, discrepancy,
                 decoder, budget=BUDGET, device="cuda"):
        coordinates = torch.as_tensor(x, dtype=torch.float64, device=device)
        predictions = torch.as_tensor(risk_predictions, dtype=torch.float64,
                                      device=device)
        self.state = CensoredRiskState(coordinates, predictions, discrepancy, budget)
        self.logits = torch.as_tensor(collision_logits, dtype=torch.float64,
                                      device=device)
        self.family = coordinates[:, 4].long()
        self.slope = predictions.new_tensor(decoder["slope"])
        self.center = predictions.new_tensor(decoder["center"])
        self.scale = predictions.new_tensor(decoder["scale"])
        self.remaining = torch.ones(len(coordinates), dtype=torch.bool, device=device)
        self.budget, self.count, self.pending = budget, 0, None
        self.records = []

    def log_probabilities(self):
        return mixture_log_probabilities(
            self.logits, self.family, self.state.log_weights,
            self.state.mean - self.state.baseline, self.state.variance,
            self.slope, self.center, self.scale)

    def probabilities(self):
        return collision_probabilities(*self.log_probabilities())

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("Observe the pending query first")
        if self.count == self.budget:
            return None
        log_collision, log_safe = self.log_probabilities()
        log_odds = log_collision - log_safe
        self.pending = int(log_odds.masked_fill(~self.remaining, -torch.inf).argmax())
        self.records.append({
            "index": self.pending, "query_number": self.count + 1,
            "collision_probability": float(collision_probabilities(
                log_collision[self.pending], log_safe[self.pending])),
            "log_safe_probability": float(log_safe[self.pending]),
            "collision_log_odds": float(log_odds[self.pending]),
        })
        return self.pending

    def observe(self, risk):
        if self.pending is None:
            raise RuntimeError("Request a query before feedback")
        self.state.observe(self.pending, risk)
        weights = self.state.log_weights.exp()
        self.records[-1].update({
            "continuous_risk": float(risk),
            "posterior_entropy": float(-(weights * self.state.log_weights).sum()),
            "largest_hypothesis_mass": float(weights.max()),
        })
        self.remaining[self.pending] = False
        self.count += 1
        self.pending = None
