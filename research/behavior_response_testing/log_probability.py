"""Rank Bernoulli probabilities accurately even close to certainty."""
import torch
from torch.nn import functional as F

from .session import BUDGET, BehaviorTestingSession


class LogProbabilityTestingSession(BehaviorTestingSession):

    def __init__(self,
                 x,
                 risk_predictions,
                 collision_logits,
                 discrepancy,
                 budget=BUDGET,
                 device="cuda"):
        logits = torch.as_tensor(collision_logits,
                                 dtype=torch.float64,
                                 device=device)
        super().__init__(x, risk_predictions, logits.sigmoid(), discrepancy,
                         budget, device)
        self.log_safe = F.logsigmoid(-logits)

    def log_safe_probabilities(self):
        return torch.logsumexp(self.log_weights[:, None] + self.log_safe,
                               dim=0)

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("Observe the pending query first")
        if self.count == self.budget:
            return None
        log_safe = self.log_safe_probabilities()
        self.pending = int(
            log_safe.masked_fill(~self.remaining, torch.inf).argmin())
        self.records.append({
            "index":
            self.pending,
            "query_number":
            self.count + 1,
            "collision_probability":
            float(-log_safe[self.pending].expm1()),
            "log_safe_probability":
            float(log_safe[self.pending])
        })
        return self.pending
