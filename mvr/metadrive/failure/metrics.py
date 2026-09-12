from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .signature import FailureSignature


@dataclass
class FixedBudgetMetrics:
    total_budget: int
    signatures: list[FailureSignature] = field(default_factory=list)

    def add(self, signature: FailureSignature) -> None:
        if len(self.signatures) >= self.total_budget:
            raise ValueError("episode budget exhausted")
        self.signatures.append(signature)

    @property
    def cumulative_unique(self) -> list[int]:
        seen, curve = set(), []
        for signature in self.signatures:
            if signature.is_failure:
                seen.add(signature.signature_id)
            curve.append(len(seen))
        return curve

    def summary(self) -> dict[str, float | int | None]:
        curve = self.cumulative_unique
        first = next((index + 1 for index, item in enumerate(self.signatures) if item.is_failure), None)
        target = lambda count: next((index + 1 for index, value in enumerate(curve) if value >= count), None)
        return {
            "episodes": len(self.signatures), "budget": self.total_budget,
            "cumulative_unique_failures": curve[-1] if curve else 0,
            "failure_discovery_auc": float(np.trapezoid(curve, dx=1.0)) if curve else 0.0,
            "tests_to_first_valid_failure": first, "tests_to_5_unique_failures": target(5), "tests_to_10_unique_failures": target(10),
            "invalid_rate": float(1.0 - sum(item.is_valid_episode for item in self.signatures) / len(self.signatures)) if self.signatures else 0.0,
            "unique_failures_per_episode": float((curve[-1] if curve else 0) / max(1, len(self.signatures))),
        }
