from __future__ import annotations

from meta_testing.evaluation.budget_protocol import BudgetProtocol
from meta_testing.evaluation.evaluator import FixedBudgetEvaluator
from meta_testing.failure.signature import FailureSignatureBuilder


def test_signature_is_deterministic_and_fixed_budget_counts_unique_failures() -> None:
    builder = FailureSignatureBuilder()
    outcome = {"target_collision": True, "min_ttc": 1.0, "min_distance": 0.5, "max_closing_speed": 12.0}
    first = builder.from_outcome(outcome, "merge", "zone")
    assert first.is_valid_episode and first.is_failure and first == builder.from_outcome(outcome, "merge", "zone")
    evaluator = FixedBudgetEvaluator(BudgetProtocol(total_episodes=3, support_shots=(0, 1, 2)))
    result = evaluator.evaluate(lambda _: first)
    assert result["cumulative_unique_failures"] == 1
    assert result["all_in_budget"] is True
