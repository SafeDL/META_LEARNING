from experiments.query import PaidExecution, QueryLedger
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.response_baselines.collision import run


def test_random_baseline_shares_paid_initialization_and_fixed_budget():
    scenes = sample("front_brake", 12, "new")

    def execute(scenario):
        return PaidExecution(
            QueryFeedback(scenario.scenario_id, False, SafetyMeasurements(30, 8), 1), []
        )

    ledger = QueryLedger(scenes, 6, execute)
    result = run(scenes, ledger, method="random", initial_indices=[1, 3], budget=6)
    assert result["selected_indices"][:2] == [1, 3]
    assert len(set(result["selected_indices"])) == 6
    assert len(ledger.records) == result["actual_queries"] == 6
