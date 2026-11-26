import numpy as np

from experiments.domain import Domain
from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from testing_methods.baselines.paired_system_active_evaluation import method as paired
from testing_methods.baselines.response_baselines.collision import run as run_collision_baseline


def test_two_dimensional_baseline_runs_through_the_paid_process_interface():
    domain = Domain(
        {"gap_m": (12, 40), "deceleration_mps2": (2, 6)},
        {"gap_m": 1, "deceleration_mps2": 0.1},
        {},
        {"ego_speed_mps": 50 / 3.6, "event_time_s": 1},
    )
    scenes = [
        Scenario(str(i), "front_brake", {"gap_m": 12 + 3 * i, "deceleration_mps2": 2 + 0.4 * i})
        for i in range(10)
    ]

    def execute(scene):
        collision = scene.parameters["gap_m"] < 20
        return PaidExecution(
            QueryFeedback(
                scene.scenario_id,
                collision,
                SafetyMeasurements(
                    0 if collision else 3,
                    0 if collision else 2,
                    impact_delta_v_mps=3 if collision else None,
                ),
                0,
            ),
            [],
        )

    public = {
        "candidates": scenes,
        "initial_indices": [5, 0, 8, 2],
        "budget": 7,
        "feature_fn": domain.features,
        "proxy": np.linspace(0.1, 0.9, len(scenes)),
        "config": paired.Config(
            batch_size=2,
            shortlist_size=4,
            training_epochs=3,
            mc_draws=4,
            local_samples=8,
            paired_samples=3,
        ),
    }
    ledger = QueryLedger(scenes, public["budget"], execute)
    result = run_isolated(paired.run, public, ledger)
    assert result["selected_indices"][:4] == public["initial_indices"]
    assert len(set(result["selected_indices"])) == result["actual_queries"] == 7
    assert [r["purpose"] for r in ledger.records] == ["initial"] * 4 + ["target"] * 3
    assert all(r["status"] == "complete" for r in ledger.records)


def test_random_baseline_shares_paid_initialization_and_fixed_budget():
    scenes = sample("front_brake", 12, "new")

    def execute(scenario):
        return PaidExecution(
            QueryFeedback(scenario.scenario_id, False, SafetyMeasurements(30, 8), 1), []
        )

    ledger = QueryLedger(scenes, 6, execute)
    result = run_collision_baseline(scenes, ledger, method="random", initial_indices=[1, 3], budget=6)
    assert result["selected_indices"][:2] == [1, 3]
    assert len(set(result["selected_indices"])) == 6
    assert len(ledger.records) == result["actual_queries"] == 6
