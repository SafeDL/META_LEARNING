from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements


def dynamic_selector(oracle, *, initial, proposed):
    queries = []
    for position, scene in enumerate([*initial, *proposed]):
        index = position if position < len(initial) else oracle.register(scene)
        feedback = oracle.query(index, purpose="initial" if position < len(initial) else "target")
        queries.append({"index": index, "feedback": feedback.to_dict()})
    return {
        "selected_indices": [q["index"] for q in queries],
        "queries": queries,
        "actual_queries": len(queries),
    }


def test_continuous_registration_discloses_no_label_and_only_executions_use_budget():
    scenes = sample("front_brake", 4, "new", replicate=27)

    def execute(scene):
        return PaidExecution(
            QueryFeedback(scene.scenario_id, False, SafetyMeasurements(10, 5), 1), []
        )

    ledger = QueryLedger(scenes[:2], 4, execute)
    result = run_isolated(dynamic_selector, {"initial": scenes[:2], "proposed": scenes[2:]}, ledger)
    assert result["actual_queries"] == 4
    assert len(ledger.candidates) == 4
    assert len(ledger.records) == 4
    assert all(r["status"] == "complete" for r in ledger.records)
    assert [r["index"] for r in ledger.records] == [0, 1, 2, 3]
