"""Audit a paid first run of the historical-graph, two-stage ScenarioFuzz adaptation."""

import time

from controllers.highway.catalog import BY_ID
from testing_methods.baselines.scenario_fuzz.model import load
from testing_methods.baselines.scenario_fuzz.pool import run
from testing_methods.common.response import archive_fingerprints

from .collision_metrics import collision_recall, summarize
from .construction import load_construction
from .discovery_config import write_json
from .domain_risk import disclose_driving_quality
from .executions import ExecutionBank, execution_contract
from .initialization import initial_indices
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .source_registration import specification_matches

OUTPUT = ROOT / "experiment_outputs/scenario_fuzz/collision/discovery/domain_comparison/stage_a"


def main():
    kind, target, seed = "cut_in", "development_idm_mobil", 11
    history, development = load_construction(kind)
    archive = next(a for a in development if a.source_id == target)
    bank = ExecutionBank(TEST_RECORDS_ROOT / "physical/development" / kind / target)
    if bank.candidates != archive.scenarios or not specification_matches(
        bank.manifest["contract"], execution_contract(BY_ID[target])
    ):
        raise ValueError("ScenarioFuzz validation requires its qualified original development bank")
    model, metadata = load(OUTPUT / "models" / kind / "risk.pt")
    if metadata["history_fingerprints"] != archive_fingerprints(history):
        raise ValueError("ScenarioFuzz model was not built from the allowed historical responses")
    initial = initial_indices(len(bank.candidates), DEFAULT.initial_count, seed)
    ledger = QueryLedger(
        bank.candidates, DEFAULT.budget, lambda scene: disclose_driving_quality(bank(scene))
    )
    started = time.perf_counter()
    result = run_isolated(
        run,
        {
            "candidates": bank.candidates,
            "initial_indices": initial,
            "classifier": model,
            "budget": DEFAULT.budget,
            "seed": seed,
        },
        ledger,
    )
    if (
        result["online_classifier_updates"]
        or metadata["training_rows"] + metadata["validation_rows"] != 4096
    ):
        raise ValueError("The fixed graph model requires the declared historical construction")
    write_json(
        OUTPUT / "first_run.json",
        {
            "stage": "A mechanism validation; not the frozen development comparison",
            "target_id": target,
            "scenario": kind,
            "algorithm_seed": seed,
            "initial_indices": initial,
            "result": result,
            "query_ledger": ledger.records,
            "metrics": {
                **summarize(bank.candidates, result["queries"], ledger.trajectories),
                **collision_recall(bank.candidates, archive.collision.tolist(), result["queries"]),
            },
            "cost": {
                "selector_and_replay_wall_s": time.perf_counter() - started,
                "new_physical_calls": 0,
            },
        },
    )
    report = {
        "status": "passed",
        "queries": len(ledger.records),
        "threshold_fill_queries": result["threshold_fill_queries"],
        "online_classifier_updates": 0,
        "new_physical_calls": 0,
        "formal_comparison_frozen": False,
    }
    write_json(OUTPUT / "audit.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    main()
