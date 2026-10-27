"""Predeclare the historical proxy and validate coverage-aware paired evaluation."""

import time
from dataclasses import asdict

from controllers.highway.catalog import BY_ID
from testing_methods.baselines.paired_system_active_evaluation.method import DEFAULT as CONFIG
from testing_methods.baselines.paired_system_active_evaluation.method import run
from testing_methods.common.response import HistoricalPredictor, archive_fingerprints

from .collision_metrics import collision_recall, summarize
from .construction import load_construction
from .discovery_config import write_json
from .executions import ExecutionBank, execution_contract
from .initialization import initial_indices
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features
from .source_registration import specification_matches

OUTPUT = (
    ROOT
    / "experiment_outputs/paired_system_active_evaluation/collision/discovery/domain_comparison/stage_a"
)


def build_proxy(candidates, history):
    started = time.perf_counter()
    values = HistoricalPredictor(history, bandwidth=0.25).predict(features(candidates)).mean(axis=0)
    return values, {
        "rule": "equal mean of the allowed historical collision predictors",
        "kernel_bandwidth": 0.25,
        "history_ids": [a.source_id for a in history],
        "history_fingerprints": archive_fingerprints(history),
        "construction_s": time.perf_counter() - started,
        "new_proxy_executions": 0,
    }


def main():
    kind, target, seed = "cut_in", "development_idm_mobil", 11
    history, development = load_construction(kind)
    archive = next(a for a in development if a.source_id == target)
    bank = ExecutionBank(TEST_RECORDS_ROOT / "physical/development" / kind / target)
    if bank.candidates != archive.scenarios or not specification_matches(
        bank.manifest["contract"], execution_contract(BY_ID[target])
    ):
        raise ValueError("Paired validation requires its qualified original development bank")
    proxy, provenance = build_proxy(bank.candidates, history)
    initial = initial_indices(len(bank.candidates), DEFAULT.initial_count, seed)
    ledger = QueryLedger(bank.candidates, DEFAULT.budget, bank)
    started = time.perf_counter()
    result = run_isolated(
        run,
        {
            "candidates": bank.candidates,
            "initial_indices": initial,
            "proxy": proxy,
            "budget": DEFAULT.budget,
            "seed": seed,
        },
        ledger,
    )
    expected_counts = list(range(DEFAULT.initial_count, DEFAULT.budget, CONFIG.batch_size))
    if result["target_fit_sample_counts"] != expected_counts:
        raise ValueError("Paired surrogate updates must use only the completed paid batches")
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
            "proxy": provenance,
            "metrics": {
                **summarize(bank.candidates, result["queries"], ledger.trajectories),
                **collision_recall(bank.candidates, archive.collision.tolist(), result["queries"]),
            },
            "cost": {
                "selector_and_replay_wall_s": time.perf_counter() - started,
                "new_physical_calls": 0,
                "new_proxy_executions": 0,
            },
        },
    )
    report = {
        "status": "passed",
        "queries": len(ledger.records),
        "config": asdict(CONFIG),
        "proxy": provenance,
        "target_fit_sample_counts": result["target_fit_sample_counts"],
        "guided_batches": max(row["batch"] for row in result["queries"]),
        "threshold_fill_batches": len(
            {row["batch"] for row in result["queries"] if row.get("batch_threshold_fill", 0) > 0}
        ),
        "new_physical_calls": 0,
        "formal_comparison_frozen": False,
    }
    write_json(OUTPUT / "audit.json", report)
    print(
        {
            "status": report["status"],
            "queries": report["queries"],
            "guided_batches": report["guided_batches"],
            "new_physical_calls": 0,
        },
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
