"""Reduced real-data training and paid-query checks; not a performance experiment."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from controllers.highway.factory import create
from simulator.highway.collision_env import execute
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.training import Config, load_model, train

from .history import build_source, load_training_archives
from .joint_validation import OUTPUT
from .prepare_joint import prepare_dataset
from .query import QueryLedger, run_isolated
from .rss import measurement_contract
from .scenarios import RECORDED_SCENARIOS, TEST_SCENARIOS, candidate_pool
from .storage import write_json

CONFIG = Config(steps=2, initial_count=2, budget=4, historical_context=8, diagnostic_interval=1)


def main():
    from controllers.highway.catalog import HISTORY_SUTS

    torch.set_num_threads(1)
    rows = []
    with TemporaryDirectory(prefix="rss-integration-") as temporary:
        systems = [s for s in HISTORY_SUTS if s.family == "rule"]
        dataset = Path(temporary) / "records"
        prepare_dataset(
            dataset,
            recorded_count=8,
            additions=dict.fromkeys(TEST_SCENARIOS, 8),
            workers=2,
            systems=systems,
        )
        for kind in TEST_SCENARIOS:
            history = load_training_archives(dataset / "manifest.json", kind)
            root = Path(temporary) / kind
            report = train(history, root, config=CONFIG)
            encoder, _ = load_model(root / "historical_encoder.pt")
            base, _ = load_model(root / "target_kernel.pt")
            candidates = candidate_pool(kind, 12, replicate=99)
            sources = build_source(encoder, history, candidates)
            assert np.isfinite(sources["means"]).all()
            ledger = QueryLedger(
                candidates,
                4,
                lambda scene: execute(create("independent_idm_mobil"), scene),
                joint_feedback=True,
            )
            result = run_isolated(
                select,
                dict(
                    candidates=candidates,
                    pretrained=base,
                    sources=sources,
                    mode="H1F1",
                    initial_indices=[0, 1],
                    budget=4,
                    device="cpu",
                ),
                ledger,
            )
            assert result["selected_indices"] == [row["index"] for row in ledger.records]
            rows.append(
                {
                    "scenario": kind,
                    "training_status": report["encoder"]["status"],
                    "historical_systems": [a.source_id for a in history],
                    "engineering_target": "independent_idm_mobil",
                    "paid_queries": len(ledger.records),
                    "selected_indices": result["selected_indices"],
                    "final_training_nll": report["encoder"]["curve"][-1]["training_nll"],
                    "queries": result["queries"],
                    "query_ledger": ledger.records,
                    "resources": result["resources"],
                    "source_blocks_shape": list(sources["blocks"].shape),
                }
            )
            if kind not in RECORDED_SCENARIOS:
                from .render_joint import render_execution

                position = max(
                    range(len(ledger.trajectories)), key=lambda i: len(ledger.trajectories[i])
                )
                render_execution(
                    candidates[ledger.records[position]["index"]],
                    ledger.trajectories[position],
                    OUTPUT / "reports/sequential_training_scenarios",
                    provenance={
                        "source_id": "independent_idm_mobil",
                        "query": ledger.records[position],
                    },
                )
            print(f"Joint integration {len(rows)}/{len(TEST_SCENARIOS)}", flush=True)
    summary = {
        "status": "passed",
        "interpretation": "Temporary historical preparation, reduced sequential training and isolated paid queries. Not a method-effectiveness experiment.",
        "continuous_response": measurement_contract(),
        "configuration": asdict(CONFIG),
        "checks": rows,
        "development_ads_used": False,
        "formal_target_bank_used": False,
        "new_simulator_calls": len(TEST_SCENARIOS) * (len(systems) * 8 + 4),
    }
    write_json(OUTPUT / "reports/sequential_training_check.json", summary)
    return summary


if __name__ == "__main__":
    main()
