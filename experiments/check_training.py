"""Reduced real-data training and paid-query checks; not a performance experiment."""

from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.training import Config, load_model, train

from .construction import OUTPUT, load_training_data
from .discovery_config import write_json
from .executions import ExecutionBank
from .protocol import TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .rss import measurement_contract
from .scenarios import DOMAINS
from .train import source_predictions

CONFIG = Config(
    steps=2, support_counts=(2, 4), query_count=4, historical_context=8, validation_interval=1
)


def subset(archive, count=32):
    return replace(
        archive,
        scenarios=archive.scenarios[:count],
        collision=archive.collision[:count],
        safety=archive.safety[:count],
        rss_margin=archive.continuous_margin[:count],
    )


def main():
    torch.set_num_threads(1)
    rows = []
    with TemporaryDirectory(prefix="rss-integration-") as temporary:
        for kind in DOMAINS:
            history, development = load_training_data(kind)
            history = [subset(a) for a in history[:3]]
            development = [subset(development[0])]
            root = Path(temporary) / kind
            report = train(history, development, root, config=CONFIG)
            encoder, _ = load_model(root / "historical_encoder.pt")
            base, _ = load_model(root / "target_kernel.pt")
            candidates = development[0].scenarios[:12]
            predictions = source_predictions(encoder, history, candidates)
            assert np.isfinite(predictions["means"]).all()
            sources = {
                "means": predictions["means"],
                "covariances": np.stack([np.diag(v) for v in predictions["variances"]]),
                "weights": predictions["weights"],
                "names": predictions["names"],
            }
            bank = ExecutionBank(TEST_RECORDS_ROOT / development[0].provenance["bank"])
            ledger = QueryLedger(candidates, 4, bank)
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
                    "development_system": development[0].source_id,
                    "paid_queries": len(ledger.records),
                    "selected_indices": result["selected_indices"],
                    "best_validation_nll": report["encoder"]["best_score"][0],
                }
            )
            print(f"RSS integration {len(rows)}/5", flush=True)
    summary = {
        "status": "passed",
        "interpretation": "Reduced training and process-isolated development replay only. Not a full retraining or an independent evaluation of testing efficiency.",
        "continuous_response": measurement_contract(),
        "configuration": asdict(CONFIG),
        "checks": rows,
        "independent_target_records_used": False,
        "new_simulator_calls": 0,
    }
    write_json(OUTPUT / "integration_check.json", summary)
    return summary


if __name__ == "__main__":
    main()
