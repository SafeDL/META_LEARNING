"""Prepare evaluator-owned truth independently of method training and queries."""

from dataclasses import asdict
from pathlib import Path

from .joint_validation import DEFAULT, OUTPUT, target_bank
from .pilot import SYSTEMS
from .prepare_joint import DATASET_ROOT
from .storage import write_json


def prepare(
    destination=DATASET_ROOT / "targets/fixed",
    *,
    config=DEFAULT,
    systems=SYSTEMS,
    candidate_replicate=10,
    report_path=OUTPUT / "pilot/reports/target_preparation.json",
):
    rows = []
    for sut in systems:
        for kind in config.scenarios:
            bank = target_bank(destination, sut, kind, config, replicate=candidate_replicate)
            rows.append(
                {
                    "sut_id": sut.sut_id,
                    "scenario": kind,
                    "candidates": len(bank.candidates),
                    "bank": str(bank.root.resolve()),
                    "status": bank.manifest["status"],
                }
            )
            write_json(
                Path(report_path),
                {
                    "status": "complete"
                    if len(rows) == len(systems) * len(config.scenarios)
                    else "running",
                    "config": asdict(config),
                    "systems": [asdict(s) for s in systems],
                    "candidate_replicate": candidate_replicate,
                    "banks": rows,
                    "completed_truth_records": sum(row["candidates"] for row in rows),
                    "method_queries": 0,
                    "truth_access": "evaluator only",
                },
            )
    return rows


if __name__ == "__main__":
    prepare()
