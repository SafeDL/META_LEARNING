"""Run the declared research sequence without selecting on the untouched pool."""

from methods.baselines.dart.train import main as train
from methods.baselines.hgt.io import write_json

from . import evaluate, freeze, prepare, report, verify
from .config import ROOT


def main():
    for stage, action in (
        ("training", train),
        ("development", lambda: evaluate.run_parallel("development")),
        ("freeze", freeze.main),
        ("measurement", prepare.main),
        ("confirmation", lambda: evaluate.run_parallel("confirmation")),
        ("verification", verify.main),
    ):
        write_json(ROOT / "status.json", {"status": "running", "stage": stage})
        print(f"PIPELINE {stage}", flush=True)
        action()
        if stage in ("development", "confirmation"):
            report.main(stage)
    write_json(ROOT / "status.json", {"status": "complete", "stage": "complete"})


if __name__ == "__main__":
    main()
