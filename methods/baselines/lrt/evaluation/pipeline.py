"""Run the declared historical training and existing-pool verification protocol."""

import time

from methods.baselines.hgt.io import write_json

from ..config import ROOT
from ..train import main as train
from .diagnostics import main as diagnostics
from .evaluate import main as evaluate
from .protocol import freeze, initialize
from .report import main as report
from .verify import main as verify


def main():
    initialize()
    started = time.perf_counter()
    for stage, action in (
        ("training", train),
        ("freeze", freeze),
        ("evaluation", evaluate),
        ("diagnostics", diagnostics),
        ("verification", verify),
        ("report", report),
    ):
        write_json(
            ROOT / "status.json",
            {
                "status": "running",
                "stage": stage,
                "elapsed_s": time.perf_counter() - started,
            },
        )
        action()
    write_json(
        ROOT / "status.json",
        {
            "status": "complete",
            "physical_calls": 0,
            "elapsed_s": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
