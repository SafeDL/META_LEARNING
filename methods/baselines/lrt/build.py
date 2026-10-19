"""Build LRT from historical responses, without target evaluation or snapshots."""

import time

import torch
from threadpoolctl import threadpool_limits

from methods.baselines.dart.config import ROOT as ENCODER_ROOT
from methods.baselines.dart.train import prepare_cases, train_seed as train_encoder
from methods.baselines.hgt.history import load_history
from methods.baselines.hgt.io import read_json, write_json
from methods.baselines.hgt.train import train_seed as train_kernel

from .config import ROOT, SEEDS
from .train import train_seed as train_value


def main():
    torch.set_num_threads(1)
    history = load_history()
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            write_json(
                ROOT / "status.json",
                {"status": "running", "stage": "build", "seed": seed},
            )
            train_kernel(history, seed)
            checkpoint = ENCODER_ROOT / "models" / f"decision_{seed}.pt"
            report = ENCODER_ROOT / "models" / f"training_decision_{seed}.json"
            if not (
                checkpoint.exists()
                and report.exists()
                and read_json(report)["status"] == "complete"
            ):
                train_encoder(seed, "decision", prepare_cases(history, seed))
            train_value(seed, "full")
    write_json(
        ROOT / "status.json",
        {
            "status": "complete",
            "stage": "build",
            "seeds": SEEDS,
            "physical_calls": 0,
            "elapsed_s": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
