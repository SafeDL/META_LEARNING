"""Development study of target-only failures in historically safe scenarios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from method_chains.core_mine.acquisition import choose
from method_chains.core_mine.config import CoreMineConfig, SUPPORT_BUDGET
from method_chains.core_mine.metrics import record_metrics
from method_chains.core_mine.oracle import CacheOracle
from method_chains.core_mine.posterior import PosteriorModel
from method_chains.core_mine import sparse_sut_experiment as sparse


CONFIG = CoreMineConfig(residual_length=.30, residual_amplitude=.15, lambda_=.10)
BETA = (0.0, 0.05, 0.10, 0.25, 0.50)


def campaign(task, branch: str | None, beta: float, repeat: int = 0,
             residual: bool = True) -> dict:
    eligible = np.flatnonzero(~task.source_event.any(axis=0))
    if len(eligible) < 50 or len(set(task.modes[eligible])) < 5:
        raise ValueError("source-safe search requires >=50 candidates in all five modes")
    oracle = CacheOracle(task)
    model = PosteriorModel(task, CONFIG, branch, residual) if branch is not None else None
    rng = np.random.default_rng(task.seed + repeat * 997 + sum(map(ord, task.target_name)))
    while len(oracle.revealed) < 50:
        if model is None:
            scores = rng.random(task.count)
        else:
            prediction = model.predict()
            scores = prediction["p_event"] + beta * np.sqrt(prediction["variance"])
        index = choose(scores, oracle.revealed, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        observed = oracle.reveal(index)
        if model is not None:
            model.observe(index, observed)
    result = record_metrics(task, oracle.revealed, 50)
    result["source_safe_candidates"] = len(eligible)
    result["source_safe_failures_in_pool"] = int(task.source_safe_target_failure.sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", choices=("v5_corrected_geometry", "v7_corrected_balanced"),
                        default="v7_corrected_balanced")
    args = parser.parse_args()
    sparse.configure_proposal(args.proposal)
    seeds = sparse.QUALIFICATION_SEEDS if args.proposal == "v7_corrected_balanced" else sparse.DEVELOPMENT_SEEDS
    rows = []
    for seed in seeds:
        bank = sparse.load_or_build(seed)
        for task in sparse.tasks_from_bank(bank, seed):
            settings = [("MeanResidual-Risk", "mean", 0.0),
                        ("CoRe-Risk", "composition", 0.0),
                        ("TargetOnly-Risk", "target", 0.0),
                        ("HistoricalMargin-Static", "mean", 0.0)]
            settings += [(f"MeanResidual-UCB-{beta:.2f}", "mean", beta) for beta in BETA[1:]]
            for method, branch, beta in settings:
                result = campaign(task, branch, beta,
                                  residual=method != "HistoricalMargin-Static")
                rows.append({"method": method, "beta": beta, **result})
            for repeat in range(10):
                result = campaign(task, None, 0.0, repeat)
                rows.append({"method": "RandomSafe", "beta": 0.0, "repeat": repeat, **result})
            print(f"completed seed={seed} target={task.heterogeneity}", flush=True)
    output = sparse.ROOT / "source_safe_development.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} campaigns to {output}")


if __name__ == "__main__":
    main()
