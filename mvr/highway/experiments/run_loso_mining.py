"""Run B=20 LOSO failure-mining evaluation with support counted in budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mvr.highway.config import ExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.mining import (
    adapted_mining,
    diagnostic_mining,
    highest_risk_support_mining,
    random_mining,
    shared_prior_mining,
)


def _row(target: str, repeat: int, trace) -> dict:
    return {
        "method": trace.method,
        "target_sut": target,
        "repeat": repeat,
        "critical_score_at_20": trace.critical_score,
        "collision_count_at_20": trace.collision_count,
        "failure_count_at_20": trace.failure_count,
        "curve": ";".join(f"{value:.3f}" for value in trace.cumulative_critical_score),
    }


def run_loso_mining(bank: ResponseBank, config: ExperimentConfig) -> list[dict]:
    """Evaluate all four methods, never revealing a target row before its support."""
    config.validate()
    rows: list[dict] = []
    seeds = np.random.SeedSequence(config.seed).spawn(len(bank.sut_names))
    for target_index, target_name in enumerate(bank.sut_names):
        prior = LowRankPrior.fit(np.delete(bank.vulnerability, target_index, axis=0), config.prior_rank)
        vulnerability = bank.vulnerability[target_index]
        collisions = bank.collisions[target_index]
        near_misses = bank.near_misses[target_index]
        rows.append(_row(target_name, 0, shared_prior_mining(prior, collisions, near_misses, config.total_budget)))
        rows.append(_row(target_name, 0, diagnostic_mining(
            prior, vulnerability, collisions, near_misses, config.support_budget, config.total_budget
        )))
        rows.append(_row(target_name, 0, highest_risk_support_mining(
            prior, vulnerability, collisions, near_misses, config.support_budget, config.total_budget
        )))
        rng = np.random.default_rng(seeds[target_index])
        for repeat in range(config.random_support_repeats):
            rows.append(_row(target_name, repeat, random_mining(
                collisions, near_misses, config.total_budget, rng
            )))
            support = rng.choice(len(vulnerability), config.support_budget, replace=False)
            rows.append(_row(target_name, repeat, adapted_mining(
                "Random Support + Adaptation", prior, vulnerability, collisions, near_misses,
                support, config.total_budget
            )))
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/diva_highway/cutin_mvp/response_bank_highway.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/diva_highway/cutin_mvp/loso_mining_highway.csv"))
    args = parser.parse_args()
    rows = run_loso_mining(ResponseBank.load(args.bank), ExperimentConfig())
    write_rows(rows, args.output)
    for method in sorted({row["method"] for row in rows}):
        values = [row["critical_score_at_20"] for row in rows if row["method"] == method]
        print(f"{method}: Critical Score@20={np.mean(values):.3f} ± {np.std(values):.3f}")


if __name__ == "__main__":
    main()
