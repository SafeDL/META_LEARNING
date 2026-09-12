"""Run leakage-free LOSO K=4 vulnerability-ranking validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mvr.highway.config import ExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.diva.acquisition import diagnostic_support_indices
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.posterior import adapt_posterior
from mvr.highway.experiments.metrics import ndcg_at_k, spearman_correlation, top_k_recall


def _metric_row(method: str, target: str, prediction: np.ndarray, truth: np.ndarray, repeat: int) -> dict:
    return {
        "method": method,
        "target_sut": target,
        "repeat": repeat,
        "ndcg_at_10": ndcg_at_k(prediction, truth),
        "top_10_recall": top_k_recall(prediction, truth),
        "spearman": spearman_correlation(prediction, truth),
    }


def run_loso_ranking(bank: ResponseBank, config: ExperimentConfig) -> list[dict]:
    """Evaluate K=0, random K=4, and diagnostic K=4 for every held-out SUT."""
    config.validate()
    rows: list[dict] = []
    seeds = np.random.SeedSequence(config.seed).spawn(len(bank.sut_names))
    for target_index, target_name in enumerate(bank.sut_names):
        source = np.delete(bank.vulnerability, target_index, axis=0)
        prior = LowRankPrior.fit(source, config.prior_rank)
        truth = bank.vulnerability[target_index]
        rows.append(_metric_row("Shared Prior", target_name, prior.mean, truth, 0))
        diagnostic = diagnostic_support_indices(prior, config.support_budget)
        diagnostic_prediction = adapt_posterior(
            prior, diagnostic, truth[diagnostic]
        ).prediction
        rows.append(
            _metric_row(
                "Diagnostic Support + Adaptation",
                target_name,
                diagnostic_prediction,
                truth,
                0,
            )
        )
        rng = np.random.default_rng(seeds[target_index])
        for repeat in range(config.random_support_repeats):
            support = rng.choice(len(truth), config.support_budget, replace=False)
            prediction = adapt_posterior(prior, support, truth[support]).prediction
            rows.append(
                _metric_row("Random Support + Adaptation", target_name, prediction, truth, repeat)
            )
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
    parser.add_argument("--output", type=Path, default=Path("results/diva_highway/cutin_mvp/loso_ranking_highway.csv"))
    args = parser.parse_args()
    rows = run_loso_ranking(ResponseBank.load(args.bank), ExperimentConfig())
    write_rows(rows, args.output)
    for method in sorted({row["method"] for row in rows}):
        values = [row["ndcg_at_10"] for row in rows if row["method"] == method]
        print(f"{method}: NDCG@10={np.mean(values):.3f} ± {np.std(values):.3f}")


if __name__ == "__main__":
    main()
