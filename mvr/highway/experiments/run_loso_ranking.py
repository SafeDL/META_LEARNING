"""Run leakage-free LOSO K=4 vulnerability-ranking validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mvr.highway.config import ExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.diva.acquisition import (
    diagnostic_support_indices,
    highest_risk_indices,
    variance_support_indices,
)
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.posterior import adapt_posterior
from mvr.highway.experiments.metrics import ndcg_at_k, spearman_correlation, top_k_recall


def _metric_row(
    method: str,
    target: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    repeat: int,
    support: np.ndarray | None = None,
) -> dict:
    return {
        "method": method,
        "target_sut": target,
        "repeat": repeat,
        "ndcg_at_10": ndcg_at_k(prediction, truth),
        "top_10_recall": top_k_recall(prediction, truth),
        "spearman": spearman_correlation(prediction, truth),
        "support_indices": "" if support is None else ";".join(map(str, support)),
    }


def run_loso_ranking(bank: ResponseBank, config: ExperimentConfig) -> list[dict]:
    """Evaluate the four K=0/K=4 ranking methods for every held-out SUT."""
    config.validate()
    rows: list[dict] = []
    seeds = np.random.SeedSequence(config.seed).spawn(len(bank.sut_names))
    for target_index, target_name in enumerate(bank.sut_names):
        source = np.delete(bank.vulnerability, target_index, axis=0)
        prior = LowRankPrior.fit(source, config.prior_rank)
        truth = bank.vulnerability[target_index]
        rows.append(_metric_row("Shared Prior", target_name, prior.mean, truth, 0))
        highest_risk = highest_risk_indices(prior.mean, config.support_budget)
        highest_risk_prediction = adapt_posterior(
            prior, highest_risk, truth[highest_risk]
        ).prediction
        rows.append(
            _metric_row(
                "Highest-Risk Support + Adaptation",
                target_name,
                highest_risk_prediction,
                truth,
                0,
                highest_risk,
            )
        )
        diagnostic = diagnostic_support_indices(prior, truth, config.support_budget)
        diagnostic_prediction = adapt_posterior(
            prior, diagnostic, truth[diagnostic]
        ).prediction
        rows.append(
            _metric_row(
                "DIVA Diagnostic + Adaptation",
                target_name,
                diagnostic_prediction,
                truth,
                0,
                diagnostic,
            )
        )
        rng = np.random.default_rng(seeds[target_index])
        for repeat in range(config.random_support_repeats):
            support = rng.choice(len(truth), config.support_budget, replace=False)
            prediction = adapt_posterior(prior, support, truth[support]).prediction
            rows.append(
                _metric_row(
                    "Random Support + Adaptation",
                    target_name,
                    prediction,
                    truth,
                    repeat,
                    support,
                )
            )
    return rows


def run_e1_diagnostic_comparison(
    bank: ResponseBank, config: ExperimentConfig
) -> list[dict]:
    """Compare sequential DIVA support with the retired variance-only rule."""
    config.validate()
    rows: list[dict] = []
    for target_index, target_name in enumerate(bank.sut_names):
        source = np.delete(bank.vulnerability, target_index, axis=0)
        prior = LowRankPrior.fit(source, config.prior_rank)
        truth = bank.vulnerability[target_index]
        variance_only = variance_support_indices(prior, config.support_budget)
        rows.append(
            _metric_row(
                "Variance-Only Support + Adaptation",
                target_name,
                adapt_posterior(prior, variance_only, truth[variance_only]).prediction,
                truth,
                0,
                variance_only,
            )
        )
        diagnostic = diagnostic_support_indices(prior, truth, config.support_budget)
        rows.append(
            _metric_row(
                "DIVA Diagnostic + Adaptation",
                target_name,
                adapt_posterior(prior, diagnostic, truth[diagnostic]).prediction,
                truth,
                0,
                diagnostic,
            )
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
