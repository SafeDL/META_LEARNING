"""Run the offline E5 oracle-headroom audit on a completed Highway response bank.

Oracle methods intentionally inspect held-out target outcomes beyond their
nominal support budget. They are diagnostic upper bounds, never baselines or
deployable methods.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mvr.highway.config import ExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.diva.acquisition import oracle_support_indices
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.mining import adapted_mining, shared_prior_mining
from mvr.highway.diva.posterior import adapt_posterior, oracle_latent_prediction
from mvr.highway.experiments.metrics import (
    ndcg_at_k,
    spearman_correlation,
    top_k_recall,
)


def _ranking_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "ndcg_at_10": ndcg_at_k(prediction, truth),
        "top_10_recall": top_k_recall(prediction, truth),
        "spearman": spearman_correlation(prediction, truth),
    }


def _row(
    method: str,
    target_name: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    critical_score: float | None,
    support: np.ndarray | None = None,
) -> dict:
    return {
        "method": method,
        "target_sut": target_name,
        **_ranking_metrics(prediction, truth),
        "critical_score_at_20": critical_score,
        "support_indices": "" if support is None else ";".join(map(str, support)),
    }


def run_oracle_headroom(bank: ResponseBank, config: ExperimentConfig) -> list[dict]:
    """Evaluate Shared Prior, full-target latent, and greedy Oracle K=4."""
    config.validate()
    rows: list[dict] = []
    for target_index, target_name in enumerate(bank.sut_names):
        prior = LowRankPrior.fit(
            np.delete(bank.vulnerability, target_index, axis=0), config.prior_rank
        )
        truth = bank.vulnerability[target_index]
        collisions = bank.collisions[target_index]
        near_misses = bank.near_misses[target_index]
        shared_trace = shared_prior_mining(
            prior, collisions, near_misses, config.total_budget
        )
        rows.append(
            _row(
                "Shared Prior", target_name, prior.mean, truth, shared_trace.critical_score
            )
        )
        latent_prediction = oracle_latent_prediction(prior, truth)
        rows.append(
            _row(
                "Oracle Latent (full target)", target_name, latent_prediction, truth, None
            )
        )
        support = oracle_support_indices(prior, truth, config.support_budget)
        oracle_prediction = adapt_posterior(prior, support, truth[support]).prediction
        oracle_trace = adapted_mining(
            "Oracle K=4 (ranking upper bound)",
            prior,
            truth,
            collisions,
            near_misses,
            support,
            config.total_budget,
        )
        rows.append(
            _row(
                "Oracle K=4 (ranking upper bound)",
                target_name,
                oracle_prediction,
                truth,
                oracle_trace.critical_score,
                support,
            )
        )
    return rows


def headroom_summary(rows: list[dict]) -> dict:
    """Apply the E5 diagnostic thresholds without treating them as method gates."""
    grouped = {
        method: [row for row in rows if row["method"] == method]
        for method in {row["method"] for row in rows}
    }
    shared = grouped["Shared Prior"]
    oracle_k4 = grouped["Oracle K=4 (ranking upper bound)"]
    oracle_latent = grouped["Oracle Latent (full target)"]
    shared_ndcg = float(np.mean([row["ndcg_at_10"] for row in shared]))
    oracle_k4_ndcg = float(np.mean([row["ndcg_at_10"] for row in oracle_k4]))
    oracle_latent_ndcg = float(np.mean([row["ndcg_at_10"] for row in oracle_latent]))
    shared_score = float(np.mean([row["critical_score_at_20"] for row in shared]))
    oracle_k4_score = float(np.mean([row["critical_score_at_20"] for row in oracle_k4]))
    k4_wins = sum(
        oracle["ndcg_at_10"] > baseline["ndcg_at_10"]
        for baseline, oracle in zip(shared, oracle_k4, strict=True)
    )
    score_wins = sum(
        oracle["critical_score_at_20"] > baseline["critical_score_at_20"]
        for baseline, oracle in zip(shared, oracle_k4, strict=True)
    )
    return {
        "schema": "highway_diva_mine_e5_oracle_headroom_v1",
        "oracle_is_diagnostic_upper_bound_only": True,
        "mean_ndcg_at_10": {
            "shared_prior": shared_ndcg,
            "oracle_k4": oracle_k4_ndcg,
            "oracle_latent": oracle_latent_ndcg,
            "oracle_k4_delta": oracle_k4_ndcg - shared_ndcg,
        },
        "mean_critical_score_at_20": {
            "shared_prior": shared_score,
            "oracle_k4": oracle_k4_score,
            "oracle_k4_delta": oracle_k4_score - shared_score,
            "oracle_k4_relative_gain": (
                (oracle_k4_score - shared_score) / shared_score if shared_score else None
            ),
        },
        "per_sut_strict_wins": {
            "oracle_k4_ndcg": k4_wins,
            "oracle_k4_critical_score": score_wins,
        },
        "acceptance": {
            "oracle_k4_ndcg_wins_at_least_4_of_6": k4_wins >= 4,
            "oracle_k4_score_mean_gain_at_least_10_percent": oracle_k4_score
            >= 1.10 * shared_score,
            "oracle_latent_strictly_beats_shared_mean_ndcg": oracle_latent_ndcg
            > shared_ndcg,
        },
    }


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("results/diva_highway/cutin_mvp_e2/response_bank_highway_e2.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/diva_highway/cutin_mvp_e2/e5_oracle_headroom.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/diva_highway/cutin_mvp_e2/e5_oracle_headroom.json"),
    )
    args = parser.parse_args()
    rows = run_oracle_headroom(ResponseBank.load(args.bank), ExperimentConfig())
    summary = headroom_summary(rows)
    write_rows(rows, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
