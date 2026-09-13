"""Evaluate PR-BRVT and fixed-budget baselines with strict LOSO replay."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mvr.highway.config import RegressionExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.diva.acquisition import highest_risk_indices
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.outcomes import formal_rewards
from mvr.highway.diva.regression_mining import (
    RegressionMiningTrace,
    run_regression_mining,
)
from mvr.highway.diva.regression_reference import (
    RegressionReference,
    build_regression_reference,
)

CSV_FIELDS = (
    "method",
    "target_sut",
    "repeat",
    "raw_critical_score_at_20",
    "regression_critical_score_at_20",
    "target_specific_failure_count_at_20",
    "queried_indices",
    "raw_curve",
    "regression_curve",
    "target_specific_curve",
)
RANDOM_METHOD = "Random"
SHARED_METHOD = "Shared Prior"
NOVELTY_METHOD = "Population-Novelty Only"
PR_BRVT_METHOD = "Bayesian Regression Vulnerability"


def _curve_text(curve: np.ndarray) -> str:
    return ";".join(f"{value:.6f}" for value in curve)


def _fixed_trace(
    queried_indices: np.ndarray,
    reference: RegressionReference,
    collisions: np.ndarray,
    near_misses: np.ndarray,
) -> RegressionMiningTrace:
    """Replay a non-adaptive ordering using the same outcome definitions."""
    queried = np.asarray(queried_indices, dtype=int)
    if len(np.unique(queried)) != len(queried):
        raise ValueError("a scenario may be queried only once")
    rewards = formal_rewards(collisions[queried], near_misses[queried])
    regression_rewards = rewards * (
        1.0 - reference.smoothed_failure_prevalence[queried]
    )
    target_specific = (
        (rewards > 0.0) & (reference.failure_prevalence[queried] <= 0.20)
    ).astype(float)
    return RegressionMiningTrace(
        queried_indices=queried,
        raw_critical_curve=np.cumsum(rewards),
        regression_critical_curve=np.cumsum(regression_rewards),
        target_specific_failure_curve=np.cumsum(target_specific),
    )


def _row(
    method: str, target_name: str, repeat: int, trace: RegressionMiningTrace
) -> dict:
    return {
        "method": method,
        "target_sut": target_name,
        "repeat": repeat,
        "raw_critical_score_at_20": float(trace.raw_critical_curve[-1]),
        "regression_critical_score_at_20": float(
            trace.regression_critical_curve[-1]
        ),
        "target_specific_failure_count_at_20": float(
            trace.target_specific_failure_curve[-1]
        ),
        "queried_indices": ";".join(str(index) for index in trace.queried_indices),
        "raw_curve": _curve_text(trace.raw_critical_curve),
        "regression_curve": _curve_text(trace.regression_critical_curve),
        "target_specific_curve": _curve_text(trace.target_specific_failure_curve),
    }


def run_loso_regression_mining(
    bank: ResponseBank, config: RegressionExperimentConfig
) -> list[dict]:
    """Compare all methods without allowing the held-out target into sources."""
    config.validate()
    if len(bank.anchors) != config.num_anchors:
        raise ValueError("response bank anchor count does not match configuration")
    rows: list[dict] = []
    target_seeds = np.random.SeedSequence(config.seed).spawn(len(bank.sut_names))
    for target_index, target_name in enumerate(bank.sut_names):
        source_vulnerability = np.delete(bank.vulnerability, target_index, axis=0)
        source_collisions = np.delete(bank.collisions, target_index, axis=0)
        source_near_misses = np.delete(bank.near_misses, target_index, axis=0)
        prior = LowRankPrior.fit(source_vulnerability, config.prior_rank)
        reference = build_regression_reference(
            source_vulnerability,
            source_collisions,
            source_near_misses,
            config.beta_prior_alpha,
            config.beta_prior_beta,
        )
        collisions = bank.collisions[target_index]
        near_misses = bank.near_misses[target_index]
        vulnerability = bank.vulnerability[target_index]

        shared_indices = highest_risk_indices(prior.mean, config.total_budget)
        rows.append(
            _row(
                SHARED_METHOD,
                target_name,
                0,
                _fixed_trace(shared_indices, reference, collisions, near_misses),
            )
        )
        novelty_scores = reference.mean_vulnerability * (
            1.0 - reference.smoothed_failure_prevalence
        )
        novelty_indices = highest_risk_indices(novelty_scores, config.total_budget)
        rows.append(
            _row(
                NOVELTY_METHOD,
                target_name,
                0,
                _fixed_trace(novelty_indices, reference, collisions, near_misses),
            )
        )
        rows.append(
            _row(
                PR_BRVT_METHOD,
                target_name,
                0,
                run_regression_mining(
                    prior,
                    reference,
                    vulnerability,
                    collisions,
                    near_misses,
                    config.total_budget,
                    config.critical_threshold,
                ),
            )
        )
        random_rng = np.random.default_rng(target_seeds[target_index])
        for repeat in range(config.random_support_repeats):
            indices = random_rng.choice(
                len(bank.anchors), size=config.total_budget, replace=False
            )
            rows.append(
                _row(
                    RANDOM_METHOD,
                    target_name,
                    repeat,
                    _fixed_trace(indices, reference, collisions, near_misses),
                )
            )
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    """Persist the reproducible LOSO replay table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], method: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["method"] == method]
    return float(np.mean(values))


def summarize_gate(rows: list[dict]) -> dict:
    """Evaluate Gate BRVT-1 without a raw-score retention constraint."""
    rcs_field = "regression_critical_score_at_20"
    tsf_field = "target_specific_failure_count_at_20"
    raw_field = "raw_critical_score_at_20"
    methods = (RANDOM_METHOD, SHARED_METHOD, NOVELTY_METHOD, PR_BRVT_METHOD)
    metrics = {
        method: {
            "mean_rcs_at_20": _mean(rows, method, rcs_field),
            "mean_tsf_at_20": _mean(rows, method, tsf_field),
            "mean_raw_critical_score_at_20": _mean(rows, method, raw_field),
        }
        for method in methods
    }
    novelty_by_target = {
        row["target_sut"]: float(row[rcs_field])
        for row in rows
        if row["method"] == NOVELTY_METHOD
    }
    proposed_by_target = {
        row["target_sut"]: float(row[rcs_field])
        for row in rows
        if row["method"] == PR_BRVT_METHOD
    }
    noninferior_targets = sum(
        proposed_by_target[target] >= novelty_score
        for target, novelty_score in novelty_by_target.items()
    )
    checks = {
        "proposed_rcs_exceeds_population_novelty": (
            metrics[PR_BRVT_METHOD]["mean_rcs_at_20"]
            > metrics[NOVELTY_METHOD]["mean_rcs_at_20"]
        ),
        "proposed_rcs_exceeds_shared": (
            metrics[PR_BRVT_METHOD]["mean_rcs_at_20"]
            > metrics[SHARED_METHOD]["mean_rcs_at_20"]
        ),
        "at_least_four_targets_noninferior_to_population_novelty": (
            noninferior_targets >= 4
        ),
        "proposed_mean_tsf_noninferior_to_population_novelty": (
            metrics[PR_BRVT_METHOD]["mean_tsf_at_20"]
            >= metrics[NOVELTY_METHOD]["mean_tsf_at_20"]
        ),
    }
    if not checks["proposed_rcs_exceeds_population_novelty"]:
        decision = "stop"
        reason = "Target Bayesian adaptation adds no RCS value over population novelty."
    elif not all(checks.values()):
        decision = "stop"
        reason = "Gate BRVT-1 was not met on the corrected E6 response bank."
    else:
        decision = "continue"
        reason = "Gate BRVT-1 passed."
    return {
        "gate": "BRVT-1",
        "metrics": metrics,
        "per_target_proposed_rcs_at_20": proposed_by_target,
        "per_target_population_novelty_rcs_at_20": novelty_by_target,
        "noninferior_target_count": noninferior_targets,
        "checks": checks,
        "decision": decision,
        "reason": reason,
    }


def write_summary(summary: dict, path: Path) -> None:
    """Write the gate outcome as machine-readable evidence."""
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def plot_curves(rows: list[dict], field: str, ylabel: str, output: Path) -> None:
    """Plot mean cumulative replay curves for the four compared methods."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(np.fromstring(row[field], sep=";"))
    figure, axis = plt.subplots(figsize=(7, 4))
    for method in (RANDOM_METHOD, SHARED_METHOD, NOVELTY_METHOD, PR_BRVT_METHOD):
        curves = np.vstack(grouped[method])
        budget = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        axis.plot(budget, mean, label=method)
        if len(curves) > 1:
            deviation = curves.std(axis=0)
            axis.fill_between(budget, mean - deviation, mean + deviation, alpha=0.14)
    axis.set(xlabel="Target test budget", ylabel=ylabel)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
