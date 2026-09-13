"""Evaluate PR-DVM and its three fixed-budget LOSO baselines."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mvr.highway.config import ExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.diva.acquisition import highest_risk_indices
from mvr.highway.diva.differential_mining import (
    DifferentialMiningTrace,
    run_differential_mining,
)
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.outcomes import formal_rewards
from mvr.highway.diva.population_reference import (
    PopulationReference,
    build_population_reference,
)

CSV_FIELDS = (
    "method",
    "target_sut",
    "repeat",
    "raw_critical_score_at_20",
    "differential_critical_score_at_20",
    "target_specific_failure_count_at_20",
    "queried_indices",
    "raw_curve",
    "differential_curve",
    "target_specific_curve",
)
RANDOM_METHOD = "Random"
SHARED_METHOD = "Shared Prior"
NOVELTY_METHOD = "Population-Novelty Only"
PR_DVM_METHOD = "PR-DVM Sequential"


def _curve_text(curve: np.ndarray) -> str:
    return ";".join(f"{value:.6f}" for value in curve)


def _fixed_query_trace(
    queried_indices: np.ndarray,
    population: PopulationReference,
    collisions: np.ndarray,
    near_misses: np.ndarray,
) -> DifferentialMiningTrace:
    """Reveal a non-adaptive baseline sequence under the same scoring protocol."""
    queried = np.asarray(queried_indices, dtype=int)
    if len(np.unique(queried)) != len(queried):
        raise ValueError("a scenario may be queried only once")
    rewards = formal_rewards(collisions[queried], near_misses[queried])
    differential_rewards = rewards * (
        1.0 - population.smoothed_failure_prevalence[queried]
    )
    target_specific = (
        (rewards > 0.0) & (population.failure_prevalence[queried] <= 0.20)
    ).astype(float)
    return DifferentialMiningTrace(
        queried_indices=queried,
        raw_critical_curve=np.cumsum(rewards),
        differential_critical_curve=np.cumsum(differential_rewards),
        target_specific_failure_curve=np.cumsum(target_specific),
    )


def _row(method: str, target: str, repeat: int, trace: DifferentialMiningTrace) -> dict:
    return {
        "method": method,
        "target_sut": target,
        "repeat": repeat,
        "raw_critical_score_at_20": float(trace.raw_critical_curve[-1]),
        "differential_critical_score_at_20": float(
            trace.differential_critical_curve[-1]
        ),
        "target_specific_failure_count_at_20": float(
            trace.target_specific_failure_curve[-1]
        ),
        "queried_indices": ";".join(str(index) for index in trace.queried_indices),
        "raw_curve": _curve_text(trace.raw_critical_curve),
        "differential_curve": _curve_text(trace.differential_critical_curve),
        "target_specific_curve": _curve_text(trace.target_specific_failure_curve),
    }


def run_loso_differential_mining(
    bank: ResponseBank, config: ExperimentConfig
) -> list[dict]:
    """Run all methods with strict LOSO sources and target reveal after selection."""
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
        population = build_population_reference(
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
                _fixed_query_trace(
                    shared_indices, population, collisions, near_misses
                ),
            )
        )
        novelty_scores = population.mean_vulnerability * (
            1.0 - population.smoothed_failure_prevalence
        )
        novelty_indices = highest_risk_indices(novelty_scores, config.total_budget)
        rows.append(
            _row(
                NOVELTY_METHOD,
                target_name,
                0,
                _fixed_query_trace(
                    novelty_indices, population, collisions, near_misses
                ),
            )
        )
        rows.append(
            _row(
                PR_DVM_METHOD,
                target_name,
                0,
                run_differential_mining(
                    prior,
                    population,
                    vulnerability,
                    collisions,
                    near_misses,
                    config.total_budget,
                    config.differential_weight,
                    config.shared_weight,
                    config.uncertainty_weight,
                ),
            )
        )
        random_rng = np.random.default_rng(target_seeds[target_index])
        for repeat in range(config.random_support_repeats):
            random_indices = random_rng.choice(
                len(bank.anchors), size=config.total_budget, replace=False
            )
            rows.append(
                _row(
                    RANDOM_METHOD,
                    target_name,
                    repeat,
                    _fixed_query_trace(
                        random_indices, population, collisions, near_misses
                    ),
                )
            )
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    """Write the required flat replay artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], method: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["method"] == method]
    return float(np.mean(values))


def summarize_gate(rows: list[dict]) -> dict:
    """Evaluate the single pre-registered PR-DVM-1 gate."""
    dcs_field = "differential_critical_score_at_20"
    tsf_field = "target_specific_failure_count_at_20"
    raw_field = "raw_critical_score_at_20"
    metrics = {
        method: {
            "mean_dcs_at_20": _mean(rows, method, dcs_field),
            "mean_tsf_at_20": _mean(rows, method, tsf_field),
            "mean_raw_critical_score_at_20": _mean(rows, method, raw_field),
        }
        for method in (RANDOM_METHOD, SHARED_METHOD, NOVELTY_METHOD, PR_DVM_METHOD)
    }
    shared_by_target = {
        row["target_sut"]: float(row[dcs_field])
        for row in rows
        if row["method"] == SHARED_METHOD
    }
    pr_dvm_by_target = {
        row["target_sut"]: float(row[dcs_field])
        for row in rows
        if row["method"] == PR_DVM_METHOD
    }
    noninferior_targets = sum(
        pr_dvm_by_target[target] >= shared_score
        for target, shared_score in shared_by_target.items()
    )
    checks = {
        "pr_dvm_dcs_exceeds_shared": (
            metrics[PR_DVM_METHOD]["mean_dcs_at_20"]
            > metrics[SHARED_METHOD]["mean_dcs_at_20"]
        ),
        "pr_dvm_dcs_exceeds_population_novelty": (
            metrics[PR_DVM_METHOD]["mean_dcs_at_20"]
            > metrics[NOVELTY_METHOD]["mean_dcs_at_20"]
        ),
        "at_least_four_targets_noninferior_to_shared": noninferior_targets >= 4,
        "pr_dvm_mean_tsf_noninferior_to_shared": (
            metrics[PR_DVM_METHOD]["mean_tsf_at_20"]
            >= metrics[SHARED_METHOD]["mean_tsf_at_20"]
        ),
        "pr_dvm_raw_score_at_least_90_percent_shared": (
            metrics[PR_DVM_METHOD]["mean_raw_critical_score_at_20"]
            >= 0.90 * metrics[SHARED_METHOD]["mean_raw_critical_score_at_20"]
        ),
    }
    if not checks["pr_dvm_dcs_exceeds_population_novelty"]:
        decision = "stop"
        reason = (
            "PR-DVM posterior adaptation adds no DCS value over population novelty."
        )
    elif not all(checks.values()):
        decision = "stop"
        reason = "Gate PR-DVM-1 was not met on the corrected E6 response bank."
    else:
        decision = "continue"
        reason = "Gate PR-DVM-1 passed."
    return {
        "gate": "PR-DVM-1",
        "metrics": metrics,
        "per_target_pr_dvm_dcs_at_20": pr_dvm_by_target,
        "per_target_shared_dcs_at_20": shared_by_target,
        "noninferior_target_count": noninferior_targets,
        "checks": checks,
        "decision": decision,
        "reason": reason,
    }


def write_summary(summary: dict, path: Path) -> None:
    """Persist the gate decision in a machine-readable artifact."""
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def plot_curves(rows: list[dict], field: str, ylabel: str, output: Path) -> None:
    """Plot mean cumulative curves for each method in the replay artifact."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(np.fromstring(row[field], sep=";"))
    figure, axis = plt.subplots(figsize=(7, 4))
    for method in (RANDOM_METHOD, SHARED_METHOD, NOVELTY_METHOD, PR_DVM_METHOD):
        curves = np.vstack(grouped[method])
        x_axis = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        axis.plot(x_axis, mean, label=method)
        if len(curves) > 1:
            standard_deviation = curves.std(axis=0)
            axis.fill_between(
                x_axis,
                mean - standard_deviation,
                mean + standard_deviation,
                alpha=0.14,
            )
    axis.set(xlabel="Target test budget", ylabel=ylabel)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
