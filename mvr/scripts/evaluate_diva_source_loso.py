"""Offline G1 leave-one-source-out diagnostic evaluation for DIVA v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.special import ndtr
from scipy.stats import rankdata, spearmanr

from ..diva.acquisition import diagnostic_scores
from ..diva.factorization import fit_low_rank_vulnerability
from ..diva.posterior import LatentVulnerabilityPosterior
from ..diva.source_bank import SourceBank, observation_from_dict
from ..provenance import content_hash
from .analyze_diva_prior import analyze_bank
from .fit_diva_prior import _select_rank


METHODS = ("shared_prior", "random_support", "diagnostic_support", "highest_shared_risk")


def _load(path: str):
    return [
        observation_from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _candidate_schedule(shots: int, rng: np.random.Generator) -> tuple[int, ...]:
    if shots == 0:
        return ()
    if shots == 1:
        return (int(rng.integers(0, 2)),)
    if shots == 2:
        return tuple(rng.permutation((0, 1)))
    if shots == 4:
        return tuple(rng.permutation((0, 0, 1, 1)))
    raise ValueError("G1 supports only K=1, 2, or 4")


def _top(values: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-np.asarray(values), kind="mergesort")[: min(count, len(values))]


def _ndcg(actual: np.ndarray, predicted: np.ndarray, count: int = 8) -> float:
    selected = _top(predicted, count)
    gains = np.power(2.0, actual[selected]) - 1.0
    discount = 1.0 / np.log2(np.arange(2, len(selected) + 2))
    ideal = (np.power(2.0, actual[_top(actual, count)]) - 1.0) * discount
    return float(np.dot(gains, discount) / np.dot(ideal, np.ones_like(ideal))) if np.sum(ideal) > 0.0 else 0.0


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(labels)) != 2:
        return None
    positives = labels.astype(bool)
    ranks = rankdata(probabilities)
    count = int(positives.sum())
    return float((ranks[positives].sum() - count * (count + 1) / 2.0) / (count * (len(labels) - count)))


def _metrics(actual: np.ndarray, mean: np.ndarray, variance: np.ndarray, posterior: LatentVulnerabilityPosterior, threshold: float) -> dict:
    std = np.sqrt(np.maximum(variance, 1e-12))
    probability = ndtr((mean - threshold) / std)
    labels = actual >= threshold
    predicted_top = _top(mean, 8)
    actual_top = _top(actual, 8)
    rho = spearmanr(actual, mean).statistic if len(actual) > 1 else float("nan")
    sign, logdet = np.linalg.slogdet(posterior.covariance)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(actual - mean)))),
        "spearman": float(rho) if np.isfinite(rho) else None,
        "ndcg_at_8": _ndcg(actual, mean),
        "top8_recall": float(len(set(predicted_top) & set(actual_top)) / max(1, len(actual_top))),
        "brier": float(np.mean(np.square(probability - labels))),
        "auroc": _auc(labels, probability),
        "posterior_logdet": float(logdet) if sign > 0 else float("-inf"),
        "posterior_entropy_reduction": float(-0.5 * logdet) if sign > 0 else float("inf"),
    }


def _one_run(bank: SourceBank, held: int, rank: int, method: str, shots: int, seed: int, threshold: float, noise_var: float) -> dict:
    train_rows = [index for index in range(len(bank.source_refs)) if index != held]
    factor = fit_low_rank_vulnerability(
        bank.responses[train_rows], bank.eligible[train_rows],
        tuple(bank.source_refs[index] for index in train_rows), bank.design_ids, rank
    )
    common = factor.common_eligible_mask & bank.eligible[held]
    available = np.flatnonzero(common)
    posterior = LatentVulnerabilityPosterior.standard_normal(rank)
    rng = np.random.default_rng(seed)
    support: list[int] = []
    for candidate in _candidate_schedule(shots, rng):
        choices = np.asarray([index for index in available if index not in support], dtype=int)
        choices = choices[np.asarray([bank.designs[index].candidate_index == candidate for index in choices])]
        if method == "random_support":
            selected = int(rng.choice(choices))
        elif method == "highest_shared_risk":
            selected = int(choices[np.argmax(factor.mean[choices])])
        elif method == "diagnostic_support":
            scores = diagnostic_scores(
                posterior,
                factor.basis[choices],
                factor.mean[choices],
                np.full(len(choices), noise_var),
                np.ones(len(choices)),
                level_set_threshold=threshold,
            )
            selected = int(choices[np.argmax(scores)])
        else:
            raise ValueError("shared_prior does not select support")
        support.append(selected)
        posterior.update(
            factor.basis[selected],
            bank.responses[held, selected] - factor.mean[selected],
            noise_var,
        )
    query = np.asarray([index for index in available if index not in support], dtype=int)
    mean = factor.mean[query] + factor.basis[query] @ posterior.mean
    variance = np.einsum("ij,jk,ik->i", factor.basis[query], posterior.covariance, factor.basis[query]) + noise_var
    metrics = _metrics(bank.responses[held, query], mean, variance, posterior, threshold)
    return {
        "held_out_source": bank.source_refs[held],
        "method": method,
        "k": shots,
        "seed": seed,
        "support_design_ids": [bank.design_ids[index] for index in support],
        "query_count": int(len(query)),
        **metrics,
    }


def _aggregate(records: list[dict]) -> dict:
    numeric = ("rmse", "ndcg_at_8", "top8_recall", "brier", "posterior_logdet", "posterior_entropy_reduction")
    result = {name: float(np.mean([row[name] for row in records])) for name in numeric}
    for name in ("spearman", "auroc"):
        values = [row[name] for row in records if row[name] is not None]
        result[name] = float(np.mean(values)) if values else None
    return result


def _bootstrap_ci(deltas: np.ndarray, seed: int = 20260910) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(5000)])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def run(config_path: str, source_path: str, output_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    observations = _load(source_path)
    bank = SourceBank.from_observations(observations, tuple(config["study"]["source_sut_refs"]))
    analysis = analyze_bank(bank, config, config["study"]["logical_domain_id"])
    if not analysis["g0_structural_viability"]["pass"]:
        raise ValueError("G0 failed; G1 evaluation is prohibited")
    rank, selection = _select_rank(
        bank,
        tuple(config["prior"]["rank_candidates"]),
        float(config["prior"]["noise_floor_var"]),
        support_shots=int(config["prior"]["rank_selection_support_shots"]),
        seeds=tuple(config["prior"]["rank_selection_seeds"]),
    )
    threshold = float(config["vulnerability_response"]["proxy_event_threshold"])
    noise = float(config["prior"]["noise_floor_var"])
    seeds = tuple(int(seed) for seed in config["gates"]["g1"]["algorithm_seeds"])
    records: list[dict] = []
    for held in range(len(bank.source_refs)):
        for seed in seeds:
            records.append(_one_run(bank, held, rank, "shared_prior", 0, seed, threshold, noise))
            for shots in (1, 2, 4):
                for method in METHODS[1:]:
                    records.append(_one_run(bank, held, rank, method, shots, seed, threshold, noise))
    aggregates = {
        f"{method}_k{k}": _aggregate([row for row in records if row["method"] == method and row["k"] == k])
        for method in METHODS for k in ((0,) if method == "shared_prior" else (1, 2, 4))
    }
    primary = int(config["gates"]["g1"]["primary_k"])
    diagnostic = [row for row in records if row["method"] == "diagnostic_support" and row["k"] == primary]
    random = [row for row in records if row["method"] == "random_support" and row["k"] == primary]
    paired_delta = np.asarray([a["ndcg_at_8"] - b["ndcg_at_8"] for a, b in zip(diagnostic, random)])
    lower, upper = _bootstrap_ci(paired_delta)
    criteria = {
        "ndcg_gain_vs_k0": aggregates[f"diagnostic_support_k{primary}"]["ndcg_at_8"] - aggregates["shared_prior_k0"]["ndcg_at_8"] >= config["gates"]["g1"]["min_ndcg_gain_vs_k0"],
        "positive_paired_ndcg_ci": lower > 0.0 if config["gates"]["g1"]["require_positive_paired_ndcg_ci_vs_random"] else True,
        "rmse_non_degradation": aggregates[f"diagnostic_support_k{primary}"]["rmse"] <= (1.0 + config["gates"]["g1"]["max_rmse_relative_degradation_vs_random"]) * aggregates[f"random_support_k{primary}"]["rmse"],
    }
    report = {
        "schema": "diva_source_loso_g1_v2",
        "config_hash": content_hash(config),
        "logical_domain_id": config["study"]["logical_domain_id"],
        "selected_rank": rank,
        "rank_selection": selection,
        "aggregates": aggregates,
        "paired_diagnostic_vs_random_k4": {"mean_delta_ndcg": float(paired_delta.mean()), "ci95": [lower, upper]},
        "g1_source_only_diagnostic": {"pass": bool(all(criteria.values())), "criteria": criteria},
        "records": records,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/diva_cutin.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.source, args.output)


if __name__ == "__main__":
    main()
