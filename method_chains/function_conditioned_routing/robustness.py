"""Run fixed-seed robustness checks for the physical function-shift benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from highway_env_benchmark.data.response_bank import ResponseBank

from .benchmark import build_functional_release_bank
from .config import RoutingExperimentConfig
from .experiment import FUNCTIONAL_OUTPUT, oracle_headroom, run_benchmark, summarize


ROBUSTNESS_SEEDS = (20260914, 20261011, 20261017)
METHODS = (
    "Mining",
    "Mining-DETOUR",
    "AdaTE Global",
    "Function Routing (risk-only)",
    "Function-Conditioned Mining",
)


def run_robustness(output: Path = FUNCTIONAL_OUTPUT) -> dict[str, object]:
    """Evaluate the frozen method on one development and two confirmation seeds."""
    seed_results = []
    for seed in ROBUSTNESS_SEEDS:
        config = RoutingExperimentConfig(seed=seed)
        if seed == RoutingExperimentConfig().seed:
            bank = ResponseBank.load(output / "response_bank.npz")
        else:
            bank = build_functional_release_bank(config.num_anchors, seed)
        rows, diagnostics = run_benchmark(bank, config)
        summary = summarize(bank, rows, diagnostics, config)
        oracle = oracle_headroom(bank)
        metrics = {
            method: summary["mean_metrics"][method]["mean_critical_recall_at_50"]
            for method in METHODS
        }
        metrics["Function Oracle"] = oracle["mean_metrics"]["function_recall_at_50"]
        seed_results.append({"seed": seed, "recall_at_50": metrics})
    aggregate = {}
    for method in METHODS + ("Function Oracle",):
        values = np.asarray(
            [row["recall_at_50"][method] for row in seed_results],
            dtype=float,
        )
        aggregate[method] = {
            "mean_recall_at_50": float(values.mean()),
            "std_recall_at_50": float(values.std(ddof=1)),
        }
    gains = np.asarray(
        [
            row["recall_at_50"]["Function-Conditioned Mining"]
            - row["recall_at_50"]["AdaTE Global"]
            for row in seed_results
        ],
        dtype=float,
    )
    result = {
        "seeds": list(ROBUSTNESS_SEEDS),
        "seed_results": seed_results,
        "aggregate": aggregate,
        "proposed_gain_over_adate": {
            "mean": float(gains.mean()),
            "std": float(gains.std(ddof=1)),
            "wins": int(np.sum(gains > 0.0)),
            "total": len(gains),
        },
    }
    (output / "robustness_summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run_robustness(), indent=2))
