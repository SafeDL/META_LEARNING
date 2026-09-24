"""Run aligned and physical function-shift routing benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from highway_env_benchmark.data.generate_anchor_bank import (
    FUNCTIONAL_MODES,
    generate_multifunction_anchor_bank,
)
from highway_env_benchmark.data.response_bank import ResponseBank, build_response_bank
from highway_env_benchmark.mining.low_rank_prior import LowRankPrior
from highway_env_benchmark.mining.mining import _trace, diagnostic_mining, random_mining
from method_chains.detour_fusion.fusion import (
    detour_guided_diagnostic_mining,
    detour_static_mining,
    hierarchy_prior,
)
from replications.adate_highway_env.adate.mixture_selector import MixtureSelector
from replications.adate_highway_env.adate.mixture import simplex_least_squares
from replications.detour_highway_env.detour.features import encode_scenarios
from replications.detour_highway_env.detour.history import scenario_specs, source_history

from .benchmark import (
    build_functional_release_bank,
    release_manifest,
)
from .config import RoutingExperimentConfig
from .routing import RoutedMiningResult, functional_routed_mining


OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "method_chains"
    / "function_conditioned_routing"
)
ALIGNED_OUTPUT = OUTPUT_DIR / "aligned_benchmark"
FUNCTIONAL_OUTPUT = OUTPUT_DIR / "functional_shift_benchmark"
CONFIG = RoutingExperimentConfig()
BUDGETS = (10, 20, 30, 50)

METHOD_ORDER = (
    "Random",
    "DETOUR",
    "Mining",
    "Mining-DETOUR",
    "AdaTE Global",
    "Global Routed Mining",
    "Function Routing (risk-only)",
    "Function-Conditioned Mining",
)
KEY_METHODS = (
    "Random",
    "DETOUR",
    "Mining",
    "Mining-DETOUR",
    "AdaTE Global",
    "Function-Conditioned Mining",
)
METHOD_COLORS = {
    "Random": "#8c8c8c",
    "DETOUR": "#e07a2f",
    "Mining": "#4c78a8",
    "Mining-DETOUR": "#59a14f",
    "AdaTE Global": "#f2cf5b",
    "Global Routed Mining": "#b279a2",
    "Function Routing (risk-only)": "#ff9da6",
    "Function-Conditioned Mining": "#d62728",
}


def severity_response(bank: ResponseBank, indices: list[int]) -> np.ndarray:
    """Use trajectory severity for adaptation without changing event labels."""
    vulnerability = np.asarray(bank.vulnerability[indices], dtype=float)
    collisions = np.asarray(bank.collisions[indices], dtype=bool)
    ttc = np.asarray(bank.min_ttc[indices], dtype=float)
    ttc_signal = np.zeros_like(ttc)
    finite = np.isfinite(ttc)
    ttc_signal[finite] = np.exp(-np.clip(ttc[finite], 0.0, 20.0) / 3.0)
    response = 0.85 * vulnerability
    response[collisions] = 0.85 + 0.15 * ttc_signal[collisions]
    return np.clip(response, 0.0, 1.0)


def build_benchmark_bank(config: RoutingExperimentConfig = CONFIG) -> ResponseBank:
    """Build the same six-SUT, 240-scenario bank as Mining-DETOUR."""
    anchors, modes = generate_multifunction_anchor_bank(config.num_anchors, config.seed)
    return build_response_bank(anchors, config.seed, modes)


def _adate_trace(
    sources: np.ndarray,
    target: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    budget: int,
):
    selector = MixtureSelector(sources, budget, "sequential")
    while True:
        chosen = selector.propose()
        if chosen is None:
            break
        selector.observe(chosen, float(target[chosen]))
    return _trace(
        "AdaTE Global",
        np.asarray(selector.selected, dtype=int),
        collisions,
        near_misses,
    )


def _recall_curve(queried: np.ndarray, critical: np.ndarray) -> np.ndarray:
    total = int(critical.sum())
    return np.cumsum(critical[queried]) / total if total else np.zeros(len(queried))


def _result_row(
    target: str,
    trace,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    repeat: int = 0,
) -> dict[str, object]:
    queried = np.asarray(trace.queried_indices, dtype=int)
    critical = collisions | near_misses
    curve = _recall_curve(queried, critical)
    row: dict[str, object] = {
        "method": trace.method,
        "target_sut": target,
        "repeat": repeat,
        "critical_events_available": int(critical.sum()),
        "collision_events_available": int(collisions.sum()),
        "critical_events_found": int(critical[queried].sum()),
        "collision_events_found": int(collisions[queried].sum()),
        "queried_indices": ";".join(str(index) for index in queried),
        "critical_recall_curve": ";".join(f"{value:.8f}" for value in curve),
    }
    for budget in BUDGETS:
        row[f"critical_recall_at_{budget}"] = float(curve[budget - 1])
        row[f"critical_count_at_{budget}"] = int(critical[queried[:budget]].sum())
    return row


def _diagnostic_row(
    target: str,
    method: str,
    result: RoutedMiningResult,
    truth: np.ndarray,
) -> dict[str, object]:
    support = result.support_indices
    unseen = np.ones(len(truth), dtype=bool)
    unseen[support] = False
    state = result.support_prediction
    return {
        "target_sut": target,
        "method": method,
        "support_indices": ";".join(str(index) for index in support),
        "held_out_mse": float(np.mean((state.prediction[unseen] - truth[unseen]) ** 2)),
        "mining_held_out_mse": float(
            np.mean((state.mining_prediction[unseen] - truth[unseen]) ** 2)
        ),
        "routed_held_out_mse": float(
            np.mean((state.routed_prediction[unseen] - truth[unseen]) ** 2)
        ),
        "revealed_rmse": state.revealed_rmse,
        "global_alpha": json.dumps(state.global_alpha.tolist()),
        "mode_alphas": json.dumps(
            {mode: alpha.tolist() for mode, alpha in state.mode_alphas.items()},
            sort_keys=True,
        ),
        "mode_gates": json.dumps(state.mode_gates, sort_keys=True),
    }


def run_benchmark(
    bank: ResponseBank,
    config: RoutingExperimentConfig = CONFIG,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run identical LOSO splits for every baseline and routing ablation."""
    config.validate()
    if len(bank.sut_names) != 6 or len(bank.anchors) != config.num_anchors:
        raise ValueError("response bank must contain six SUTs and the frozen scenarios")
    labels = np.asarray(bank.modes, dtype=str)
    features = encode_scenarios(scenario_specs(bank))
    responses = severity_response(bank, list(range(len(bank.sut_names))))
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    random_seeds = np.random.SeedSequence(config.seed).spawn(len(bank.sut_names))
    for target_order, target_name in enumerate(bank.sut_names):
        target_index = bank.index_of(target_name)
        source_names = tuple(name for name in bank.sut_names if name != target_name)
        source_indices = [bank.index_of(name) for name in source_names]
        sources = responses[source_indices]
        truth = responses[target_index]
        collisions = bank.collisions[target_index]
        near_misses = bank.near_misses[target_index]
        prior = LowRankPrior.fit(sources, config.prior_rank)
        history = source_history(bank, target_name, "collision", source_names)
        hierarchy = hierarchy_prior(
            encode_scenarios(tuple(item.scenario for item in history)),
            features,
            np.asarray([item.failed for item in history], dtype=bool),
            config.seed + target_order,
        )
        traces = [
            detour_static_mining(hierarchy, collisions, near_misses, config.total_budget),
            diagnostic_mining(
                prior,
                truth,
                collisions,
                near_misses,
                config.support_budget,
                config.total_budget,
            ),
            detour_guided_diagnostic_mining(
                prior,
                hierarchy,
                truth,
                collisions,
                near_misses,
                config.support_budget,
                config.total_budget,
                config.hierarchy_weight,
            ),
            _adate_trace(sources, truth, collisions, near_misses, config.total_budget),
        ]
        for index, name in enumerate(("DETOUR", "Mining", "Mining-DETOUR")):
            traces[index] = _trace(
                name,
                traces[index].queried_indices,
                collisions,
                near_misses,
            )
        routed_specs = (
            ("Global Routed Mining", False, True, "adate"),
            ("Function Routing (risk-only)", True, False, "adate"),
            ("Function-Conditioned Mining", True, True, "coverage"),
        )
        routed_results = []
        for method, functional, trusted, support_policy in routed_specs:
            result = functional_routed_mining(
                method,
                sources,
                labels,
                prior,
                truth,
                collisions,
                near_misses,
                config,
                functional=functional,
                trusted=trusted,
                support_policy=support_policy,
            )
            routed_results.append(result)
            diagnostics.append(_diagnostic_row(target_name, method, result, truth))
        rows.extend(_result_row(target_name, trace, collisions, near_misses) for trace in traces)
        rows.extend(
            _result_row(target_name, result.trace, collisions, near_misses)
            for result in routed_results
        )
        rng = np.random.default_rng(random_seeds[target_order])
        for repeat in range(config.random_repeats):
            rows.append(
                _result_row(
                    target_name,
                    random_mining(collisions, near_misses, config.total_budget, rng),
                    collisions,
                    near_misses,
                    repeat,
                )
            )
    return rows, diagnostics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    bank: ResponseBank,
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    config: RoutingExperimentConfig = CONFIG,
) -> dict[str, object]:
    metrics = {}
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        metrics[method] = {
            f"mean_critical_recall_at_{budget}": float(
                np.mean([float(row[f"critical_recall_at_{budget}"]) for row in selected])
            )
            for budget in BUDGETS
        }
    per_target = {
        target: {
            method: {
                f"critical_recall_at_{budget}": float(
                    next(
                        row[f"critical_recall_at_{budget}"]
                        for row in rows
                        if row["target_sut"] == target
                        and row["method"] == method
                        and int(row["repeat"]) == 0
                    )
                )
                for budget in (20, 50)
            }
            for method in METHOD_ORDER
            if method != "Random"
        }
        for target in bank.sut_names
    }
    proposed = [
        row for row in diagnostics if row["method"] == "Function-Conditioned Mining"
    ]
    return {
        "protocol": {
            "task": "LOSO multi-function critical-event mining",
            "alignment_reference": "method_chains/detour_fusion",
            "scenario_count": config.num_anchors,
            "sut_names": list(bank.sut_names),
            "source_suts_per_fold": len(bank.sut_names) - 1,
            "functional_modes": list(FUNCTIONAL_MODES),
            "balanced_count_per_mode": config.num_anchors // len(FUNCTIONAL_MODES),
            "diagnostic_budget": config.support_budget,
            "total_budget": config.total_budget,
            "random_repeats": config.random_repeats,
            "target_truth_visibility": "selected reveals only; offline scoring",
            "adaptation_response": "continuous trajectory severity; event labels unchanged",
        },
        "event_prevalence": {
            name: int(
                (bank.collisions[index] | bank.near_misses[index]).sum()
            )
            for index, name in enumerate(bank.sut_names)
        },
        "mean_metrics": metrics,
        "per_target": per_target,
        "proposed_support_prediction": {
            "mean_held_out_mse": float(
                np.mean([float(row["held_out_mse"]) for row in proposed])
            ),
            "mean_mining_held_out_mse_on_same_support": float(
                np.mean([float(row["mining_held_out_mse"]) for row in proposed])
            ),
            "mean_routed_held_out_mse": float(
                np.mean([float(row["routed_held_out_mse"]) for row in proposed])
            ),
        },
    }


def oracle_headroom(bank: ResponseBank) -> dict[str, object]:
    """Measure whether function-specific transfer can change the top-B ranking."""
    labels = np.asarray(bank.modes, dtype=str)
    responses = severity_response(bank, list(range(len(bank.sut_names))))
    critical = bank.collisions | bank.near_misses
    per_target = {}
    for target_index, target_name in enumerate(bank.sut_names):
        source_indices = [
            index for index in range(len(bank.sut_names)) if index != target_index
        ]
        sources = responses[source_indices]
        truth = responses[target_index]
        global_alpha = simplex_least_squares(sources.T, truth).alpha
        global_prediction = global_alpha @ sources
        function_prediction = np.empty_like(truth)
        for mode in dict.fromkeys(labels):
            mask = labels == mode
            alpha = simplex_least_squares(sources[:, mask].T, truth[mask]).alpha
            function_prediction[mask] = alpha @ sources[:, mask]
        target_critical = critical[target_index]
        available = int(target_critical.sum())
        target_metrics = {
            "global_mse": float(np.mean((global_prediction - truth) ** 2)),
            "function_mse": float(np.mean((function_prediction - truth) ** 2)),
        }
        for budget in BUDGETS:
            global_indices = np.argsort(-global_prediction, kind="stable")[:budget]
            function_indices = np.argsort(-function_prediction, kind="stable")[:budget]
            target_metrics[f"global_recall_at_{budget}"] = float(
                target_critical[global_indices].sum() / available
            )
            target_metrics[f"function_recall_at_{budget}"] = float(
                target_critical[function_indices].sum() / available
            )
        per_target[target_name] = target_metrics
    mean_metrics = {
        key: float(np.mean([metrics[key] for metrics in per_target.values()]))
        for key in next(iter(per_target.values()))
    }
    mean_metrics["function_gain_at_50"] = (
        mean_metrics["function_recall_at_50"]
        - mean_metrics["global_recall_at_50"]
    )
    return {"mean_metrics": mean_metrics, "per_target": per_target}


def _mean_curve(rows: list[dict[str, object]], method: str) -> np.ndarray:
    curves = [
        np.asarray(str(row["critical_recall_curve"]).split(";"), dtype=float)
        for row in rows
        if row["method"] == method
    ]
    return np.mean(np.vstack(curves), axis=0)


def write_figures(
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    output: Path,
    config: RoutingExperimentConfig = CONFIG,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, config.total_budget + 1)
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for method in KEY_METHODS:
        axis.plot(
            steps,
            _mean_curve(rows, method),
            label=method,
            color=METHOD_COLORS[method],
            linewidth=2.4 if method == "Function-Conditioned Mining" else 1.7,
        )
    axis.set(
        xlabel="Target executions consumed",
        ylabel="Mean critical-event recall",
        title="Function-conditioned routing under fixed-budget LOSO",
        xlim=(1, config.total_budget),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False, ncol=2)
    figure.savefig(output / "critical_recall_curve.png", dpi=180)
    plt.close(figure)

    methods = (
        "Mining",
        "Mining-DETOUR",
        "AdaTE Global",
        "Function-Conditioned Mining",
    )
    targets = tuple(dict.fromkeys(str(row["target_sut"]) for row in rows))
    positions = np.arange(len(targets))
    width = 0.19
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), constrained_layout=True)
    for axis, budget in zip(axes, (20, 50)):
        for method_index, method in enumerate(methods):
            values = [
                float(
                    next(
                        row[f"critical_recall_at_{budget}"]
                        for row in rows
                        if row["target_sut"] == target
                        and row["method"] == method
                        and int(row["repeat"]) == 0
                    )
                )
                for target in targets
            ]
            axis.bar(
                positions + (method_index - (len(methods) - 1) / 2) * width,
                values,
                width,
                label=method,
                color=METHOD_COLORS[method],
            )
        axis.set(
            xticks=positions,
            xticklabels=targets,
            ylabel=f"Recall at B={budget}",
            title=f"Held-out SUT performance after {budget} executions",
            ylim=(0, 1.02),
        )
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Held-out target SUT")
    axes[0].legend(fontsize=8, frameon=False, ncol=3)
    figure.savefig(output / "target_recall_by_budget.png", dpi=180)
    plt.close(figure)

    proposed = [
        row for row in diagnostics if row["method"] == "Function-Conditioned Mining"
    ]
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(proposed))
    axis.bar(
        x - 0.18,
        [float(row["mining_held_out_mse"]) for row in proposed],
        0.36,
        label="Mining prediction",
        color=METHOD_COLORS["Mining"],
    )
    axis.bar(
        x + 0.18,
        [float(row["held_out_mse"]) for row in proposed],
        0.36,
        label="Trusted routed prediction",
        color=METHOD_COLORS["Function-Conditioned Mining"],
    )
    axis.set(
        xticks=x,
        xticklabels=[str(row["target_sut"]) for row in proposed],
        ylabel="Held-out severity-response MSE after K=10",
        title="Prediction quality on unqueried scenarios",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output / "held_out_prediction_error.png", dpi=180)
    plt.close(figure)


def write_manifest(
    bank: ResponseBank,
    output: Path,
    benchmark: str,
    design: dict[str, object] | None = None,
    config: RoutingExperimentConfig = CONFIG,
) -> None:
    content = {
        "benchmark": benchmark,
        "seed": config.seed,
        "sut_names": list(bank.sut_names),
        "anchor_count": len(bank.anchors),
        "functional_modes": list(FUNCTIONAL_MODES),
        "balanced_count_per_mode": len(bank.anchors) // len(FUNCTIONAL_MODES),
        "loso_source_count": len(bank.sut_names) - 1,
        "scenario_controls": (
            "timing and intensity" if bank.scenario_controls is not None else None
        ),
        "config": config.__dict__,
        "design": design,
    }
    output.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def _run_and_write(
    bank: ResponseBank,
    output: Path,
    benchmark: str,
    design: dict[str, object] | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    bank.save(output / "response_bank.npz")
    rows, diagnostics = run_benchmark(bank)
    _write_csv(output / "mining_results.csv", rows)
    _write_csv(output / "routing_diagnostics.csv", diagnostics)
    summary = summarize(bank, rows, diagnostics)
    summary["protocol"]["benchmark"] = benchmark
    oracle = oracle_headroom(bank)
    proposed = summary["mean_metrics"]["Function-Conditioned Mining"]
    adate = summary["mean_metrics"]["AdaTE Global"]
    summary["validation"] = {
        "oracle_function_gain_at_50": oracle["mean_metrics"]["function_gain_at_50"],
        "proposed_gain_over_adate_at_50": (
            proposed["mean_critical_recall_at_50"]
            - adate["mean_critical_recall_at_50"]
        ),
        "gap_to_function_oracle_at_50": (
            oracle["mean_metrics"]["function_recall_at_50"]
            - proposed["mean_critical_recall_at_50"]
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "oracle_headroom.json").write_text(
        json.dumps(oracle, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(bank, output / "benchmark_manifest.json", benchmark, design)
    write_figures(rows, diagnostics, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reuse-bank", action="store_true")
    args = parser.parse_args()
    CONFIG.validate()
    aligned_output = args.output_dir / ALIGNED_OUTPUT.name
    functional_output = args.output_dir / FUNCTIONAL_OUTPUT.name
    if args.reuse_bank:
        aligned_bank = ResponseBank.load(aligned_output / "response_bank.npz")
        functional_bank = ResponseBank.load(functional_output / "response_bank.npz")
    else:
        aligned_bank = build_benchmark_bank()
        functional_bank = build_functional_release_bank(CONFIG.num_anchors, CONFIG.seed)
    summaries = {
        "aligned_benchmark": _run_and_write(
            aligned_bank,
            aligned_output,
            "aligned six-SUT reference",
            {"alignment_reference": "method_chains/detour_fusion"},
        ),
        "functional_shift_benchmark": _run_and_write(
            functional_bank,
            functional_output,
            "physical function-shift releases",
            release_manifest(),
        ),
    }
    print(
        json.dumps(
            {
                name: summary["validation"]
                for name, summary in summaries.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
