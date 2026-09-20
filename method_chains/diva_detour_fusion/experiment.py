"""Run the pre-registered B=50 multi-function DETOUR/DIVA comparison.

The evaluation is leave-one-SUT-out: a target SUT's outcomes are hidden until
a method spends one of its 50 tests.  The only source of transfer labels is
the other five SUTs.  The comparison deliberately contains no RSS or
Shared-Prior arm; its purpose is to separate the original DETOUR hierarchy
from target-specific DIVA diagnosis and their two specified combinations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from method_chains.diva_detour_fusion.config import FusionExperimentConfig
from diva_highway_env.data.generate_anchor_bank import (
    FUNCTIONAL_MODES,
    generate_multifunction_anchor_bank,
)
from diva_highway_env.data.response_bank import ResponseBank, build_response_bank
from method_chains.diva_detour_fusion.fusion import (
    detour_fused_diagnostic_mining,
    detour_guided_diagnostic_mining,
    detour_static_mining,
    hierarchy_prior,
)
from diva_highway_env.diva.low_rank_prior import LowRankPrior
from diva_highway_env.diva.mining import diagnostic_mining, random_mining
from replications.detour_highway_env.detour.features import encode_scenarios
from replications.detour_highway_env.detour.history import scenario_specs, source_history

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "method_chains"
    / "diva_detour_fusion"
)
CONFIG = FusionExperimentConfig(
    num_anchors=240,
    prior_rank=2,
    support_budget=10,
    total_budget=50,
    random_support_repeats=20,
    seed=20260914,
    detour_hierarchy_weight=0.10,
)

METHOD_ORDER = (
    "Random",
    "DETOUR Hierarchy (K=0)",
    "DIVA Diagnostic + Adaptation",
    "DIVA Diagnostic + DETOUR Hierarchy",
    "DIVA DETOUR-Guided Diagnosis",
)
METHOD_COLORS = {
    "Random": "#8c8c8c",
    "DETOUR Hierarchy (K=0)": "#e07a2f",
    "DIVA Diagnostic + Adaptation": "#4c78a8",
    "DIVA Diagnostic + DETOUR Hierarchy": "#59a14f",
    "DIVA DETOUR-Guided Diagnosis": "#b279a2",
}


def _recall(selected: np.ndarray, event_mask: np.ndarray) -> float:
    total = int(event_mask.sum())
    return float(event_mask[selected].sum() / total) if total else 0.0


def _row(target: str, repeat: int, trace, collisions: np.ndarray,
         near_misses: np.ndarray) -> dict[str, object]:
    selected = np.asarray(trace.queried_indices, dtype=int)
    critical = np.asarray(collisions | near_misses, dtype=bool)
    return {
        "method":
        trace.method,
        "target_sut":
        target,
        "repeat":
        repeat,
        "critical_score_at_50":
        trace.critical_score,
        "collision_count_at_50":
        trace.collision_count,
        "critical_event_count_at_50":
        trace.failure_count,
        "critical_recall_at_50":
        _recall(selected, critical),
        "collision_recall_at_50":
        _recall(selected, collisions),
        "queried_indices":
        ";".join(str(index) for index in selected),
        "critical_score_curve":
        ";".join(f"{value:.3f}" for value in trace.cumulative_critical_score),
        "critical_recall_curve":
        ";".join(f"{_recall(selected[:index], critical):.6f}"
                 for index in range(1,
                                    len(selected) + 1)),
    }


def run_loso_multifunction(bank: ResponseBank,
                           config: FusionExperimentConfig = CONFIG,
                           target_names: tuple[str, ...] | None = None,
                           source_names: tuple[str, ...] | None = None) -> list[dict[str, object]]:
    """Evaluate only paper-grounded DETOUR/DIVA methods under one B=50 budget."""
    config.validate()
    if len(bank.anchors) != config.num_anchors:
        raise ValueError("response bank anchor count does not match configuration")
    specs = scenario_specs(bank)
    candidate_features = encode_scenarios(specs)
    rows: list[dict[str, object]] = []
    targets = bank.sut_names if target_names is None else target_names
    if not targets or not set(targets).issubset(bank.sut_names):
        raise ValueError("target_names must be non-empty members of the response bank")
    seeds = np.random.SeedSequence(config.seed).spawn(len(targets))
    for target_order, target_name in enumerate(targets):
        target_index = bank.index_of(target_name)
        selected_sources = (
            tuple(name for name in bank.sut_names if name != target_name)
            if source_names is None else source_names
        )
        source = source_history(bank, target_name, "collision", selected_sources)
        hierarchy = hierarchy_prior(
            encode_scenarios(tuple(item.scenario for item in source)),
            candidate_features,
            np.asarray([item.failed for item in source], dtype=bool),
            config.seed + target_order,
        )
        source_indices = [bank.index_of(name) for name in selected_sources]
        prior = LowRankPrior.fit(bank.vulnerability[source_indices], config.prior_rank)
        vulnerability = bank.vulnerability[target_index]
        collisions, near_misses = bank.collisions[target_index], bank.near_misses[target_index]
        traces = (
            detour_static_mining(hierarchy, collisions, near_misses, config.total_budget),
            diagnostic_mining(
                prior,
                vulnerability,
                collisions,
                near_misses,
                config.support_budget,
                config.total_budget,
            ),
            detour_fused_diagnostic_mining(
                prior,
                hierarchy,
                vulnerability,
                collisions,
                near_misses,
                config.support_budget,
                config.total_budget,
                config.detour_hierarchy_weight,
            ),
            detour_guided_diagnostic_mining(
                prior,
                hierarchy,
                vulnerability,
                collisions,
                near_misses,
                config.support_budget,
                config.total_budget,
                config.detour_hierarchy_weight,
            ),
        )
        rows.extend(_row(target_name, 0, trace, collisions, near_misses) for trace in traces)
        rng = np.random.default_rng(seeds[target_order])
        for repeat in range(config.random_support_repeats):
            rows.append(
                _row(
                    target_name,
                    repeat,
                    random_mining(collisions, near_misses, config.total_budget, rng),
                    collisions,
                    near_misses,
                ))
    return rows


def mode_discovery_rows(bank: ResponseBank, rows: list[dict[str,
                                                            object]]) -> list[dict[str, object]]:
    """Break final discovery into fixed functional modes without re-querying data."""
    modes = np.asarray(bank.modes if bank.modes is not None else ["fast_intrusion"] *
                       len(bank.anchors))
    output: list[dict[str, object]] = []
    for row in rows:
        target_index = bank.index_of(str(row["target_sut"]))
        selected = np.asarray([int(value) for value in str(row["queried_indices"]).split(";")])
        critical = bank.collisions[target_index] | bank.near_misses[target_index]
        for mode in FUNCTIONAL_MODES:
            in_mode = modes == mode
            output.append({
                "method":
                row["method"],
                "target_sut":
                row["target_sut"],
                "repeat":
                row["repeat"],
                "mode":
                mode,
                "queries":
                int(in_mode[selected].sum()),
                "critical_events_found":
                int(critical[selected][in_mode[selected]].sum()),
                "critical_events_available":
                int(critical[in_mode].sum()),
                "critical_recall_at_50":
                (float(critical[selected][in_mode[selected]].sum() /
                       critical[in_mode].sum()) if critical[in_mode].sum() else 0.0),
            })
    return output


def _mean_curve(rows: list[dict[str, object]], method: str, column: str) -> np.ndarray:
    curves = [
        np.asarray(str(row[column]).split(";"), dtype=float) for row in rows
        if row["method"] == method
    ]
    return np.mean(np.vstack(curves), axis=0)


def summarize(bank: ResponseBank,
              rows: list[dict[str, object]],
              mode_rows: list[dict[str, object]],
              config: FusionExperimentConfig = CONFIG) -> dict[str, object]:
    """Summarize target-hidden test performance and functional-mode coverage."""
    method_metrics = {}
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        method_metrics[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "critical_score_at_50",
                "collision_count_at_50",
                "critical_event_count_at_50",
                "critical_recall_at_50",
                "collision_recall_at_50",
            )
        }
    per_mode = {}
    for method in METHOD_ORDER:
        per_mode[method] = {}
        for mode in FUNCTIONAL_MODES:
            selected = [
                row for row in mode_rows if row["method"] == method and row["mode"] == mode
            ]
            per_mode[method][mode] = {
                "mean_queries":
                float(np.mean([float(row["queries"]) for row in selected])),
                "mean_critical_recall_at_50":
                float(np.mean([float(row["critical_recall_at_50"]) for row in selected])),
            }
    return {
        "protocol": {
            "task": "LOSO multi-function critical-event mining",
            "scenario_count": config.num_anchors,
            "functional_modes": list(FUNCTIONAL_MODES),
            "total_budget": config.total_budget,
            "diagnostic_budget": config.support_budget,
            "random_repeats": config.random_support_repeats,
            "detour_hierarchy_weight": config.detour_hierarchy_weight,
            "target_truth_visibility": "support/query reveal only; offline scoring",
            "excluded_methods": ["RSS", "Shared-Prior", "Shared-Risk"],
        },
        "event_prevalence": {
            name: {
                "collisions": int(bank.collisions[index].sum()),
                "critical_events": int((bank.collisions[index] | bank.near_misses[index]).sum()),
            }
            for index, name in enumerate(bank.sut_names)
        },
        "mean_metrics": method_metrics,
        "mean_mode_coverage": per_mode,
    }


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(rows: list[dict[str, object]], mode_rows: list[dict[str, object]],
                  output_dir: Path) -> None:
    """Write curves and per-function recall figures for the final artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, len(str(rows[0]["critical_score_curve"]).split(";")) + 1)
    for column, title, ylabel, filename in (
        ("critical_score_curve", "Critical-event discovery under B=50", "Mean critical score",
         "01_critical_score_curve.png"),
        ("critical_recall_curve", "Critical-event recall under B=50", "Mean critical-event recall",
         "02_critical_recall_curve.png"),
    ):
        figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        for method in METHOD_ORDER:
            axis.plot(steps,
                      _mean_curve(rows, method, column),
                      label=method,
                      color=METHOD_COLORS[method],
                      linewidth=2)
        axis.set(xlabel="Test queries consumed", ylabel=ylabel, title=title, xlim=(1, steps[-1]))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    positions = np.arange(len(FUNCTIONAL_MODES))
    width = 0.15
    for index, method in enumerate(METHOD_ORDER):
        values = [
            np.mean([
                float(row["critical_recall_at_50"]) for row in mode_rows
                if row["method"] == method and row["mode"] == mode
            ]) for mode in FUNCTIONAL_MODES
        ]
        axis.bar(positions + (index - 2) * width,
                 values,
                 width,
                 label=method,
                 color=METHOD_COLORS[method])
    axis.set(
        xticks=positions,
        xticklabels=[mode.replace("_", "\n") for mode in FUNCTIONAL_MODES],
        xlabel="Functional scenario mechanism",
        ylabel="Mean critical-event recall",
        title="Functional-mode coverage at B=50",
    )
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    figure.savefig(output_dir / "03_mode_coverage.png", dpi=180)
    plt.close(figure)


def write_manifest(anchors: np.ndarray,
                   modes: np.ndarray,
                   output: Path,
                   config: FusionExperimentConfig = CONFIG) -> None:
    mechanism = {
        "fast_intrusion":
        "Adjacent-lane lead vehicle cuts into the ego lane at 0.45 s; no prescribed brake.",
        "cutin_braking":
        "Adjacent-lane lead vehicle completes a 1.5-s cut-in, then applies -4.5 m/s² for 1 s.",
        "lead_braking":
        "Lead vehicle starts in the ego lane and applies -6.5 m/s² from 1.0 s for 1 s.",
        "stop_and_go":
        "Lead vehicle starts in the ego lane, brakes at -3.5 m/s² from "
        "1.0 s for 2 s, then resumes car-following control.",
        "slow_lead_following":
        "A slower lead vehicle starts in the ego lane and remains in "
        "longitudinal car-following interaction.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {
            "seed": config.seed,
            "coordinates": {
                "initial_gap_m": [5.0, 40.0],
                "relative_speed_mps": [-8.0, 2.0]
            },
            "modes": mechanism,
            "balanced_count_per_mode": int(len(anchors) / len(FUNCTIONAL_MODES)),
            "anchor_count": int(len(anchors)),
            "mode_sequence": [str(mode) for mode in modes],
        },
        indent=2) + "\n",
                      encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reuse-bank",
                        action="store_true",
                        help="Reuse an existing response bank in output-dir.")
    args = parser.parse_args()
    CONFIG.validate()
    output_dir = args.output_dir
    bank_path = output_dir / "response_bank.npz"
    if args.reuse_bank:
        bank = ResponseBank.load(bank_path)
    else:
        anchors, modes = generate_multifunction_anchor_bank(CONFIG.num_anchors, CONFIG.seed)
        write_manifest(anchors, modes, output_dir / "scenario_manifest.json")
        bank = build_response_bank(anchors, CONFIG.seed, modes)
        bank.save(bank_path)
    rows = run_loso_multifunction(bank)
    mode_rows = mode_discovery_rows(bank, rows)
    _write_csv(rows, output_dir / "loso_mining.csv")
    _write_csv(mode_rows, output_dir / "mode_discovery.csv")
    summary = summarize(bank, rows, mode_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8")
    write_figures(rows, mode_rows, output_dir / "figures")
    for method in METHOD_ORDER:
        metrics = summary["mean_metrics"][method]
        print(
            f"{method}: score@50={metrics['critical_score_at_50']:.3f}; "
            f"recall@50={metrics['critical_recall_at_50']:.3f}"
        )


if __name__ == "__main__":
    main()
