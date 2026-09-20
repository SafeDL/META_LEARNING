"""Generate AdaTE figures only from saved A0/A1 logs, never from hidden state."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_dense_batch(run_dir: Path) -> list[Path]:
    """Render cross-target summaries from immutable per-case CSV evidence."""
    rows = _csv(run_dir / "case_strategy_summary.csv")
    if not rows:
        return []
    created: list[Path] = []
    targets = sorted({row["target_profile"] for row in rows})
    preferred = ["NDE-phi-H", "Equal-Mixture-H", "Uniform-QP-H", "AdaTE-QP-H"]
    strategies = [name for name in preferred if any(row["strategy"] == name for row in rows)]
    fig, axes = plt.subplots(1,
                             len(targets),
                             figsize=(4.2 * len(targets), 4),
                             sharey=True,
                             squeeze=False)
    for target, axis in zip(targets, axes.flat):
        means, spreads = [], []
        for strategy in strategies:
            values = np.asarray([
                float(row["estimate"]) for row in rows
                if row["target_profile"] == target and row["strategy"] == strategy
            ])
            means.append(values.mean())
            spreads.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        axis.bar(np.arange(len(strategies)), means, yerr=spreads, capsize=3)
        axis.set(title=target,
                 xlabel="frozen policy",
                 xticks=np.arange(len(strategies)),
                 xticklabels=strategies)
        axis.tick_params(axis="x", labelrotation=32)
    axes[0, 0].set_ylabel("mean IS estimate across seeds")
    fig.suptitle("A1 cross-target estimates (error bars: seed SD, not IS CI)")
    path = run_dir / "cross_target_estimates.png"
    _save(fig, path)
    created.append(path)

    fig, axes = plt.subplots(1,
                             len(targets),
                             figsize=(4.2 * len(targets), 4),
                             sharey=True,
                             squeeze=False)
    for target, axis in zip(targets, axes.flat):
        means = [
            np.mean([
                float(row["relative_half_width"]) for row in rows
                if row["target_profile"] == target and row["strategy"] == strategy
            ]) for strategy in strategies
        ]
        axis.bar(np.arange(len(strategies)), means)
        axis.set(title=target,
                 xlabel="frozen policy",
                 xticks=np.arange(len(strategies)),
                 xticklabels=strategies)
        axis.tick_params(axis="x", labelrotation=32)
    axes[0, 0].set_ylabel("mean per-run relative half-width")
    fig.suptitle("A1 precision diagnostic by held-out target")
    path = run_dir / "cross_target_precision.png"
    _save(fig, path)
    created.append(path)

    alpha_rows = _csv(run_dir / "final_mixture_coefficients.csv")
    if alpha_rows:
        variants = [
            name for name in ("Uniform-QP-H", "AdaTE-QP-H")
            if any(row["method_variant"] == name for row in alpha_rows)
        ]
        alpha_keys = sorted(key for key in alpha_rows[0] if key.startswith("alpha_"))
        fig, axes = plt.subplots(1,
                                 len(targets),
                                 figsize=(4.2 * len(targets), 4),
                                 sharey=True,
                                 squeeze=False)
        for target, axis in zip(targets, axes.flat):
            x = np.arange(len(alpha_keys))
            width = 0.8 / max(1, len(variants))
            for index, variant in enumerate(variants):
                selected = [
                    row for row in alpha_rows
                    if row["target_profile"] == target and row["method_variant"] == variant
                ]
                values = np.asarray([[float(row[key]) for key in alpha_keys] for row in selected])
                axis.bar(x + (index - (len(variants) - 1) / 2) * width,
                         values.mean(axis=0),
                         width=width,
                         label=variant)
            axis.set(title=target,
                     xlabel="source surrogate",
                     xticks=x,
                     xticklabels=alpha_keys,
                     ylim=(0, 1.02))
            axis.legend(fontsize=7)
        axes[0, 0].set_ylabel("mean final simplex coefficient")
        fig.suptitle("A1 final QP mixture by held-out target")
        path = run_dir / "cross_target_coefficients.png"
        _save(fig, path)
        created.append(path)

    # Preserve the scenario-mode dimension rather than reporting only an
    # aggregate that could hide a policy working in one Cut-in mode alone.
    mode_rows: list[dict] = []
    for case in sorted(run_dir.glob("seed_*/target_*")):
        for row in _csv(case / "evaluation_draws.csv"):
            if row["strategy"] not in strategies:
                continue
            mode_rows.append({
                "target_profile":
                case.name.removeprefix("target_"),
                "strategy":
                row["strategy"],
                "mode":
                row["scenario_mode"],
                "weighted_event":
                (row["collision"].lower() == "true") * np.exp(float(row["log_weight"])),
            })
    if mode_rows:
        modes = sorted({row["mode"] for row in mode_rows})
        fig, axes = plt.subplots(1,
                                 len(targets),
                                 figsize=(4.4 * len(targets), 4),
                                 sharey=True,
                                 squeeze=False)
        for target, axis in zip(targets, axes.flat):
            x = np.arange(len(modes))
            width = 0.8 / max(1, len(strategies))
            for index, strategy in enumerate(strategies):
                means = []
                for mode in modes:
                    values = [
                        row["weighted_event"] for row in mode_rows
                        if row["target_profile"] == target and row["strategy"] == strategy
                        and row["mode"] == mode
                    ]
                    means.append(float(np.mean(values)) if values else np.nan)
                axis.bar(x + (index - (len(strategies) - 1) / 2) * width,
                         means,
                         width=width,
                         label=strategy)
            axis.set(title=target, xlabel="Cut-in mode", xticks=x, xticklabels=modes)
            axis.tick_params(axis="x", labelrotation=35)
            axis.legend(fontsize=6)
        axes[0, 0].set_ylabel("mean weighted collision contribution")
        fig.suptitle("A1 mode-stratified IS samples (pooled draws; no additional CI)")
        path = run_dir / "mode_stratified_estimates.png"
        _save(fig, path)
        created.append(path)
    return created


def render(run_dir: Path) -> list[Path]:
    """Render available figures; a mixture or dense run may be rendered alone."""
    run_dir = Path(run_dir)
    if (run_dir / "case_strategy_summary.csv").exists():
        return _render_dense_batch(run_dir)
    created: list[Path] = []
    alpha = _csv(run_dir / "mixture_coefficients_trace.csv")
    if alpha:
        a0 = "response" in alpha[0]
        groups: dict[str, list[dict]] = {}
        for row in alpha:
            key = row.get("response", "DenseRL") + "/" + row.get(
                "method_variant", "DenseRL") + "/" + row.get("target_sut", "target")
            groups.setdefault(key, []).append(row)
        if a0:
            response_names = sorted({row["response"] for row in alpha})
            target_names = sorted({row["target_sut"] for row in alpha})
            columns = min(3, len(target_names))
            rows_count = int(np.ceil(len(target_names) / columns))
            for response_name in response_names:
                fig, axes = plt.subplots(rows_count,
                                         columns,
                                         figsize=(4 * columns, 3 * rows_count),
                                         sharex=True,
                                         sharey=True,
                                         squeeze=False)
                for target_name, axis in zip(target_names, axes.flat):
                    rows = groups[f"{response_name}/AdaTE-Mixture-H-sequential/{target_name}"]
                    steps = [int(row["step"]) for row in rows]
                    for key in sorted(name for name in rows[0] if name.startswith("alpha_")):
                        axis.plot(steps, [float(row[key]) for row in rows], label=key)
                    axis.set(title=target_name, ylim=(-0.02, 1.02), xlabel="reveal", ylabel="α")
                    axis.legend(fontsize=6, ncol=2)
                for axis in axes.flat[len(target_names):]:
                    axis.remove()
                fig.suptitle(f"A0 sequential α traces: {response_name}")
                path = run_dir / ("mixture_coefficients_vulnerability.png" if response_name ==
                                  "vulnerability" else f"mixture_coefficients_{response_name}.png")
                _save(fig, path)
                created.append(path)
            fig, axes = plt.subplots(rows_count,
                                     columns,
                                     figsize=(4 * columns, 3 * rows_count),
                                     sharex=True,
                                     sharey=True,
                                     squeeze=False)
            for target_name, axis in zip(target_names, axes.flat):
                for response_name in response_names:
                    rows = groups[f"{response_name}/AdaTE-Mixture-H-sequential/{target_name}"]
                    values = np.asarray([[
                        float(row[key]) for key in sorted(name for name in rows[0]
                                                          if name.startswith("alpha_"))
                    ] for row in rows])
                    window = 10
                    asd = [
                        np.nan if index < window else float(
                            np.abs((values[index - window + 1:index + 1] -
                                    values[index - window:index]).sum(axis=0)).mean())
                        for index in range(len(values))
                    ]
                    axis.plot(range(1, len(asd) + 1), asd, label=response_name)
                axis.axhline(0.02, color="black", linestyle="--", linewidth=1, label="threshold")
                axis.set(title=target_name, xlabel="reveal", ylabel="ASD")
                axis.legend(fontsize=6)
            for axis in axes.flat[len(target_names):]:
                axis.remove()
            fig.suptitle("A0 sequential ASD by held-out target")
            path = run_dir / "coefficient_stability.png"
            _save(fig, path)
            created.append(path)
        else:
            fig, axis = plt.subplots(figsize=(8, 4))
            for label, rows in groups.items():
                steps = [int(row.get("step", row.get("episode", 0))) for row in rows]
                for key in sorted(name for name in rows[0] if name.startswith("alpha_")):
                    axis.plot(steps, [float(row[key]) for row in rows],
                              label=f"{label}:{key}",
                              alpha=0.8)
            axis.set(xlabel="target episode",
                     ylabel="combination coefficient",
                     ylim=(-0.02, 1.02),
                     title="AdaTE DenseRL simplex mixture trace")
            axis.legend(fontsize=6, ncol=3)
            path = run_dir / "mixture_coefficients.png"
            _save(fig, path)
            created.append(path)
            fig, axis = plt.subplots(figsize=(8, 3))
            for label, rows in groups.items():
                asd = [
                    float(row["asd"]) if row.get("asd") not in (None, "", "nan") else np.nan
                    for row in rows
                ]
                axis.plot(range(1, len(asd) + 1), asd, label=label)
            axis.axhline(0.02, color="black", linestyle="--", label="paper ASD threshold")
            axis.set(xlabel="target episode", ylabel="ASD", title="DenseRL stopping diagnostic")
            axis.legend(fontsize=6)
            path = run_dir / "coefficient_stability.png"
            _save(fig, path)
            created.append(path)
    traces = _csv(run_dir / "target_query_trace.csv")
    bank_path = run_dir / "source_response_bank.npz"
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as bank:
            anchors, vulnerability = bank["anchors"], bank["vulnerability"]
        fig, axis = plt.subplots(figsize=(7, 4))
        scatter = axis.scatter(anchors[:, 0],
                               anchors[:, 1],
                               c=vulnerability.mean(axis=0),
                               cmap="magma",
                               vmin=0,
                               vmax=1)
        fig.colorbar(scatter, ax=axis, label="mean source vulnerability")
        axis.set(xlabel="configured gap (m)",
                 ylabel="relative speed (m/s)",
                 title="A0 source response boundary (fast_intrusion bank)")
        path = run_dir / "source_response_boundary.png"
        _save(fig, path)
        created.append(path)
    if traces:
        fig, axis = plt.subplots(figsize=(7, 4))
        groups: dict[str, list[dict]] = {}
        for row in traces:
            groups.setdefault(
                row.get("response", "vulnerability") + "/" + row["method_variant"], []).append(row)
        for label, rows in groups.items():
            if "/" in label and not (label.endswith("/Uniform-Mixture-H")
                                     or label.endswith("/AdaTE-Mixture-H-sequential")):
                continue
            by_target: dict[str, list[dict]] = {}
            for row in rows:
                by_target.setdefault(row["target_sut"], []).append(row)
            curves = [
                np.asarray([float(item["cumulative_critical"]) for item in target_rows])
                for target_rows in by_target.values()
            ]
            axis.plot(range(1, len(curves[0]) + 1), np.mean(curves, axis=0), label=label)
        axis.set(xlabel="target-query budget",
                 ylabel="mean cumulative critical events",
                 title="A0 discovery is separate from A1 estimation")
        axis.legend(fontsize=7)
        path = run_dir / "critical_discovery.png"
        _save(fig, path)
        created.append(path)
    evaluation = _csv(run_dir / "evaluation_draws.csv")
    if evaluation:
        fig, axis = plt.subplots(figsize=(7, 4))
        by_strategy: dict[str, list[dict]] = {}
        for row in evaluation:
            by_strategy.setdefault(row["strategy"], []).append(row)
        for name, rows in by_strategy.items():
            weighted = np.asarray([
                (row["collision"].lower() == "true") * np.exp(float(row["log_weight"]))
                for row in rows
            ])
            axis.plot(range(1,
                            len(weighted) + 1),
                      np.cumsum(weighted) / np.arange(1,
                                                      len(weighted) + 1),
                      label=name)
        axis.set(xlabel="independent evaluation draw",
                 ylabel="weighted reference event estimate",
                 title="A1 importance-sampling estimates")
        axis.legend()
        path = run_dir / "importance_sampling_estimates.png"
        _save(fig, path)
        created.append(path)
        summary_path = run_dir / "importance_sampling_summary.json"
        summary = json.loads(summary_path.read_text(
            encoding="utf-8")) if summary_path.exists() else {}
        fig, axis = plt.subplots(figsize=(8, 4))
        names = list(summary)
        estimates = [float(summary[name]["estimate"]) for name in names]
        errors = [float(summary[name].get("ci95_half_width", 0.0)) for name in names]
        axis.bar(np.arange(len(names)), estimates, yerr=errors, capsize=3)
        axis.set(xlabel="frozen evaluation policy",
                 ylabel="reference event estimate",
                 title="Independent IS estimates with 95% normal half-width")
        axis.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
        path = run_dir / "estimate_precision.png"
        _save(fig, path)
        created.append(path)
    gap_rows = _csv(run_dir / "q_gap_visitation.csv")
    if gap_rows:
        fig, axis = plt.subplots(figsize=(7, 4))
        grouped: dict[str, list[dict]] = {}
        for row in gap_rows:
            grouped.setdefault(row.get("method_variant", "DenseRL"), []).append(row)
        for label, rows in grouped.items():
            gap = [max(json.loads(row["gap_scores"])) for row in rows]
            axis.plot(range(1, len(rows) + 1), gap, label=f"{label}: max gap/UCB")
        axis.set(xlabel="physical transition",
                 ylabel="gap/UCB score",
                 title="DenseRL sampling diagnostics")
        axis.legend()
        path = run_dir / "q_gap_visitation.png"
        _save(fig, path)
        created.append(path)
    support = _csv(run_dir / "proposal_support.csv")
    if support:
        fig, axis = plt.subplots(figsize=(8, 4))
        grouped: dict[str, list[dict]] = {}
        for row in support:
            grouped.setdefault(row.get("strategy", "adaptive"), []).append(row)
        x = np.arange(len(next(iter(grouped.values()))))
        for label, rows in grouped.items():
            axis.plot(x, [float(row.get("mean_q", row.get("q_eps", 0.0))) for row in rows],
                      marker="o",
                      label=label)
        axis.plot(x, [float(row["p_phi"]) for row in next(iter(grouped.values()))],
                  color="black",
                  linestyle="--",
                  label="p = φ")
        axis.set(xlabel="background action",
                 ylabel="probability mass",
                 xticks=x,
                 xticklabels=[row["action"] for row in next(iter(grouped.values()))],
                 ylim=(0, 1.05),
                 title="Support diagnostic from logged frozen policies")
        axis.legend()
        path = run_dir / "proposal_support.png"
        _save(fig, path)
        created.append(path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()
