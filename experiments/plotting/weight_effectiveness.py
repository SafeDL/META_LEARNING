"""Compare held-out and next-query collision probability errors for E03."""

import csv

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from experiments.weight_ablation import run as study
from experiments.paper_results import paper_study
from experiments.plotting.budget_discovery import panel_label
from experiments.plotting.style import STYLE, audit_figure, plt
from experiments.weight_ablation.effectiveness import METRICS
from experiments.storage import read_json

PUBLIC_FROZEN = study.ROOT / "results/public_ads/weight_adaptation/reports"
PUBLIC_ADS = (
    ("online_mc_ppo", "OnlineMC PPO*"),
    ("social_attention_ddqn", "Attention DDQN*"),
    ("decision_transformer", "Decision Trans.*"),
)


def _load_public_scores(public_root=PUBLIC_FROZEN):
    summary = read_json(public_root / "summary.json")
    if summary.get("status") != "complete" or summary.get("prediction_pairs") != 12:
        raise ValueError("Incomplete public E03 probability summary")
    with (public_root / "paired_scores.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12:
        raise ValueError("Incomplete public E03 paired scores")
    by_target = {}
    for sut_id, _ in PUBLIC_ADS:
        subset = [r for r in rows if r["target"] == sut_id]
        if len(subset) != 4:
            raise ValueError(f"Incomplete public E03 scenes for {sut_id}")
        by_target[sut_id] = subset
    return summary, by_target


def plot(report_root=None, destination=None, *, include_public=True):
    directory = report_root or paper_study(study.OUTPUT) / "reports"
    report = read_json(directory / "probability_summary.json")
    if (report["status"] != "complete" or report["metrics"] != list(METRICS)
            or report["probability_scores_sha256"] != study.sha(directory / "probability_scores.csv")):
        raise ValueError("Requires the complete paired E03 scoring table")
    public = _load_public_scores() if (include_public and (PUBLIC_FROZEN / "summary.json").is_file()) else None
    evaluations = (("heldout", "Held-out candidates", "#995037", "o", -0.14),
                   ("sequential", "Next query", "#315D86", "s", 0.14))
    scopes = [report["scopes"][key] for key in ("overall", *study.SCENES)]
    row_labels = ["All scenes (8 ADS)", "Following", "Cut-in", "Cut-out", "Encroachment"]
    if public is not None:
        row_labels.extend(["Public mean (3 ADS)", *(label for _, label in PUBLIC_ADS)])
    height = 4.8 if public is not None else 3.14
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, height), sharey=True)
        fig.subplots_adjust(left=0.24 if public is not None else 0.16, right=0.98,
                            bottom=0.255 if public is not None else 0.30, top=0.93, wspace=0.17)
        for index, (ax, metric, label) in enumerate(zip(
            axes, ("nll", "brier"), ("Δ NLL (nats) ↓", "Δ Brier score ↓"), strict=True
        )):
            ax.axvline(0, color="#6C757D", linestyle="--", linewidth=0.85, zorder=1)
            if public is not None:
                ax.axhline(len(scopes) - 0.5, color="#B0B7BF", linestyle=":", linewidth=0.9, zorder=1)
            for mode, _, color, marker, offset in evaluations:
                column = METRICS.index(f"{mode}_{metric}")
                for row, scope in enumerate(scopes):
                    low, mean, high = [scope[key][column] for key in ("low", "delta", "high")]
                    y = row + offset
                    ax.hlines(y, low, high, color=color, linewidth=1.3)
                    ax.plot(mean, y, marker, color=color, markersize=4.2)
                if public is not None:
                    pub_summary, by_target = public
                    field = f"{mode}_{metric}"
                    mean_y = len(scopes) + offset
                    ads_means = [
                        pub_summary["scopes"][sut_id][field]["dynamic_minus_frozen"] for sut_id, _ in PUBLIC_ADS
                    ]
                    pub_mean = sum(ads_means) / len(ads_means)
                    ax.hlines(mean_y, min(ads_means), max(ads_means), color=color, linewidth=1.0, linestyle=":")
                    ax.plot(pub_mean, mean_y, marker, color=color, markersize=4.3,
                            markerfacecolor="white", markeredgewidth=1.15)
                    for idx, (sut_id, _) in enumerate(PUBLIC_ADS):
                        y = len(scopes) + 1 + idx + offset
                        deltas = [
                            float(r[f"dynamic_{field}"]) - float(r[f"frozen_{field}"]) for r in by_target[sut_id]
                        ]
                        ax.hlines(y, min(deltas), max(deltas), color=color, linewidth=1.0, linestyle=":")
                        ax.plot(sum(deltas) / len(deltas), y, marker, color=color, markersize=4.3,
                                markerfacecolor="white", markeredgewidth=1.15)
            x_low, x_high = ax.get_xlim()
            ax.axvspan(x_low, 0, color="#F3F7F4", zorder=0)
            ax.set_xlim(x_low, max(x_high, abs(x_low) * 0.04))
            ax.set_yticks(range(len(row_labels)), row_labels)
            ax.set_ylim(len(row_labels) - 0.5, -0.5)
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel(label)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            panel_label(ax, index)
        handles = [Line2D([], [], color=color, marker=marker, markersize=4, label=name)
                   for _, name, color, marker, _ in evaluations]
        if public is not None:
            handles.extend([
                Line2D([], [], color="#444444", linestyle="-", marker="o", markersize=4,
                       label="Primary 8 ADS (95% CI)"),
                Line2D([], [], color="#444444", linestyle=":", marker="o", markersize=4,
                       markerfacecolor="white", markeredgewidth=1.1, label="Public 3 ADS (range)"),
            ])
        fig.legend(handles=handles, title="Ours method − frozen weights", ncol=2,
                   loc="lower center", bbox_to_anchor=(0.55 if public is not None else 0.53, 0.005), frameon=False)
        fig.canvas.draw()
        name = "weight_effectiveness"
        audit_figure(fig, list(axes), name, report_root=directory / "figure_quality")
        path = (destination or study.ROOT / "figures/weight_adaptation") / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=600, facecolor="white")
        plt.close(fig)
        return path


if __name__ == "__main__":
    plot()
