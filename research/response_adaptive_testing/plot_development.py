"""Visualize development component controls and changing response coefficients."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods.history_guided_testing.io import read_json

from .config import BUDGET, OUTPUT

POOLS = ("original", "nominal", "delayed_fvdm", "limited_fvdm", "delayed_idm")
POOL_LABELS = ("Original", "Nominal", "Delayed FVDM", "Limited FVDM",
               "Delayed IDM")
METHODS = ("full", "source_contrasts_only", "local_transfer_only",
           "no_risk_failure_coupling", "static_posterior")
METHOD_LABELS = ("Full model", "Source contrasts only", "Local transfer only",
                 "No cross coupling", "Static posterior")
COLORS = ("#146A8B", "#D68B37", "#479B73", "#AB6981", "#838B97")
FIGURES = OUTPUT / "figures"


def plot_components():
    records = read_json(OUTPUT / "component_controls/summary.json")
    figure, axes = plt.subplots(1,
                                2,
                                figsize=(12, 4.5),
                                constrained_layout=True)
    positions = np.arange(len(POOLS))
    width = 0.15
    for method_index, method in enumerate(METHODS):
        groups = [[
            r for r in records if r["pool"] == pool and r["method"] == method
        ] for pool in POOLS]
        for axis, metric in zip(axes, ("normalized_area", "recall")):
            values = [
                np.array([
                    r["mean_cumulative_collisions"] / r["pool_collisions"]
                    if metric == "normalized_area" else r["recall"]
                    for r in group
                ]) for group in groups
            ]
            axis.bar(positions + (method_index - 2) * width,
                     [value.mean() for value in values],
                     width,
                     yerr=[value.std(ddof=1) for value in values],
                     capsize=2,
                     color=COLORS[method_index],
                     label=METHOD_LABELS[method_index],
                     error_kw={"linewidth": 0.8})
    totals = [
        next(r["pool_collisions"] for r in records if r["pool"] == pool)
        for pool in POOLS
    ]
    area_bound = [
        np.minimum(np.arange(1, BUDGET + 1), total).mean() / total
        for total in totals
    ]
    recall_bound = [min(BUDGET, total) / total for total in totals]
    for axis, bounds, title in zip(axes, (area_bound, recall_bound),
                                   ("Mean discovery curve / all pool failures",
                                    "Final failure recall at budget 200")):
        axis.scatter(positions,
                     bounds,
                     color="black",
                     marker="_",
                     s=300,
                     label="Budget ceiling",
                     zorder=5)
        axis.set_xticks(positions, POOL_LABELS, rotation=15)
        axis.set_ylim(0, 1.05)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Fraction")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Development evidence: five predictor seeds; error bars are seed SD",
        fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles,
                  labels,
                  ncol=3,
                  fontsize=9,
                  loc="upper center",
                  bbox_to_anchor=(0.5, -0.04))
    figure.savefig(FIGURES / "development_components.png",
                   dpi=200,
                   bbox_inches="tight")
    figure.savefig(FIGURES / "development_components.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_coefficients():
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    source_colors = ("#146A8B", "#D68B37", "#479B73", "#AB6981", "#7358A1",
                     "#898F39")
    for axis, pool, title in zip(axes,
                                 ("nominal", "delayed_fvdm", "limited_fvdm"),
                                 ("Nominal", "Delayed FVDM", "Limited FVDM")):
        value = read_json(OUTPUT / "response_coefficients" / f"{pool}.json")
        queries = [row["query_count"] for row in value["trace"]]
        means = np.array([
            row["effective_response_coefficients"][1] for row in value["trace"]
        ])
        for source_index, source in enumerate(value["sources"]):
            axis.plot(queries,
                      means[:, source_index],
                      label=source.replace("_", " "),
                      color=source_colors[source_index])
        axis.axhline(1 / len(value["sources"]),
                     color="gray",
                     linestyle="--",
                     linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Queried risk observations")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Effective latent response coefficient")
    axes[-1].legend(fontsize=8)
    figure.suptitle(
        "Development brake family, seed 11: response coefficients are not source probabilities",
        fontsize=11)
    figure.savefig(FIGURES / "development_response_coefficients.png", dpi=200)
    figure.savefig(FIGURES / "development_response_coefficients.pdf")
    plt.close(figure)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_components()
    plot_coefficients()
    print("DEVELOPMENT FIGURES", FIGURES, flush=True)


if __name__ == "__main__":
    main()
