"""Static evaluator-only figures for the finite NL-IDM grid experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


STATE = {"STABLE_PASS": 0, "PERSISTENT_FAILURE": 1,
         "REGRESSION": 2, "IMPROVEMENT": 3, "UNKNOWN": 4}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def render(root: Path, *, query_dir: Path | None = None,
           highlighted_method: str = "directed_residual") -> list[str]:
    query_dir = query_dir or root
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    builds = tuple(protocol["builds"])
    if len(builds) != 3:
        raise ValueError("expected three release builds")
    pairs = tuple(zip(builds, builds[1:]))
    cases = _jsonl(root / "scenario_manifest.jsonl")
    truth = _csv(query_dir / "transition_truth.csv")
    queries = _jsonl(query_dir / "queries.jsonl")
    out = query_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    contexts = sorted({case["context_id"] for case in cases})
    case_by_id = {case["scenario_id"]: case for case in cases}
    n = cases[0]["grid_resolution"]
    colors = ListedColormap(["#d9edf7", "#555555", "#e45756", "#58a55c", "#c47cc8"])
    norm = BoundaryNorm(np.arange(-.5, 5.5), colors.N)
    paths = []
    for parent, target in pairs:
        lookup = {row["scenario_id"]: STATE[row["transition"]] for row in truth
                  if row["parent_build_id"] == parent and row["target_build_id"] == target}
        fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
        for axis, context in zip(axes.flat, contexts):
            grid = np.full((n, n), 4, dtype=int)
            for case in cases:
                if case["context_id"] == context:
                    i, j = case["grid_index"]
                    grid[i, j] = lookup[case["scenario_id"]]
            axis.imshow(grid, cmap=colors, norm=norm, origin="lower", interpolation="nearest")
            axis.set_title(context, fontsize=9)
            axis.set_xlabel("parameter 2 grid index")
            axis.set_ylabel("parameter 1 grid index")
        for axis in list(axes.flat)[len(contexts):]:
            axis.set_axis_off()
        fig.suptitle(f"Evaluator truth: {parent} → {target}")
        handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=colors(i),
                              label=label, markersize=10) for label, i in STATE.items()]
        fig.legend(handles=handles, loc="lower center", ncol=5,
                   bbox_to_anchor=(.5, -.075), fontsize=8)
        path = out / f"truth_{parent}_to_{target}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

        selected = [row for row in queries if row["parent"] == parent and
                    row["target"] == target and row["method"] == highlighted_method and
                    row["repeat"] == 0]
        fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
        for axis, context in zip(axes.flat, contexts):
            local = [row for row in selected if row["context_id"] == context]
            for direction, color in (("R", "#e45756"), ("I", "#58a55c")):
                subset = [row for row in local if row["direction"] == direction]
                axis.scatter([case_by_id[row["scenario_id"]]["grid_index"][1] for row in subset],
                             [case_by_id[row["scenario_id"]]["grid_index"][0] for row in subset],
                             c=color, label=direction, s=70, alpha=.8)
                for row in subset:
                    i, j = case_by_id[row["scenario_id"]]["grid_index"]
                    axis.annotate(str(row["rank"]), (j, i), xytext=(3, 3),
                                  textcoords="offset points", fontsize=7)
            axis.set(xlim=(-.5, n-.5), ylim=(-.5, n-.5),
                     xlabel="parameter 2 grid index", ylabel="parameter 1 grid index")
            axis.grid(alpha=.2)
            axis.set_title(context, fontsize=9)
        for axis in list(axes.flat)[len(contexts):]:
            axis.set_axis_off()
        fig.suptitle(f"{highlighted_method} queries: {parent} → {target} (rank labels)")
        path = out / f"queries_{parent}_to_{target}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.subplots_adjust(left=.08, right=.98, top=.94, bottom=.25,
                        hspace=.35, wspace=.2)
    methods = sorted({row["method"] for row in queries})
    focus = {highlighted_method, "static_risk", "coordinate_residual",
             "static_margin_coverage", "coordinate_margin_frontier",
             "directed_context_laplace", "directed_context_ucb"}
    palette = {highlighted_method: "#4b2c73", "static_risk": "#1f77b4",
               "coordinate_residual": "#ff7f0e", "static_margin_coverage": "#17becf",
               "coordinate_margin_frontier": "#2ca02c",
               "directed_context_laplace": "#d62728",
               "directed_context_ucb": "#9467bd"}
    for pair_index, (parent, target) in enumerate(pairs):
        for direction_index, direction in enumerate(("R", "I")):
            axis = axes[pair_index, direction_index]
            transition = "REGRESSION" if direction == "R" else "IMPROVEMENT"
            change_count = sum(row["parent_build_id"] == parent and
                               row["target_build_id"] == target and
                               row["transition"] == transition for row in truth)
            for method in methods:
                repeats = sorted({row["repeat"] for row in queries if row["method"] == method})
                curves = []
                for repeat in repeats:
                    local = [row for row in queries if row["parent"] == parent and
                             row["target"] == target and row["method"] == method and
                             row["direction"] == direction and row["repeat"] == repeat]
                    curves.append(np.cumsum([0] + [row["discovery"] for row in local]))
                min_len = min(map(len, curves))
                mean = np.mean(np.asarray([curve[:min_len] for curve in curves]), axis=0)
                axis.plot(range(min_len), mean, label=method,
                          color=palette.get(method, "#808080"),
                          linewidth=3 if method == highlighted_method else
                          1.8 if method in focus else 1,
                          alpha=1 if method in focus else .4,
                          zorder=5 if method == highlighted_method else 2)
            axis.set(title=f"{parent} → {target}: {direction}", xlabel="direction queries",
                     ylabel="discoveries", xlim=(0, 20))
            if change_count == 0:
                axis.set_ylim(-.1, 1)
                axis.text(.5, .55, "No true changes (NA comparison)",
                          ha="center", transform=axis.transAxes, color="#555555")
            else:
                axis.set_ylim(bottom=-.1)
            axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .015), fontsize=8, frameon=False)
    path = out / "discovery_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path)
    parser.add_argument("--highlighted-method", default="directed_residual")
    args = parser.parse_args()
    print(json.dumps(render(args.root, query_dir=args.query_dir,
                            highlighted_method=args.highlighted_method), ensure_ascii=False))


if __name__ == "__main__":
    main()
