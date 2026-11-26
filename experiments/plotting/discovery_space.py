"""Observed E02 discovery in the PPO restricted cut-in parameter space.

The three duration slices partition all 2,000 candidates; no coordinate or
response is interpolated. Both cut-in directions retain distinct markers.
Each panel shows a prefix of an existing 200-query trace, not a new run.
The comparison explains spatial allocation in this one task, rather than
estimating an aggregate method effect or a continuous failure boundary.
"""

import gzip
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.budget import protocol
from experiments.plotting.budget_discovery import STYLE, audit_figure, save
from experiments.storage import read_json

DESTINATION = protocol.ROOT / "figures/budget_discovery/discovery_space"
SCENE = "cut_in_onset"
TARGET = "target_ppo"
BUDGETS = (20, 100, 200)
METHODS = {"full": "Ours method", "target_only": "Target-only GP"}
DURATIONS = {1: "rapid", 2: "moderate", 3: "gradual"}
COLORS = {"unknown": "#B7BEC5", "safe": "#367E9B",
          "earlier": "#E6ADA4", "new": "#C44437"}
MARKERS = {"left": "o", "right": "^"}


def sources():
    bank = read_json(protocol.ROOT / "results/k_shot_adaptation"
                     / "candidates/target" / SCENE / "manifest.json")["records"]
    coordinates = np.array([[r["independent"][key] for key in
                             ("ego_speed_mps", "closing_speed_mps", "gap_m")]
                            for r in bank])
    duration = np.array([r["independent"]["lateral_duration_s"] for r in bank])
    sides = np.array([r["scenario"]["categories"]["side"] for r in bank])
    traces = {}
    for method in METHODS:
        path = protocol.OUTPUT / "budget" / TARGET / SCENE / "11" / method / "result.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            result = json.load(stream)["result"]
        queries = result["queries"]
        indices = np.array([q["index"] for q in queries])
        assert len(queries) == len(set(indices)) == 200
        assert result["selected_indices"] == indices.tolist()
        order = np.full(len(bank), 201, dtype=int)
        collisions = np.zeros(len(bank), dtype=bool)
        for sequence, query in enumerate(queries, 1):
            index = query["index"]
            assert query["query"] == sequence
            assert query["feedback"]["scenario_id"] == bank[index]["scenario"]["scenario_id"]
            order[index] = sequence
            collisions[index] = query["feedback"]["collision"]
        traces[method] = order, collisions
    assert len(bank) == 2000 and set(duration) == set(DURATIONS)
    assert set(sides) == set(MARKERS)
    for field in (0, 1):
        a, b = (traces[m][field] for m in METHODS)
        initial = traces["full"][0] <= 20
        assert np.array_equal(a[initial], b[initial])
    return coordinates, duration, sides, traces


def spatial_axes(ax):
    ax.patch.set_visible(False)
    ax.view_init(elev=23, azim=-57)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 0.95), zoom=0.89)
    ax.set(xlim=(19, 35), ylim=(-1, 13), zlim=(3, 47),
           xticks=(20, 27, 34), yticks=(0, 6, 12), zticks=(5, 25, 45))
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.set_xlabel("Ego speed\n(m/s)", labelpad=7)
    ax.set_ylabel("Closing speed\n(m/s)", labelpad=9)
    ax.set_zlabel("Gap (m)", labelpad=0)
    ax.tick_params(labelsize=9, pad=-2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#E1E5E9")
        axis.line.set_color("#8C949D")
        axis.line.set_linewidth(0.55)
        axis._axinfo["grid"].update(color="#DCE1E6", linewidth=0.4)
        axis._axinfo["tick"].update(inward_factor=0, outward_factor=0.15)


def cloud(ax, coordinates, mask, sides, *, face, edge, size, alpha, zorder):
    for side, marker in MARKERS.items():
        points = coordinates[mask & (sides == side)]
        ax.scatter(*points.T, s=size, marker=marker, facecolors=face,
                   edgecolors=edge, linewidths=0.45, alpha=alpha,
                   depthshade=False, zorder=zorder, rasterized=True)


def legend(fig):
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3,
               color=COLORS["unknown"], label="Unqueried"),
        Line2D([], [], marker="o", linestyle="none", markersize=4,
               markerfacecolor="white", markeredgecolor=COLORS["safe"],
               label="Queried, no collision"),
        Line2D([], [], marker="o", linestyle="none", markersize=4,
               color=COLORS["earlier"], label="Earlier collision"),
        Line2D([], [], marker="o", linestyle="none", markersize=4,
               color=COLORS["new"], label="New collision"),
        Line2D([], [], marker="o", linestyle="none", markersize=4,
               color="#616A73", label="Cut-in from left"),
        Line2D([], [], marker="^", linestyle="none", markersize=4,
               color="#616A73", label="Cut-in from right"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=False, handlelength=1.0, columnspacing=1.4,
               labelspacing=0.65, prop={"family": "Arial", "size": 9})


def draw(coordinates, duration, sides, traces, seconds, audit):
    subset = duration == seconds
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 6.0),
                             subplot_kw={"projection": "3d", "computed_zorder": False})
    fig.subplots_adjust(left=0.065, right=0.955, bottom=0.16, top=0.88,
                        wspace=0.10, hspace=0.51)
    for row, (method, label) in enumerate(METHODS.items()):
        order, collisions = traces[method]
        row_box = axes[row, 0].get_position()
        fig.text(0.025, row_box.y0 + row_box.height / 2, label,
                 rotation=90, rotation_mode="anchor",
                 va="center", ha="left", fontsize=9)
        for col, budget in enumerate(BUDGETS):
            ax = axes[row, col]
            spatial_axes(ax)
            observed = subset & (order <= budget)
            previous = 0 if col == 0 else BUDGETS[col - 1]
            masks = {"unknown": subset & ~observed,
                     "safe": observed & ~collisions,
                     "earlier": observed & collisions & (order <= previous),
                     "new": observed & collisions & (order > previous)}
            cloud(ax, coordinates, masks["unknown"], sides, face=COLORS["unknown"],
                  edge="none", size=3, alpha=0.20, zorder=1)
            cloud(ax, coordinates, masks["safe"], sides, face="none",
                  edge=COLORS["safe"], size=13, alpha=0.9, zorder=2)
            cloud(ax, coordinates, masks["earlier"], sides, face=COLORS["earlier"],
                  edge="white", size=15, alpha=1, zorder=3)
            cloud(ax, coordinates, masks["new"], sides, face=COLORS["new"],
                  edge="white", size=19, alpha=1, zorder=4)
            index = row * 3 + col
            ax.annotate(chr(97 + index), (0, 1.025), xycoords="axes fraction", fontsize=9,
                        fontweight="bold", va="bottom")
            if row == 0:
                ax.text2D(0.50, 1.16, f"{budget} total queries", transform=ax.transAxes,
                          ha="center", va="bottom")
            count = int((observed & collisions).sum())
            ax.text2D(0.12, 1.025, f"Queried {observed.sum()} · Collisions {count}",
                      transform=ax.transAxes, fontsize=9, color="#4D555D", va="bottom")
            print({"duration_s": seconds, "method": method, "budget": budget,
                   "slice_queries": int(observed.sum()), "slice_collisions": count,
                   "total_collisions": int((collisions & (order <= budget)).sum())})
    legend(fig)
    return save(fig, axes, f"discovery_space_{DURATIONS[seconds]}_cut_in", audit, destination=DESTINATION)


def render(*, audit=audit_figure):
    data = sources()
    with plt.rc_context({**STYLE, "axes.labelsize": 9.5}):
        return [draw(*data, seconds, audit) for seconds in DURATIONS]


if __name__ == "__main__":
    print([str(path) for path in render()])
