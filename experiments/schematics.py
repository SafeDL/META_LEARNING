"""Conceptual traffic and method diagrams; no simulated outcomes are represented."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .joint_figures import STYLE, save_figure
from .joint_validation import OUTPUT
from .scenarios import DOMAINS, TEST_SCENARIOS
from .storage import write_json

EGO, OTHER, INK = "#D55E00", "#0072B2", "#263746"


def vehicle(axis, x, y, color, name, speed=None):
    axis.add_patch(
        FancyBboxPatch(
            (x - 3.4, y - 0.75),
            6.8,
            1.5,
            boxstyle="round,pad=0.08,rounding_size=0.3",
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    )
    axis.plot([x - 1.4, x - 1.4], [y - 0.48, y + 0.48], color="white", lw=1, zorder=4)
    axis.text(x, y + 1.12, name, ha="center", va="top", fontsize=8, color=INK)
    if speed:
        axis.annotate(
            "",
            (x + 12, y - 1.1),
            (x + 1, y - 1.1),
            arrowprops={"arrowstyle": "->", "lw": 1, "color": color},
        )
        axis.text(x + 6.5, y - 1.28, speed, ha="center", va="bottom", color=color, fontsize=8)


def distance(axis, start, end, y, label):
    axis.annotate(
        "", (end, y), (start, y), arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": INK}
    )
    axis.text(
        (start + end) / 2,
        y - 0.22 if y < 0 else y + 0.22,
        label,
        ha="center",
        va="bottom" if y < 0 else "top",
        fontsize=8,
    )


def path_arrow(axis, start, end, color, rad=0.22):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="->",
            mutation_scale=9,
            linestyle="--",
            linewidth=1.2,
            color=color,
            zorder=2,
        )
    )


def traffic_scenarios(path):
    figure, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 112 / 25.4))
    titles = (
        "(a) Car following · 3 parameters",
        "(b) Lead braking · 6 parameters",
        "(c) Cut-out and hidden obstacle · 6 parameters",
        "(d) Ego lane change · 6 parameters",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_xlim(-4, 104)
        axis.set_ylim(13, -4.5)
        axis.add_patch(Rectangle((-4, -2), 108, 8, facecolor="#EEF1F3", edgecolor="none"))
        for y in (-2, 6):
            axis.axhline(y, color="#BBC3CB", lw=0.8)
        axis.axhline(2, color="white", lw=1.5, linestyle=(0, (6, 4)))
        axis.set_title(title, loc="left", fontsize=8.5, pad=8)
        axis.axis("off")
    axis = axes[0, 0]
    vehicle(axis, 16, 4, EGO, "Ego", r"$v_e$")
    vehicle(axis, 65, 4, OTHER, "Lead", r"$v_l$")
    distance(axis, 19.4, 61.6, 8.4, r"Net gap $g$")
    axis.text(50, 12, r"Constant lead speed; $D_v=v_l-v_e$", ha="center", fontsize=8)
    axis = axes[0, 1]
    vehicle(axis, 16, 4, EGO, "Ego", r"$v_e$")
    vehicle(axis, 65, 4, OTHER, "Lead", r"$v_l=v_e-\Delta v$")
    distance(axis, 19.4, 61.6, 8.4, r"Net gap $g$")
    axis.text(
        50, 12, r"Brake: start $t_b$, deceleration $a_b$, rise $\tau_b$", ha="center", fontsize=8
    )
    axis = axes[1, 0]
    vehicle(axis, 12, 4, EGO, "Ego", r"$v_e$")
    vehicle(axis, 43, 4, OTHER, "Lead", r"$v_l$")
    axis.add_patch(Rectangle((80, 3.2), 3, 1.6, facecolor=INK, hatch="////", zorder=3))
    axis.text(81.5, 5.12, "Stopped\nobstacle", ha="center", va="top", fontsize=8)
    path_arrow(axis, (46, 3.6), (69, 0), OTHER)
    axis.text(68, -1.15, r"Cut-out: $t_c,\tau_c$", ha="center", fontsize=8, color=OTHER)
    distance(axis, 15.4, 39.6, 8.4, r"$g$")
    distance(axis, 12, 81.5, 11.1, r"$x_o-x_e$")
    axis = axes[1, 1]
    vehicle(axis, 7, 0, OTHER, "Rear", r"$v_r$")
    vehicle(axis, 87, 0, OTHER, "Adj. lead", r"$v_e$")
    vehicle(axis, 30, 4, EGO, "Ego", r"$v_e$")
    vehicle(axis, 65, 4, OTHER, "Lead", r"$v_l$")
    path_arrow(axis, (33, 3.6), (56, 0), EGO)
    distance(axis, 10.4, 26.6, -2.7, r"$g_r$")
    distance(axis, 33.4, 83.6, -2.7, r"$g_a$")
    distance(axis, 33.4, 61.6, 8.4, r"$g_l$")
    axis.text(50, 12, "Lane change is selected by the ADS", ha="center", fontsize=8)
    figure.subplots_adjust(
        left=0.025, right=0.985, top=0.91, bottom=0.055, hspace=0.38, wspace=0.14
    )
    save_figure(figure, path)


def method_flow(path):
    figure, axis = plt.subplots(figsize=(180 / 25.4, 112 / 25.4))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    boxes = (
        (
            0.025,
            0.57,
            "1",
            "RSS-assisted historical\nrisk prior construction",
            "Historical collision + RSS\nShared features; distinct ADS sources\nJoint predictions and uncertainty",
            OTHER,
        ),
        (
            0.565,
            0.57,
            "2",
            "Few-shot target\nrisk adaptation",
            "All paid target collision + RSS\nSix scalars/kernel; frozen features\nPrequery-likelihood source weights",
            EGO,
        ),
        (
            0.565,
            0.12,
            "3",
            "Budget-constrained\nfailure-oriented testing",
            "20 common initial queries\nHighest-risk unqueried candidate\n200 total queries; true collisions",
            EGO,
        ),
        (
            0.025,
            0.12,
            "4",
            "Post-task\nknowledge accumulation",
            "Archive executed, paid records\nBuild same-scenario sources\nUpdate shared models between tasks",
            "#009E73",
        ),
    )
    for x, y, number, title, body, color in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                0.41,
                0.30,
                boxstyle="round,pad=0.012",
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
            )
        )
        axis.text(
            x + 0.025,
            y + 0.257,
            number,
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "circle,pad=0.22", "fc": color, "ec": color},
        )
        axis.text(x + 0.062, y + 0.255, title, ha="left", va="center", fontsize=9, color=INK)
        axis.text(
            x + 0.205,
            y + 0.105,
            body,
            ha="center",
            va="center",
            fontsize=8,
            color=INK,
            linespacing=1.65,
        )
    for start, end in (
        ((0.45, 0.72), (0.55, 0.72)),
        ((0.80, 0.55), (0.80, 0.44)),
        ((0.55, 0.27), (0.45, 0.27)),
        ((0.23, 0.44), (0.23, 0.55)),
    ):
        axis.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=INK
            )
        )
    axis.text(0.5, 0.80, "Historical\npriors", ha="center", fontsize=8)
    axis.text(0.84, 0.49, "Risk\nprediction", va="center", fontsize=8)
    axis.text(0.5, 0.19, "Task\ncomplete", ha="center", fontsize=8)
    axis.text(0.19, 0.49, "Future\ntasks", ha="right", va="center", fontsize=8)
    axis.add_patch(
        FancyArrowPatch(
            (0.65, 0.44),
            (0.65, 0.55),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=EGO,
        )
    )
    axis.text(0.61, 0.49, "Paid\nfeedback", ha="right", va="center", fontsize=8, color=EGO)
    axis.text(
        0.5, 0.95, "One independent pipeline per functional scenario", ha="center", fontsize=10
    )
    axis.text(
        0.5,
        0.025,
        "Model and source snapshots remain fixed within each target task.",
        ha="center",
        fontsize=8,
        color=INK,
    )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.01)
    save_figure(figure, path)


def render(destination=OUTPUT / "reports/figures/schematics"):
    destination = Path(destination)
    with plt.rc_context(STYLE):
        traffic_scenarios(destination / "functional_scenarios.png")
        method_flow(destination / "four_stage_method.png")
    write_json(
        destination / "sources.json",
        {
            "purpose": "conceptual diagrams; geometry is illustrative, no outcome data",
            "scenarios": {kind: list(DOMAINS[kind].bounds) for kind in TEST_SCENARIOS},
            "collision_definition": "actual simulation collision event",
            "initial_count": 20,
            "total_budget": 200,
            "format": "PNG",
            "dpi": 600,
        },
    )


if __name__ == "__main__":
    render()
