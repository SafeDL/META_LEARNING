"""Plot the audited E05 sequence comparisons at double-column width."""

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from experiments.plotting.style import FIGURE_SCRIPTS, STYLE, plt
from experiments.accumulation.report import REPORTS
from experiments.accumulation.run import ROOT, RUN_SCENARIOS, sha
from experiments.storage import read_json, write_json

OUTPUT = ROOT / "figures/knowledge_accumulation"
FIGURE = "paired_accumulation_effects"
SCENES = dict(zip(RUN_SCENARIOS, (
    "Car-following braking", "Constrained cut-in", "Cut-out with obstacle", "Side encroachment"
)))
GROUP_LABELS = {"group_a": "A", "group_b": "B"}
ORDER_LABELS = {"forward": "→", "reverse": "←"}
APPEND = "#14776F"
UPDATE = "#80578D"
LATER = "#2F597E"
RANGE = "#9FBAB6"
INK = "#243238"


def number(row, field):
    return float(row[field]) if row[field] else None


def panel_limits(rows, fields):
    values = [0.0]
    for row in rows:
        values.extend(float(row[field]) for field in fields if row[field])
    lower, upper = min(values), max(values)
    margin = max(0.05, (upper - lower) * 0.085)
    return lower - margin, upper + margin


def export(fig, axes):
    locator = importlib.util.spec_from_file_location(
        "sequence_panel_alignment", FIGURE_SCRIPTS / "audit_panel_alignment.py"
    )
    alignment_module = importlib.util.module_from_spec(locator)
    sys.modules[locator.name] = alignment_module
    locator.loader.exec_module(alignment_module)
    require_matplotlib_panel_alignment = alignment_module.require_matplotlib_panel_alignment
    with TemporaryDirectory(prefix="accumulation-figure-qa-") as name:
        temporary = Path(name)
        alignment = temporary / "alignment.json"
        require_matplotlib_panel_alignment(
            fig, axes=axes, json_out=alignment, tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
        )
        fig.savefig(f"{temporary}/figure.svg")
        fig.savefig(f"{temporary}/figure.pdf")
        pdf = temporary / "figure.pdf"
        collision = temporary / "collision.json"
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        subprocess.run([sys.executable, "-B", str(FIGURE_SCRIPTS / "audit_pdf_text.py"),
                        str(pdf), "--min-pt", "5"], check=True, env=env)
        subprocess.run([sys.executable, "-B", str(FIGURE_SCRIPTS / "audit_figure_collisions.py"),
                        str(pdf), "--json-out", str(collision), "--strict"], check=True, env=env)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / f"{FIGURE}.png"
        fig.savefig(path, dpi=600)
        write_json(REPORTS / "figure_quality" / f"{FIGURE}.json", {
            "alignment": read_json(alignment), "collision_audit": read_json(collision),
            "png_sha256": sha(path), "width_in": 7.16, "height_in": 5.8, "dpi": 600,
        })
    return path


def render():
    completion = read_json(REPORTS / "completion.json")
    if completion["positions"] != 192 or completion["distinct_paths"] != 112:
        raise ValueError("E05 plot requires the full registered matrix")
    for filename, checksum in completion["files"].items():
        if sha(REPORTS / filename) != checksum:
            raise ValueError(f"E05 plotted source changed after audit: {filename}")
    with (REPORTS / "sequences.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = completion["summary"]["scenes"]
    archive_limits = panel_limits(rows, ("task2", "task3", "task4"))
    update_limits = panel_limits(rows, ("task3_update_append_auc", "task4_update_append_auc"))
    with plt.rc_context({
        **STYLE, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8,
        "axes.labelsize": 8.3, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "legend.fontsize": 7.5, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    }):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 5.8), sharey=True)
        fig.subplots_adjust(left=0.219, right=0.98, bottom=0.16, top=0.95, wspace=0.18)
        tick_positions, tick_labels = [], []
        for scene_index, scene in enumerate(RUN_SCENARIOS):
            first = scene_index * 5.65
            scene_rows = [row for row in rows if row["scene"] == scene]
            if len(scene_rows) != 4:
                raise ValueError("Each traffic scene must have four registered sequences")
            tick_positions.append(first)
            tick_labels.append(SCENES[scene])
            for ax in (a, b):
                ax.axhspan(first - 0.43, first + 0.43, color="#F0F4F4", linewidth=0, zorder=0)
            scenes = summary[scene]
            mean_a = scenes["append_balanced_auc"]
            if mean_a is not None:
                a.plot(mean_a, first, marker="D", ms=5.3, mfc=APPEND, mec="white", mew=0.6, zorder=4)
            for task, offset, color, marker in (("task3", -.14, UPDATE, "o"), ("task4", .14, LATER, "s")):
                value = scenes[task]["mean"]
                if value is not None and scenes[task]["paired"] == scenes[task]["total"]:
                    b.plot(value, first + offset, marker=marker, ms=4.5, mfc=color,
                           mec="white", mew=0.6, zorder=4)
            for offset, row in enumerate(scene_rows, 1):
                y = first + offset
                tick_positions.append(y)
                tick_labels.append(f'{GROUP_LABELS[row["group"]]} {ORDER_LABELS[row["order"]]}')
                all_task_values = [number(row, f"task{i}") for i in range(2, 5)]
                paired = [v for v in all_task_values if v is not None]
                if paired:
                    a.hlines(y + .20, min(paired), max(paired), color=RANGE, linewidth=0.9, zorder=2)
                    a.scatter(paired, [y + .20] * len(paired), s=9, color=RANGE, linewidths=0, zorder=2)
                value = number(row, "mean_append_fixed_auc")
                if value is not None:
                    a.plot(value, y - .20, marker="D", ms=4.1, mfc="white", mec=APPEND, mew=0.95, zorder=4)
                for name, offset, color, marker in (
                    ("task3_update_append_auc", -.17, UPDATE, "o"),
                    ("task4_update_append_auc", .17, LATER, "s"),
                ):
                    value = number(row, name)
                    if value is not None:
                        b.plot(value, y + offset, marker=marker, ms=4, mfc="white",
                               mec=color, mew=1, zorder=4)
        a.set_yticks(tick_positions, tick_labels)
        for position, label in enumerate(a.get_yticklabels()):
            if position % 5 == 0:
                label.set_weight("bold")
        a.set_ylim(tick_positions[-1] + 0.7, -0.65)
        a.set_xlabel("ΔAUC  (Append sources − Fixed history)")
        b.set_xlabel("ΔAUC  (Ours method − Append sources)")
        b.tick_params(axis="y", labelleft=False)
        for ax, panel, limits in ((a, "a", archive_limits), (b, "b", update_limits)):
            ax.text(0.0, 1.02, panel, transform=ax.transAxes, fontsize=9, fontweight="bold",
                    ha="left", va="bottom")
            ax.set_xlim(limits)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=(1, 2, 2.5, 5, 10)))
            ax.set_axisbelow(True)
            ax.xaxis.grid(color="#E5EBED", linewidth=0.48)
            ax.axvline(0, color="#77858B", linewidth=0.72, zorder=1)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.tick_params(axis="x", length=2.8, width=0.6)
            ax.spines["left"].set_visible(False)
        handles = [
            Line2D([], [], color=RANGE, marker="o", ms=3.5, linewidth=0.9, label="Tasks 2–4, paired range"),
            Line2D([], [], color=APPEND, marker="D", mfc="white", ms=4.3, linestyle="none", label="Sequence mean"),
            Line2D([], [], color=UPDATE, marker="o", mfc="white", ms=4.8, linestyle="none", label="Task 3, model update"),
            Line2D([], [], color=LATER, marker="s", mfc="white", ms=4.8, linestyle="none", label="Task 4, accumulated effect"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.58, .021),
                   ncol=2, columnspacing=1.7, handletextpad=.55, frameon=False)
        fig.canvas.draw()
        path = export(fig, [a, b])
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(render())
