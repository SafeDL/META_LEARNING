"""Budget discovery and scene composition by target ADS.

The five testing strategies use all eight ADS, four scenes and 200 budget
positions. Pooled GP and static prior remain in the complete result tables.
Curves show descriptive means without uncertainty bands, as requested;
paired uncertainty remains in the audited statistical tables.
"""

import csv
from functools import partial
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from controllers.highway.catalog import FORMAL_TARGET_SUTS

from experiments.budget import protocol
from experiments.paper_results import paper_study
from experiments.storage import read_json, write_json

DESTINATION = protocol.ROOT / "figures/budget_discovery"
PUBLIC_BUDGET = protocol.ROOT / "results/public_ads/budget_discovery"
SCENES = {
    "car_following_braking_onset": "Single-lane braking",
    "cut_in_onset": "Restricted cut-in",
    "cut_out_rear_gap": "Cut-out with rear traffic",
    "side_encroachment": "Side encroachment",
}
METHODS = {
    "full": ("Ours method", "#D04444", "-", "o"),
    "target_only": ("Target-only GP", "#3979B5", "--", "s"),
    "scenario_fuzz": ("ScenarioFuzz", "#559A69", "-.", "^"),
    "paired_system_active_evaluation": ("Paired-system evaluation", "#9470B3", ":", "D"),
    "random": ("Random", "#888888", (0, (5, 2, 1, 2)), "x"),
}
ADS_NAMES = {
    "idm_mobil": "IDM–MOBIL", "fvdm_lane": "FVDM",
    "linear_mpc": "Linear MPC", "nonlinear_trajectory": "Nonlinear planner",
    "uct": "UCT", "beam": "Beam search", "ppo": "PPO", "dqn": "DQN",
}
PUBLIC_ADS = (
    ("online_mc_ppo", "OnlineMC PPO*"),
    ("social_attention_ddqn", "Attention DDQN*"),
    ("decision_transformer", "Decision Trans.*"),
)
SCENE_COLORS = ("#668DB6", "#DBA066", "#9B89B6", "#82AA98")
SCENE_HATCHES = ("", "///", "...", "\\\\")
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
}


def sources(source_root=None):
    report = (source_root or paper_study(protocol.OUTPUT)) / "reports"
    completion = read_json(report / "completion.json")
    if completion["status"] != "complete" or completion["cases"] != 224 or completion["curve_rows"] != 44800:
        raise ValueError("Budget figures require all 224 independently audited results")
    for name, digest in completion["files"].items():
        if protocol.sha(report / name) != digest:
            raise ValueError("Audited budget figure source changed")
    with (report / "discovery_curves.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["sut_id"], r["scenario"], r["method"], int(r["query"])): int(r["collisions"]) for r in rows}
    if len(lookup) != 44800 or len(rows) != len(lookup):
        raise ValueError("Budget figure source has missing or duplicate curve positions")
    data = np.asarray([[[[lookup[s.sut_id, kind, method, query] for query in range(1, 201)]
                         for method in protocol.METHODS] for kind in protocol.RUN_SCENARIOS]
                       for s in FORMAL_TARGET_SUTS])
    if data.shape != (8, 4, 7, 200) or (np.diff(data, axis=-1) < 0).any():
        raise ValueError("Discovery curves must be complete cumulative counts")
    if not (data[:, :, :, :20] == data[:, :, :1, :20]).all():
        raise ValueError("Methods must share all 20 initialization responses")
    return data


def public_sources(public_root=PUBLIC_BUDGET):
    report = Path(public_root) / "reports"
    summary = read_json(report / "summary.json")
    if summary.get("status") != "complete" or summary.get("tasks") != 84 or summary.get("curve_rows") != 16800:
        raise ValueError("Public budget figures require all 84 independently audited results")
    with (report / "discovery_curves.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["target"], r["scenario"], r["method"], int(r["query"])): int(r["collisions"]) for r in rows}
    if len(lookup) != 16800 or len(rows) != len(lookup):
        raise ValueError("Public budget curve table has missing or duplicate positions")
    data = np.asarray([[[[lookup[sut_id, kind, method, query] for query in range(1, 201)]
                         for method in protocol.METHODS] for kind in protocol.RUN_SCENARIOS]
                       for sut_id, _ in PUBLIC_ADS])
    if data.shape != (3, 4, 7, 200) or (np.diff(data, axis=-1) < 0).any():
        raise ValueError("Public discovery curves must be complete cumulative counts")
    if not (data[:, :, :, :20] == data[:, :, :1, :20]).all():
        raise ValueError("Public methods must share all 20 initialization responses")
    return data


def panel_label(ax, index, annotation=""):
    ax.annotate(chr(97 + index), (0, 1), xycoords="axes fraction",
                xytext=(0, 6), textcoords="offset points", fontsize=9,
                fontweight="bold", ha="left", va="bottom")
    if annotation:
        ax.annotate(annotation, (0, 1), xycoords="axes fraction",
                    xytext=(12, 6), textcoords="offset points", fontsize=9,
                    ha="left", va="bottom")


def audit_figure(fig, axes, filename, *, report_root=None):
    """Use the installed figure QA tools; only PNG is retained as a figure asset."""
    scripts = Path(os.environ.get("NATURE_FIGURE_SCRIPTS", Path.home() / ".codex/skills/nature-figure/scripts"))
    source = scripts / "audit_panel_alignment.py"
    spec = importlib.util.spec_from_file_location("budget_panel_alignment", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require_matplotlib_panel_alignment = module.require_matplotlib_panel_alignment
    with tempfile.TemporaryDirectory(prefix="budget-figure-qa-") as temporary:
        directory = Path(temporary)
        alignment = directory / "alignment.json"
        require_matplotlib_panel_alignment(fig, axes=list(axes), json_out=alignment, tolerance_pt=1.5,
                                           gutter_tolerance_pt=1.5,
                                           require_panel_labels=True, strict=True)
        pdf = directory / "figure.pdf"
        fig.savefig(pdf, dpi=600)
        fig.savefig(directory / "figure.svg", dpi=600)
        subprocess.run([sys.executable, "-B", str(scripts / "audit_pdf_text.py"), str(pdf),
                        "--min-pt", "9"], check=True)
        collision = directory / "collisions.json"
        subprocess.run([sys.executable, "-B", str(scripts / "audit_figure_collisions.py"),
                        str(pdf), "--json-out", str(collision)], check=True)
        quality = {"alignment": read_json(alignment),
                   "collision_audit": read_json(collision)}
        write_json((report_root or protocol.OUTPUT / "reports") / "figure_quality" / f"{filename}.json", quality)
        print({"figure": filename, "alignment": quality["alignment"]["verdict"],
               "collision_audit": quality["collision_audit"]["verdict"]}, flush=True)


def save(fig, axes, filename, audit, destination=None):
    visible_axes = [ax for ax in np.ravel(axes) if ax.get_visible()]
    fig.canvas.draw()
    boxes = np.asarray([ax.get_window_extent().bounds for ax in visible_axes])
    tolerance = fig.dpi * 1.5 / 72
    if np.ptp(boxes[:, 2]) > tolerance or np.ptp(boxes[:, 3]) > tolerance:
        raise ValueError("Comparable final plot areas have unequal dimensions")
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.x1 + 1 or bounds.y1 > fig.bbox.y1 + 1:
        raise ValueError("Figure content falls outside its final canvas")
    if audit is not None:
        audit(fig, visible_axes, filename)
    destination = destination or DESTINATION
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{filename}.png"
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)
    return path


def draw_curves(ax, means):
    """Draw every budget position; markers only aid grayscale identification."""
    x = np.arange(201)
    ax.axvspan(0, 20, color="#F2F4F7", zorder=0)
    ax.axvline(20, color="#B5BCC4", linestyle=":", linewidth=0.7, zorder=1)
    for method in reversed(METHODS):
        label, color, line, marker = METHODS[method]
        values = np.r_[0, means[protocol.METHODS.index(method)]]
        ax.step(x, values, where="post", color=color, linestyle=line,
                linewidth=1.65 if method == "full" else 1.0, label=label,
                marker=marker, markevery=[50, 100, 150, 200], markersize=2.5,
                markerfacecolor="white", markeredgewidth=0.65,
                zorder=5 if method == "full" else 3)
    ax.set_xlim(0, 203)
    ax.set_ylim(0, 205)
    ax.grid(color="#E0E0E0", linewidth=0.4)
    ax.set_axisbelow(True)


def shared_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered = [METHODS[m][0] for m in METHODS]
    fig.legend([by_label[label] for label in ordered], ordered, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False,
               prop={"family": "Arial", "size": 9},
               handlelength=2.5, columnspacing=1.6, labelspacing=0.5)


def discovery_curves(data, audit, destination=None):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.21, top=0.92,
                        wspace=0.16, hspace=0.40)
    for index, (kind, ax) in enumerate(zip(protocol.RUN_SCENARIOS, axes.flat)):
        draw_curves(ax, data[:, index].mean(axis=0))
        panel_label(ax, index, SCENES[kind])
        ax.set_xticks((0, 50, 100, 150, 200))
        ax.set_yticks((0, 50, 100, 150, 200))
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean collisions ↑")
    for ax in axes[-1]:
        ax.set_xlabel("Target queries")
    shared_legend(fig, axes.flat[0])
    return save(fig, axes, "discovery_curves", audit, destination)


def ads_discovery_curves(data, audit, destination=None, *, public_data=None):
    """Each ADS panel averages its four independent, equally budgeted scenes."""
    combined = np.concatenate([data, public_data], axis=0) if public_data is not None else data
    labels = [ADS_NAMES[sut.lineage] for sut in FORMAL_TARGET_SUTS]
    if public_data is not None:
        labels.extend(name for _, name in PUBLIC_ADS)
    rows = 3 if public_data is not None else 2
    fig, axes = plt.subplots(rows, 4, figsize=(7.16, 5.85 if rows == 3 else 4.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16 if rows == 3 else 0.22, top=0.93 if rows == 3 else 0.91,
                        wspace=0.25, hspace=0.42)
    flat = list(axes.flat)
    for index, (label, ax) in enumerate(zip(labels, flat)):
        draw_curves(ax, combined[index].mean(axis=0))
        panel_label(ax, index, label)
        ax.set_xticks((0, 100, 200))
        ax.set_yticks((0, 100, 200))
    for ax in flat[len(labels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean collisions ↑")
    if rows == 3:
        for col in range(3):
            axes[2, col].set_xlabel("Queries per scene")
        axes[1, 3].tick_params(labelbottom=True)
        axes[1, 3].set_xlabel("Queries per scene")
    else:
        for ax in axes[-1]:
            ax.set_xlabel("Queries per scene")
    shared_legend(fig, axes.flat[0])
    return save(fig, axes, "ads_discovery_curves", audit, destination)


def ads_collision_composition(data, audit, destination=None, *, public_data=None):
    """Decompose each ADS's four independent 200-query runs by scene."""
    combined = np.concatenate([data, public_data], axis=0) if public_data is not None else data
    labels = [ADS_NAMES[sut.lineage] for sut in FORMAL_TARGET_SUTS]
    if public_data is not None:
        labels.extend(name for _, name in PUBLIC_ADS)
    rows = 3 if public_data is not None else 2
    fig, axes = plt.subplots(rows, 4, figsize=(7.16, 6.25 if rows == 3 else 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.16 if rows == 3 else 0.22, top=0.94 if rows == 3 else 0.92,
                        wspace=0.22, hspace=0.42)
    selected = [protocol.METHODS.index(method) for method in METHODS]
    positions = np.arange(len(selected))
    flat = list(axes.flat)
    for index, (label, ax) in enumerate(zip(labels, flat)):
        bottom = np.zeros(len(selected))
        for si, (color, hatch) in enumerate(zip(SCENE_COLORS, SCENE_HATCHES)):
            counts = combined[index, si, selected, -1]
            ax.barh(positions, counts, left=bottom, height=0.65, color=color,
                    edgecolor="#FFFFFF", linewidth=0.35, hatch=hatch, rasterized=True)
            bottom += counts
        panel_label(ax, index, label)
        ax.set_yticks(positions, [METHODS[m][0].replace(" evaluation", "") for m in METHODS])
        ax.set_ylim(4.65, -0.65)
        ax.set_xlim(0, 825)
        ax.set_xticks((0, 400, 800))
        ax.grid(axis="x", color="#E2E2E2", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "Ours method":
                tick.set_fontweight("bold")
    for ax in flat[len(labels):]:
        ax.set_visible(False)
    if rows == 3:
        for col in range(3):
            axes[2, col].set_xlabel("Total collisions ↑")
        axes[1, 3].tick_params(labelbottom=True)
        axes[1, 3].set_xlabel("Total collisions ↑")
    else:
        for ax in axes[-1]:
            ax.set_xlabel("Total collisions ↑")
    handles = [Patch(facecolor=c, edgecolor="white", hatch=h, label=name)
               for c, h, name in zip(SCENE_COLORS, SCENE_HATCHES, SCENES.values())]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                        ncol=2, frameon=False, prop={"family": "Arial", "size": 9},
                        handlelength=2.5, columnspacing=1.8)
    for patch in legend.get_patches():
        patch.set_rasterized(True)
    return save(fig, axes, "ads_collision_composition", audit, destination)


def render(*, audit=audit_figure, source_root=None, destination=None, include_public=True):
    source_root = source_root or paper_study(protocol.OUTPUT)
    data = sources(source_root)
    public_data = public_sources() if (include_public and (PUBLIC_BUDGET / "reports/summary.json").is_file()) else None
    if audit is audit_figure:
        audit = partial(audit_figure, report_root=source_root / "reports")
    with plt.rc_context(STYLE):
        return [
            discovery_curves(data, audit, destination),
            ads_discovery_curves(data, audit, destination, public_data=public_data),
            ads_collision_composition(data, audit, destination, public_data=public_data),
        ]


if __name__ == "__main__":
    print([str(p) for p in render()])
