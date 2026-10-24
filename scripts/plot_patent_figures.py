"""Render figures 5--7 from the saved collision report without rerunning experiments."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ("history_response", "本发明", "-", "o", 2.1),
    ("random", "（a）随机选择", (0, (5, 3)), "s", 1.15),
    ("space_filling", "（b）空间填充", (0, (1, 2)), "^", 1.15),
    ("history_ranking", "（c）历史危险排序", (0, (6, 2, 1, 2)), "D", 1.15),
    ("gp_collision", "（d）不使用历史的高斯过程分类", (0, (8, 3)), "v", 1.15),
    ("nearest_residual", "（e）最相近历史系统＋被测系统残差", (0, (5, 2, 1, 2, 1, 2)), "x", 1.15),
)
CHECKPOINTS = (16, 32, 64, 128)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def configure_style() -> str:
    chinese_font = next(
        (name for name in ("STSong", "SimSun", "Noto Serif CJK SC", "SimHei", "Microsoft YaHei")
         if any(font.name == name and Path(font.fname).suffix.lower() == ".ttf"
                for font in font_manager.fontManager.ttflist)),
        None,
    )
    require(chinese_font is not None, "A Simplified Chinese font is required.")
    plt.rcParams.update({
        "font.family": [chinese_font, "Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 0.85,
        "axes.edgecolor": "black",
        "axes.unicode_minus": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 10.5,
        "legend.frameon": False,
        "legend.handlelength": 3.0,
        "legend.handletextpad": 0.7,
        "legend.labelspacing": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "patent-figures",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "lines.dash_capstyle": "butt",
        "lines.solid_capstyle": "round",
    })
    return chinese_font


def load_data(path: Path) -> tuple[dict, dict, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report["status"] == "complete", "The report is incomplete.")
    require(report["run_count"] == 10880, "Expected the report for 10,880 runs.")
    require(report["protocol"]["checkpoints"] == list(CHECKPOINTS), "Unexpected checkpoints.")
    require(report["protocol"]["budget"] == 128, "Expected a 128-query budget.")
    methods = report["studies"]["overall"]["new"]["methods"]
    selected = {}
    for name, label, *_ in METHODS:
        row = methods[name]
        require(row["runs"] == 160 and row["independent_lineages"] == 8,
                f"Unexpected sampling population for {name}.")
        values = [row["checkpoint_collisions"][str(step)] for step in CHECKPOINTS]
        require(np.isfinite(values).all() and np.all(np.diff(values) >= 0),
                f"Invalid discovery checkpoints for {name}.")
        survival = row["first_collision"]
        require(survival["aggregation"] == "equal_weight_lineage_kaplan_meier"
                and survival["restriction"] == 128, f"Unexpected survival estimator for {name}.")
        times = np.array([point["time"] for point in survival["curve"]])
        probabilities = np.array([point["survival"] for point in survival["curve"]])
        require(times[0] == 0 and times[-1] == 128 and np.all(np.diff(times) > 0),
                f"Invalid survival times for {name}.")
        require(probabilities[0] == 1 and np.isfinite(probabilities).all()
                and np.all((probabilities >= 0) & (probabilities <= 1))
                and np.all(np.diff(probabilities) <= 1e-12), f"Invalid survival curve for {name}.")
        selected[name] = {"label": label, "checkpoint_collisions": values,
                          "first_collision_curve": survival["curve"],
                          "runs": row["runs"], "independent_lineages": row["independent_lineages"]}

    groups = {"same_family": [], "alternative_family": []}
    scopes = set()
    for row in report["source_adaptation"]:
        if row["method"] != "history_response":
            continue
        _, target, alternative, source = row["scope"].split("_")
        require(target != alternative and source in (target, alternative), "Invalid source-swap scope.")
        require(row["scope"] not in scopes, "Duplicate source-adaptation scope.")
        scopes.add(row["scope"])
        weights = np.asarray(row["mean_weight_after_each_paid_query"], dtype=float)
        columns = row["columns"]
        require(weights.shape == (128, 5) and len(set(columns)) == 5,
                "Unexpected source-weight dimensions.")
        require(np.isfinite(weights).all() and np.all(weights >= 0)
                and np.allclose(weights.sum(axis=1), 1), "Invalid fusion weights.")
        require(row["runs"] == 40 and row["independent_lineages"] == 2,
                "Unexpected source-swap sampling population.")
        excluded = alternative if source == target else target
        require(np.allclose(weights[:, columns.index(excluded)], 0),
                "An excluded family has nonzero weight.")
        key = "same_family" if source == target else "alternative_family"
        groups[key].append({
            "scope": row["scope"], "runs": row["runs"],
            "source_weight": weights[:, columns.index(source)].tolist(),
            "self_learning_weight": weights[:, columns.index("target_only")].tolist(),
        })
    require(all(len(rows) == 12 for rows in groups.values()), "Expected 12 scopes in each source-swap arm.")
    for rows in groups.values():
        rows.sort(key=lambda row: row["scope"])
    return report, selected, groups


def format_axes(ax: plt.Axes, *, survival: bool = False, weights: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="black", linestyle=(0, (1, 5)), linewidth=0.35)
    # The requested artwork contains only numerical ticks and essential legends.
    ax.set_xlabel("")
    ax.set_ylabel("")
    if survival or weights:
        ax.set_xlim(0, 128)
        ax.xaxis.set_major_locator(FixedLocator([0, 32, 64, 96, 128]))
        ax.set_ylim(0, 1.025)
        ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 1, 6)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    else:
        ax.set_xlim(12, 132)
        ax.xaxis.set_major_locator(FixedLocator(CHECKPOINTS))
        ax.set_ylim(0, 36)
        ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 36, 5)))
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily("Times New Roman")


def method_style(index: int) -> dict:
    _, label, linestyle, marker, width = METHODS[index]
    return {"label": label, "color": "black", "linestyle": linestyle,
            "marker": marker, "linewidth": width, "markersize": 5.2,
            "markerfacecolor": "white", "markeredgecolor": "black",
            "markeredgewidth": 0.9, "zorder": 6 if index == 0 else 3}


def method_legend(fig: plt.Figure, handles: list) -> None:
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=2, columnspacing=2.2, borderaxespad=0)


def discovery_figure(selected: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.09, top=0.755)
    handles = []
    for index, (name, *_) in enumerate(METHODS):
        line, = ax.plot(CHECKPOINTS, selected[name]["checkpoint_collisions"], **method_style(index))
        handles.append(line)
    format_axes(ax)
    method_legend(fig, handles)
    return fig


def survival_figure(selected: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.09, top=0.755)
    handles = []
    for index, (name, *_) in enumerate(METHODS):
        curve = selected[name]["first_collision_curve"]
        times = np.array([point["time"] for point in curve])
        values = np.array([point["survival"] for point in curve])
        # Each method has its own marker positions; no vertical offset is applied.
        marker_indices = []
        for position in (12 + 3 * index, 44 + 3 * index, 76 + 3 * index, 108 + 3 * index):
            marker_indices.append(int(np.searchsorted(times, position, side="right") - 1))
        line, = ax.step(times, values, where="post", markevery=sorted(set(marker_indices)),
                        **method_style(index))
        handles.append(line)
    format_axes(ax, survival=True)
    method_legend(fig, handles)
    return fig


def adaptation_figure(groups: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.11, top=0.745, wspace=0.16)
    queries = np.arange(1, 129)
    for ax, (key, rows), history_label in zip(
        axes, groups.items(), ("同范式历史分量", "异范式来源分量")
    ):
        history = np.array([row["source_weight"] for row in rows])
        self_learning = np.array([row["self_learning_weight"] for row in rows])
        handles = []
        if key == "same_family":
            for curve in history:
                ax.plot(queries, curve, color="black", linewidth=0.45, zorder=2)
            handles.append(Line2D([], [], color="black", linewidth=0.45,
                                  label=f"{history_label}（各配置）"))
        mean_line, = ax.plot(queries, history.mean(axis=0), color="black", linewidth=2.1,
                             label=f"{history_label}（均值）", zorder=4)
        self_line, = ax.plot(queries, self_learning.mean(axis=0), color="black",
                             linewidth=1.4, linestyle=(0, (6, 2, 1, 2)),
                             label="自学习分量（均值）", zorder=5)
        handles.extend((mean_line, self_line))
        format_axes(ax, weights=True)
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.065),
                  borderaxespad=0, fontsize=10, handlelength=2.5)
    axes[1].tick_params(labelleft=False)
    return fig


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for legend in (*fig.legends, *(ax.get_legend() for ax in fig.axes)):
        if legend is None:
            continue
        bounds = legend.get_window_extent(renderer)
        require(fig.bbox.contains(bounds.x0, bounds.y0) and fig.bbox.contains(bounds.x1, bounds.y1),
                f"Legend extends outside the canvas: {stem.name}")
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.convert("L").save(stem.with_suffix(".png"), dpi=(dpi, dpi), optimize=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results/gcrt/collision/report/report.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "patent_figures")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    require(args.dpi >= 300, "Use at least 300 dpi for publication artwork.")
    report, selected, groups = load_data(args.report)
    chinese_font = configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        ("fig05_cumulative_collisions", discovery_figure(selected)),
        ("fig06_first_collision_survival", survival_figure(selected)),
        ("fig07_fusion_weights", adaptation_figure(groups)),
    )
    for name, fig in figures:
        save_figure(fig, args.output_dir / name, args.dpi)
        print(f"Saved {name}: SVG, PDF, PNG ({args.dpi} dpi)")
    provenance = {
        "report": str(args.report.resolve()),
        "report_sha256": sha256(args.report.read_bytes()).hexdigest(),
        "report_run_count": report["run_count"],
        "checkpoints": CHECKPOINTS,
        "chinese_font": chinese_font,
        "artwork_text": "essential_legends_and_numeric_ticks_only",
        "methods": selected,
        "source_adaptation": groups,
    }
    (args.output_dir / "figure_data.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
