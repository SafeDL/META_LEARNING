"""Publication PNGs and numerical summaries for candidate and compute scaling."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .joint_figures import MARKERS, STYLE, STYLES, defined_mean, save_figure
from .joint_validation import OUTPUT
from .report_joint import report
from .storage import read_json, write_json

METRICS = (
    ("unique_collisions", "Discovered failures"),
    ("collision_recall", "Candidate-library\nrecall"),
    ("pair_distance", "Mean pairwise\ndistance"),
    ("nearest_neighbor_distance", "Mean nearest-neighbor\ndistance"),
)


@plt.rc_context(STYLE)
def sensitivity(root=OUTPUT / "sensitivity"):
    root = Path(root)
    data = read_json(root / "reports/summary.json")
    if data["status"] != "complete":
        raise ValueError("Candidate sensitivity figures require the completed comparison")
    rows = data["runs"]
    sizes = sorted({r["candidate_count"] for r in rows})
    systems = list(dict.fromkeys(r["sut_id"] for r in rows))
    methods = [m for m in STYLES if any(r["method"] == m for r in rows)]
    indexed = {(r["sut_id"], r["method"], r["candidate_count"]): r for r in rows}
    if len(indexed) != len(rows) or len(indexed) != len(systems) * len(methods) * len(sizes):
        raise ValueError("Sensitivity figures require one paired run per ADS, method and pool size")
    pools = []
    for system in systems:
        for n in sizes:
            counts = {indexed[system, m, n]["metrics"]["pool_unique_collisions"] for m in methods}
            budgets = {len(indexed[system, m, n]["metrics"]["discovery_curve"]) for m in methods}
            if len(counts) != 1 or len(budgets) != 1:
                raise ValueError("Compared methods use different candidate truth or budgets")
            count, budget = counts.pop(), budgets.pop()
            pools.append(
                {
                    "sut_id": system,
                    "candidates": n,
                    "true_failures": count,
                    "failure_rate": count / n,
                    "budget": budget,
                    "budget_fraction": budget / n,
                }
            )
    for n in sizes:
        report(root / str(n))
    files, defined = [], {}
    for metric, label in METRICS:
        panels = [*systems, None]
        columns = min(3, len(panels))
        nrows = math.ceil(len(panels) / columns)
        legend_height = 0.23 * math.ceil(len(methods) / 2) + 0.15
        figure, axes = plt.subplots(
            nrows,
            columns,
            figsize=(180 / 25.4, 2.0 * nrows + legend_height + 0.2),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        handles, values_shown = {}, []
        defined[metric] = {}
        for panel, (axis, system) in enumerate(zip(axes.flat, panels)):
            panel_defined = False
            for method in methods:
                title, color, linestyle = STYLES[method]
                if system is None:
                    values, counts = defined_mean(
                        [[indexed[s, method, n]["metrics"][metric] for n in sizes] for s in systems]
                    )
                    defined[metric][method] = counts.tolist()
                    title += " (n=" + "/".join(str(n) for n in counts) + ")"
                else:
                    values = np.asarray(
                        [indexed[system, method, n]["metrics"][metric] for n in sizes], dtype=float
                    )
                (line,) = axis.plot(
                    sizes,
                    values,
                    label=title,
                    color=color,
                    linestyle=linestyle,
                    marker=MARKERS[method],
                    markersize=3.5,
                    linewidth=1.25,
                )
                values_shown.extend(v for v in values if np.isfinite(v))
                panel_defined = panel_defined or np.isfinite(values).any()
                handles[method] = line
            title = "Equal-ADS mean" if system is None else system.removeprefix("independent_")
            axis.set(
                title=f"({chr(97 + panel)}) {title}",
                xlabel="Candidate count",
                ylabel=label,
                xticks=sizes,
            )
            axis.tick_params(axis="x", labelsize=8)
            axis.grid(alpha=0.18, linewidth=0.5)
            if not panel_defined:
                reason = (
                    "no failures in library"
                    if metric == "collision_recall"
                    else "fewer than two discoveries"
                )
                axis.text(
                    0.5,
                    0.5,
                    "Undefined:\n" + reason,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#777777",
                )
        for axis in axes.flat[len(panels) :]:
            axis.set_visible(False)
        axes.flat[0].set_ylim(
            0, 1 if metric == "collision_recall" else max(values_shown or [1]) * 1.1 or 1
        )
        figure.legend(
            handles=list(handles.values()), loc="lower center", ncol=2, frameon=False, fontsize=8
        )
        figure.tight_layout(rect=(0, (legend_height + 0.1) / figure.get_figheight(), 1, 1))
        path = root / "reports/figures/sensitivity" / f"front_brake_{metric}.png"
        save_figure(figure, path)
        files.append(path.relative_to(root / "reports").as_posix())
    result = {"figures": files, "candidate_sizes": sizes, "pools": pools, "defined_ads": defined}
    write_json(root / "reports/figure_data.json", result)
    return result


@plt.rc_context(STYLE)
def scaling(root=OUTPUT / "scaling"):
    root = Path(root)
    data = read_json(root / "reports/scaling.json")
    if data["status"] != "complete":
        raise ValueError("Compute figures require completed measurements")
    rows = data["runs"]
    axes_names = list(dict.fromkeys(r["axis"] for r in rows))
    devices = list(dict.fromkeys(r["device"] for r in rows))
    factors = {
        "sources": "Historical sources",
        "source_records": "Records per source",
        "candidates": "Candidates",
        "target_records": "Paid target records",
    }
    metrics = (
        ("source_construction_s", "Source construction (s)", False),
        ("adaptation_s", "Target adaptation (s)", False),
        ("prediction_s", "Candidate prediction (s)", False),
        ("sampled_peak_cpu_bytes", "Peak host RSS (MiB)", True),
        ("peak_cuda_allocated_bytes", "Peak CUDA allocation (MiB)", True),
    )
    files = []
    for metric, label, memory in metrics:
        if metric == "peak_cuda_allocated_bytes" and not any(r["resources"][metric] for r in rows):
            continue
        columns = min(2, len(axes_names))
        nrows = math.ceil(len(axes_names) / columns)
        figure, axes = plt.subplots(
            nrows, columns, figsize=(180 / 25.4, 2.4 * nrows + 0.45), squeeze=False
        )
        handles = {}
        for panel, (axis, factor) in enumerate(zip(axes.flat, axes_names)):
            if factor == "source_records":
                axis.set_xscale("log", base=2)
            for index, device in enumerate(devices):
                measurements = sorted(
                    (r for r in rows if r["axis"] == factor and r["device"] == device),
                    key=lambda r: r["settings"][factor],
                )
                values = [
                    (r["resources"][metric] / 2**20 if memory else r[metric]) for r in measurements
                ]
                if metric == "peak_cuda_allocated_bytes" and not any(values):
                    continue
                x = [r["settings"][factor] for r in measurements]
                (line,) = axis.plot(
                    x,
                    values,
                    label=device.upper(),
                    marker=("o", "s")[index % 2],
                    color=("#0072B2", "#D55E00")[index % 2],
                    linestyle=("--", "-")[index % 2],
                    linewidth=1.4,
                )
                handles[device] = line
                axis.set_xticks(x, [str(value) for value in x])
            xlabel = factors[factor] + (" (log scale)" if factor == "source_records" else "")
            axis.set(title=f"({chr(97 + panel)}) {factors[factor]}", xlabel=xlabel, ylabel=label)
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.18, linewidth=0.5)
        for axis in axes.flat[len(axes_names) :]:
            axis.set_visible(False)
        figure.legend(
            handles=list(handles.values()), loc="lower center", ncol=len(handles), frameon=False
        )
        figure.tight_layout(rect=(0, 0.07, 1, 1))
        path = root / "reports/figures/scaling" / f"{metric}.png"
        save_figure(figure, path)
        files.append(path.relative_to(root / "reports").as_posix())
    result = {
        "figures": files,
        "hardware": data["hardware"],
        "warmups_excluded": True,
        "runs": len(rows),
        "performance_evidence": "compute only; historical response replicas",
    }
    write_json(root / "reports/figure_data.json", result)
    return result


if __name__ == "__main__":
    sensitivity()
    scaling()
