"""Publication figures from evaluator statistics, with separate comparison groups."""

import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

STYLES = {
    "joint": ("History transfer + joint", "#D55E00", "-"),
    "random": ("Random", "#777777", ":"),
    "av_bo_rf_ei": ("RF-EI", "#0072B2", "--"),
    "samota": ("SAMOTA", "#009E73", "-."),
    "scenario_fuzz": ("ScenarioFuzz", "#CC79A7", "-"),
    "paired_system_active_evaluation": ("Paired-system", "#56B4E9", "--"),
    "indago": ("Indago (history)", "#8C564B", "-."),
    "behavexplor": ("BehAVExplor", "#BCBD22", ":"),
    "collision": ("Target collision only", "#332288", "--"),
    "no_history": ("Target joint GP", "#44AA99", "-."),
    "historical_collision": ("History collision only", "#332288", "--"),
    "embedding_only": ("Shared features only", "#44AA99", "-."),
    "aggregate": ("Aggregate history", "#CC79A7", ":"),
    "fixed_weights": ("Fixed source weights", "#0072B2", "--"),
    "fixed_history": ("Fixed history", "#777777", ":"),
    "append_sources": ("Append sources", "#0072B2", "--"),
    "update_shared": ("Update shared model", "#D55E00", "-"),
}
SYSTEM_MARKERS = ("o", "s", "^", "D", "v", "P", "h", "*")
MARKERS = {name: SYSTEM_MARKERS[i % 8] for i, name in enumerate(STYLES)}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
CURVES = (
    ("discovery", "discovery_curve", "Discovered failures", "queries"),
    ("recall", "recall_curve", "Candidate-library recall", "queries"),
    ("pair_distance", "pair_distance", "Mean pairwise distance", "queries"),
    ("nearest_neighbor", "nearest_neighbor_distance", "Mean nearest-neighbor distance", "queries"),
    ("elapsed", "discovery_curve", "Discovered failures", "elapsed"),
    ("equal_count_pair", "pair_distance", "Mean pairwise distance", "discoveries"),
    (
        "equal_count_neighbor",
        "nearest_neighbor_distance",
        "Mean nearest-neighbor distance",
        "discoveries",
    ),
)


def comparison_sets(rows):
    if not any("comparison" in row for row in rows):
        return {"all": rows}
    main = [r for r in rows if r["comparison"] == "main"]
    controls = [r for r in rows if r["comparison"] == "control"]
    groups = {"main": main} if len({r["method"] for r in main}) > 1 or not controls else {}
    if controls:
        groups["controls"] = [
            r for r in rows if r["comparison"] == "control" or r["method"] == "joint"
        ]
    return groups


@plt.rc_context(STYLE)
def plot_sequence(rows, root, updates):
    files = []
    for kind in dict.fromkeys(row["scenario"] for row in rows):
        selected = [row for row in rows if row["scenario"] == kind]
        reference = {row["ordinal"]: row for row in selected if row["regime"] == "fixed_history"}
        order = sorted(reference)
        figure, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 135 / 25.4))
        panels = (
            ("unique_collisions", "Discovered failures", False),
            ("collision_recall", "Candidate-library recall", False),
            ("unique_collisions", "Failure difference vs. fixed", True),
            ("collision_recall", "Recall difference vs. fixed", True),
        )
        handles = {}
        for panel, (axis, (metric, label, difference)) in enumerate(zip(axes.flat, panels)):
            for regime in ("fixed_history", "append_sources", "update_shared"):
                if difference and regime == "fixed_history":
                    continue
                runs = {row["ordinal"]: row for row in selected if row["regime"] == regime}
                if runs.keys() != reference.keys():
                    raise ValueError("Chronological comparison requires the same ADS order")
                values = []
                for ordinal in order:
                    value, baseline = (
                        runs[ordinal]["metrics"][metric],
                        reference[ordinal]["metrics"][metric],
                    )
                    values.append(
                        np.nan
                        if value is None or (difference and baseline is None)
                        else value - baseline
                        if difference
                        else value
                    )
                title, color, linestyle = STYLES[regime]
                (line,) = axis.plot(
                    order,
                    values,
                    marker=MARKERS[regime],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.3,
                    label=title,
                )
                handles[regime] = line
            for update in updates:
                if update["scenario"] == kind:
                    axis.axvline(
                        update["after_systems"] + 0.5, linestyle=":", color="#999999", linewidth=0.7
                    )
                    axis.text(
                        update["after_systems"] + 0.5,
                        0.98,
                        "Update" if update["publication"]["accepted"] else "Update failed",
                        transform=axis.get_xaxis_transform(),
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=8,
                    )
            if difference:
                axis.axhline(0, color="#999999", linewidth=0.7)
            elif metric == "collision_recall":
                axis.set_ylim(0, 1)
            else:
                axis.set_ylim(bottom=0)
            axis.set(title=f"({chr(97 + panel)}) {label}", ylabel=label, xticks=order)
            axis.set_xticklabels(
                [f"{n}. {reference[n]['sut_id'].removeprefix('independent_')}" for n in order],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            axis.grid(axis="y", alpha=0.18, linewidth=0.5)
        figure.legend(handles=list(handles.values()), loc="lower center", ncol=3, frameon=False)
        figure.tight_layout(rect=(0, 0.065, 1, 1))
        path = root / "figures/sequential" / f"{kind}_task_order.png"
        save_figure(figure, path)
        files.append(path.relative_to(root).as_posix())
    return files


@plt.rc_context(STYLE)
def plot_retention(rows, root):
    files = []
    for kind in dict.fromkeys(row["scenario"] for row in rows):
        selected = [row for row in rows if row["scenario"] == kind]
        updates = sorted({row["after_systems"] for row in selected})
        counts = [sum(row["after_systems"] == n for row in selected) for n in updates]
        figure, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 3.1))
        for panel, (axis, metric) in enumerate(zip(axes, ("nll_change", "brier_change"))):
            means = []
            for update in updates:
                values = [row[metric] for row in selected if row["after_systems"] == update]
                offsets = np.linspace(-0.1, 0.1, len(values)) if len(values) > 1 else [0]
                dots = axis.scatter(
                    update + np.asarray(offsets),
                    values,
                    color="#777777",
                    s=14,
                    alpha=0.7,
                    label="Completed tasks",
                )
                means.append(float(np.mean(values)))
            (line,) = axis.plot(
                updates, means, marker="D", color="#D55E00", label="Task mean", linewidth=1.2
            )
            axis.axhline(0, color="#999999", linewidth=0.7)
            axis.set(
                title=f"({chr(97 + panel)}) {'NLL' if panel == 0 else 'Brier'} change",
                ylabel="After update minus before update",
                xticks=updates,
                xticklabels=[f"After {n} ADS\n(n={count})" for n, count in zip(updates, counts)],
            )
            axis.grid(axis="y", alpha=0.18, linewidth=0.5)
        figure.suptitle("Completed-task replay (including shared-training records)", fontsize=9)
        figure.legend(handles=[dots, line], loc="lower center", ncol=2, frameon=False)
        figure.tight_layout(rect=(0, 0.09, 1, 0.98))
        path = root / "figures/sequential" / f"{kind}_completed_task_replay.png"
        save_figure(figure, path)
        files.append(path.relative_to(root).as_posix())
    return files


def defined_mean(values):
    values = np.asarray(values, dtype=float)
    counts = np.isfinite(values).sum(axis=0)
    means = np.divide(
        np.nansum(values, axis=0), counts, out=np.full(counts.shape, np.nan), where=counts > 0
    )
    return means, counts


def curve_data(measured, metric, abscissa, common_count):
    if abscissa == "discoveries":
        values = measured["distances_by_discovery_count"][metric][1:common_count]
        return np.arange(2, common_count + 1), np.asarray(values, dtype=float)
    values = (
        measured["distance_curves"][metric]
        if metric in ("pair_distance", "nearest_neighbor_distance")
        else measured[metric]
    )
    x = (
        measured["equivalent_testing_curve_s"]
        if abscissa == "elapsed"
        else np.arange(1, len(values) + 1)
    )
    return np.asarray(x), np.asarray(values, dtype=float)


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=600)
    plt.close(figure)


def plot_comparison(rows, destination, group_key):
    systems = list(dict.fromkeys(r["sut_id"] for r in rows))
    methods = [name for name in STYLES if any(r[group_key] == name for r in rows)]
    runs = {(r["sut_id"], r[group_key]): r for r in rows}
    if len(runs) != len(rows) or len(runs) != len(systems) * len(methods):
        raise ValueError("Single-seed plots require exactly one run per task and method")
    common = {s: min(runs[s, m]["metrics"]["unique_collisions"] for m in methods) for s in systems}
    files, diagnostics = [], {}
    for name, metric, label, abscissa in CURVES:
        panels = [*systems, None] if len(systems) > 1 and abscissa == "queries" else systems
        columns = 3 if len(panels) >= 8 else min(2, len(panels))
        nrows = math.ceil(len(panels) / columns)
        legend_in_panel = nrows * columns > len(panels)
        legend_height = 0 if legend_in_panel else 0.24 * math.ceil(len(methods) / 2) + 0.3
        height = 2.3 * nrows + legend_height
        figure, axes = plt.subplots(
            nrows, columns, figsize=(180 / 25.4, height), squeeze=False, sharey=True
        )
        handles = {}
        diagnostics[name] = {}
        for panel, (axis, system) in enumerate(zip(axes.flat, panels)):
            has_defined_values = False
            for method in methods:
                title, color, linestyle = STYLES[method]
                if system is None:
                    series = [
                        curve_data(runs[s, method]["metrics"], metric, abscissa, common[s])
                        for s in systems
                    ]
                    x = series[0][0]
                    if any(not np.array_equal(x, entry[0]) for entry in series):
                        raise ValueError("Mean curves require the same query budget")
                    values, counts = defined_mean([entry[1] for entry in series])
                    diagnostics[name][method] = {"defined_tasks": counts.tolist()}
                else:
                    x, values = curve_data(
                        runs[system, method]["metrics"], metric, abscissa, common[system]
                    )
                defined_values = bool(np.isfinite(values).any())
                has_defined_values |= defined_values
                line = axis.step(
                    x,
                    values,
                    where="post",
                    label=title,
                    color=color,
                    linestyle=linestyle,
                    marker=MARKERS[method],
                    markersize=2.5,
                    markevery=max(1, len(values) // 8),
                    linewidth=1.6 if method in ("joint", "update_shared") else 1.2,
                    # An empty unclipped line contributes an origin bbox to tight_layout.
                    clip_on=not defined_values,
                )[0]
                handles[title] = line
            if abscissa == "queries":
                initial = {r["metrics"]["initial_count"] for r in rows}
                if len(initial) != 1:
                    raise ValueError("Compared methods must share initialization")
                axis.axvline(next(iter(initial)), color="#999999", linewidth=0.6, linestyle=":")
            title = "Mean across ADSs" if system is None else system.removeprefix("independent_")
            undefined_reason = None
            if system is not None and abscissa == "discoveries" and common[system] < 2:
                undefined_reason = "Undefined: fewer than two\ncommon discoveries"
            elif not has_defined_values:
                if metric == "recall_curve":
                    undefined_reason = "Undefined: no true failures\nin candidate libraries"
                elif metric in ("pair_distance", "nearest_neighbor_distance"):
                    undefined_reason = "Undefined: fewer than two\ndiscovered failures"
            if undefined_reason is not None:
                if columns == 3:
                    undefined_reason = textwrap.fill(undefined_reason.replace("\n", " "), width=22)
                axis.text(
                    0.5,
                    0.5,
                    undefined_reason,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
            axis.set(
                title=f"({chr(97 + panel)}) {title}",
                ylabel=label,
                xlabel={
                    "queries": "Paid queries",
                    "elapsed": "Equivalent testing time (s)",
                    "discoveries": "First k discovered failures",
                }[abscissa],
            )
            if metric == "recall_curve":
                axis.set_ylim(0, 1)
            elif metric in ("pair_distance", "nearest_neighbor_distance"):
                distances = [
                    value
                    for row in rows
                    for value in curve_data(
                        row["metrics"], metric, abscissa, common[row["sut_id"]]
                    )[1]
                    if np.isfinite(value)
                ]
                axis.set_ylim(0, min(1, max(0.01, 1.06 * max(distances, default=1))))
            else:
                maximum = max(r["metrics"]["unique_collisions"] for r in rows)
                axis.set_ylim(0, max(1, 1.06 * maximum))
                if system is not None:
                    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            if abscissa == "discoveries":
                axis.set_xlim(1, max(2, max(common.values())))
            elif abscissa == "elapsed":
                elapsed = [r["metrics"]["equivalent_testing_curve_s"] for r in rows]
                maximum = max(values[-1] for values in elapsed)
                minimum = min(values[0] for values in elapsed)
                if minimum > 0 and maximum >= 10 * min(values[-1] for values in elapsed):
                    axis.set_xscale("log")
                    axis.set_xlim(0.9 * minimum, 1.03 * maximum)
                    axis.set_xlabel("Equivalent testing time (s; log scale)")
                else:
                    axis.set_xlim(0, max(1e-6, 1.03 * maximum))
            else:
                axis.set_xlim(0, max(len(r["metrics"]["discovery_curve"]) for r in rows))
            if abscissa != "elapsed":
                axis.xaxis.set_major_locator(
                    MaxNLocator(nbins=4 if columns == 3 else 10, integer=True)
                )
            if columns == 3:
                axis.set_xlabel(textwrap.fill(axis.get_xlabel(), width=24))
                axis.set_ylabel(textwrap.fill(axis.get_ylabel(), width=24))
            axis.grid(alpha=0.18, linewidth=0.5)
        legend_labels = []
        for method in methods:
            label = STYLES[method][0]
            if method in diagnostics[name]:
                counts = diagnostics[name][method]["defined_tasks"]
                span = (
                    str(min(counts))
                    if min(counts) == max(counts)
                    else f"{min(counts)}–{max(counts)}"
                )
                label += f" (n={span})"
            legend_labels.append(label)
        legend_title = "Contributing ADSs at each query" if diagnostics[name] else None
        if legend_in_panel:
            legend_axis = axes.flat[-1]
            legend_axis.set_axis_off()
            legend_axis.legend(
                handles.values(),
                legend_labels,
                loc="center",
                frameon=False,
                title=legend_title,
                title_fontsize=8,
            )
        else:
            figure.legend(
                handles.values(),
                legend_labels,
                loc="lower center",
                ncol=min(2, len(handles)),
                frameon=False,
                title=legend_title,
                title_fontsize=8,
            )
        figure.tight_layout(rect=(0, legend_height / height, 1, 1))
        path = destination / f"{rows[0]['scenario']}_{name}.png"
        save_figure(figure, path)
        files.append(path)
    files.append(plot_endpoints(rows, destination, group_key, methods, systems))
    files.append(plot_online_costs(rows, destination, group_key, methods, systems))
    return files, diagnostics


def plot_endpoints(rows, destination, group_key, methods, systems):
    figure, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 7.3))
    definitions = (
        ("unique_collisions", "Discovered failures"),
        ("collision_recall", "Candidate-library recall"),
        ("discovery_auc", "Discovery AUC"),
        ("first_collision_query", "First failure query"),
    )
    for panel, (axis, (metric, label)) in enumerate(zip(axes.flat, definitions)):
        for ordinal, system in enumerate(systems):
            values = [
                next(r["metrics"] for r in rows if r["sut_id"] == system and r[group_key] == method)
                for method in methods
            ]
            axis.plot(
                range(len(methods)),
                [np.nan if v[metric] is None else v[metric] for v in values],
                color="#999999",
                alpha=0.5,
                linewidth=0.6,
            )
            for index, (method, value) in enumerate(zip(methods, values)):
                number = value[metric]
                censored = metric == "first_collision_query" and number is None
                if censored:
                    number = len(value["discovery_curve"])
                if number is not None:
                    axis.scatter(
                        index,
                        number,
                        color=STYLES[method][1],
                        s=25,
                        marker="x" if censored else SYSTEM_MARKERS[ordinal % 8],
                        clip_on=False,
                        zorder=3,
                    )
        axis.set(title=f"({chr(97 + panel)}) {label}", ylabel=label, xticks=range(len(methods)))
        axis.set_xticklabels([STYLES[m][0] for m in methods], rotation=55, ha="right", fontsize=8)
        axis.set_ylim(bottom=0)
        if metric in ("unique_collisions", "discovery_auc") and all(
            row["metrics"][metric] == 0 for row in rows
        ):
            axis.set_ylim(0, 1)
        if metric == "unique_collisions":
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        if metric == "collision_recall":
            axis.set_ylim(0, 1)
            if all(row["metrics"][metric] is None for row in rows):
                axis.text(
                    0.5,
                    0.5,
                    "Undefined: no true failures\nin candidate libraries",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)
    handles = [
        plt.Line2D(
            [],
            [],
            color="#777777",
            marker=SYSTEM_MARKERS[i % 8],
            linestyle="",
            label=s.removeprefix("independent_"),
        )
        for i, s in enumerate(systems)
    ]
    if any(row["metrics"]["first_collision_query"] is None for row in rows):
        handles.append(
            plt.Line2D(
                [], [], color="#777777", marker="x", linestyle="", label="No failure within budget"
            )
        )
    figure.legend(handles=handles, loc="lower center", ncol=min(4, len(systems)), frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path = destination / f"{rows[0]['scenario']}_endpoints.png"
    save_figure(figure, path)
    return path


def plot_online_costs(rows, destination, group_key, methods, systems):
    figure, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 4.7))
    definitions = (
        ("wall_s", 1, "Online replay wall time", "Wall time (s)"),
        ("sampled_peak_cpu_bytes", 2**20, "Peak selector CPU memory", "Memory (MiB)"),
    )
    runs = {(row["sut_id"], row[group_key]): row["resources"] for row in rows}
    for panel, (axis, (metric, unit, title, label)) in enumerate(zip(axes, definitions)):
        all_values = []
        for ordinal, system in enumerate(systems):
            values = [runs[system, method][metric] / unit for method in methods]
            all_values.extend(values)
            axis.plot(range(len(methods)), values, color="#999999", alpha=0.5, linewidth=0.6)
            for index, (method, value) in enumerate(zip(methods, values)):
                axis.scatter(
                    index,
                    value,
                    color=STYLES[method][1],
                    marker=SYSTEM_MARKERS[ordinal % 8],
                    s=25,
                    clip_on=False,
                    zorder=3,
                )
        if min(all_values) > 0 and max(all_values) >= 10 * min(all_values):
            axis.set_yscale("log")
            label += "; log scale"
        else:
            axis.set_ylim(bottom=0)
        axis.set(title=f"({chr(97 + panel)}) {title}", ylabel=label, xticks=range(len(methods)))
        axis.set_xticklabels([STYLES[m][0] for m in methods], rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)
    handles = [
        plt.Line2D(
            [],
            [],
            color="#777777",
            marker=SYSTEM_MARKERS[i % 8],
            linestyle="",
            label=system.removeprefix("independent_"),
        )
        for i, system in enumerate(systems)
    ]
    figure.legend(handles=handles, loc="lower center", ncol=min(4, len(systems)), frameon=False)
    figure.tight_layout(rect=(0, 0.06 + 0.035 * (len(systems) > 4), 1, 1))
    path = destination / f"{rows[0]['scenario']}_online_costs.png"
    save_figure(figure, path)
    return path


def plot_curves(rows, root, group_key):
    files, diagnostics = [], {}
    stage = "sequential" if group_key == "regime" else "fixed_history"
    with plt.rc_context(STYLE):
        for comparison, members in comparison_sets(rows).items():
            scopes = sorted({(r["scenario"], r.get("held_family") is not None) for r in members})
            for scenario, held_out in scopes:
                selected = [
                    r
                    for r in members
                    if r["scenario"] == scenario and (r.get("held_family") is not None) == held_out
                ]
                destination = root / "figures" / stage
                if group_key == "method":
                    destination = (
                        destination
                        / comparison
                        / ("leave_family_out" if held_out else "all_sources")
                    )
                paths, counts = plot_comparison(selected, destination, group_key)
                files.extend(p.relative_to(root).as_posix() for p in paths)
                diagnostics[f"{comparison}/{scenario}/{held_out}"] = counts
    return files, diagnostics
