"""Evaluator-only spatial and adaptation figures from saved candidate truth and predictions."""

import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .joint_figures import STYLE, STYLES, comparison_sets, save_figure
from .scenarios import features
from .schema import Scenario
from .storage import read_json


def component_label(name):
    labels = {
        "independent_target": "Target GP",
        "physical_target": "Raw target GP",
        "raw_target": "Raw target GP",
        "aggregate_history_teacher": "Aggregate history",
    }
    if name in labels:
        return labels[name]
    if name.startswith("history_"):
        label = "History: " + name.removeprefix("history_")
    elif "@" in name and ":" in name:
        system, suffix = name.split("@", 1)
        version, _ = suffix.rsplit(":", 1)
        label = f"Completed: {system.removeprefix('independent_')} ({version})"
    else:
        label = name
    label = label.replace("_", " ")
    for acronym in ("idm", "mobil", "fvdm", "mpc", "uct", "ppo", "dqn"):
        label = label.replace(acronym, acronym.upper())
    return textwrap.fill(label, width=30, break_long_words=False, break_on_hyphens=False)


def candidate_projection(scenes):
    """Fix PCA and physical probability bins using only public candidate coordinates."""
    x = features(scenes)
    mean = x.mean(axis=0)
    _, singular, basis = np.linalg.svd(x - mean, full_matrices=False)
    components = basis[:2].copy()
    for axis in components:
        axis *= 1 if axis[np.abs(axis).argmax()] >= 0 else -1
    variance = singular[:2] ** 2 / np.square(singular).sum()
    physical, labels = physical_projection(scenes)
    edges = [
        np.histogram_bin_edges(axis, bins=min(20, len(np.unique(axis)))) for axis in physical.T
    ]
    counts = np.histogram2d(*physical.T, bins=edges)[0].T
    return {
        "coordinates": ((x - mean) @ components.T).tolist(),
        "mean": mean.tolist(),
        "components": components.tolist(),
        "explained_variance_ratio": variance.tolist(),
        "candidate_ids": [s.scenario_id for s in scenes],
        "input": "public normalized candidate coordinates only",
        "probability_map": {
            "coordinates": physical.tolist(),
            "axis_labels": list(labels),
            "x_edges": edges[0].tolist(),
            "y_edges": edges[1].tolist(),
            "counts": counts.astype(int).tolist(),
            "aggregation": "mean of saved candidate probabilities within each physical projection bin",
            "empty_bins": "undefined; no candidate prediction in this bin",
        },
    }


def physical_projection(scenes):
    kind = scenes[0].kind
    p = [s.parameters for s in scenes]
    if kind in ("car_following", "front_brake"):
        gap = [v["gap_m"] for v in p]
        closing = [
            v["closing_speed_mps"]
            if kind == "front_brake"
            else v["ego_speed_mps"] - v["lead_speed_mps"]
            for v in p
        ]
        return np.column_stack((gap, closing)), ("Net gap (m)", "Closing speed (m/s)")
    if kind == "cut_out":
        return np.array([[v["gap_m"], v["obstacle_distance_m"]] for v in p]), (
            "Lead net gap (m)",
            "Obstacle coordinate difference (m)",
        )
    if kind == "lane_change_rear":
        return np.array([[v["rear_gap_m"], v["rear_speed_mps"] - v["ego_speed_mps"]] for v in p]), (
            "Rear net gap (m)",
            "Rear closing speed (m/s)",
        )
    raise ValueError("No registered physical projection for this scenario")


def load_run(row, root):
    saved = read_json(root.parent / row["result_path"])
    candidates = [Scenario(**s) for s in saved["specification"]["candidates"]]
    truth = np.array(saved["evaluation"]["candidate_collisions"], dtype=bool)
    result = saved["result"]
    queried = result["selected_indices"]
    if len(truth) != len(candidates) or any(c.kind != row["scenario"] for c in candidates):
        raise ValueError("Reported truth must match the task's full candidate library")
    discovered = {candidates[i].scenario_id for i in queried if truth[i]}
    if discovered != set(row["metrics"]["discovered_ids"]):
        raise ValueError("Spatial discoveries differ from the candidate-ID metrics")
    return {
        "candidates": candidates,
        "truth": truth,
        "selected": queried,
        "snapshots": result.get("prediction_snapshots", []),
        "component_names": result.get("component_names", []),
        "weights": [
            q["component_weights_after_query"]
            for q in result["queries"]
            if "component_weights_after_query" in q
        ],
        "prequery": [
            (q["query"], q["prequery_probability"], q["feedback"]["collision"])
            for q in result["queries"]
            if q.get("prequery_probability") is not None
        ],
    }


def prediction_scores(probability, truth, mask):
    if not np.any(mask):
        return {"count": 0, "nll": None, "brier": None}
    p, y = probability[mask], truth[mask]
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "count": int(mask.sum()),
        "nll": float(-np.mean(np.log(np.where(y, bounded, 1 - bounded)))),
        "brier": float(np.mean((p - y.astype(float)) ** 2)),
    }


def snapshot_diagnostics(run):
    truth = run["truth"]
    diagnostics = []
    for snapshot in run["snapshots"]:
        count = snapshot["paid_queries"]
        if snapshot["observed_indices"] != run["selected"][:count]:
            raise ValueError("Prediction snapshot differs from its paid query prefix")
        probability = np.asarray(snapshot["probabilities"], dtype=float)
        if (
            probability.shape != truth.shape
            or not np.all(np.isfinite(probability))
            or np.any((probability < 0) | (probability > 1))
        ):
            raise ValueError("Prediction snapshot requires a probability per candidate")
        paid = np.zeros(len(truth), dtype=bool)
        paid[snapshot["observed_indices"]] = True
        diagnostics.append(
            {
                "paid_queries": count,
                "full_library": prediction_scores(
                    probability, truth, np.ones(len(truth), dtype=bool)
                ),
                "paid": prediction_scores(probability, truth, paid),
                "unqueried": prediction_scores(probability, truth, ~paid),
            }
        )
    return diagnostics


def plot_space(members, runs, projection, destination, group_key="method"):
    files = []
    reference = runs[members[0]["result_path"]]
    physical, labels = physical_projection(reference["candidates"])
    ratio = projection["explained_variance_ratio"]
    for name, xy, axis_labels in (
        ("physical", physical, labels),
        (
            "pca",
            np.asarray(projection["coordinates"]),
            (f"PC1 ({ratio[0]:.1%})", f"PC2 ({ratio[1]:.1%})"),
        ),
    ):
        columns = min(2, len(members))
        figure, axes = plt.subplots(
            math.ceil(len(members) / columns),
            columns,
            figsize=(180 / 25.4, 2.15 * math.ceil(len(members) / columns) + 0.35),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for panel, (axis, row) in enumerate(zip(axes.flat, members)):
            run = runs[row["result_path"]]
            truth = run["truth"]
            found = np.array([i for i in run["selected"] if truth[i]], dtype=int)
            axis.scatter(*xy.T, s=3, color="#DDDDDD", label="Candidates")
            axis.scatter(
                *xy[truth].T,
                s=12,
                facecolors="none",
                edgecolors="#A65A36",
                linewidths=0.5,
                label="True failures",
            )
            axis.scatter(*xy[found].T, s=9, color="#0072B2", label="Discovered failures")
            count_label = f"{len(found)}/{int(truth.sum())}" if truth.any() else "0 discovered"
            axis.set(
                title=f"({chr(97 + panel)}) {STYLES[row[group_key]][0]} | {count_label}",
                xlabel=axis_labels[0],
                ylabel=textwrap.fill(axis_labels[1], width=24),
            )
            if not truth.any():
                axis.text(
                    0.5,
                    0.5,
                    "No failures in library",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                )
        for axis in list(axes.flat)[len(members) :]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        path = destination / f"failure_space_{name}.png"
        save_figure(figure, path)
        files.append(path)
    return files


def projected_probability(probability, mapping):
    xy = np.asarray(mapping["coordinates"])
    counts = np.asarray(mapping["counts"])
    total = np.histogram2d(
        *xy.T, bins=(mapping["x_edges"], mapping["y_edges"]), weights=probability
    )[0].T
    return np.divide(total, counts, out=np.full(total.shape, np.nan), where=counts > 0)


def plot_snapshots(run, projection, destination):
    snapshots = run["snapshots"]
    if not snapshots:
        return []
    mapping = projection["probability_map"]
    xy = np.asarray(mapping["coordinates"])
    figure, axes = plt.subplots(
        math.ceil(len(snapshots) / 2),
        min(2, len(snapshots)),
        figsize=(180 / 25.4, 2.6 * math.ceil(len(snapshots) / 2)),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for panel, (axis, snapshot) in enumerate(zip(axes.flat, snapshots)):
        mesh = axis.pcolormesh(
            mapping["x_edges"],
            mapping["y_edges"],
            projected_probability(snapshot["probabilities"], mapping),
            cmap="viridis",
            vmin=0,
            vmax=1,
            shading="flat",
        )
        paid = snapshot["observed_indices"]
        axis.scatter(
            *xy[run["truth"]].T,
            marker="+",
            s=14,
            color="#D55E00",
            linewidths=0.6,
            label="True failures",
        )
        axis.scatter(
            *xy[paid].T,
            s=20,
            facecolors="none",
            edgecolors="black",
            linewidths=0.7,
            label="Paid queries",
        )
        axis.set(
            title=f"({chr(97 + panel)}) {snapshot['paid_queries']} paid queries",
            xlabel=mapping["axis_labels"][0],
            ylabel=textwrap.fill(mapping["axis_labels"][1], width=24),
        )
        for key, setter in (("x_edges", axis.set_xlim), ("y_edges", axis.set_ylim)):
            lo, hi = mapping[key][0], mapping[key][-1]
            pad = 0.02 * (hi - lo)
            setter(lo - pad, hi + pad)
    for axis in list(axes.flat)[len(snapshots) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="#BBBBBB"))
    labels.append("No candidates")
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.colorbar(
        mesh,
        ax=list(axes.flat),
        label="Mean predicted collision probability\nwithin projected bin",
        shrink=0.8,
    )
    path = destination / "prediction_snapshots.png"
    save_figure(figure, path)
    files = [path]
    if run["weights"]:
        names = run["component_names"]
        labels = [component_label(name) for name in names]
        row_height = max(0.3, 0.15 * max(label.count("\n") + 1 for label in labels))
        figure, axis = plt.subplots(figsize=(180 / 25.4, max(2.5, row_height * len(names) + 1)))
        weights = np.asarray(run["weights"])
        mesh = axis.imshow(
            weights.T,
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
            origin="upper",
            extent=(0.5, len(weights) + 0.5, len(names) - 0.5, -0.5),
        )
        axis.set(
            xlabel="Paid queries (weights after feedback)",
            yticks=np.arange(len(names)),
            yticklabels=labels,
        )
        figure.colorbar(mesh, ax=axis, label="Mixture weight")
        figure.tight_layout()
        path = destination / "source_weights.png"
        save_figure(figure, path)
        files.append(path)
    return files


def plot_calibration(members, runs, destination, group_key="method"):
    predictors = [row for row in members if runs[row["result_path"]]["prequery"]]
    if not predictors:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 3.2))
    handles = {}
    for row in predictors:
        observed = np.asarray(runs[row["result_path"]]["prequery"], dtype=float)
        queries, p, y = observed.T
        bounded = np.clip(p, 1e-12, 1 - 1e-12)
        losses = (-np.log(np.where(y, bounded, 1 - bounded)), (p - y) ** 2)
        title, color, linestyle = STYLES[row[group_key]]
        for axis, values in zip(axes, losses):
            line = axis.step(
                queries,
                np.cumsum(values) / np.arange(1, len(values) + 1),
                where="post",
                label=title,
                color=color,
                linestyle=linestyle,
            )[0]
        handles[title] = line
    for panel, (axis, metric) in enumerate(zip(axes, ("collision NLL", "Brier score"))):
        axis.set(
            title=f"({chr(97 + panel)}) Query-before-feedback",
            xlabel="Paid queries",
            ylabel=f"Running mean {metric}",
        )
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.18, linewidth=0.5)
    figure.legend(
        handles.values(), handles, loc="lower center", ncol=min(3, len(handles)), frameon=False
    )
    legend_height = 0.08 * math.ceil(len(handles) / 3)
    figure.tight_layout(rect=(0, legend_height, 1, 1))
    path = destination / "prequery_calibration.png"
    save_figure(figure, path)
    return [path]


def risk_figures(rows, root):
    if not any("result_path" in row for row in rows):
        return [], {
            "status": "unavailable",
            "reason": "summary does not link saved predictions and truth",
        }
    group_key = "regime" if "regime" in rows[0] else "method"
    suite = "sequential" if group_key == "regime" else "fixed_history"
    runs = {r["result_path"]: load_run(r, root) for r in rows}
    projections, coordinates = {}, {}
    for row in rows:
        scenes = runs[row["result_path"]]["candidates"]
        kind = row["scenario"]
        if kind not in projections:
            projections[kind] = candidate_projection(scenes)
            coordinates[kind] = [(s.scenario_id, s.coordinate()) for s in scenes]
        if coordinates[kind] != [(s.scenario_id, s.coordinate()) for s in scenes]:
            raise ValueError("All methods and ADSs must share the same scenario projection")
    files, scores = [], {}
    # Use the first registered ADS in each scene, independent of outcomes.
    representatives = {}
    for row in rows:
        representatives.setdefault(row["scenario"], row["sut_id"])
    with plt.rc_context(STYLE):
        for comparison, members in comparison_sets(rows).items():
            for kind, sut, held in dict.fromkeys(
                (r["scenario"], r["sut_id"], r.get("held_family")) for r in members
            ):
                if sut != representatives[kind]:
                    continue
                tasks = [
                    r
                    for r in members
                    if r["scenario"] == kind and r["sut_id"] == sut and r.get("held_family") == held
                ]
                tasks.sort(key=lambda r: list(STYLES).index(r[group_key]))
                destination = (
                    root
                    / "figures"
                    / suite
                    / comparison
                    / ("leave_family_out" if held else "all_sources")
                    / kind
                    / sut
                )
                files.extend(plot_space(tasks, runs, projections[kind], destination, group_key))
                files.extend(plot_calibration(tasks, runs, destination, group_key))
        for row in rows:
            run = runs[row["result_path"]]
            scores[row["result_path"]] = snapshot_diagnostics(run)
            if row["sut_id"] != representatives[row["scenario"]]:
                continue
            destination = (
                root
                / "figures/adaptation"
                / ("leave_family_out" if row.get("held_family") else "all_sources")
                / row["scenario"]
                / row["sut_id"]
                / row[group_key]
            )
            files.extend(plot_snapshots(run, projections[row["scenario"]], destination))
    return [p.relative_to(root).as_posix() for p in files], {
        "status": "complete",
        "figure_systems": representatives,
        "figure_selection": "first registered ADS per functional scenario; all tasks retain numeric diagnostics",
        "projections": projections,
        "snapshot_scores": scores,
        "weight_component_labels": {
            key: {name: component_label(name) for name in run["component_names"]}
            for key, run in runs.items()
            if run["weights"]
        },
    }
