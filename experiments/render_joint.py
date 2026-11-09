"""Render paid executions at their measured frame rate, retaining execution provenance."""

import gzip
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator

from .animation import body_vertices, capture_frame, save_gif
from .execution_events import analyze_events
from .joint_figures import STYLE
from .protocol import RUNS_ROOT
from .rss import measurement_contract, trajectory_margin
from .scenarios import TEST_SCENARIOS
from .schema import Scenario
from .storage import read_json, read_trajectory, write_json

OUTPUT = RUNS_ROOT / "joint_response"


def place_actor_labels(road, labels, actors, visible_ids):
    """Keep names in a separate row, with leaders to their actual vehicle positions."""
    renderer = road.figure.canvas.get_renderer()
    origin = actors["ego"]["x"]
    placed = []
    for name, actor in actors.items():
        label = labels[name]
        label.set_text(
            name.replace("_", " ").capitalize()
            + (" *" if name != "ego" and name not in visible_ids else "")
        )
        label.xy = (actor["x"] - origin, actor["y"])
        width = renderer.get_text_width_height_descent(
            label.get_text(), label.get_fontproperties(), False
        )[0]
        center = road.transData.transform(label.xy)[0]
        placed.append([center, width / 2, label])
    placed.sort(key=lambda item: item[0])
    edge = road.bbox.x0 + 3
    for item in placed:
        item[0] = max(item[0], edge + item[1])
        edge = item[0] + item[1] + 8
    edge = road.bbox.x1 - 3
    for item in reversed(placed):
        item[0] = min(item[0], edge - item[1])
        edge = item[0] - item[1] - 8
        item[2].set_position(((item[0] - road.bbox.x0) / road.bbox.width, 1.04))


def render_execution(
    scenario,
    trajectory,
    destination,
    *,
    provenance=None,
    static=False,
    events=None,
):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / scenario.kind
    diagnosis = trajectory_margin(trajectory)
    events = analyze_events(scenario, trajectory) if events is None else events
    payload = {
        "scenario": asdict(scenario),
        "trajectory": trajectory,
        "events": events,
        "provenance": provenance,
        "rss_contract": measurement_contract(),
    }
    if provenance is not None and "execution_ref" in provenance:
        payload.pop("trajectory")
        payload["execution_ref"] = provenance["execution_ref"]
    stem.with_suffix(".execution.json.gz").write_bytes(
        gzip.compress(json.dumps(payload, allow_nan=False).encode("utf-8"), mtime=0)
    )
    times = np.array([frame["time_s"] for frame in trajectory])
    margins = np.array([frame["margin"] for frame in diagnosis["series"]])
    states = [{actor["actor_id"]: actor for actor in frame["actors"]} for frame in trajectory]
    names = list(states[0])
    speeds = {name: np.array([state[name]["speed"] for state in states]) for name in names}
    colors = dict(zip(names, ("#3D78A6", "#E19332", "#CE5656", "#8192A1", "#398A83")))
    relative = [actor["x"] - state["ego"]["x"] for state in states for actor in state.values()]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            3, 1, figsize=(18 / 2.54, 5.2), gridspec_kw={"height_ratios": (1, 1, 1)}
        )
    road, margin_axis, speed_axis = axes
    road.set_xlim(min(-20, min(relative) - 10), max(60, max(relative) + 10))
    road.set_ylim(11, -3)
    road.set_aspect("equal", adjustable="box")
    road.set_facecolor("#E9ECEF")
    for y in (2, 6):
        road.axhline(y, color="white", linestyle="--")
    road.set_xlabel("Longitudinal position relative to ego (m)")
    road.set_ylabel("Lateral position (m)")
    scenario_name = scenario.kind.replace("_", " ").capitalize()
    title = road.set_title(f"(a) {scenario_name}", loc="left", fontsize=9, pad=25)
    if any(
        actor != "ego" and actor not in frame["visible_ids"]
        for frame, state in zip(trajectory, states)
        for actor in state
    ):
        figure.text(
            0.99,
            0.015,
            "* Outside visible set",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#52616B",
        )
    bodies, labels, speed_lines = {}, {}, {}
    for name in names:
        bodies[name] = Polygon(
            body_vertices(states[0][name], states[0]["ego"]["x"]),
            facecolor=colors[name],
            edgecolor="white",
        )
        road.add_patch(bodies[name])
        labels[name] = road.annotate(
            name,
            xy=(0, 0),
            xytext=(0.5, 1.04),
            textcoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[name],
            arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": colors[name]},
            annotation_clip=False,
        )
        (speed_lines[name],) = speed_axis.plot(
            [], [], color=colors[name], label=name.replace("_", " ").capitalize()
        )
    margin_axis.plot(times, margins, color="#CCD3D8", linestyle="--")
    (margin_line,) = margin_axis.plot([], [], color="#398A83")
    margin_axis.axhline(0, color="#CE5656", linestyle="--")
    margin_axis.set_title("(b) RSS response", loc="left", fontsize=9)
    margin_axis.set_ylabel("Normalized RSS margin")
    margin_axis.set_ylim(min(-0.1, margins.min() - 0.1), max(0.1, margins.max() + 0.1))
    speed_axis.set_ylabel("Speed (m/s)")
    speed_axis.set_title("(c) Vehicle speeds", loc="left", fontsize=9)
    speed_axis.set_xlabel("Simulation time (s)")
    speed_axis.set_ylim(0, max(values.max() for values in speeds.values()) * 1.1 + 1)
    speed_axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=3))
    speed_axis.legend(
        loc="lower right",
        bbox_to_anchor=(1, 1.01),
        ncol=min(4, len(names)),
        fontsize=8,
        frameon=False,
    )
    cursors = []
    for axis in (margin_axis, speed_axis):
        axis.set_xlim(0, max(0.05, times[-1]))
        cursors.append(axis.axvline(0, color="#243746", alpha=0.4))
        if "event_time_s" in scenario.parameters:
            axis.axvline(scenario.parameters["event_time_s"], color="#E19332", linestyle=":")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.canvas.draw()
    images = []

    def draw_frame(i):
        frame, ego = trajectory[i], states[i]["ego"]
        for name, actor in states[i].items():
            bodies[name].set_xy(body_vertices(actor, ego["x"]))
            speed_lines[name].set_data(times[: i + 1], speeds[name][: i + 1])
        place_actor_labels(road, labels, states[i], frame["visible_ids"])
        collision = " | Collision" if frame["collision"] else ""
        title.set_text(f"(a) {scenario_name} | t = {times[i]:.2f} s{collision}")
        margin_line.set_data(times[: i + 1], margins[: i + 1])
        for cursor in cursors:
            cursor.set_xdata([times[i], times[i]])

    try:
        for i, frame in enumerate(trajectory):
            draw_frame(i)
            images.append(capture_frame(figure))
        if static:
            figure.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    finally:
        plt.close(figure)
    playback = save_gif(images, times, stem.with_suffix(".gif"))
    result = {
        "scenario": asdict(scenario),
        "events": events,
        "rss_contract": measurement_contract(),
        "minimum_margin": diagnosis["minimum_margin"],
        **playback,
        "gif": stem.with_suffix(".gif").name,
        "source": "actual paid execution",
        "static_time_s": float(times[-1]) if static else None,
    }
    write_json(stem.with_suffix(".json"), result)
    return result


def select_cases(root):
    """Select paid cases from the full method in the registered ADS execution order."""
    root = Path(root)
    summary = read_json(root / "reports/summary.json")
    if summary["status"] != "complete":
        raise ValueError("Case selection requires the completed registered comparison")
    protocol = read_json(root / "reports/protocol.json")
    sequential = protocol["suite"] == "sequential"
    if protocol["suite"] == "sensitivity" and "candidate_sizes" in protocol:
        raise ValueError("Case selection requires a single registered candidate size")
    order = {s["sut_id"]: i for i, s in enumerate(protocol["systems"])}
    kinds = protocol["config"]["scenarios"]
    selected = {kind: {"collision": None, "safe": None} for kind in kinds}
    if sequential:
        rows = sorted(
            (r for r in summary["runs"] if r["regime"] == "update_shared"),
            key=lambda r: (r["ordinal"], r["seed"]),
        )
    else:
        rows = sorted(
            (r for r in summary["runs"] if r["method"] == "joint"),
            key=lambda r: order[r["sut_id"]],
        )
    for row in rows:
        if sequential:
            path = (
                root
                / "methods"
                / row["regime"]
                / str(row["seed"])
                / "completed"
                / f"{row['ordinal']:02d}_{row['sut_id']}_{row['scenario']}.json.gz"
            )
        else:
            path = (root / row["result_path"]).parent / "completed.json.gz"
        task = json.loads(gzip.decompress(path.read_bytes()))
        if not task["completed"] or task["sut_id"] != row["sut_id"]:
            raise ValueError("Case archive differs from the completed task")
        if sequential and (
            task["branch_id"] != row["branch_id"]
            or row["branch_id"] != f"update_shared:{row['seed']}"
            or task["completed_order"] != row["ordinal"]
        ):
            raise ValueError("Case archive differs from the chronological task snapshot")
        for execution in sorted(task["executions"], key=lambda e: e["query"]):
            scene = Scenario(**execution["scenario"])
            if scene.kind != row["scenario"] or scene.kind not in TEST_SCENARIOS:
                raise ValueError("Case belongs to a different functional scenario")
            outcome = "collision" if execution["feedback"]["collision"] else "safe"
            if selected[scene.kind][outcome] is None:
                provenance = {
                    "task_id": task["task_id"],
                    "record_id": execution["record_id"],
                    "source_id": task["sut_id"],
                    "original_role": task["original_role"],
                    "query": execution["query"],
                    "result_path": row["result_path"],
                    "selection": "first outcome in registered ADS and paid-query order",
                }
                if "execution_ref" in execution:
                    provenance["execution_ref"] = execution["execution_ref"]
                if sequential:
                    provenance.update(
                        selection="first outcome in chronological ADS and paid-query order of the update_shared branch",
                        branch_id=task["branch_id"],
                        ordinal=task["completed_order"],
                        model_snapshot=task["model_snapshot"],
                    )
                selected[scene.kind][outcome] = {
                    "scene": scene,
                    "trace": read_trajectory(execution),
                    "provenance": provenance,
                }
    return selected


def render_cases(root=OUTPUT / "pilot"):
    root = Path(root)
    destination = root / "reports/figures/cases"
    cases, missing = [], []
    for kind, outcomes in select_cases(root).items():
        for outcome, case in outcomes.items():
            if case is None:
                missing.append({"scenario": kind, "outcome": outcome, "reason": "not observed"})
                continue
            case_dir = destination / kind / outcome
            rendered = render_execution(
                case["scene"], case["trace"], case_dir, provenance=case["provenance"], static=True
            )
            cases.append(
                {
                    **rendered,
                    "outcome": outcome,
                    "directory": case_dir.relative_to(destination).as_posix(),
                    "provenance": case["provenance"],
                }
            )
    manifest = {"cases": cases, "missing": missing}
    write_json(destination / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    render_cases()
