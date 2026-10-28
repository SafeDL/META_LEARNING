"""Render recorded discoveries with synchronized ADS-centred RSS margins."""

import csv
import gzip
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ImageDraw, ImageFont

from .protocol import TEST_RECORDS_ROOT
from .rss import DEFAULT, measurement_contract, trajectory_margin

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiment_outputs/benchmark/collision/discovered_scenarios"
RUNS = ROOT / "experiment_outputs/gcrt/collision/discovery/independent_validation/gcrt"
BANKS = TEST_RECORDS_ROOT / "independent_targets"
CASES = (
    ("front_brake", "independent_dqn", "Front braking", "lead"),
    ("cut_in", "independent_dqn", "Vehicle cut-in", "cut_in"),
    ("cut_out", "independent_linear_mpc", "Cut-out reveals an obstacle", "obstacle"),
    ("lane_change_rear", "independent_dqn", "Lane change and rear-end impact", "rear"),
    ("cut_in_interaction", "independent_dqn", "Cut-in with restricted adjacent space", "cut_in"),
)
BLUE, RED, TEAL, GRAY = "#3D78A6", "#CE5656", "#398A83", "#8192A1"
INK, GRID = "#243746", "#E5E9EC"
ACTOR_LABELS = {
    "ego": "ADS",
    "lead": "Lead vehicle",
    "cut_in": "Cut-in vehicle",
    "rear": "Rear vehicle",
    "obstacle": "Obstacle",
    "adjacent_front": "Adjacent front",
    "guard_front": "Adjacent front",
    "guard_rear": "Adjacent rear",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contact_direction(frame):
    actors = {a["actor_id"]: a for a in frame["actors"]}
    ego, other = actors["ego"], actors[frame["collision_partner"]]
    dx, dy = other["x"] - ego["x"], other["y"] - ego["y"]
    c, s = math.cos(ego["heading"]), math.sin(ego["heading"])
    x = (c * dx + s * dy) / ((ego["length"] + other["length"]) / 2)
    y = (-s * dx + c * dy) / ((ego["width"] + other["width"]) / 2)
    return ("front" if x >= 0 else "rear") if abs(x) >= abs(y) else "side"


def select_case(kind, target, title, partner):
    """Earliest post-initialization discovery satisfying the named mechanism."""
    run_path = RUNS / target / kind / "all_sources/11.json"
    run = read_json(run_path)
    bank = BANKS / target / kind / "env_81001"
    manifest = read_json(bank / "manifest.json")
    indices = {s["scenario_id"]: i for i, s in enumerate(manifest["scenarios"])}
    for query in run["result"]["queries"]:
        if not query["feedback"]["collision"] or query["index"] in run["initial_indices"]:
            continue
        index = indices[query["feedback"]["scenario_id"]]
        source = bank / f"executions/{index:06d}.json.gz"
        execution = json.loads(gzip.decompress(source.read_bytes()))
        frames = execution["trajectory"]
        impact = next(f for f in frames if f["collision"])
        events = execution["events"]
        if impact["collision_partner"] != partner:
            continue
        if kind == "lane_change_rear":
            if contact_direction(impact) != "rear" or events["ego_lane_change_onset_s"] is None:
                continue
        else:
            onset = events["measured_background_onset_s"]
            if onset is None or impact["time_s"] <= onset:
                continue
        if kind == "cut_out" and events["obstacle_visible_s"] is None:
            continue
        if kind == "cut_in_interaction":
            if execution["scenario"]["categories"]["adjacent_space"] != "restricted":
                continue
        assert execution["scenario"] == manifest["scenarios"][index]
        assert execution["feedback"]["scenario_id"] == query["feedback"]["scenario_id"]
        assert execution["feedback"]["collision"] == query["feedback"]["collision"]
        assert (
            run["specification"]["candidate_ids"][query["index"]]
            == execution["scenario"]["scenario_id"]
        )
        ledger = run["query_ledger"][query["query"] - 1]
        assert ledger["scenario_id"] == execution["scenario"]["scenario_id"]
        assert ledger["purpose"] != "initial" and ledger["status"] == "complete"
        summary = {
            "kind": kind,
            "title": title,
            "target": target,
            "method": "gcrt",
            "condition": "all_sources",
            "algorithm_seed": 11,
            "query_number": query["query"],
            "budget": run["result"]["maximum_queries"],
            "candidate_index": query["index"],
            "physical_bank_index": index,
            "run_path": run_path.relative_to(ROOT).as_posix(),
            "execution_path": source.relative_to(ROOT).as_posix(),
            "scenario": execution["scenario"],
            "collision_time_s": impact["time_s"],
            "collision_partner": partner,
            "contact_direction": contact_direction(impact),
            "events": events,
        }
        return summary, execution
    raise ValueError(f"No paid discovery matches the requested mechanism: {kind}, {target}")


def body_vertices(actor, origin):
    c, s = math.cos(actor["heading"]), math.sin(actor["heading"])
    vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    vertices *= [actor["length"] / 2, actor["width"] / 2]
    return vertices @ np.array([[c, s], [-s, c]]) + [actor["x"] - origin, actor["y"]]


def render_case(number, summary, execution):
    frames = execution["trajectory"]
    diagnosis = trajectory_margin(frames)
    summary["rss"] = {k: v for k, v in diagnosis.items() if k != "series"}
    summary["ego_response_sensitivity"] = {
        str(response): trajectory_margin(frames, replace(DEFAULT, ego_response_s=response))[
            "minimum_margin"
        ]
        for response in (0.3, 0.5, 1.0, 1.5)
    }
    times = np.array([f["time_s"] for f in frames])
    margins = np.array([r["margin"] for r in diagnosis["series"]])
    states = [{a["actor_id"]: a for a in f["actors"]} for f in frames]
    partner = summary["collision_partner"]
    speeds = {key: [s[key]["speed"] for s in states] for key in ("ego", partner)}
    relative = [a["x"] - s["ego"]["x"] for s in states for a in s.values()]
    xmin, xmax = min(-28, min(relative) - 9), max(48, max(relative) + 9)
    lanes = (0, 4) if summary["kind"] == "lane_change_rear" else (0, 4, 8)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(12, 7.6), dpi=110, facecolor="white")
    fig.text(0.055, 0.945, f"{number:02d}   {summary['title']}", fontsize=20, weight="bold")
    target_name = (
        "Independent DQN" if summary["target"] == "independent_dqn" else "Independent linear MPC"
    )
    fig.text(
        0.055,
        0.902,
        f"GCRT  /  {target_name}  /  Query {summary['query_number']} of {summary['budget']}  /  Seed 11",
        fontsize=11,
        color="#637584",
    )
    timer = fig.text(0.055, 0.847, "", fontsize=14, weight="bold")
    phase = fig.text(0.945, 0.847, "", fontsize=12, ha="right", color=TEAL)
    road = fig.add_axes([0.055, 0.48, 0.89, 0.32])
    risk_ax = fig.add_axes([0.075, 0.145, 0.53, 0.245])
    speed_ax = fig.add_axes([0.69, 0.145, 0.255, 0.245])
    risk_ax.plot(times, margins, color="#D7DEE3", lw=2, ls="--")
    (risk_line,) = risk_ax.plot([], [], color=TEAL, lw=2.3)
    (risk_dot,) = risk_ax.plot([], [], "o", color=TEAL, ms=5)
    risk_ax.axhline(0, color=RED, ls="--", lw=1)
    lower, upper = min(-0.1, margins.min() - 0.1), max(0.3, margins.max() + 0.1)
    risk_ax.set_ylim(lower, upper)
    risk_ax.axhspan(lower, 0, facecolor=RED, alpha=0.07)
    risk_ax.set_title(
        "Post-hoc normalized RSS margin", loc="left", fontsize=12, weight="bold", pad=13
    )
    risk_ax.set_ylabel("Margin (dimensionless)")
    risk_ax.text(
        0.98,
        0.93,
        "Below 0: safety envelope violated",
        transform=risk_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=RED,
    )
    speed_lines = {}
    for key, color in (("ego", BLUE), (partner, RED)):
        speed_ax.plot(times, speeds[key], color=color, alpha=0.18, lw=1.5, ls="--")
        (speed_lines[key],) = speed_ax.plot([], [], color=color, lw=2, label=ACTOR_LABELS[key])
    speed_ax.set_title("Vehicle speed", loc="left", fontsize=12, weight="bold", pad=13)
    speed_ax.set_ylabel("Speed (m/s)")
    speed_ax.set_ylim(0, max(max(v) for v in speeds.values()) * 1.25 + 1)
    speed_ax.legend(loc="upper right", frameon=False, fontsize=8)
    event_time = (
        summary["events"]["ego_lane_change_onset_s"]
        if summary["kind"] == "lane_change_rear"
        else summary["events"]["measured_background_onset_s"]
    )
    event_label = {
        "front_brake": "Lead braking",
        "cut_in": "Cut-in motion",
        "cut_out": "Lead cut-out",
        "lane_change_rear": "ADS lane change",
        "cut_in_interaction": "Cut-in motion",
    }[summary["kind"]]
    event_note = fig.text(0.055, 0.802, "", fontsize=9, color="#637584")
    for ax in (risk_ax, speed_ax):
        ax.set_xlim(0, times[-1])
        ax.set_xlabel("Simulation time (s)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.axvline(event_time, color=GRAY, ls=":", lw=1)
        for spine in ax.spines.values():
            spine.set_color("#C6CFD6")
    cursors = [ax.axvline(0, color=INK, alpha=0.45, lw=1) for ax in (risk_ax, speed_ax)]
    fig.text(
        0.055,
        0.057,
        "Blue: tested ADS     Red: collision partner     Gray: surrounding traffic",
        fontsize=10,
        color="#637584",
    )
    fig.text(
        0.055,
        0.025,
        "Playback 0.5x  |  RSS uses global traffic state  |  * Outside ADS visible set  |  Faint dashed curves: full recorded trace",
        fontsize=9,
        color="#637584",
    )
    images = []
    indices = list(range(0, len(frames), 2))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
    for index in indices:
        frame, actors = frames[index], states[index]
        ego = actors["ego"]
        road.clear()
        road.set_xlim(xmin, xmax)
        road.set_ylim(max(lanes) + 3.4, min(lanes) - 3.4)
        road.set_aspect("equal", adjustable="box")
        road.add_patch(Rectangle((xmin, -2), xmax - xmin, max(lanes) + 4, color="#F0F3F5"))
        for y in (-2, max(lanes) + 2):
            road.axhline(y, color="#A7B4BF", lw=1.4)
        for y in [v + 2 for v in lanes[:-1]]:
            road.axhline(y, color="white", ls=(0, (10, 8)), lw=2)
        for name, actor in actors.items():
            color = BLUE if name == "ego" else RED if name == partner else GRAY
            start = max(0, index - 20)
            road.plot(
                [s[name]["x"] - ego["x"] for s in states[start : index + 1]],
                [s[name]["y"] for s in states[start : index + 1]],
                color=color,
                alpha=0.35,
                lw=1.3,
            )
            road.add_patch(
                Polygon(
                    body_vertices(actor, ego["x"]),
                    closed=True,
                    facecolor=color,
                    edgecolor=INK if frame["collision"] and name in ("ego", partner) else "white",
                    linewidth=1.4,
                    zorder=4,
                )
            )
            if name != "obstacle":
                c, s = math.cos(actor["heading"]), math.sin(actor["heading"])
                x, y = actor["x"] - ego["x"], actor["y"]
                road.plot(
                    [x + 0.7 * c - 0.6 * s, x + 0.7 * c + 0.6 * s],
                    [y + 0.7 * s + 0.6 * c, y + 0.7 * s - 0.6 * c],
                    color="white",
                    lw=2,
                    alpha=0.8,
                    zorder=5,
                )
            offset = -2.2 if name == "ego" else 2.2
            label = ACTOR_LABELS.get(name, name)
            if name not in frame["visible_ids"] and name != "ego":
                label += " *"
            road.text(
                actor["x"] - ego["x"],
                actor["y"] + offset,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                zorder=6,
            )
        road.set_yticks(lanes, [f"Lane {i + 1}" for i in range(len(lanes))])
        road.tick_params(length=0, labelsize=9)
        road.set_xlabel("Longitudinal position relative to the ADS (m)", fontsize=10, labelpad=8)
        for spine in road.spines.values():
            spine.set_visible(False)
        timer.set_text(f"t = {frame['time_s']:4.2f} s     |     Margin = {margins[index]:+.2f}")
        critical = ACTOR_LABELS[diagnosis["series"][index]["critical_actor"]]
        event_note.set_text(
            f"{event_label} onset: {event_time:.2f} s (dotted line)     |     Current critical object: {critical}"
        )
        if frame["collision"]:
            phase.set_text(f"RECORDED COLLISION  /  {summary['contact_direction'].upper()} CONTACT")
            phase.set_color(RED)
        else:
            phase.set_text(
                "MANEUVER IN PROGRESS" if frame["time_s"] >= event_time else "BEFORE MANEUVER"
            )
            phase.set_color(TEAL)
        risk_line.set_data(times[: index + 1], margins[: index + 1])
        risk_dot.set_data([times[index]], [margins[index]])
        for key, line in speed_lines.items():
            line.set_data(times[: index + 1], speeds[key][: index + 1])
        for cursor in cursors:
            cursor.set_xdata([times[index]] * 2)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    plt.close(fig)
    stem = OUTPUT / summary["kind"]
    durations = [round((times[b] - times[a]) * 2000) for a, b in zip(indices, indices[1:])] + [1800]
    durations[0] += 700
    images[0].save(
        stem.with_suffix(".gif"),
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    images[-1].save(stem.with_suffix(".png"))
    summary["gif"] = stem.with_suffix(".gif").name
    summary["gif_frames"] = len(images)
    summary["playback_speed"] = 0.5
    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "rss_margin",
                "critical_actor",
                "ego_speed_mps",
                "partner_speed_mps",
                "collision",
            ]
        )
        writer.writerows(
            (
                t,
                m,
                row["critical_actor"],
                speeds["ego"][i],
                speeds[partner][i],
                frames[i]["collision"],
            )
            for i, (t, m, row) in enumerate(zip(times, margins, diagnosis["series"]))
        )
    return images[-1]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, posters = [], []
    for number, specification in enumerate(CASES, 1):
        summary, execution = select_case(*specification)
        posters.append(render_case(number, summary, execution))
        summaries.append(summary)
        print(
            f"Rendered {summary['kind']}: query {summary['query_number']}, bank {summary['physical_bank_index']}",
            flush=True,
        )
    canvas = Image.new("RGB", (1320, 3 * 440), "#F5F7F8")
    for i, poster in enumerate(posters):
        tile = poster.resize((660, 418), Image.Resampling.LANCZOS)
        canvas.paste(tile, ((i % 2) * 660, (i // 2) * 440))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    draw.multiline_text(
        (710, 1010),
        "Five recorded GCRT discoveries\nIndependent ADS / seed 11\n\nRSS margin: post-hoc analysis\nCollision labels: simulator records\n* Actor outside the ADS visible set",
        font=font,
        fill=INK,
        spacing=12,
    )
    canvas.save(OUTPUT / "overview.png")
    manifest = {
        "selection": "Fixed seed 11, all_sources; earliest collision after initialization matching each stated mechanism; cut-out uses linear MPC because DQN has no paid obstacle impact in this run.",
        "interpretation": "Illustrative discoveries, not representative frequencies. RSS is post-hoc and did not select these cases. Full actor state includes occluded actors; no new selector queries or simulator calls.",
        "style_reference": "simulator/metadrive/scripts/render_stage_a.py (blue/red palette and scientific panels); historical SAC renderer (blue SUT/red adversary).",
        "rss_parameters": asdict(DEFAULT),
        "rss_definition": measurement_contract(),
        "cases": summaries,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    rows = "\n".join(
        f"<article><h2>{s['title']}</h2><p>{s['target']} · Query {s['query_number']}/128 · {s['contact_direction']} contact</p><img src='{s['gif']}' alt='{s['title']}'><p><a href='{s['kind']}.csv'>Trajectory measurements</a></p></article>"
        for s in summaries
    )
    (OUTPUT / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>GCRT discovered scenarios</title><style>body{font:16px system-ui;background:#f4f6f8;color:#243746;margin:32px auto;max-width:1600px;padding:0 24px}h1{font-size:30px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(580px,1fr));gap:24px}article{background:white;padding:20px;border-radius:8px}img{width:100%}h2{font-size:21px}p{line-height:1.6;color:#637584}a{color:#3d78a6}</style><h1>GCRT: discovered collision scenarios</h1><p>Five mechanism-specific examples from paid queries on independent ADS. Seed 11; 0.5x playback. The RSS margin is a post-hoc analysis, not the score used in the original search. * marks an actor outside the ADS visible set. <a href='manifest.json'>Provenance and parameters</a></p><main>"
        + rows
        + "</main></html>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
