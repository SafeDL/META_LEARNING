"""Sample, inspect and animate safe archived ADS executions without simulation."""

import math
import random
from itertools import combinations

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageSequence

from simulator.highway.observation import VehicleState, observe

from experiments.plotting.animation import body_vertices, save_gif
from experiments.scene_design.audit_interactions import check_dynamics
from experiments.execution_events import analyze_events
from experiments.protocol import ROOT
from experiments.schema import Scenario
from experiments.storage import read_json, write_json

SEED = 20261115
DESTINATION = ROOT / "figures/scene_replays/safe"
RESULTS = ROOT / "results/scene_replays/safe"
FRAME_LABELS = ("Initial", "Event frame", "Closest approach", "Final")
SCENES = (
    ROOT / "results/scene_design/braking/critical_following",
    ROOT / "results/scene_design/interaction/cut_in_onset/initial",
    ROOT / "results/scene_design/interaction/cut_out_rear_gap/initial",
    ROOT / "results/scene_design/interaction/side_encroachment/initial",
)
COLORS = (
    "#FFFFFF",
    "#243746",
    "#E8EDF0",
    "#AFBCC4",
    "#2674B8",
    "#E28A27",
    "#647B8E",
    "#A3649D",
    "#CC4747",
    "#1C865B",
)
ACTORS = {
    "ego": (4, "E"),
    "lead": (5, "L"),
    "cut_in": (5, "C"),
    "intruder": (5, "I"),
    "guard_front": (6, "F"),
    "guard_rear": (7, "R"),
    "obstacle": (8, "O"),
}
PALETTE = [v for color in COLORS for v in bytes.fromhex(color[1:])] + [0] * (
    768 - 3 * len(COLORS)
)
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
SHEET_FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 27)
WIDTH, HEIGHT = 1120, 430


def road_bounds(scene):
    kind = scene["kind"]
    if kind == "car_following_braking_onset":
        return 2, 6
    if kind.startswith("cut_out") or (
        kind == "side_encroachment" and scene["categories"]["side"] == "left"
    ):
        return -2, 6
    return (2, 10) if kind == "side_encroachment" else (-2, 10)


def select_cases():
    """Uniformly sample one saved Safe record per configuration in fixed order."""
    rng = random.Random(SEED)
    cases = []
    for directory in SCENES:
        summary = read_json(
            directory
            / (
                "secondary_summary.json"
                if directory.parent.name == "cut_out_rear_gap"
                else "summary.json"
            )
        )
        for row in summary["rows"]:
            candidates = []
            for path in sorted(
                (directory / row["sut_id"]).glob("[0-9][0-9][0-9].json.gz")
            ):
                payload = read_json(path)
                duration = (
                    14
                    if payload["scenario"]["kind"] == "car_following_braking_onset"
                    else 10
                )
                if (
                    not payload["feedback"]["collision"]
                    and not payload["feedback"]["safety"]["violations"]
                    and payload["trajectory"][-1]["time_s"] == duration
                ):
                    candidates.append(path)
            assert len(candidates) == row.get(
                "safe", row["records"] - row["collisions"]
            )
            path = rng.choice(candidates)
            cases.append((row["sut_id"], path, len(candidates)))
    return cases


def inspect_safe(payload):
    """Check all actor footprints and native integration at every physics frame."""
    trace = payload["trajectory"]
    lower, upper = road_bounds(payload["scenario"])
    clearance, road_margin, background_margin = [], [], []
    overlap_frames = []
    visibility_mismatches = []
    for frame in trace:
        assert not frame["collision"] and frame["collision_partner"] is None
        assert all(
            math.isfinite(a[k])
            for a in frame["actors"]
            for k in ("x", "y", "speed", "heading", "length", "width")
        )
        assert all(a["speed"] >= -1e-8 for a in frame["actors"])
        bodies = {a["actor_id"]: VehicleState(**a).polygon() for a in frame["actors"]}
        ego = bodies["ego"]
        others = [b for name, b in bodies.items() if name != "ego"]
        clearance.append(min(ego.distance(b) for b in others))
        road_margin.append(min(ego.bounds[1] - lower, upper - ego.bounds[3]))
        background_margin.append(
            min(min(b.bounds[1] - lower, upper - b.bounds[3]) for b in others)
        )
        if any(a.intersects(b) for a, b in combinations(bodies.values(), 2)):
            overlap_frames.append(frame["time_s"])
        observation = observe(
            frame["time_s"], [VehicleState(**a) for a in frame["actors"]]
        )
        if {a.actor_id for a in observation.vehicles} != set(frame["visible_ids"]):
            visibility_mismatches.append(frame["time_s"])
    issues = []
    if overlap_frames:
        issues.append("actor_footprints_overlap")
    if min(road_margin) < -1e-8:
        issues.append("ego_body_outside_road")
    if min(background_margin) < -1e-8:
        issues.append("background_body_outside_road")
    if visibility_mismatches:
        issues.append("visible_actor_set_mismatch")
    try:
        check_dynamics(payload)
    except AssertionError:
        issues.append("native_dynamics_mismatch")
    ego_states = [next(a for a in f["actors"] if a["actor_id"] == "ego") for f in trace]
    events = analyze_events(Scenario(**payload["scenario"]), trace)
    if (
        events["measured_background_onset_s"] is None
        or events["background_complete_s"] is None
    ):
        issues.append("scripted_event_missing_or_incomplete")
    closest = int(np.argmin(clearance))
    reveal = next(
        (i for i, f in enumerate(trace) if "obstacle" in f["visible_ids"]), None
    )
    event = (
        reveal
        if reveal is not None
        else min(
            len(trace) - 1,
            round(payload["scenario"]["parameters"].get("lateral_duration_s", 2) * 10),
        )
    )
    return {
        "passed": not issues,
        "issues": issues,
        "physics_frames": len(trace),
        "duration_s": trace[-1]["time_s"],
        "minimum_body_clearance_m": min(clearance),
        "minimum_road_margin_m": min(road_margin),
        "minimum_background_road_margin_m": min(background_margin),
        "closest_time_s": trace[closest]["time_s"],
        "obstacle_first_visible_s": (
            trace[reveal]["time_s"] if reveal is not None else None
        ),
        "ego_speed_initial_final_mps": [
            ego_states[0]["speed"],
            ego_states[-1]["speed"],
        ],
        "ego_lateral_range_m": [
            min(a["y"] for a in ego_states),
            max(a["y"] for a in ego_states),
        ],
        "ego_final_y_m": ego_states[-1]["y"],
        "events": events,
        "keyframe_indices": [0, event, closest, len(trace) - 1],
        "clearance_series_m": clearance,
    }


def draw_frame(payload, sut_id, index, inspection, extent):
    frame = payload["trajectory"][index]
    states = {a["actor_id"]: a for a in frame["actors"]}
    origin = states["ego"]["x"]
    lower, upper = road_bounds(payload["scenario"])
    image = Image.new("P", (WIDTH, HEIGHT), 0)
    image.putpalette(PALETTE)
    draw = ImageDraw.Draw(image)
    status = (
        f"COLLISION: {frame['collision_partner']}"
        if frame["collision"]
        else "Before collision" if payload["feedback"]["collision"] else "Safe"
    )
    draw.text(
        (12, 9),
        f"{sut_id}   |   record {payload['scenario']['scenario_id'].rsplit(':', 1)[-1]}   |   t = {frame['time_s']:.2f} / {payload['trajectory'][-1]['time_s']:g} s   |   {status}",
        fill=8 if frame["collision"] else 1,
        font=FONT,
    )
    control = frame["control"]
    draw.text(
        (12, 34),
        f"ego speed {states['ego']['speed']:.2f} m/s   |   acceleration {control['acceleration']:+.2f} m/s2   |   steering {control['steering']:+.3f} rad   |   clearance {inspection['clearance_series_m'][index]:.3f} m",
        fill=1,
        font=SMALL,
    )

    def panel(left, right, top, bottom, *, overview):
        xscale = (WIDTH - 60) / (right - left)
        yscale = 5 if overview else xscale
        middle = (top + bottom) / 2

        def project(x, y):
            return 30 + (x - left) * xscale, middle + (y - (lower + upper) / 2) * yscale

        road_top, road_bottom = project(0, lower)[1], project(0, upper)[1]
        draw.rectangle((30, road_top, WIDTH - 30, road_bottom), fill=2)
        for y in range(lower, upper + 1, 4):
            py = project(0, y)[1]
            if y in (lower, upper):
                draw.line((30, py, WIDTH - 30, py), fill=1, width=2)
            else:
                for x in range(30, WIDTH - 30, 20):
                    draw.line((x, py, x + 10, py), fill=0, width=2)
        for dx in np.arange(math.ceil(left / 20) * 20, right, 20):
            px = project(dx, 0)[0]
            draw.line((px, road_bottom + 2, px, road_bottom + 6), fill=3)
            draw.text((px - 9, road_bottom + 8), f"{dx:.0f}", fill=1, font=SMALL)
        for name, actor in states.items():
            color, short = ACTORS[name]
            visible = name == "ego" or name in frame["visible_ids"]
            dx = actor["x"] - origin
            if left + actor["length"] / 2 <= dx <= right - actor["length"] / 2:
                points = [project(x, y) for x, y in body_vertices(actor, origin)]
                draw.polygon(
                    points, fill=color if visible else 0, outline=color, width=2
                )
                if overview:
                    label_y = road_top - 15 - list(states).index(name) % 2 * 14
                    draw.line(
                        (
                            project(dx, actor["y"])[0],
                            label_y + 13,
                            *project(dx, actor["y"]),
                        ),
                        fill=color,
                    )
                else:
                    label_y = (
                        project(dx, actor["y"])[1] - actor["width"] * yscale / 2 - 16
                    )
                draw.text(
                    (project(dx, actor["y"])[0] - 4, label_y),
                    short + ("*" if not visible else ""),
                    fill=color,
                    font=SMALL,
                )
                if not overview:
                    front = project(
                        dx + math.cos(actor["heading"]) * actor["length"] * 0.38,
                        actor["y"]
                        + math.sin(actor["heading"]) * actor["length"] * 0.38,
                    )
                    draw.ellipse(
                        (front[0] - 2, front[1] - 2, front[0] + 2, front[1] + 2), fill=0
                    )
            elif not overview:
                x = 32 if dx < left else WIDTH - 195
                y = 73 + list(states).index(name) * 15
                draw.text(
                    (x, y),
                    f"{'<' if dx < left else '>'} {short}{'' if visible else '*'} {dx:+.1f} m",
                    fill=color,
                    font=SMALL,
                )
        draw.text((WIDTH - 100, road_bottom + 8), "travel  --->", fill=1, font=SMALL)

    panel(-30, 90, 68, 224, overview=False)
    draw.text(
        (12, 237),
        "All actors, relative x (m); overview has a separate longitudinal scale",
        fill=1,
        font=SMALL,
    )
    panel(*extent, 278, 342, overview=True)
    legend = "   ".join(
        f"{short}={name.replace('_', ' ')}"
        for name, (_, short) in ACTORS.items()
        if name in states
    )
    draw.text((12, 383), legend, fill=1, font=SMALL)
    draw.text(
        (12, 405),
        "* not visible to ADS; outlined actor is still physically present.  Main view uses equal x/y scale.  Every saved 0.05 s frame is retained.",
        fill=1,
        font=SMALL,
    )
    if frame["collision"]:
        draw.rectangle((1, 1, WIDTH - 2, HEIGHT - 2), outline=8, width=3)
    return image


def render_case(
    sut_id,
    path,
    candidate_count=None,
    *,
    inspect_record=inspect_safe,
    destination=DESTINATION,
):
    payload = read_json(path)
    inspection = inspect_record(payload)
    trace = payload["trajectory"]
    relative = [
        a["x"] - next(v["x"] for v in f["actors"] if v["actor_id"] == "ego")
        for f in trace
        for a in f["actors"]
    ]
    extent = (min(-30, min(relative) - 8), max(90, max(relative) + 8))
    destination = destination / payload["scenario"]["kind"]
    destination.mkdir(parents=True, exist_ok=True)
    images = [
        draw_frame(payload, sut_id, i, inspection, extent) for i in range(len(trace))
    ]
    gif_path = destination / f"{sut_id}.gif"
    playback = save_gif(images, [f["time_s"] for f in trace], gif_path)
    with Image.open(gif_path) as gif:
        assert gif.n_frames == len(images)
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            assert frame.info["duration"] == (500 if i == len(images) - 1 else 50)
            assert (
                ImageChops.difference(
                    frame.convert("RGB"), images[i].convert("RGB")
                ).getbbox()
                is None
            )
    inspection.pop("clearance_series_m")
    result = {
        "sut_id": sut_id,
        "scenario": payload["scenario"],
        "source": path.relative_to(ROOT).as_posix(),
        "gif": gif_path.relative_to(ROOT).as_posix(),
        "inspection": inspection,
        "gif_frames": playback["frames"],
        "gif_pixels_and_timestamps_verified": True,
        "warnings": payload.get("warnings", []),
    }
    if candidate_count is not None:
        result["safe_candidates"] = candidate_count
    return result


def render_contact_sheets(results, destination, frame_labels):
    groups = {}
    for result in results:
        kind = result["scenario"]["kind"]
        group = "target" if result["sut_id"].startswith("target_") else "history"
        groups.setdefault((kind, group), []).append(result)

    paths = []
    frame_size = (560, 89)
    gap, margin, row_height = 12, 12, 157
    for (kind, group), cases in groups.items():
        for start in range(0, len(cases), 4):
            page = cases[start : start + 4]
            columns = [margin + i * (frame_size[0] + gap) for i in range(len(frame_labels))]
            width = columns[-1] + frame_size[0] + margin
            bottom = 48 + row_height * len(page)
            sheet = Image.new("RGB", (width, bottom + 45), "white")
            draw = ImageDraw.Draw(sheet)
            actors = set()
            has_occlusion = False
            for x, label in zip(columns, frame_labels):
                draw.text((x + frame_size[0] / 2, 7), label, anchor="mt",
                          fill=COLORS[1], font=SHEET_FONT)
            for row, result in enumerate(page):
                y = 48 + row * row_height
                record = result["scenario"]["scenario_id"].rsplit(":", 1)[-1]
                draw.text((margin, y), f"{result['sut_id']} · record {record}",
                          fill=COLORS[1], font=SHEET_FONT)
                trajectory = read_json(ROOT / result["source"])["trajectory"]
                with Image.open(ROOT / result["gif"]) as gif:
                    for x, index in zip(columns, result["inspection"]["keyframe_indices"], strict=True):
                        snapshot = trajectory[index]
                        ids = {actor["actor_id"] for actor in snapshot["actors"]}
                        actors.update(ids)
                        has_occlusion |= bool(ids.difference(snapshot["visible_ids"], {"ego"}))
                        gif.seek(index)
                        frame = gif.convert("RGB").crop((0, 57, WIDTH, 235))
                        sheet.paste(frame.resize(frame_size, Image.Resampling.LANCZOS), (x, y + 34))
                        if snapshot["collision"]:
                            draw.rectangle((x, y + 34, x + frame_size[0] - 1, y + 122),
                                           outline=COLORS[8], width=2)
                        draw.text((x + frame_size[0] / 2, y + 125),
                                  f"{snapshot['time_s']:.2f} s", anchor="mt",
                                  fill=COLORS[1], font=SHEET_FONT)
            labels = [(name.replace("_", " ").capitalize(), COLORS[color]) for name, (color, _) in
                      ACTORS.items() if name in actors]
            widths = [24 + SHEET_FONT.getlength(label) + 36 for label, _ in labels]
            note = "Outline / *  Outside ADS view" if has_occlusion else ""
            x = (width - sum(widths) - SHEET_FONT.getlength(note)) / 2
            for (label, color), item_width in zip(labels, widths, strict=True):
                draw.rectangle((x, bottom + 9, x + 18, bottom + 25), fill=color)
                draw.text((x + 24, bottom + 4), label, fill=COLORS[1], font=SHEET_FONT)
                x += item_width
            if note:
                draw.text((x, bottom + 4), note, fill=COLORS[1], font=SHEET_FONT)
            path = destination / kind / f"{group}_{start // 4 + 1}_frames.png"
            sheet.save(path)
            paths.append(path.relative_to(ROOT).as_posix())
    return paths


def render_details(results):
    paths = []
    for result in results:
        if result["inspection"]["minimum_body_clearance_m"] >= 0.1:
            continue
        with Image.open(ROOT / result["gif"]) as gif:
            indices = sorted(
                {
                    0,
                    10,
                    20,
                    30,
                    40,
                    60,
                    100,
                    150,
                    gif.n_frames - 1,
                    result["inspection"]["keyframe_indices"][2],
                }
            )
            sheet = Image.new("RGB", (WIDTH, 235 * len(indices)), "white")
            for row, index in enumerate(indices):
                gif.seek(index)
                sheet.paste(gif.convert("RGB").crop((0, 0, WIDTH, 235)), (0, 235 * row))
            path = (ROOT / result["gif"]).with_name(
                result["sut_id"] + "_near_miss_frames.png"
            )
            sheet.save(path)
            paths.append(path.relative_to(ROOT).as_posix())
    for kind in ("cut_out_rear_gap", "side_encroachment"):
        cases = [
            row
            for row in results
            if row["scenario"]["kind"] == kind
            and row["sut_id"].split("_", 1)[1] in ("uct", "beam")
        ]
        sheet = Image.new("RGB", (2400, 120 * len(cases)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, result in enumerate(cases):
            draw.text(
                (8, row * 120 + 3),
                result["sut_id"]
                + " | record "
                + result["scenario"]["scenario_id"].rsplit(":", 1)[-1],
                fill="black",
                font=FONT,
            )
            with Image.open(ROOT / result["gif"]) as gif:
                for column, index in enumerate((0, 40, 80, 120, 160, 200)):
                    gif.seek(index)
                    frame = gif.convert("RGB").crop((0, 0, WIDTH, 235))
                    sheet.paste(
                        frame.resize((400, 84), Image.Resampling.LANCZOS),
                        (column * 400, row * 120 + 29),
                    )
        path = DESTINATION / kind / "planner_motion_frames.png"
        sheet.save(path)
        paths.append(path.relative_to(ROOT).as_posix())
    return paths


def main():
    results = []
    for sut_id, path, count in select_cases():
        result = render_case(sut_id, path, count)
        results.append(result)
        check = result["inspection"]
        kind = result["scenario"]["kind"]
        print(
            f"{len(results):02d}/64 {kind} {sut_id} {path.name}: clearance={check['minimum_body_clearance_m']:.4f}, road={check['minimum_road_margin_m']:.4f}, issues={check['issues']}",
            flush=True,
        )
    sheet_paths = render_contact_sheets(results, DESTINATION, FRAME_LABELS)
    write_json(
        RESULTS / "summary.json",
        {
            "seed": SEED,
            "selection": "one uniformly random archived Safe record per ADS configuration; fixed scene and registered summary order; sorted source filenames",
            "new_simulator_executions": 0,
            "cases": results,
            "contact_sheets": sheet_paths,
            "detail_sheets": render_details(results),
            "all_passed": all(r["inspection"]["passed"] for r in results),
            "physics_frames": sum(r["inspection"]["physics_frames"] for r in results),
            "gif_frame_pixels_and_timestamps_verified": all(
                r["gif_pixels_and_timestamps_verified"] for r in results
            ),
        },
    )


if __name__ == "__main__":
    main()
