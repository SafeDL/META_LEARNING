"""Export verified highway-env GIFs of collisions already found in S01."""
import gzip
import json

from PIL import Image, ImageDraw, ImageFont

from highway_sim_env.envs.unified_env import PHYSICS_HZ, UnifiedHighwayEnv
from .common import REPO, read_json, write_json
from .risk import PassiveRiskObserver
from .s01 import ROOT, TARGET, build_spec, configuration, response_rows


SEED = 11
EXAMPLE_COUNT = 3
OUTPUT = ROOT / "visualizations"
WIDTH = 1000
ROAD_HEIGHT = 200
HEADER_HEIGHT = 144
HEIGHT = 460
EGO_COLOR = (47, 118, 211)
LEAD_COLOR = (226, 158, 42)
TITLE_FONT = ImageFont.load_default(size=25)
FONT = ImageFont.load_default(size=19)
SMALL_FONT = ImageFont.load_default(size=17)


def annotated_frame(env, query, measurement, expected):
    for role, actor in env.actors.items():
        actor.color = (211, 54, 61) if actor.crashed else (
            EGO_COLOR if role == "ego" else LEAD_COLOR)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f6fa")
    image.paste(Image.fromarray(env.render()), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 52), fill="#17263c")
    draw.text((22, 13),
              f"SRD-TNP-BQD | S01 | seed {SEED} | query {query['query_number']:03d}"
              f" | scene {query['index']:04d}",
              font=TITLE_FONT, fill="white")
    params = env.scenario["active_parameters"]
    draw.text((22, 65),
              f"Initial clearance {params['initial_clearance_m']:.2f} m   |   "
              f"Cut-in speed {params['lead_speed_mps']:.2f} m/s   |   "
              f"Start {params['event_start_s']:.2f} s   |   "
              f"Duration {params['lane_change_time_scale_s']:.2f} s",
              font=SMALL_FONT, fill="#26374f")
    draw.rounded_rectangle((22, 103, 43, 124), radius=4, fill=EGO_COLOR)
    draw.text((52, 104), "Ego: original FVDM target", font=SMALL_FONT, fill="#26374f")
    draw.rounded_rectangle((358, 103, 379, 124), radius=4, fill=LEAD_COLOR)
    draw.text((388, 104), "Scripted cut-in vehicle", font=SMALL_FONT, fill="#26374f")
    draw.text((716, 104), "Red = collision", font=SMALL_FONT, fill="#bc2836")
    pair = measurement["pairs"]["lead"]
    time_s = env.steps / PHYSICS_HZ
    footer = HEADER_HEIGHT + ROAD_HEIGHT
    draw.text((22, footer + 14),
              f"t = {time_s:.2f} s   |   Ego {env.vehicle.speed:.2f} m/s   |   "
              f"Cut-in {env.actors['lead'].speed:.2f} m/s   |   "
              f"Body distance {pair['body_distance']:.3f} m",
              font=FONT, fill="#26374f")
    collided = env.vehicle.crashed
    draw.text((22, footer + 48),
              f"{'COLLISION' if collided else 'RUNNING'}   |   "
              f"Risk now {pair['risk']:.4f}   |   "
              f"Episode peak risk {expected['risk']:.4f}   |   cell {query['cell']}",
              font=FONT, fill="#bc2836" if collided else "#176746")
    draw.text((22, footer + 84),
              "Original 20 Hz physics; playback at half speed; final frame held 1.5 s.",
              font=SMALL_FONT, fill="#66748a")
    return image


def export_collision(scene, expected, query):
    with gzip.open(REPO / expected["trace_path"], "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    env = UnifiedHighwayEnv(build_spec(TARGET), scene)
    env.render_mode = "rgb_array"
    env.config.update(offscreen_rendering=True, screen_width=WIDTH,
                      screen_height=ROAD_HEIGHT, scaling=12.0,
                      centering_position=[.28, .35])
    observer = PassiveRiskObserver(configuration()["measurement"])
    frames = []
    try:
        env.reset(seed=scene["simulator_seed"])
        observer.observe(env)
        frames.append(annotated_frame(env, query, observer.frames[-1], expected))
        limit = int(round(env.config["duration"] * PHYSICS_HZ))
        while env.steps < limit and not any(actor.crashed for actor in env.actors.values()):
            env._advance()
            observer.observe(env)
            frames.append(annotated_frame(env, query, observer.frames[-1], expected))
        result = env.result()
        summary = observer.summary()
        assert result["ego_collision"] and query["label"] == 1
        assert result["build_fingerprint"] == expected["build_fingerprint"]
        assert env.trace == saved["original_runner_trace"]
        assert observer.frames == saved["frames"]
        assert summary["valid_risk"] and abs(summary["risk"] - query["risk"]) < 1e-12
    finally:
        env.close()
    stem = f"seed_{SEED}_q{query['query_number']:03d}_scene_{query['index']:04d}_collision"
    path = OUTPUT / f"{stem}.gif"
    durations = [100] * len(frames)
    durations[0], durations[-1] = 500, 1500
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2, optimize=False)
    with Image.open(path) as gif:
        assert gif.n_frames == len(frames) and gif.size == (WIDTH, HEIGHT)
        for index in range(gif.n_frames):
            gif.seek(index)
            gif.load()
        assert gif.info["duration"] == 1500
    row = {
        "gif": path.name, "seed": SEED,
        "query_number": query["query_number"], "candidate_index": query["index"],
        "scenario_id": scene["scenario_id"], "cell": query["cell"],
        "active_parameters": scene["active_parameters"], "target": TARGET,
        "ego_initial_speed_mps": scene["fixed_context"]["ego_speed_mps"],
        "ego_collision": True, "collision_time_s": result["collision_time_s"],
        "collision_partner_role": result["collision_partner_role"],
        "collision_type": result["collision_type"],
        "observed_maneuver_phase": result["observed_maneuver_phase"],
        "risk": summary["risk"], "peak_time_s": summary["peak_time_s"],
        "frames": len(frames), "physics_hz": PHYSICS_HZ, "playback_speed": .5,
        "saved_runner_trace_equal": True, "saved_measurement_frames_equal": True,
        "collision_and_risk_equal": True, "saved_trace": expected["trace_path"],
    }
    print(path.relative_to(REPO), f"collision at {result['collision_time_s']:.2f} s", flush=True)
    return row


def main():
    run_path = ROOT / "comparison/runs" / f"SRD_TNP_BQD_seed_{SEED}.json"
    run = read_json(run_path)
    scenes, rows = response_rows("D")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    examples = []
    cells = set()
    for query in run["queries"][:50]:
        if not query["label"] or query["cell"] in cells:
            continue
        index = query["index"]
        assert query["scenario_id"] == scenes[index]["scenario_id"]
        examples.append(export_collision(scenes[index], rows[index], query))
        cells.add(query["cell"])
        if len(examples) == EXAMPLE_COUNT:
            break
    assert len(examples) == EXAMPLE_COUNT
    write_json(OUTPUT / "examples.json", {
        "method": "SRD-TNP-BQD", "selection_log": str(run_path.relative_to(REPO)),
        "selection": "Three already discovered collisions before query 50, in distinct cells",
        "visualization_only_physical_replays": len(examples),
        "experiment_budget_changed": False, "verification": "PASS", "examples": examples,
    })


if __name__ == "__main__":
    main()
