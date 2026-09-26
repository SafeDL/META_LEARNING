"""Show the same source-safe scenario under a target and a historical controller.

This is a visual audit of already selected B=50 discoveries, not a new search
experiment. Every displayed state comes from the original 20 Hz physics step;
the external policies still choose a high-level action at 5 Hz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from highway_sim_env.envs.cutin_env import CutInScenario
from methods.core_mine.replay_source_safe_discoveries import Case, ReplayEnv, ROOT, _cases
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


HISTORICAL_CONTROLLER = "ppo_ece"
OUTPUT = ROOT / "gifs" / "comparisons"


class ComparisonEnv(ReplayEnv):
    def __init__(self, *args, **kwargs) -> None:
        self.ego_crashed: list[bool] = []
        super().__init__(*args, **kwargs)

    def capture(self) -> None:
        self.ego_crashed.append(bool(self.vehicle.crashed))
        super().capture()


def _run(case: Case, sut: str) -> tuple[ComparisonEnv, object]:
    policy = policy_factory(sut, ASSETS)
    scenario = CutInScenario(case.gap, case.relative_speed, case.mode,
                             case.timing, case.intensity)
    env = ComparisonEnv(scenario, ego_kind=policy.ego_kind, render_mode="rgb_array")
    try:
        env.reset(seed=case.seed + case.index)
        policy.reset()
        env.capture()
        env.capture_enabled = True
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy.act(env)
            _, _, terminated, truncated, _ = env.step(action)
        result = env.external_result()
    finally:
        env.close()
    if len(env.physics_frames) != len(env.ego_crashed):
        raise RuntimeError("frame/crash trace mismatch")
    return env, result


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _panel(env: ComparisonEnv, position: int, label: str, *, target: bool,
           last_index: int) -> Image.Image:
    frame = env.physics_frames[min(position, last_index)]
    crashed = env.ego_crashed[min(position, last_index)]
    ttc = float(np.min(env.instant_ttc[:min(position, last_index) + 1]))
    raw = Image.fromarray(frame.image).convert("RGB")
    # The camera keeps the ego near x=180. This crop doubles vehicle size while
    # retaining the scheduled vehicle throughout this fixed, short-gap pool.
    road = raw.crop((70, 0, min(400, raw.width), raw.height))
    road = road.resize((660, 2 * raw.height), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (660, road.height + 92), (18, 25, 36))
    panel.paste(road, (0, 92))
    draw = ImageDraw.Draw(panel)
    font = _font(17)
    small = _font(15)
    status = "EGO COLLISION REGISTERED" if crashed else (
        "DANGER: clearance < 1 m or TTC < 1.5 s"
        if frame.polygon_clearance < 1 or ttc < 1.5 else "no event yet")
    if not target and not crashed and position >= last_index:
        status = "completed safely"
    color = (255, 100, 100) if crashed else (
        (255, 214, 110) if status.startswith("DANGER") else (180, 230, 205))
    draw.text((10, 7), f"{label}  |  t={frame.time:.2f} s", font=font, fill=(255, 255, 255))
    draw.text((10, 33), status, font=font, fill=color)
    ttc_text = "inf" if not np.isfinite(ttc) else f"{ttc:.2f} s"
    draw.text((10, 63), f"current clearance {frame.polygon_clearance:.2f} m   min TTC so far {ttc_text}",
              font=small, fill=(210, 225, 245))
    if crashed:
        draw.rectangle((0, 92, panel.width - 1, panel.height - 1),
                       outline=(255, 70, 70), width=5)
    return panel


def _render(case: Case) -> dict:
    target_env, target_result = _run(case, case.target)
    source_env, source_result = _run(case, HISTORICAL_CONTROLLER)
    if not target_result.ego_collision or target_result.near_miss:
        raise RuntimeError("selected target case is not an ego collision")
    if source_result.ego_collision or source_result.near_miss:
        raise RuntimeError("selected historical controller is not safe")
    target_end, source_end = len(target_env.physics_frames) - 1, len(source_env.physics_frames) - 1
    if not target_env.ego_crashed[-1]:
        raise RuntimeError("target collision was not captured")
    frames = []
    for index in range(max(target_end, source_end) + 1):
        target = _panel(target_env, index, f"NEW TARGET: {case.target}", target=True,
                        last_index=target_end)
        source = _panel(source_env, index, f"HISTORICAL: {HISTORICAL_CONTROLLER}", target=False,
                        last_index=source_end)
        image = Image.new("RGB", (target.width + source.width, target.height + 38), (18, 25, 36))
        image.paste(target, (0, 38))
        image.paste(source, (target.width, 38))
        ImageDraw.Draw(image).text(
            (10, 9), f"{case.mode} | same scenario, same seed | physics 20 Hz; external decisions 5 Hz",
            font=_font(17), fill=(255, 255, 255))
        frames.append(image)
    path = OUTPUT / f"{case.mode}_target_vs_history.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[50] * (len(frames) - 1) + [1000], loop=0,
                   disposal=2, optimize=True)
    # The side-by-side still at the instant of the target collision is easier
    # to inspect than a GIF's initial frame in static previews.
    frames[target_end].save(path.with_suffix(".png"))
    # A three-timepoint strip makes the closing trajectory inspectable even
    # when a GIF viewer drops frames or displays only its first frame.
    timeline_indices = [max(0, target_end - 16), max(0, target_end - 8), target_end]
    timeline = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for row, frame_index in enumerate(timeline_indices):
        timeline.paste(frames[frame_index], (0, row * frames[0].height))
    timeline_path = OUTPUT / f"{case.mode}_event_timeline.png"
    timeline.save(timeline_path)
    return {"mode": case.mode, "target": case.target, "historical": HISTORICAL_CONTROLLER,
            "seed": case.seed, "index": case.index, "gif": path.name,
            "event_timeline_png": timeline_path.name,
            "target_collision_time_s": target_env.physics_frames[-1].time,
            "target_min_clearance_m": target_result.min_distance,
            "target_min_ttc_s": target_result.min_ttc,
            "historical_ego_collision": source_result.ego_collision,
            "historical_near_miss": source_result.near_miss,
            "historical_min_clearance_m": source_result.min_distance,
            "historical_min_ttc_s": source_result.min_ttc if np.isfinite(source_result.min_ttc) else None,
            "physics_frame_interval_s": 0.05, "external_action_interval_s": 0.2}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    items = [_render(cases[mode]) for mode in cases]
    (OUTPUT / "manifest.json").write_text(json.dumps({"budget": 50, "cases": items},
                                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} verified comparisons to {OUTPUT}")


if __name__ == "__main__":
    main()
