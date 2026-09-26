"""20 Hz source-vs-target GIF audit of five charged collision modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from highway_sim_env.envs.cutin_env import CutInScenario
from highway_sim_env.envs.single_lane_longitudinal import (
    _SingleLaneLongitudinalMixin,
)
from methods.core_mine.multimode20_experiment import BOUNDS, ROOT, SEEDS, _read
from methods.core_mine.replay_source_safe_comparison import ComparisonEnv, _font, _panel
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


OUTPUT = ROOT / "gifs" / "modeshift_collisions_20hz"
METHOD = "ModeShift-Risk"
TARGET = "vi_ttc"
SOURCE = "idm_mobil"


class SingleLaneComparisonEnv(_SingleLaneLongitudinalMixin, ComparisonEnv):
    """Capture actual 20 Hz frames without changing either controller."""


def cases() -> dict[str, dict]:
    found = {}
    for seed in SEEDS:
        with np.load(ROOT / str(seed) / "source_bank.npz", allow_pickle=False) as bank:
            anchors, modes, controls = (bank[key].copy() for key in
                                        ("anchors", "modes", "controls"))
            source_safe = ~((bank["ego_collision"] | bank["near_miss"]).any(axis=0))
            source_safe &= bank["completed"].all(axis=0)
        rows = _read(ROOT / str(seed) / TARGET / f"{METHOD}.csv")
        for row in rows:
            mode = row["mode"]
            if mode in found or row["ego_collision"] != "True":
                continue
            index = int(row["index"])
            if not source_safe[index] or mode != str(modes[index]):
                raise RuntimeError("displayed target collision is not source-safe")
            found[mode] = {
                "seed": seed, "query": int(row["query"]), "index": index,
                "mode": mode, "gap": float(anchors[index, 0]),
                "relative_speed": float(anchors[index, 1]),
                "timing": float(controls[index, 0]),
                "intensity": float(controls[index, 1]),
                "charged_min_ttc": float(row["min_ttc"]),
                "charged_min_clearance": float(row["min_clearance"]),
            }
    return found


def _run(case: dict, sut: str):
    policy = policy_factory(sut, ASSETS)
    scenario = CutInScenario(case["gap"], case["relative_speed"],
                             case["mode"], case["timing"], case["intensity"])
    env = SingleLaneComparisonEnv(scenario, ego_kind=policy.ego_kind,
                                  render_mode="rgb_array")
    env.config["policy_frequency"] = 20
    try:
        env.reset(seed=case["seed"] + case["index"])
        policy.reset()
        env.capture()
        env.capture_enabled = True
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(policy.act(env))
        result = env.external_result()
    finally:
        env.close()
    if (len(env.physics_frames) != len(env.ego_crashed)
            or len(env.physics_frames) != len(env.instant_ttc)):
        raise RuntimeError("20 Hz physical replay ledger mismatch")
    return env, result


def render(case: dict) -> dict:
    target, target_result = _run(case, TARGET)
    source, source_result = _run(case, SOURCE)
    if (not target_result.ego_collision or source_result.ego_collision
            or source_result.near_miss or not source_result.completed
            or not np.isclose(target_result.min_ttc,
                              case["charged_min_ttc"], equal_nan=True)
            or not np.isclose(target_result.min_distance,
                              case["charged_min_clearance"], atol=1e-9)):
        raise RuntimeError("visual replay differs from charged outcome")
    end = next(index for index, crashed in enumerate(target.ego_crashed) if crashed)
    lane_count = 2 if case["mode"] in {"fast_intrusion", "cutin_braking"} else 1
    frames = []
    # Keep the failed target at its collision frame while the historical
    # controller finishes its full episode. Otherwise the animation stops
    # before the viewer can see that the historical run completed safely.
    for index in range(max(end + 1, len(source.physics_frames))):
        left = _panel(target, index, f"NEW TARGET: {TARGET}",
                      target=True, last_index=end)
        right = _panel(source, index, f"HISTORY: {SOURCE}",
                       target=False, last_index=len(source.physics_frames) - 1)
        image = Image.new("RGB", (left.width + right.width, left.height + 38),
                          (18, 25, 36))
        image.paste(left, (0, 38))
        image.paste(right, (left.width, 38))
        ImageDraw.Draw(image).text(
            (10, 9), f"{case['mode']} | {lane_count} lane(s) | ego + physics 20 Hz (0.05 s) | query {case['query']}/50",
            font=_font(17), fill=(255, 255, 255))
        frames.append(image)
    path = OUTPUT / f"{case['mode']}_20hz.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[50] * (len(frames) - 1) + [1000], loop=0,
                   disposal=2, optimize=True)
    outcome_path = OUTPUT / f"{case['mode']}_20hz_outcome.png"
    frames[-1].save(outcome_path)
    timeline = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for row, index in enumerate((max(0, end - 16), max(0, end - 8), end)):
        timeline.paste(frames[index], (0, row * frames[0].height))
    timeline_path = OUTPUT / f"{case['mode']}_20hz_timeline.png"
    timeline.save(timeline_path)
    return {**case, "gif": path.name, "timeline_png": timeline_path.name,
            "outcome_png": outcome_path.name,
            "lane_count": lane_count,
            "target_collision_time_s": float(target.physics_frames[end].time),
            "source_completion_time_s": float(source.physics_frames[-1].time),
            "target_min_clearance_m": float(target_result.min_distance),
            "source_min_clearance_m": float(source_result.min_distance),
            "source_completed_safely": True,
            "ego_control_interval_s": .05, "physics_frame_interval_s": .05}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = cases()
    if set(selected) != set(BOUNDS):
        raise RuntimeError("frozen B=50 ModeShift traces lack a collision mode")
    items = [render(selected[mode]) for mode in BOUNDS]
    (OUTPUT / "manifest.json").write_text(
        json.dumps({"method": METHOD, "target": TARGET, "history": SOURCE,
                    "budget": 50,
                    "selection": "first charged ego collision per mode in frozen seed/query order",
                    "cases": items}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"wrote {len(items)} verified 20 Hz collision GIFs to {OUTPUT}")


if __name__ == "__main__":
    main()
