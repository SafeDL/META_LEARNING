"""20 Hz side-by-side physical collision audit for charged CoRe queries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from highway_env_benchmark.envs.cutin_env import CutInScenario
from method_chains.core_mine.heterogeneous20_replication import ROOT, SEEDS
from method_chains.core_mine.replay_source_safe_comparison import ComparisonEnv, _font, _panel
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


OUTPUT = ROOT / "gifs" / "core_collisions_20hz"
MODES = ("fast_intrusion", "cutin_braking", "lead_braking", "stop_and_go",
         "slow_lead_following")
TARGET = "vi_ttc"
SOURCE = "idm_mobil"


def _cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for seed in SEEDS:
        with np.load(ROOT / str(seed) / "source_bank.npz", allow_pickle=False) as bank:
            anchors, modes, controls = (bank[key].copy() for key in
                                        ("anchors", "modes", "controls"))
            source_event = bank["ego_collision"] | bank["near_miss"]
            source_completed = bank["completed"].copy()
        with (ROOT / str(seed) / "CoRe-Residual.csv").open(
                encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            mode = row["mode"]
            if mode in cases or row["ego_collision"] != "True":
                continue
            index = int(row["index"])
            if source_event[:, index].any() or not source_completed[:, index].all():
                raise RuntimeError("charged collision was not source-safe")
            cases[mode] = {
                "seed": seed, "query": int(row["query"]), "index": index,
                "mode": mode, "gap": float(anchors[index, 0]),
                "relative_speed": float(anchors[index, 1]),
                "timing": float(controls[index, 0]),
                "intensity": float(controls[index, 1]),
            }
    return cases


def _run(case: dict, sut: str) -> tuple[ComparisonEnv, object]:
    policy = policy_factory(sut, ASSETS)
    scenario = CutInScenario(case["gap"], case["relative_speed"], case["mode"],
                             case["timing"], case["intensity"])
    env = ComparisonEnv(scenario, ego_kind=policy.ego_kind, render_mode="rgb_array")
    env.config["policy_frequency"] = 20
    try:
        env.reset(seed=case["seed"] + case["index"])
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
        raise RuntimeError("physical replay frame ledger mismatch")
    return env, result


def _render(case: dict) -> dict:
    target, target_result = _run(case, TARGET)
    source, source_result = _run(case, SOURCE)
    if (not target_result.ego_collision or source_result.ego_collision
            or source_result.near_miss or not source_result.completed):
        raise RuntimeError("physical replay disagrees with charged target/source record")
    end = next(index for index, crashed in enumerate(target.ego_crashed) if crashed)
    frames = []
    for index in range(end + 1):
        left = _panel(target, index, f"NEW TARGET: {TARGET}",
                      target=True, last_index=end)
        right = _panel(source, index, f"HISTORY: {SOURCE}", target=False,
                       last_index=len(source.physics_frames) - 1)
        image = Image.new("RGB", (left.width + right.width, left.height + 38),
                          (18, 25, 36))
        image.paste(left, (0, 38))
        image.paste(right, (left.width, 38))
        ImageDraw.Draw(image).text(
            (10, 9), f"{case['mode']} | same scenario and seed | ego + physics 20 Hz | query {case['query']}/50",
            font=_font(17), fill=(255, 255, 255))
        frames.append(image)
    path = OUTPUT / f"{case['mode']}_core_20hz.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[50] * (len(frames) - 1) + [1000], loop=0,
                   disposal=2, optimize=True)
    timeline = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    indices = [max(0, end - 16), max(0, end - 8), end]
    for row, index in enumerate(indices):
        timeline.paste(frames[index], (0, row * frames[0].height))
    timeline_path = OUTPUT / f"{case['mode']}_core_20hz_timeline.png"
    timeline.save(timeline_path)
    return {**case, "gif": path.name, "timeline_png": timeline_path.name,
            "target_collision_time_s": target.physics_frames[end].time,
            "target_min_clearance_m": float(target_result.min_distance),
            "source_min_clearance_m": float(source_result.min_distance),
            "source_completed_safely": True,
            "ego_control_interval_s": .05, "physics_frame_interval_s": .05}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    if len(cases) < 2:
        raise RuntimeError("expected charged collisions in at least two modes")
    items = [_render(cases[mode]) for mode in MODES if mode in cases]
    (OUTPUT / "manifest.json").write_text(
        json.dumps({"method": "CoRe-Residual", "budget": 50,
                    "selection": "first charged ego collision per mode in frozen seed/query order",
                    "missing_collision_modes": [mode for mode in MODES if mode not in cases],
                    "cases": items}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"wrote {len(items)} 20 Hz verified collision replays to {OUTPUT}")


if __name__ == "__main__":
    main()
