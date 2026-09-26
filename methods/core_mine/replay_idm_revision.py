"""Replay first charged collision per mode in the 20 Hz IDM revision study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from highway_sim_env.envs.cutin_env import CutInEnv, CutInScenario
from methods.core_mine.idm_revision_pilot import BUILDS
from methods.core_mine.idm_revision_validation import ROOT, TARGETS
from methods.core_mine.replay_sparse_critical_cases import PhysicsFrame
from methods.core_mine.replay_source_safe_comparison import _font, _panel


OUTPUT = ROOT / "gifs" / "comparisons"
METHOD = "HistoryMargin-Residual"
TARGET = "idm_delay07_brake3"


class IDMReplayEnv(CutInEnv):
    def __init__(self, *args, **kwargs) -> None:
        self.physics_frames: list[PhysicsFrame] = []
        self.ego_crashed: list[bool] = []
        self.instant_ttc: list[float] = []
        self.capture_enabled = False
        super().__init__(*args, **kwargs)

    def capture(self) -> None:
        ego, lead = self.vehicle, self._cutin_vehicle
        gap = float(lead.position[0] - ego.position[0])
        closing = float(ego.speed - lead.speed)
        lateral = abs(float(lead.position[1] - ego.position[1]))
        ttc = gap / closing if gap > 0 and lateral < ego.WIDTH and closing > 1e-6 else float("inf")
        polygon_clearance = float(Polygon(ego.polygon()).distance(Polygon(lead.polygon())))
        self.physics_frames.append(PhysicsFrame(np.asarray(self.render()).copy(),
                                                float(lead.elapsed), polygon_clearance,
                                                polygon_clearance))
        self.ego_crashed.append(bool(ego.crashed))
        self.instant_ttc.append(ttc)

    def _record_lead_trace(self) -> None:
        super()._record_lead_trace()
        if self.capture_enabled:
            self.capture()


def _cases() -> dict[str, dict]:
    cases = {}
    with (ROOT / "records.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["method"] == METHOD and row["heterogeneity"] == TARGET
                and int(row["repeat"]) == 0]
    for row in rows:
        seed = int(row["seed"])
        with np.load(ROOT / str(seed) / "candidate_pool.npz", allow_pickle=False) as pool, \
             np.load(ROOT / str(seed) / "target_bank.npz", allow_pickle=False) as bank:
            target_index = list(bank["target_names"].astype(str)).index(TARGET)
            for index in map(int, row["queried_indices"].split(";")):
                mode = str(pool["modes"][index])
                if mode in cases or not bool(bank["ego_collision"][target_index, index]):
                    continue
                if bool(pool["source_ego_collision"][index]) or bool(pool["source_near_miss"][index]):
                    raise RuntimeError("displayed case is not historically safe")
                cases[mode] = {"seed": seed, "index": index,
                               "original_index": int(pool["original_indices"][index]),
                               "mode": mode, "gap": float(pool["anchors"][index, 0]),
                               "relative_speed": float(pool["anchors"][index, 1]),
                               "timing": float(pool["controls"][index, 0]),
                               "intensity": float(pool["controls"][index, 1])}
    return cases


def _run(case: dict, build: str) -> tuple[IDMReplayEnv, object]:
    scenario = CutInScenario(case["gap"], case["relative_speed"], case["mode"],
                             case["timing"], case["intensity"])
    env = IDMReplayEnv(BUILDS[build], scenario, render_mode="rgb_array")
    try:
        env.reset(seed=case["seed"] + case["original_index"])
        env.capture()
        env.capture_enabled = True
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
        result = env.episode_result()
    finally:
        env.close()
    return env, result


def _render(case: dict) -> dict:
    target, target_result = _run(case, TARGET)
    source, source_result = _run(case, "idm_ref")
    if not target.ego_crashed[-1] or source.ego_crashed[-1] or source_result.near_miss:
        raise RuntimeError("replay outcomes disagree with stored collision/source-safe labels")
    end = next(i for i, crashed in enumerate(target.ego_crashed) if crashed)
    frames = []
    for index in range(end + 1):
        left = _panel(target, index, f"MUTANT: {TARGET}", target=True, last_index=end)
        right = _panel(source, index, "HISTORY: idm_ref", target=False,
                       last_index=len(source.physics_frames) - 1)
        frame = Image.new("RGB", (left.width + right.width, left.height + 38), (18, 25, 36))
        frame.paste(left, (0, 38))
        frame.paste(right, (left.width, 38))
        ImageDraw.Draw(frame).text((10, 9),
            f"{case['mode']} | same scenario and seed | ego control + physics 20 Hz | B=50",
            font=_font(17), fill=(255, 255, 255))
        frames.append(frame)
    path = OUTPUT / f"{case['mode']}_idm_revision_20hz.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[50] * (len(frames) - 1) + [1000], loop=0,
                   disposal=2, optimize=True)
    indices = [max(0, end - 16), max(0, end - 8), end]
    timeline = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for row, index in enumerate(indices):
        timeline.paste(frames[index], (0, row * frames[0].height))
    timeline_path = OUTPUT / f"{case['mode']}_idm_revision_20hz_timeline.png"
    timeline.save(timeline_path)
    return {**case, "gif": path.name, "timeline_png": timeline_path.name,
            "target_collision_time_s": target.physics_frames[end].time,
            "target_min_clearance_m": float(target_result.min_distance),
            "source_min_clearance_m": float(source_result.min_distance),
            "source_completed_safely": True, "physics_frame_interval_s": .05,
            "ego_control_interval_s": .05,
            "note": "seeded IDM parameter mutant, not a deployed software release"}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    if len(cases) != 5:
        raise RuntimeError(f"expected five collision modes, found {tuple(cases)}")
    items = [_render(case) for case in cases.values()]
    (OUTPUT / "idm_revision_manifest.json").write_text(
        json.dumps({"method": METHOD, "target": TARGET, "budget": 50,
                    "selection": "first charged ego collision per mode in seed/trace order",
                    "cases": items}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} 20 Hz IDM comparisons to {OUTPUT}")


if __name__ == "__main__":
    main()
