"""Replay one physically verified B=50 new failure in each functional mode."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from highway_sim_env.envs.cutin_env import CutInScenario
from methods.core_mine.replay_sparse_critical_cases import DiagnosticReplayEnv
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


ROOT = Path("results/method_chains/core_mine/studies/source_safe")
MODES = ("fast_intrusion", "cutin_braking", "lead_braking", "stop_and_go", "slow_lead_following")


@dataclass(frozen=True)
class Case:
    seed: int
    target: str
    index: int
    mode: str
    gap: float
    relative_speed: float
    timing: float
    intensity: float
    collision: bool
    near_miss: bool


class ReplayEnv(DiagnosticReplayEnv):
    def __init__(self, *args, **kwargs) -> None:
        self.instant_ttc: list[float] = []
        super().__init__(*args, **kwargs)

    def capture(self) -> None:
        ego, lead = self.vehicle, self.scheduled_vehicle
        gap = float(lead.position[0] - ego.position[0])
        lateral = abs(float(lead.position[1] - ego.position[1]))
        closing = float(ego.speed - lead.speed)
        ttc = gap / closing if gap > 0 and lateral < ego.WIDTH and closing > 1e-6 else float("inf")
        self.instant_ttc.append(ttc)
        super().capture()


def _cases() -> dict[str, Case]:
    candidates: dict[str, list[Case]] = {mode: [] for mode in MODES}
    with (ROOT / "validate" / "records.csv").open(encoding="utf-8", newline="") as handle:
        records = [row for row in csv.DictReader(handle)
                   if row["method"] == "HistoryMargin-Residual" and int(row["repeat"]) == 0]
    for row in records:
        seed, target = int(row["seed"]), row["heterogeneity"]
        with np.load(ROOT / "banks" / "source_safe" / f"sparse_sut_bank_{seed}.npz", allow_pickle=False) as bank:
            sut_index = list(bank["sut_names"].astype(str)).index(target)
            sources = np.arange(len(bank["sut_names"])) != sut_index
            source_events = bank["ego_collision"][sources] | bank["near_miss"][sources]
            for index in (int(value) for value in row["queried_indices"].split(";")):
                mode = str(bank["modes"][index])
                collision = bool(bank["ego_collision"][sut_index, index])
                near_miss = bool(bank["near_miss"][sut_index, index])
                if not (collision or near_miss):
                    continue
                if source_events[:, index].any():
                    raise RuntimeError("displayed discovery was not source-safe")
                candidates[mode].append(Case(seed, target, index, mode,
                                             float(bank["anchors"][index, 0]),
                                             float(bank["anchors"][index, 1]),
                                             float(bank["controls"][index, 0]),
                                             float(bank["controls"][index, 1]),
                                             collision, near_miss))
    return {mode: max(values, key=lambda item: (item.collision, item.gap, item.seed, item.index))
            for mode, values in candidates.items() if values}


def _annotate(raw: np.ndarray, case: Case, time: float, clearance: float, ttc: float) -> Image.Image:
    source = Image.fromarray(raw).convert("RGB")
    canvas = Image.new("RGB", (source.width, source.height + 100), (18, 25, 36))
    canvas.paste(source, (0, 100))
    draw = ImageDraw.Draw(canvas)
    warning = clearance < 1.0 or ttc < 1.5
    label = "EGO COLLISION" if case.collision else "PHYSICAL NEAR MISS"
    draw.text((8, 6), f"{case.mode} | {label} | t={time:.2f}s", fill=(255, 120, 120) if warning else (255, 255, 255))
    draw.text((8, 29), f"vehicle clearance: {clearance:.2f} m | instantaneous TTC: {'inf' if not np.isfinite(ttc) else f'{ttc:.2f} s'}",
              fill=(255, 225, 145))
    draw.text((8, 52), f"target: {case.target} | all three historical controllers safe | selected within B=50",
              fill=(195, 225, 255))
    draw.text((8, 75), f"physics 20 Hz | gap={case.gap:.1f} m | relative speed={case.relative_speed:.1f} m/s",
              fill=(195, 225, 255))
    if warning:
        draw.rectangle((0, 100, canvas.width - 1, canvas.height - 1), outline=(255, 70, 70), width=3)
    return canvas


def _render(case: Case, path: Path) -> dict:
    policy = policy_factory(case.target, ASSETS)
    scenario = CutInScenario(case.gap, case.relative_speed, case.mode, case.timing, case.intensity)
    env = ReplayEnv(scenario, ego_kind=policy.ego_kind, render_mode="rgb_array")
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
    if result.ego_collision != case.collision or result.near_miss != case.near_miss:
        raise RuntimeError(f"replay mismatch: {case}")
    frames = [_annotate(frame.image, case, frame.time, frame.polygon_clearance, ttc)
              for frame, ttc in zip(env.physics_frames, env.instant_ttc, strict=True)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0,
                   disposal=2, optimize=True)
    frames[-1].save(path.with_suffix(".png"))
    return {**asdict(case), "gif": path.name, "frames": len(frames), "frame_interval_seconds": .05,
            "min_polygon_clearance": result.min_distance,
            "min_ttc": result.min_ttc if np.isfinite(result.min_ttc) else None}


def main() -> None:
    output = ROOT / "gifs"
    output.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    manifest = [_render(cases[mode], output / f"{mode}.gif") for mode in MODES if mode in cases]
    (output / "manifest.json").write_text(json.dumps({"method": "HistoryMargin-Residual", "budget": 50,
                                                        "missing_modes": [mode for mode in MODES if mode not in cases],
                                                        "replays": manifest}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} verified discovery GIFs to {output}")


if __name__ == "__main__":
    main()
