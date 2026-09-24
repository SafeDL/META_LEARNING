"""Audit one archived v4 near-miss label per available mode at physics rate.

The cases are selected only from scenarios actually queried by
MeanResidual-Risk within the B=50 validation campaigns.  This is a replay
utility, not an additional selection study: it verifies each stored event
label and writes 20 Hz replay GIFs with true vehicle-polygon clearance.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

from highway_env_benchmark.envs.cutin_env import CutInScenario
from highway_env_benchmark.envs.external_cutin import ExternalCutInEnv
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


ROOT = Path("results/method_chains/core_mine/studies/sparse_continuous_suts")
METHOD = "MeanResidual-Risk"
BUDGET = 50
MODES = (
    "fast_intrusion",
    "cutin_braking",
    "lead_braking",
    "stop_and_go",
    "slow_lead_following",
)
DISPLAY_NAMES = {
    "fast_intrusion": "Fast intrusion",
    "cutin_braking": "Cut-in braking",
    "lead_braking": "Lead braking",
    "stop_and_go": "Stop and go",
    "slow_lead_following": "Slow lead following",
}


@dataclass(frozen=True)
class ReplayCase:
    seed: int
    target_sut: str
    index: int
    mode: str
    initial_gap: float
    relative_speed: float
    timing: float
    intensity: float
    collision: bool
    near_miss: bool


@dataclass(frozen=True)
class PhysicsFrame:
    image: np.ndarray
    time: float
    polygon_clearance: float
    legacy_clearance: float


class DiagnosticReplayEnv(ExternalCutInEnv):
    """Capture the original 20 Hz dynamics without changing policy calls."""

    def __init__(self, *args, **kwargs) -> None:
        self.physics_frames: list[PhysicsFrame] = []
        self.capture_enabled = False
        super().__init__(*args, **kwargs)

    def capture(self) -> None:
        ego, lead = self.vehicle, self.scheduled_vehicle
        polygon_clearance = Polygon(ego.polygon()).distance(Polygon(lead.polygon()))
        center_distance = float(np.linalg.norm(lead.position - ego.position))
        legacy_clearance = max(0.0, center_distance - (lead.LENGTH + ego.LENGTH) / 2)
        self.physics_frames.append(
            PhysicsFrame(np.asarray(self.render()).copy(), float(self.time), float(polygon_clearance), legacy_clearance)
        )

    def _record_lead_trace(self) -> None:
        super()._record_lead_trace()
        if self.capture_enabled:
            self.capture()


def _read_records() -> list[dict[str, str]]:
    with (ROOT / "validate" / "records.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_cases() -> dict[str, ReplayCase]:
    candidates: dict[str, list[ReplayCase]] = {mode: [] for mode in MODES}
    for row in _read_records():
        if row["method"] != METHOD or int(row["budget"]) != BUDGET or int(row["repeat"]) != 0:
            continue
        seed = int(row["seed"])
        bank_path = ROOT / "banks" / "sparse_continuous" / f"sparse_sut_bank_{seed}.npz"
        with np.load(bank_path, allow_pickle=False) as bank:
            target = row["heterogeneity"]
            target_index = list(bank["sut_names"].astype(str)).index(target)
            events = bank["ego_collision"][target_index] | bank["near_miss"][target_index]
            for index in map(int, row["queried_indices"].split(";")):
                mode = str(bank["modes"][index])
                if mode not in candidates or not events[index]:
                    continue
                candidates[mode].append(
                    ReplayCase(
                        seed=seed,
                        target_sut=target,
                        index=index,
                        mode=mode,
                        initial_gap=float(bank["anchors"][index, 0]),
                        relative_speed=float(bank["anchors"][index, 1]),
                        timing=float(bank["controls"][index, 0]),
                        intensity=float(bank["controls"][index, 1]),
                        collision=bool(bank["ego_collision"][target_index, index]),
                        near_miss=bool(bank["near_miss"][target_index, index]),
                    )
                )
    # The largest initial gap makes each visual example a nontrivial discovered
    # boundary case rather than the most extreme point in a function.
    return {mode: max(values, key=lambda item: (item.initial_gap, item.seed, item.index)) for mode, values in candidates.items() if values}


def _annotate(frame: PhysicsFrame, case: ReplayCase, closest_time: float) -> Image.Image:
    raw = Image.fromarray(frame.image).convert("RGB")
    image = Image.new("RGB", (raw.width, raw.height + 104), (18, 25, 36))
    image.paste(raw, (0, 104))
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), f"{DISPLAY_NAMES[case.mode]} | archived near-miss flag | t={frame.time:.2f}s", fill=(255, 255, 255))
    draw.text((8, 29), f"vehicle-outline clearance: {frame.polygon_clearance:.2f} m | legacy metric: {frame.legacy_clearance:.2f} m", fill=(255, 225, 145))
    draw.text((8, 52), f"closest outline approach at t={closest_time:.2f}s | no collision in replay", fill=(195, 225, 255))
    controller_rate = "IDM/MOBIL internal 20 Hz" if case.target_sut == "idm_mobil" else "external policy 5 Hz"
    draw.text((8, 75), f"target: {case.target_sut} | B={BUDGET} | physics 20 Hz | {controller_rate}", fill=(195, 225, 255))
    return image


def _render_case(case: ReplayCase, output: Path) -> tuple[list[Image.Image], dict[str, object]]:
    policy = policy_factory(case.target_sut, ASSETS)
    scenario = CutInScenario(case.initial_gap, case.relative_speed, case.mode, case.timing, case.intensity)
    env = DiagnosticReplayEnv(scenario, ego_kind=policy.ego_kind, render_mode="rgb_array")
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
    legacy_near_miss = not result.ego_collision and (
        result.min_ttc < 1.5 or min(item.legacy_clearance for item in env.physics_frames) < 2.0
    )
    if result.ego_collision != case.collision or legacy_near_miss != case.near_miss:
        raise RuntimeError(f"archived label mismatch for {case}")
    closest = min(env.physics_frames, key=lambda item: item.polygon_clearance)
    frames = [_annotate(item, case, closest.time) for item in env.physics_frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=50, loop=0, disposal=2, optimize=True)
    return frames, {**asdict(case), "gif": output.name, "frames": len(frames), "frame_interval_seconds": 0.05,
                    "min_polygon_clearance": closest.polygon_clearance, "min_polygon_clearance_time": closest.time,
                    "min_legacy_clearance": min(item.legacy_clearance for item in env.physics_frames),
                    "legacy_near_miss_reproduced": legacy_near_miss,
                    "replay_result": asdict(result)}


def _panel(frame: Image.Image, mode: str, size: tuple[int, int]) -> Image.Image:
    resized = frame.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0], size[1] + 26), (18, 25, 36))
    canvas.paste(resized, (0, 26))
    ImageDraw.Draw(canvas).text((8, 6), DISPLAY_NAMES[mode], fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


def _write_overview(replays: dict[str, list[Image.Image]], output: Path) -> None:
    modes = tuple(replays)
    size = (384, 230)
    count = max(len(frames) for frames in replays.values())
    frames: list[Image.Image] = []
    for frame_index in range(count):
        canvas = Image.new("RGB", (size[0] * 2, (size[1] + 26) * 2), (8, 12, 20))
        for index, mode in enumerate(modes):
            source = replays[mode][min(frame_index, len(replays[mode]) - 1)]
            canvas.paste(_panel(source, mode, size), ((index % 2) * size[0], (index // 2) * (size[1] + 26)))
        frames.append(canvas)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=50, loop=0, disposal=2, optimize=True)
    frames[0].save(output.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "gifs")
    args = parser.parse_args()
    cases = _selected_cases()
    missing = [mode for mode in MODES if mode not in cases]
    rendered: dict[str, list[Image.Image]] = {}
    manifest = []
    for mode, case in cases.items():
        frames, item = _render_case(case, args.output_dir / f"{mode}.gif")
        rendered[mode] = frames
        manifest.append(item)
    if rendered:
        _write_overview(rendered, args.output_dir / "critical_cases_overview.gif")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "method": METHOD,
                "budget": BUDGET,
                "selection_rule": "one highest-initial-gap archived near-miss label actually queried in each mode",
                "warning": "Archived near-miss labels use a center-distance-minus-length proxy that can report zero clearance in adjacent lanes; these GIFs audit that label rather than assert danger.",
                "missing_modes": missing,
                "replays": manifest,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rendered)} label-audit GIFs to {args.output_dir}; no archived near-miss case selected for: {', '.join(missing) or 'none'}")


if __name__ == "__main__":
    main()
