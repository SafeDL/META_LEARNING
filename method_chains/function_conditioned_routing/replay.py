"""Render one dangerous scenario discovered in each functional mode."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from highway_env_benchmark.data.generate_anchor_bank import FUNCTIONAL_MODES
from highway_env_benchmark.data.response_bank import ResponseBank
from highway_env_benchmark.envs.cutin_env import CutInEnv, CutInScenario
from sut_algorithms.highway_env.idm_profiles import get_profile

from .benchmark import functional_releases
from .config import RoutingExperimentConfig


RESULT_DIR = Path(__file__).resolve().parent / "results" / "functional_shift_benchmark"
METHOD = "Function-Conditioned Mining"


@dataclass(frozen=True)
class ReplayCase:
    """One dangerous test actually selected within the target budget."""

    mode: str
    target_sut: str
    module_profile: str
    anchor_index: int
    initial_gap_m: float
    relative_speed_mps: float
    timing: float
    intensity: float
    collision: bool
    near_miss: bool
    min_ttc_seconds: float
    min_distance_m: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_replay_cases(
    bank: ResponseBank,
    rows: list[dict[str, str]],
) -> tuple[ReplayCase, ...]:
    """Select one clearly observable discovered event per functional mode."""
    if bank.modes is None or bank.scenario_controls is None:
        raise ValueError("functional replay requires modes and scenario controls")
    releases = {release.name: release for release in functional_releases()}
    selected_rows = [row for row in rows if row["method"] == METHOD]
    if {row["target_sut"] for row in selected_rows} != set(bank.sut_names):
        raise ValueError("expected one proposed-method row per target release")
    candidates: list[ReplayCase] = []
    for row in selected_rows:
        target = row["target_sut"]
        target_index = bank.index_of(target)
        for text_index in row["queried_indices"].split(";"):
            index = int(text_index)
            collision = bool(bank.collisions[target_index, index])
            near_miss = bool(bank.near_misses[target_index, index])
            if not collision and not near_miss:
                continue
            mode = str(bank.modes[index])
            candidates.append(
                ReplayCase(
                    mode=mode,
                    target_sut=target,
                    module_profile=releases[target].modules[mode],
                    anchor_index=index,
                    initial_gap_m=float(bank.anchors[index, 0]),
                    relative_speed_mps=float(bank.anchors[index, 1]),
                    timing=float(bank.scenario_controls[index, 0]),
                    intensity=float(bank.scenario_controls[index, 1]),
                    collision=collision,
                    near_miss=near_miss,
                    min_ttc_seconds=float(bank.min_ttc[target_index, index]),
                    min_distance_m=float(bank.min_distance[target_index, index]),
                )
            )
    cases = []
    for mode in FUNCTIONAL_MODES:
        matches = [case for case in candidates if case.mode == mode]
        if not matches:
            raise RuntimeError(f"no dangerous {mode} scenario was found within B=50")
        collisions = [case for case in matches if case.collision]
        observable = collisions or matches
        cases.append(
            max(
                observable,
                key=lambda case: (
                    case.initial_gap_m,
                    case.min_ttc_seconds,
                    -case.min_distance_m,
                    -case.anchor_index,
                ),
            )
        )
    return tuple(cases)


def _scenario(case: ReplayCase) -> CutInScenario:
    return CutInScenario(
        case.initial_gap_m,
        case.relative_speed_mps,
        case.mode,
        case.timing,
        case.intensity,
    )


def _annotate(frame: np.ndarray, case: ReplayCase, time_seconds: float) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    outcome = "collision" if case.collision else "near miss"
    draw.rectangle((0, 0, image.width, 76), fill=(20, 20, 20))
    draw.text(
        (8, 6),
        f"{METHOD} | {case.mode} | t={time_seconds:.1f} s",
        fill=(255, 255, 255),
    )
    draw.text(
        (8, 29),
        (
            f"target={case.target_sut} | active module={case.module_profile} "
            f"| recorded outcome={outcome}"
        ),
        fill=(190, 225, 255),
    )
    draw.text(
        (8, 52),
        (
            f"gap={case.initial_gap_m:.1f} m | dv={case.relative_speed_mps:.1f} m/s "
            f"| timing={case.timing:.2f} | intensity={case.intensity:.2f}"
        ),
        fill=(255, 220, 130),
    )
    return image


def render_case(case: ReplayCase, output: Path) -> dict[str, object]:
    """Replay one stored scenario and verify its physical event labels."""
    seed = RoutingExperimentConfig().seed + case.anchor_index
    env = CutInEnv(get_profile(case.module_profile), _scenario(case), render_mode="rgb_array")
    frames: list[Image.Image] = []
    try:
        env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            frames.append(_annotate(env.render(), case, env.time))
            _, _, terminated, truncated, _ = env.step(1)
        frames.append(_annotate(env.render(), case, env.time))
        outcome = env.episode_result()
        if outcome.collision != case.collision or outcome.near_miss != case.near_miss:
            raise RuntimeError("replay outcome does not match the response bank")
        physical_duration = float(env.time)
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return {
        **asdict(case),
        "gif": output.name,
        "frames": len(frames),
        "physical_duration_seconds": physical_duration,
        "selection_method": METHOD,
        "within_budget": 50,
    }


def main() -> None:
    bank = ResponseBank.load(RESULT_DIR / "response_bank.npz")
    cases = select_replay_cases(bank, read_rows(RESULT_DIR / "mining_results.csv"))
    gif_dir = RESULT_DIR / "gifs"
    replays = [render_case(case, gif_dir / f"{case.mode}.gif") for case in cases]
    (gif_dir / "manifest.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "For each functional mode, choose the collision with the largest "
                    "initial gap actually queried by Function-Conditioned Mining within "
                    "B=50. Fall back to a near miss only if no collision was found."
                ),
                "replays": replays,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(replays)} GIFs to {gif_dir}")


if __name__ == "__main__":
    main()
