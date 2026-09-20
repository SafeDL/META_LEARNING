"""Render representative, actually discovered B=50 scenarios as GIF replays."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from diva_highway_env.data.generate_anchor_bank import FUNCTIONAL_MODES
from diva_highway_env.data.response_bank import ResponseBank
from diva_highway_env.envs.cutin_env import CutInEnv, CutInScenario
from method_chains.diva_detour_fusion.experiment import CONFIG, OUTPUT_DIR
from diva_highway_env.sut.idm_profiles import get_profile


RESULT_DIR = OUTPUT_DIR
METHOD = "DIVA DETOUR-Guided Diagnosis"


@dataclass(frozen=True)
class ReplayCase:
    """One target-hidden B=50 discovery selected for faithful visual replay."""

    mode: str
    target_sut: str
    anchor_index: int
    initial_gap_m: float
    relative_speed_mps: float
    collision: bool
    near_miss: bool
    vulnerability: float


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_replay_cases(
    bank: ResponseBank, rows: list[dict[str, str]]
) -> list[ReplayCase]:
    """Choose one highest-severity event per mode from fusion-selected queries."""
    return [
        _ranked_cases_by_mode(bank, rows)[mode][0]
        for mode in FUNCTIONAL_MODES
    ]


def _ranked_cases_by_mode(
    bank: ResponseBank, rows: list[dict[str, str]]
) -> dict[str, list[ReplayCase]]:
    """Return every B=50-selected case ordered by replay severity per mode."""
    modes = bank.modes
    if modes is None:
        raise ValueError("the multi-function response bank must include interaction modes")
    selected_rows = [row for row in rows if row["method"] == METHOD]
    if len(selected_rows) != len(bank.sut_names):
        raise ValueError("expected exactly one fusion row for every target SUT")
    candidates: list[ReplayCase] = []
    for row in selected_rows:
        sut_index = bank.index_of(row["target_sut"])
        for text_index in row["queried_indices"].split(";"):
            anchor_index = int(text_index)
            candidates.append(
                ReplayCase(
                    mode=str(modes[anchor_index]),
                    target_sut=row["target_sut"],
                    anchor_index=anchor_index,
                    initial_gap_m=float(bank.anchors[anchor_index, 0]),
                    relative_speed_mps=float(bank.anchors[anchor_index, 1]),
                    collision=bool(bank.collisions[sut_index, anchor_index]),
                    near_miss=bool(bank.near_misses[sut_index, anchor_index]),
                    vulnerability=float(bank.vulnerability[sut_index, anchor_index]),
                )
            )
    output: dict[str, list[ReplayCase]] = {}
    for mode in FUNCTIONAL_MODES:
        mode_candidates = [case for case in candidates if case.mode == mode]
        if not mode_candidates:
            raise RuntimeError(f"the B=50 fusion selected no {mode} scenario")
        output[mode] = sorted(
            mode_candidates,
            key=lambda case: (
                case.collision, case.near_miss, case.vulnerability, -case.anchor_index
            ),
            reverse=True,
        )
    return output


def _minimum_showcase_duration(mode: str) -> float:
    """Require a replay to reach the mechanism it is intended to demonstrate."""
    if mode == CutInEnv.FAST_INTRUSION:
        return CutInEnv.CUTIN_START + CutInEnv.FAST_CUTIN_DURATION
    if mode == CutInEnv.CUTIN_BRAKING:
        return (
            CutInEnv.CUTIN_START
            + CutInEnv.DEFAULT_CUTIN_DURATION
            + CutInEnv.BRAKE_DURATION
        )
    if mode == CutInEnv.LEAD_BRAKING:
        return 1.0 + CutInEnv.BRAKE_DURATION
    if mode == CutInEnv.STOP_AND_GO:
        return 1.0 + 2.0
    if mode == CutInEnv.SLOW_LEAD_FOLLOWING:
        return 3.0
    raise ValueError(f"unsupported functional mode: {mode}")


def replay_duration(case: ReplayCase, seed: int = CONFIG.seed) -> float:
    """Return the physical episode duration before rendering the selected replay."""
    env = CutInEnv(
        get_profile(case.target_sut),
        CutInScenario(case.initial_gap_m, case.relative_speed_mps, case.mode),
    )
    try:
        env.reset(seed=seed + case.anchor_index)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
        return float(env.time)
    finally:
        env.close()


def _annotate_frame(
    frame, case: ReplayCase, time_seconds: float
) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    outcome = (
        "Collision" if case.collision else "Near miss" if case.near_miss else "Completed"
    )
    draw.rectangle((0, 0, image.width, 56), fill=(20, 20, 20))
    draw.text(
        (8, 6),
        f"Highway-env | {case.mode} | {case.target_sut} | t={time_seconds:.1f} s",
        fill=(255, 255, 255),
    )
    draw.text(
        (8, 29),
        (
            f"gap={case.initial_gap_m:.1f} m, "
            f"dv={case.relative_speed_mps:.1f} m/s | {outcome}"
        ),
        fill=(255, 220, 130) if case.collision else (210, 230, 255),
    )
    return image


def render_case(case: ReplayCase, output: Path, seed: int = CONFIG.seed) -> dict[str, object]:
    """Replay exactly the selected bank scenario through highway-env RGB rendering."""
    scenario = CutInScenario(
        case.initial_gap_m, case.relative_speed_mps, case.mode
    )
    env = CutInEnv(get_profile(case.target_sut), scenario, render_mode="rgb_array")
    frames: list[Image.Image] = []
    physical_duration = 0.0
    try:
        env.reset(seed=seed + case.anchor_index)
        terminated = truncated = False
        while not (terminated or truncated):
            frames.append(_annotate_frame(env.render(), case, env.time))
            _, _, terminated, truncated, _ = env.step(1)
        frames.append(_annotate_frame(env.render(), case, env.time))
        physical_duration = float(env.time)
        outcome = env.episode_result()
        if outcome.collision != case.collision or outcome.near_miss != case.near_miss:
            raise RuntimeError("replay outcome does not match the stored response bank")
    finally:
        env.close()
    if not frames:
        raise RuntimeError("highway-env replay produced no frames")
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
        "duration_seconds": len(frames) / 5,
        "physical_episode_duration_seconds": physical_duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    result_dir = args.result_dir
    replay_dir = result_dir / "gifs"
    bank = ResponseBank.load(result_dir / "response_bank.npz")
    ranked_cases = _ranked_cases_by_mode(
        bank, _read_rows(result_dir / "loso_mining.csv")
    )
    cases = []
    for mode in FUNCTIONAL_MODES:
        minimum = _minimum_showcase_duration(mode)
        eligible = [
            case for case in ranked_cases[mode]
            if replay_duration(case) >= minimum
        ]
        if not eligible:
            raise RuntimeError(f"no B=50-selected {mode} replay reaches its mechanism")
        cases.append(eligible[0])
    replays = [
        render_case(case, replay_dir / f"{index:02d}_{case.mode}.gif")
        for index, case in enumerate(cases, start=1)
    ]
    (replay_dir / "gif_manifest.json").write_text(
        json.dumps(
            {
                "method_that_selected_cases": METHOD,
                "selection_rule": (
                    "highest severity among cases selected within B=50 that survive "
                    "long enough to expose the declared functional mechanism, one per mode"
                ),
                "replays": replays,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(replays)} highway-env GIF replays to {replay_dir}")


if __name__ == "__main__":
    main()
