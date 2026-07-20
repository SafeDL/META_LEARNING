"""Show a SAC rollout as SUT-following and top-down views side by side."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sac_scenario_mining.src.env import Stage1AdversarialMergeEnv
from sac_scenario_mining.src.utils import load_config

DEFAULT_MODEL = PROJECT_ROOT / "results/sac_scenario_mining/merge_sac_seed2/best_model.zip"
DEFAULT_CONFIG = PROJECT_ROOT / "sac_scenario_mining/configs/merge_sac.yaml"


def _label_panel(frame_bgr: np.ndarray, title: str, subtitle: str, color: tuple[int, int, int]) -> np.ndarray:
    """Add a compact title strip while preserving equal panel dimensions."""
    import cv2

    panel = frame_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 54), (25, 25, 25), thickness=-1)
    cv2.putText(panel, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    return panel


def _dual_view_frame(chase_rgb: np.ndarray, topdown_rgb: np.ndarray, case_id: str,
                     step_info: dict, action: np.ndarray) -> np.ndarray:
    """Compose the left SUT chase panel and the right global top-down panel."""
    import cv2

    height = min(chase_rgb.shape[0], topdown_rgb.shape[0])
    chase = cv2.resize(chase_rgb, (round(chase_rgb.shape[1] * height / chase_rgb.shape[0]), height))
    topdown = cv2.resize(topdown_rgb, (round(topdown_rgb.shape[1] * height / topdown_rgb.shape[0]), height))
    left = _label_panel(
        cv2.cvtColor(chase, cv2.COLOR_RGB2BGR),
        "SUT · IDM  |  blue",
        f"{case_id}  |  min TTC: {step_info['min_ttc']:.2f} s  |  distance: {step_info['distance']:.1f} m",
        (255, 170, 40),
    )
    right = _label_panel(
        cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR),
        "Global top-down  |  ADV · SAC = red",
        f"SAC action: [{action[0]:+.2f}, {action[1]:+.2f}]",
        (70, 70, 255),
    )
    separator = np.full((height, 4, 3), 245, dtype=np.uint8)
    return np.concatenate((left, separator, right), axis=1)


def _case_ids(env: Stage1AdversarialMergeEnv, case_id: str | None,
              num_cases: int, selection_seed: int) -> list[str]:
    """Choose distinct cases from the fixed held-out table, never regenerate it."""
    if case_id is not None:
        return [case_id]
    table = env.case_table()
    if num_cases < 1 or num_cases > len(table):
        raise ValueError(f"num_cases must be in [1, {len(table)}]")
    indices = np.random.default_rng(selection_seed).choice(len(table), size=num_cases, replace=False)
    return [str(table[int(index)]["case_id"]) for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sampled deterministic SAC on-ramp-merge rollouts.")
    parser.add_argument("--policy-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--case-id", help="Visualize exactly one named held-out case; overrides sampling.")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of distinct held-out cases to show.")
    parser.add_argument(
        "--selection-seed", type=int, default=0,
        help="Seed for sampling cases from the fixed held-out table (default: 0).",
    )
    parser.add_argument("--frame-delay-ms", type=int, default=30, help="Display delay per frame in milliseconds.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    if args.frame_delay_ms < 1:
        parser.error("--frame-delay-ms must be at least 1")
    from stable_baselines3 import SAC
    import cv2

    config = load_config(args.config)
    config["environment"]["dual_view"] = True
    env = Stage1AdversarialMergeEnv(config, split="test", seed=0)
    model = SAC.load(Path(args.policy_path), device="cpu")
    try:
        case_ids = _case_ids(env, args.case_id, args.num_cases, args.selection_seed)
        print({"selection_seed": args.selection_seed, "case_ids": case_ids})
        records = []
        stop_requested = False
        for position, selected_case_id in enumerate(case_ids, start=1):
            observation, info = env.reset(options={"case_id": selected_case_id})
            env.set_camera_target("sut")
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, step_info = env.step(action)
                topdown = env.render(view="topdown", screen_size=(640, 360), window=False, center_on_map=True)
                case_label = f"[{position}/{len(case_ids)}] {info['case_id']}"
                frame = _dual_view_frame(env.camera_frame(), topdown, case_label, step_info, action)
                cv2.imshow("SAC on-ramp merge: SUT view + global top-down", frame)
                if cv2.waitKey(args.frame_delay_ms) & 0xFF in (27, ord("q")):
                    stop_requested = True
                    break
                done = terminated or truncated
            record = {"case_id": info["case_id"], **env.episode_record()}
            records.append(record)
            print(record)
            if stop_requested:
                break
        print({"displayed_cases": len(records), "stopped_early": stop_requested})
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
