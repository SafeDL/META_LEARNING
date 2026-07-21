"""Verify that PEARL work has not changed the Stage 1 SAC environment."""
from __future__ import annotations

from pearl_learning.src.io import write_json


def main() -> None:
    from sac_scenario_mining.src.env import Stage1AdversarialMergeEnv
    from sac_scenario_mining.src.utils import load_config

    config = load_config("sac_scenario_mining/configs/merge_sac.yaml")
    env = Stage1AdversarialMergeEnv(config, split="validation", seed=0)
    try:
        obs, info = env.reset(options={"case_id": "validation_000"})
        _, _, terminated, truncated, step_info = env.step([0.0, 0.0])
        report = {
            "passed": obs.shape == (38,),
            "observation_schema": info["observation_schema"],
            "case_id": info["case_id"],
            "step_terminated": bool(terminated),
            "step_truncated": bool(truncated),
            "target_collision": bool(step_info["target_collision"]),
            "baseline_tag": "stage1-on-ramp-sac-v1",
        }
    finally:
        env.close()
    write_json("results/pearl_learning/stage1_compatibility.json", report)
    if not report["passed"]:
        raise SystemExit(f"Stage 1 compatibility verification failed: {report}")
    print(report)


if __name__ == "__main__":
    main()
