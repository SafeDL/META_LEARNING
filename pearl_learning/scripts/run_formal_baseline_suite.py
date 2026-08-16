"""Run the formal SAC baseline suite reproducibly, resuming completed artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from pearl_learning.src.baselines import PRETRAIN_BASELINE_NAMES
from pearl_learning.src.io import content_hash, write_json
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


SUITE_SCHEMA = "logical_merge_formal_baseline_suite_v1"


def _is_complete(root: Path, baseline: str, taskbook_hash: str, environment_steps: int) -> bool:
    manifest = root / baseline / "baseline_manifest.json"
    if not manifest.exists():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        payload.get("baseline") == baseline
        and payload.get("status") == "completed"
        and payload.get("taskbook_hash") == taskbook_hash
        and int(payload.get("environment_steps", -1)) == environment_steps
    )


def baseline_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    common = [
        "--config", args.config, "--taskbook", args.taskbook,
        "--casebook-root", args.casebook_root, "--output", args.output,
        "--seed", str(args.seed), "--env-steps", str(args.environment_steps), "--resume", "--formal-run",
    ]
    checkpoint_interval_steps = int(getattr(args, "checkpoint_interval_steps", 0))
    if checkpoint_interval_steps:
        common.extend(["--checkpoint-interval-steps", str(checkpoint_interval_steps)])
    root = Path(args.output)
    return [
        ("per_task_sac", ["--baseline", "per_task_sac", *common]),
        ("cross_task_policy_matrix", [
            "--baseline", "cross_task_policy_matrix", *common,
            "--per-task-policy-root", str(root / "per_task_sac" / "policies"),
        ]),
        ("topology_conditioned_pooled_sac", [
            "--baseline", "topology_conditioned_pooled_sac", *common,
            "--pooled-steps-per-task", str(args.environment_steps),
        ]),
        ("scratch_sac", ["--baseline", "scratch_sac", *common]),
        ("pooled_finetune_sac", [
            "--baseline", "pooled_finetune_sac", *common,
            "--pooled-pretrain-model", str(root / "topology_conditioned_pooled_sac" / "model.zip"),
        ]),
        ("oracle_task_conditioned_sac", [
            "--baseline", "oracle_task_conditioned_sac", *common,
            "--pooled-steps-per-task", str(args.environment_steps),
        ]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--environment-steps", type=int, required=True)
    parser.add_argument(
        "--checkpoint-interval-steps", type=int, default=0,
        help="save resumable per-task SAC checkpoints every N steps",
    )
    parser.add_argument("--progress-output")
    parser.add_argument("--formal-run", action="store_true", help="explicitly authorize this long-running comparison suite")
    args = parser.parse_args()
    if not args.formal_run:
        parser.error("formal baseline suites are disabled by default; pass --formal-run only after approving a separate experiment and resource plan")
    if args.environment_steps <= 0:
        raise ValueError("--environment-steps must be positive")
    if args.checkpoint_interval_steps < 0:
        raise ValueError("--checkpoint-interval-steps must be non-negative")
    taskbook = load_taskbook(args.taskbook)
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    root = Path(args.output)
    progress_path = Path(args.progress_output) if args.progress_output else root / "baseline_suite_progress.json"
    commands = baseline_commands(args)
    if {name for name, _ in commands} != set(PRETRAIN_BASELINE_NAMES):
        raise RuntimeError("formal suite does not cover the required pre-training baselines")
    completed: list[str] = []
    for baseline, command in commands:
        if _is_complete(root, baseline, taskbook_hash, args.environment_steps):
            completed.append(baseline)
        else:
            subprocess.run([sys.executable, "-m", "pearl_learning.scripts.run_baselines", *command], check=True)
            if not _is_complete(root, baseline, taskbook_hash, args.environment_steps):
                raise RuntimeError(f"baseline {baseline} did not write a matching completed manifest")
            completed.append(baseline)
        if baseline == "per_task_sac":
            subprocess.run([
                sys.executable, "-m", "pearl_learning.scripts.select_per_task_sac_checkpoints",
                "--config", args.config,
                "--taskbook", args.taskbook,
                "--casebook-root", args.casebook_root,
                "--baseline-root", args.output,
            ], check=True)
        if baseline == "topology_conditioned_pooled_sac":
            subprocess.run([
                sys.executable, "-m", "pearl_learning.scripts.select_pooled_sac_checkpoint",
                "--config", args.config,
                "--taskbook", args.taskbook,
                "--casebook-root", args.casebook_root,
                "--baseline-root", args.output,
            ], check=True)
        write_json(progress_path, {
            "schema": SUITE_SCHEMA,
            "taskbook_hash": taskbook_hash,
            "environment_steps": int(args.environment_steps),
            "completed_baselines": completed,
            "required_baselines": list(PRETRAIN_BASELINE_NAMES),
        })
    print(f"formal baseline suite completed: {progress_path}")


if __name__ == "__main__":
    main()
