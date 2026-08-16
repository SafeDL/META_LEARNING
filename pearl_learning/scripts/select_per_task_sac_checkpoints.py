"""Select per-task SAC checkpoints on validation cases, then report test cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from stable_baselines3 import SAC

from pearl_learning.scripts.run_baselines import _evaluate_sac
from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def selection_key(summary: Mapping[str, Any], steps: int) -> tuple[float, float, float, int]:
    """Primary strict success, then validity, return, and sample efficiency."""
    return (
        float(summary["valid_critical_strict_rate"]),
        -float(summary["invalid_rate"]),
        float(summary["mean_episode_return"]),
        -int(steps),
    )


def checkpoint_steps(path: Path, task_id: str) -> int:
    return int(path.stem.removeprefix(f"{task_id}_").removesuffix("_steps"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    args = parser.parse_args()

    cfg = read_config(args.config)
    taskbook = load_taskbook(args.taskbook)
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    root = Path(args.baseline_root) / "per_task_sac"
    manifest_path = root / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("taskbook_hash") != taskbook_hash:
        raise SystemExit("per-task SAC training is incomplete or belongs to another taskbook")
    if manifest.get("config_hash") != content_hash(cfg):
        raise SystemExit("per-task SAC checkpoints belong to another resolved configuration")

    selected_metrics: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for task in taskbook["meta_train"]:
        book = load_casebook(task, args.casebook_root)
        candidates = sorted(
            (root / "checkpoints" / task.task_id).glob(f"{task.task_id}_*_steps.zip"),
            key=lambda path: checkpoint_steps(path, task.task_id),
        )
        if not candidates:
            raise SystemExit(f"no selectable checkpoints for {task.task_id}")
        validation: list[dict[str, Any]] = []
        for candidate in candidates:
            steps = checkpoint_steps(candidate, task.task_id)
            model = SAC.load(candidate, device="auto")
            summary = _evaluate_sac(model, task, cfg, book["validation_query"])["summary"]
            validation.append({"steps": steps, "checkpoint": str(candidate), "summary": summary})
        chosen = max(validation, key=lambda row: selection_key(row["summary"], row["steps"]))
        model = SAC.load(chosen["checkpoint"], device="auto")
        policy_path = root / "policies" / f"{task.task_id}.zip"
        model.save(policy_path)
        selected_metrics[task.task_id] = _evaluate_sac(model, task, cfg, book["test_query"])
        selection[task.task_id] = {
            "selected_steps": int(chosen["steps"]),
            "selected_checkpoint": str(chosen["checkpoint"]),
            "selection_split": "validation_query",
            "candidates": validation,
        }

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selection_payload = {
        "schema": "per_task_sac_checkpoint_selection",
        "taskbook_hash": taskbook_hash,
        "config_hash": content_hash(cfg),
        "selector_implementation_hash": source_hash,
        "primary_metric": "valid_critical_strict_rate",
        "tie_breakers": ["invalid_rate", "mean_episode_return", "lower_environment_steps"],
        "selection_split": "validation_query",
        "report_split": "test_query",
        "tasks": selection,
    }
    selection_path = root / "checkpoint_selection.json"
    metrics_path = root / "per_task_metrics.json"
    write_json(selection_path, selection_payload)
    write_json(metrics_path, {"tasks": selected_metrics, "checkpoint_selection": str(selection_path)})

    partial_path = root / "per_task_partial_metrics.json"
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        partial["tasks"] = selected_metrics
        partial["checkpoint_selection"] = str(selection_path)
        write_json(partial_path, partial)
    manifest["artifacts"]["checkpoint_selection"] = str(selection_path)
    manifest["artifacts"]["metrics"] = str(metrics_path)
    manifest["selection_protocol"] = {
        "selection_split": "validation_query",
        "report_split": "test_query",
        "primary_metric": "valid_critical_strict_rate",
        "tie_breakers": ["invalid_rate", "mean_episode_return", "lower_environment_steps"],
    }
    manifest["selected_environment_steps"] = {
        task_id: int(row["selected_steps"]) for task_id, row in selection.items()
    }
    write_json(manifest_path, manifest)
    print(f"selected validation checkpoints for {len(selection)} per-task SAC policies")


if __name__ == "__main__":
    main()
