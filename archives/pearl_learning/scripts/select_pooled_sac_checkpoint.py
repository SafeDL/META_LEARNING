"""Select a pooled SAC checkpoint on meta-validation, then report frozen test cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from stable_baselines3 import SAC

from archives.pearl_learning.scripts.run_baselines import _evaluate_sac
from archives.pearl_learning.src.casebook import load_casebook
from archives.pearl_learning.src.io import content_hash, read_config, write_json
from archives.pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def aggregate_key(task_metrics: Mapping[str, Mapping[str, Any]], steps: int) -> tuple[float, float, float, int]:
    summaries = [row["summary"] for row in task_metrics.values()]
    return (
        sum(float(row["valid_critical_strict_rate"]) for row in summaries) / len(summaries),
        -sum(float(row["invalid_rate"]) for row in summaries) / len(summaries),
        sum(float(row["mean_episode_return"]) for row in summaries) / len(summaries),
        -int(steps),
    )


def checkpoint_steps(path: Path) -> int:
    return int(path.stem.removeprefix("topology_conditioned_pooled_sac_").removesuffix("_steps"))


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
    root = Path(args.baseline_root) / "topology_conditioned_pooled_sac"
    manifest_path = root / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("taskbook_hash") != taskbook_hash:
        raise SystemExit("pooled SAC training is incomplete or belongs to another taskbook")
    if manifest.get("config_hash") != content_hash(cfg):
        raise SystemExit("pooled SAC checkpoints belong to another resolved configuration")

    all_tasks = [task for tasks in taskbook.values() for task in tasks]
    books = {task.task_id: load_casebook(task, args.casebook_root) for task in all_tasks}
    candidates = sorted((root / "checkpoints").glob("topology_conditioned_pooled_sac_*_steps.zip"), key=checkpoint_steps)
    if not candidates:
        raise SystemExit("no selectable pooled SAC checkpoints")
    validation: list[dict[str, Any]] = []
    for candidate in candidates:
        model = SAC.load(candidate, device="auto")
        metrics = {
            task.task_id: _evaluate_sac(model, task, cfg, books[task.task_id]["validation_query"])
            for task in taskbook["meta_validation"]
        }
        steps = checkpoint_steps(candidate)
        key = aggregate_key(metrics, steps)
        validation.append({
            "steps": steps,
            "checkpoint": str(candidate),
            "mean_valid_critical_strict_rate": key[0],
            "mean_invalid_rate": -key[1],
            "mean_episode_return": key[2],
            "tasks": metrics,
        })
    chosen = max(validation, key=lambda row: (
        row["mean_valid_critical_strict_rate"], -row["mean_invalid_rate"],
        row["mean_episode_return"], -row["steps"],
    ))
    model = SAC.load(chosen["checkpoint"], device="auto")
    model_path = root / "model.zip"
    model.save(model_path)
    report_tasks = list(taskbook["meta_train"]) + list(taskbook["meta_test_template"]) + list(taskbook["meta_test_logical"])
    report = {
        task.task_id: _evaluate_sac(model, task, cfg, books[task.task_id]["test_query"])
        for task in report_tasks
    }
    selection_path = root / "checkpoint_selection.json"
    payload = {
        "schema": "pooled_sac_checkpoint_selection",
        "taskbook_hash": taskbook_hash,
        "config_hash": content_hash(cfg),
        "selector_implementation_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_split": "meta_validation/validation_query",
        "report_split": "meta_train+meta_test/test_query",
        "primary_metric": "mean_valid_critical_strict_rate",
        "tie_breakers": ["mean_invalid_rate", "mean_episode_return", "lower_environment_steps"],
        "selected_steps": int(chosen["steps"]),
        "selected_checkpoint": str(chosen["checkpoint"]),
        "candidates": validation,
    }
    metrics_path = root / "pooled_metrics.json"
    write_json(selection_path, payload)
    write_json(metrics_path, {"tasks": report, "checkpoint_selection": str(selection_path)})
    partial_path = root / "pooled_partial_metrics.json"
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        partial["tasks"] = report
        partial["checkpoint_selection"] = str(selection_path)
        write_json(partial_path, partial)
    manifest["artifacts"]["model"] = str(model_path)
    manifest["artifacts"]["metrics"] = str(metrics_path)
    manifest["artifacts"]["checkpoint_selection"] = str(selection_path)
    manifest["selected_environment_steps"] = int(chosen["steps"])
    manifest["selection_protocol"] = {
        "selection_split": "meta_validation/validation_query",
        "report_split": "meta_train+meta_test/test_query",
        "primary_metric": "mean_valid_critical_strict_rate",
        "tie_breakers": ["mean_invalid_rate", "mean_episode_return", "lower_environment_steps"],
    }
    write_json(manifest_path, manifest)
    print(f"selected pooled SAC checkpoint at {chosen['steps']} environment steps")


if __name__ == "__main__":
    main()
