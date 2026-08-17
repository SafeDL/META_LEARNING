from __future__ import annotations

import argparse
import json
from pathlib import Path

from pearl_learning.src.benchmark_calibration import apply_calibration_manifest
from pearl_learning.src.casebook import CASEBOOK_SCHEMA, load_casebook
from pearl_learning.src.io import (
    assert_method_variant_contract,
    content_hash,
    file_sha256,
    git_commit_sha,
    prepare_run_manifest,
    read_config,
)
from pearl_learning.src.pearl_trainer import train
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload
from pearl_learning.src.task_representation import configure_disentangled_representation


def _pilot_tasks(cfg, taskbook, split):
    requested = cfg.get("method_flow_pilot", {}).get("task_ids", {}).get(split)
    if requested is None:
        return list(taskbook[split])
    wanted = set(map(str, requested))
    selected = [task for task in taskbook[split] if task.geometry_id in wanted]
    if {task.geometry_id for task in selected} != wanted:
        raise ValueError(f"method-flow pilot {split} task ids do not match frozen taskbook")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--critical-thresholds")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--run-name", required=True)
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--smoke", action="store_true")
    run_mode.add_argument(
        "--mechanism-gate",
        action="store_true",
        help="run the non-paper 20k mechanism gate with full training settings",
    )
    run_mode.add_argument(
        "--formal-run",
        action="store_true",
        help="explicitly authorize a non-smoke training run after a separate resource plan has been approved",
    )
    parser.add_argument("--formal-validation")
    parser.add_argument("--output-root")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--checkpoint-interval-steps", type=int)
    parser.add_argument("--no-topology", action="store_true")
    parser.add_argument("--topology-dropout", type=float, default=0.0)
    parser.add_argument("--topology-dropout-warmup-steps", type=int, default=0)
    parser.add_argument("--disentangled-representation", action="store_true")
    parser.add_argument("--representation-latent-dims", nargs=3, type=int, default=[2, 2, 1])
    parser.add_argument("--geometry-aux-weight", type=float, default=0.1)
    parser.add_argument("--interaction-aux-weight", type=float, default=0.1)
    parser.add_argument("--rule-aux-weight", type=float, default=0.1)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if str(cfg.get("critical_metric", {}).get("schema")) == "spatiotemporal_near_miss_v2":
        if not args.critical_thresholds:
            raise ValueError("v2 training requires --critical-thresholds from validation calibration")
        manifest = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
        cfg = apply_calibration_manifest(cfg, manifest)
    run_kind = "smoke" if args.smoke else ("mechanism_gate" if args.mechanism_gate else "formal")
    cfg["experiment"] = {**cfg["experiment"], "run_kind": run_kind}
    if args.output_root:
        cfg["project"] = {**cfg["project"], "output_root": args.output_root}
    if args.no_topology:
        cfg["ablation"] = {**cfg.get("ablation", {}), "no_topology": True}
    if not 0.0 <= args.topology_dropout <= 1.0:
        raise ValueError("--topology-dropout must lie in [0, 1]")
    if args.topology_dropout_warmup_steps < 0:
        raise ValueError("--topology-dropout-warmup-steps must be non-negative")
    if args.topology_dropout:
        cfg["regularization"] = {
            **cfg.get("regularization", {}),
            "topology_dropout_probability": args.topology_dropout,
            "topology_dropout_warmup_steps": args.topology_dropout_warmup_steps,
        }
    cfg = configure_disentangled_representation(
        cfg,
        enabled=args.disentangled_representation,
        latent_dims=args.representation_latent_dims,
        geometry_weight=args.geometry_aux_weight,
        interaction_weight=args.interaction_aux_weight,
        rule_weight=args.rule_aux_weight,
    )
    method_variant = assert_method_variant_contract(cfg, args.run_name, run_kind)
    taskbook = load_taskbook(args.taskbook)
    tasks = _pilot_tasks(cfg, taskbook, "meta_train")
    validation = _pilot_tasks(cfg, taskbook, "meta_validation")
    if args.smoke and "method_flow_pilot" not in cfg:
        tasks = tasks[: min(2, len(tasks))]
        validation = validation[:1]
    selected = tasks + validation
    required_casebook_schema = (
        CASEBOOK_SCHEMA
        if str(cfg.get("critical_metric", {}).get("schema")) == "spatiotemporal_near_miss_v2"
        else None
    )
    casebooks = {
        task.task_id: load_casebook(task, args.casebook_root, required_schema=required_casebook_schema)
        for task in selected
    }
    max_steps = args.max_env_steps or int(cfg["meta_training"]["total_environment_steps"])
    if args.mechanism_gate and (args.max_env_steps is None or not 0 < max_steps <= 20_000):
        raise ValueError("--mechanism-gate requires --max-env-steps in [1, 20000]")
    run_directory = "smoke" if args.smoke else ("mechanism_gate" if args.mechanism_gate else "models")
    output_dir = Path(cfg["project"]["output_root"]) / run_directory / args.run_name
    calibration_hash = cfg.get("critical_metric", {}).get("calibration_hash")
    run_manifest = {
        "schema": "pearl_run_manifest_v1",
        "requested_config_path": str(Path(args.config).resolve()),
        "source_config_sha256": file_sha256(args.config),
        "resolved_config_sha256": content_hash(cfg),
        "git_commit_sha": git_commit_sha(),
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": {task_id: content_hash(book) for task_id, book in casebooks.items()},
        "critical_threshold_hash": calibration_hash,
        "run_name": args.run_name,
        "run_kind": run_kind,
        "method_variant": method_variant,
        "training_seed": int(args.seed),
    }
    prepare_run_manifest(output_dir, run_manifest, resume=bool(args.resume_checkpoint))
    result = train(
        cfg,
        tasks,
        validation,
        casebooks,
        content_hash(taskbook_payload(taskbook)),
        max_steps,
        args.seed,
        args.run_name,
        args.smoke,
        args.formal_validation,
        args.resume_checkpoint,
        args.checkpoint_interval_steps,
        mechanism_gate=args.mechanism_gate,
    )
    print(f"PEARL training completed: {result}")


if __name__ == "__main__":
    main()
