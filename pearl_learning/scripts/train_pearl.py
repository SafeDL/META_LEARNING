from __future__ import annotations

import argparse

from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.io import content_hash, read_config
from pearl_learning.src.pearl_trainer import train
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload
from pearl_learning.src.task_representation import configure_disentangled_representation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--formal-run",
        action="store_true",
        help="explicitly authorize a non-smoke training run after a separate resource plan has been approved",
    )
    parser.add_argument("--gate-manifest")
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
    if not args.smoke and not args.formal_run:
        parser.error(
            "non-smoke PEARL training is disabled by default; use --smoke for a main-flow check, "
            "or pass --formal-run only after approving a separate experiment and resource plan"
        )
    cfg = read_config(args.config)
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
    taskbook = load_taskbook(args.taskbook)
    tasks = taskbook["meta_train"][: min(2, len(taskbook["meta_train"]))] if args.smoke else taskbook["meta_train"]
    validation = taskbook["meta_validation"][:1] if args.smoke else taskbook["meta_validation"]
    selected = tasks + validation
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in selected}
    max_steps = args.max_env_steps or int(cfg["meta_training"]["total_environment_steps"])
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
        args.gate_manifest,
        args.resume_checkpoint,
        args.checkpoint_interval_steps,
    )
    print(f"PEARL training completed: {result}")


if __name__ == "__main__":
    main()
