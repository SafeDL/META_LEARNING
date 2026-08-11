from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.evaluator import compact_fewshot_result, evaluate_fewshot
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload
from pearl_learning.src.task_representation import configure_disentangled_representation
from pearl_learning.src.validation_freeze import verify_validation_freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument(
        "--split",
        choices=["meta_test_template", "meta_test_logical", "meta_validation"],
        required=True,
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--query-cases", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--no-topology", action="store_true")
    parser.add_argument("--disentangled-representation", action="store_true")
    parser.add_argument("--representation-latent-dims", nargs=3, type=int, default=[2, 2, 1])
    parser.add_argument("--geometry-aux-weight", type=float, default=0.1)
    parser.add_argument("--interaction-aux-weight", type=float, default=0.1)
    parser.add_argument("--rule-aux-weight", type=float, default=0.1)
    parser.add_argument(
        "--support-selection",
        choices=["fixed", "random", "initial_condition_diversity", "posterior_action_disagreement"],
        default="fixed",
    )
    parser.add_argument(
        "--adaptation-mode",
        choices=["posterior_sampled", "posterior_deterministic", "no_context"],
        default="posterior_sampled",
    )
    parser.add_argument(
        "--query-latent-mode",
        choices=["adaptive", "frozen_prior"],
        default="adaptive",
    )
    parser.add_argument(
        "--query-route-mode",
        choices=["adaptive", "frozen_prior", "uniform"],
        default="adaptive",
    )
    parser.add_argument("--knockout-expert", type=int)
    parser.add_argument("--mechanism-audit", action="store_true")
    parser.add_argument("--validation-freeze-manifest", help="require this validation freeze before a holdout split")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.output_root:
        cfg["project"] = {**cfg["project"], "output_root": args.output_root}
    if args.no_topology:
        cfg["ablation"] = {**cfg.get("ablation", {}), "no_topology": True}
    cfg = configure_disentangled_representation(
        cfg,
        enabled=args.disentangled_representation,
        latent_dims=args.representation_latent_dims,
        geometry_weight=args.geometry_aux_weight,
        interaction_weight=args.interaction_aux_weight,
        rule_weight=args.rule_aux_weight,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["experiment"].get("device") != "cpu" else "cpu")
    agent = PEARLAgent(int(cfg["environment"]["observation_dim"]), int(cfg["environment"]["action_dim"]), cfg, device)
    checkpoint = load_checkpoint(args.checkpoint, agent, device)
    # Support policies sample posterior latents.  Restore the checkpoint RNG
    # before evaluation so a frozen checkpoint/taskbook/casebook triple has a
    # reproducible few-shot trajectory and cannot vary across CLI invocations.
    rng_state = checkpoint["rng_state"]
    torch.set_rng_state(torch.as_tensor(rng_state["torch"], dtype=torch.uint8, device="cpu").clone())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state, dtype=torch.uint8, device="cpu").clone() for state in rng_state["cuda"]]
        )
    if bool(checkpoint.get("no_topology_ablation", False)) != bool(args.no_topology):
        raise ValueError("checkpoint topology-ablation mode does not match evaluation mode")
    if bool(checkpoint.get("no_context_training", False)) != bool(
        cfg.get("ablation", {}).get("no_context_training", False)
    ):
        raise ValueError("checkpoint no-context-training mode does not match evaluation configuration")
    taskbook = load_taskbook(args.taskbook)
    expected_hash = content_hash(taskbook_payload(taskbook))
    if checkpoint["taskbook_hash"] != expected_hash:
        raise ValueError("checkpoint was trained with a different frozen taskbook")
    tasks = taskbook[args.split][:1] if args.smoke else taskbook[args.split]
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in tasks}
    manifest_path = Path(args.checkpoint).with_suffix(".manifest.json")
    checkpoint_hash = json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_hash"]
    if args.validation_freeze_manifest and args.split != "meta_validation":
        verify_validation_freeze(
            json.loads(Path(args.validation_freeze_manifest).read_text(encoding="utf-8")),
            taskbook_hash=expected_hash,
            checkpoint_hash=checkpoint_hash,
        )
    root = Path(cfg["project"]["output_root"]) / ("smoke" if args.smoke else "evaluations") / args.split / args.run_name
    provenance = {
        "git_commit": checkpoint["git_commit"],
        "config_hash": checkpoint["config_hash"],
        "taskbook_hash": expected_hash,
        "casebook_hashes": checkpoint["casebook_hashes"],
        "checkpoint_hash": checkpoint_hash,
        "training_seed": checkpoint["training_seed"],
        "no_context_training": bool(checkpoint.get("no_context_training", False)),
        "evaluation_rng": "checkpoint_rng_state",
    }
    result = evaluate_fewshot(
        agent,
        cfg,
        tasks,
        casebooks,
        args.split,
        args.query_cases,
        provenance,
        args.support_selection,
        args.adaptation_mode,
        args.query_latent_mode,
        args.query_route_mode,
        args.knockout_expert,
        args.mechanism_audit,
    )
    write_json(root / "metrics.json", compact_fewshot_result(result))
    print(f"no-gradient few-shot result: {root / 'metrics.json'}")


if __name__ == "__main__":
    main()
