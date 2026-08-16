"""Run support-only PEARL posterior or representation audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.evaluator import audit_task_representation, infer_support_posteriors
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.task_representation import configure_disentangled_representation
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


AUDIT_SPLITS = ("meta_validation", "meta_test_template", "meta_test_logical")


def _add_common_arguments(parser: argparse.ArgumentParser, *, default_shots: list[int] | None) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--split", choices=AUDIT_SPLITS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=default_shots)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="audit", required=True)

    posterior = commands.add_parser("posterior", help="export support-only posterior statistics")
    _add_common_arguments(posterior, default_shots=None)
    posterior.add_argument("--no-topology", action="store_true")
    posterior.add_argument("--disentangled-representation", action="store_true")
    posterior.add_argument("--representation-latent-dims", nargs=3, type=int, default=[2, 2, 1])
    posterior.add_argument("--geometry-aux-weight", type=float, default=0.1)
    posterior.add_argument("--interaction-aux-weight", type=float, default=0.1)
    posterior.add_argument("--rule-aux-weight", type=float, default=0.1)

    representation = commands.add_parser("representation", help="audit a disentangled support representation")
    _add_common_arguments(representation, default_shots=[1, 2, 5])
    representation.add_argument("--representation-latent-dims", nargs=3, type=int, default=[2, 2, 1])
    representation.add_argument("--geometry-aux-weight", type=float, default=0.1)
    representation.add_argument("--interaction-aux-weight", type=float, default=0.1)
    representation.add_argument("--rule-aux-weight", type=float, default=0.1)
    representation.add_argument("--smoke", action="store_true", help="audit the first split task only")
    return parser


def _restore_checkpoint_rng(checkpoint: dict[str, Any]) -> None:
    rng_state = checkpoint["rng_state"]
    torch.set_rng_state(torch.as_tensor(rng_state["torch"], dtype=torch.uint8, device="cpu").clone())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state, dtype=torch.uint8, device="cpu").clone() for state in rng_state["cuda"]]
        )


def _checkpoint_hash(checkpoint_path: str) -> str:
    manifest_path = Path(checkpoint_path).with_suffix(".manifest.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_hash"]


def _without_output_location(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("project", None)
    # ``train_pearl`` adds this execution label after reading the YAML. It is
    # provenance, not a model/representation hyperparameter, and the audit
    # subcommand has no smoke/formal switch from which to reconstruct it.
    experiment = dict(result.get("experiment", {}))
    experiment.pop("run_kind", None)
    result["experiment"] = experiment
    return result


def _load_inputs(args: argparse.Namespace, config: dict[str, Any]) -> tuple[PEARLAgent, dict[str, Any], list[Any], dict[str, Any], str]:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["experiment"].get("device") != "cpu" else "cpu"
    )
    agent = PEARLAgent(
        int(config["environment"]["observation_dim"]),
        int(config["environment"]["action_dim"]),
        config,
        device,
    )
    checkpoint = load_checkpoint(args.checkpoint, agent, device)
    if bool(checkpoint.get("no_topology_ablation", False)) != bool(getattr(args, "no_topology", False)):
        raise ValueError("checkpoint topology-ablation mode does not match audit mode")

    taskbook = load_taskbook(args.taskbook)
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    if checkpoint["taskbook_hash"] != taskbook_hash:
        raise ValueError("checkpoint was trained with a different frozen taskbook")

    _restore_checkpoint_rng(checkpoint)
    tasks = taskbook[args.split][:1] if getattr(args, "smoke", False) else taskbook[args.split]
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in tasks}
    return agent, checkpoint, tasks, casebooks, taskbook_hash


def _provenance(checkpoint: dict[str, Any], taskbook_hash: str, checkpoint_path: str) -> dict[str, Any]:
    return {
        "taskbook_hash": taskbook_hash,
        "checkpoint_hash": _checkpoint_hash(checkpoint_path),
        "config_hash": checkpoint["config_hash"],
        "evaluation_rng": "checkpoint_rng_state",
    }


def _posterior_config(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    if args.no_topology:
        config["ablation"] = {**config.get("ablation", {}), "no_topology": True}
    return configure_disentangled_representation(
        config,
        enabled=args.disentangled_representation,
        latent_dims=args.representation_latent_dims,
        geometry_weight=args.geometry_aux_weight,
        interaction_weight=args.interaction_aux_weight,
        rule_weight=args.rule_aux_weight,
    )


def _run_posterior(args: argparse.Namespace) -> dict[str, Any]:
    config = _posterior_config(args)
    agent, checkpoint, tasks, casebooks, taskbook_hash = _load_inputs(args, config)
    return infer_support_posteriors(
        agent,
        config,
        tasks,
        casebooks,
        args.split,
        args.shots,
        _provenance(checkpoint, taskbook_hash, args.checkpoint),
    )


def _representation_config(args: argparse.Namespace) -> dict[str, Any]:
    return configure_disentangled_representation(
        read_config(args.config),
        enabled=True,
        latent_dims=args.representation_latent_dims,
        geometry_weight=args.geometry_aux_weight,
        interaction_weight=args.interaction_aux_weight,
        rule_weight=args.rule_aux_weight,
    )


def _run_representation(args: argparse.Namespace) -> dict[str, Any]:
    config = _representation_config(args)
    agent, checkpoint, tasks, casebooks, taskbook_hash = _load_inputs(args, config)
    resolved_path = Path(args.checkpoint).parent / "config_resolved.json"
    if not resolved_path.exists():
        raise ValueError("checkpoint lacks config_resolved.json for task-representation audit")
    resolved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    if content_hash(_without_output_location(resolved_config)) != content_hash(_without_output_location(config)):
        raise ValueError("checkpoint task-representation configuration does not match this audit")
    provenance = _provenance(checkpoint, taskbook_hash, args.checkpoint)
    provenance["smoke"] = bool(args.smoke)
    return audit_task_representation(
        agent,
        config,
        tasks,
        casebooks,
        args.split,
        args.shots,
        provenance,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.audit == "posterior":
        result = _run_posterior(args)
        description = "support-only posterior diagnostic"
    else:
        result = _run_representation(args)
        description = "task-representation support-only audit"
    write_json(args.output, result)
    print(f"{description}: {args.output}")


if __name__ == "__main__":
    main()
