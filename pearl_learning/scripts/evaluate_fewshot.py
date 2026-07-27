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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--taskbook", required=True); parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--split", choices=["meta_test_template", "meta_test_logical", "meta_validation"], required=True); parser.add_argument("--run-name", required=True); parser.add_argument("--query-cases", type=int); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--output-root"); parser.add_argument("--no-topology", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.output_root:
        cfg["project"] = {**cfg["project"], "output_root": args.output_root}
    if args.no_topology:
        cfg["ablation"] = {**cfg.get("ablation", {}), "no_topology": True}
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["experiment"].get("device") != "cpu" else "cpu")
    agent = PEARLAgent(int(cfg["environment"]["observation_dim"]), int(cfg["environment"]["action_dim"]), cfg, device)
    checkpoint = load_checkpoint(args.checkpoint, agent, device)
    # Support policies sample posterior latents.  Restore the checkpoint RNG
    # before evaluation so a frozen checkpoint/taskbook/casebook triple has a
    # reproducible few-shot trajectory and cannot vary across CLI invocations.
    rng_state = checkpoint["rng_state"]
    torch.set_rng_state(torch.as_tensor(rng_state["torch"], dtype=torch.uint8, device="cpu").clone())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([torch.as_tensor(state, dtype=torch.uint8, device="cpu").clone() for state in rng_state["cuda"]])
    if bool(checkpoint.get("no_topology_ablation", False)) != bool(args.no_topology):
        raise ValueError("checkpoint topology-ablation mode does not match evaluation mode")
    taskbook = load_taskbook(args.taskbook); expected_hash = content_hash(taskbook_payload(taskbook))
    if checkpoint["taskbook_hash"] != expected_hash:
        raise ValueError("checkpoint was trained with a different frozen taskbook")
    tasks = taskbook[args.split][:1] if args.smoke else taskbook[args.split]
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in tasks}
    checkpoint_hash = json.loads(Path(args.checkpoint).with_suffix(".manifest.json").read_text(encoding="utf-8"))["checkpoint_hash"]
    root = Path(cfg["project"]["output_root"]) / ("smoke" if args.smoke else "evaluations") / args.split / args.run_name
    provenance = {"git_commit": checkpoint["git_commit"], "config_hash": checkpoint["config_hash"], "taskbook_hash": expected_hash, "casebook_hashes": checkpoint["casebook_hashes"], "checkpoint_hash": checkpoint_hash, "training_seed": checkpoint["training_seed"], "evaluation_rng": "checkpoint_rng_state"}
    result = evaluate_fewshot(agent, cfg, tasks, casebooks, args.split, args.query_cases, provenance)
    write_json(root / "metrics.json", compact_fewshot_result(result))
    print(f"no-gradient few-shot result: {root / 'metrics.json'}")


if __name__ == "__main__":
    main()
