"""Gate 3 causal audit: prior / correct / wrong / zero latent interventions.

Runs the vanilla PEARL causal chain check on the two matched logical-order
mechanism tasks: support evidence from the correct task and from the opposite
task is mapped to posteriors under the unchanged target-task prior, then the
resulting policies are compared on identical matched query cases.  This is a
post-hoc, no-gradient audit of a mechanism-gate checkpoint, not a benchmark
evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pearl_learning.src.benchmark_calibration import resolve_calibration
from pearl_learning.src.casebook import MECHANISM_CASEBOOK_SCHEMA, load_casebook
from pearl_learning.src.causal_audit import audit_task_context_interventions
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _select_pair(taskbook, requested: list[str]):
    wanted = set(map(str, requested))
    tasks = [
        task
        for split in taskbook.values()
        for task in split
        if task.task_id in wanted or task.geometry_id in wanted
    ]
    if len(tasks) != 2 or len(tasks) != len(wanted):
        raise ValueError("Gate 3 requires exactly two unique frozen task ids/geometry ids")
    return tasks


# Deterministic mechanism-gate thresholds for the Gate 3 causal chain.  They
# are diagnostics of the chain links, not benchmark performance criteria.
STAGE_A_MIN_CORRECT_WRONG_LATENT_L2 = 0.5
STAGE_B_MIN_CORRECT_WRONG_ACTION_L2 = 0.1
STAGE_C_REQUIRES_STRICT_ADVANTAGE = True


def gate3_causal_chain_verdict(suite: dict) -> dict:
    """Judge the support -> posterior -> policy -> outcome chain per stage."""
    task_values: dict[str, dict[str, list[float]]] = {}
    for task_id, task in suite["tasks"].items():
        latent = []
        action = []
        vcsr_advantage = []
        return_advantage = []
        for shot in sorted(task["shots"], key=int):
            row = task["shots"][shot]
            latent.append(float(row["latent_l2"]["correct_wrong_l2"]))
            action.append(float(row["action_adaptation"]["correct_wrong"]["action_l2"]["mean"]))
            vcsr_advantage.append(
                float(row["trajectory_summaries"]["correct"]["valid_critical_strict_rate"])
                - float(row["trajectory_summaries"]["wrong"]["valid_critical_strict_rate"])
            )
            return_advantage.append(float(row["paired_gain_means"]["correct_minus_wrong_return"]))
        task_values[task_id] = {
            "max_correct_wrong_latent_l2": max(latent),
            "max_correct_wrong_action_l2": max(action),
            "max_correct_wrong_vcsr_advantage": max(vcsr_advantage),
            "max_correct_wrong_return_advantage": max(return_advantage),
            "correct_vcsr_by_shot": [
                float(task["shots"][shot]["trajectory_summaries"]["correct"]["valid_critical_strict_rate"])
                for shot in sorted(task["shots"], key=int)
            ],
        }
    stage_a = all(
        values["max_correct_wrong_latent_l2"] >= STAGE_A_MIN_CORRECT_WRONG_LATENT_L2
        for values in task_values.values()
    )
    stage_b = all(
        values["max_correct_wrong_action_l2"] >= STAGE_B_MIN_CORRECT_WRONG_ACTION_L2
        for values in task_values.values()
    )
    if STAGE_C_REQUIRES_STRICT_ADVANTAGE:
        stage_c = any(
            values["max_correct_wrong_vcsr_advantage"] > 0.0 and any(rate > 0.0 for rate in values["correct_vcsr_by_shot"])
            for values in task_values.values()
        )
    else:
        stage_c = any(values["max_correct_wrong_return_advantage"] > 0.0 for values in task_values.values())
    passed = [name for name, ok in (("stage_a", stage_a), ("stage_b", stage_b), ("stage_c", stage_c)) if ok]
    return {
        "schema": "gate3_vanilla_pearl_causal_chain_gate_v1",
        "gate_name": "gate3_vanilla_pearl_causal_chain",
        "status": "pass" if stage_a and stage_b and stage_c else "fail",
        "next_allowed_stage": "gate4_structure_aware_transfer" if stage_a and stage_b and stage_c else None,
        "stages": {
            "stage_a_context_to_posterior": {"status": "pass" if stage_a else "fail",
                                             "criterion": "correct/wrong posterior mean L2 >= 0.5 for both tasks",
                                             "minimum_latent_l2": STAGE_A_MIN_CORRECT_WRONG_LATENT_L2},
            "stage_b_posterior_to_action": {"status": "pass" if stage_b else "fail",
                                            "criterion": "correct/wrong deterministic action L2 >= 0.1 for both tasks",
                                            "minimum_action_l2": STAGE_B_MIN_CORRECT_WRONG_ACTION_L2},
            "stage_c_action_to_outcome": {"status": "pass" if stage_c else "fail",
                                          "criterion": "correct context strictly improves VCSR over wrong on at least one task",
                                          "requires_strict_advantage": STAGE_C_REQUIRES_STRICT_ADVANTAGE},
        },
        "stage_values": task_values,
        "passed_stages": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args()
    cfg = resolve_calibration(read_config(args.config), args.critical_thresholds)
    # train_pearl stamps the run kind into the resolved configuration before
    # hashing it into the checkpoint; replicate that so the provenance guard
    # compares identical payloads.
    cfg["experiment"] = {**cfg["experiment"], "run_kind": "mechanism_gate"}
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["experiment"].get("device") != "cpu" else "cpu"
    )
    agent = PEARLAgent(
        int(cfg["environment"]["observation_dim"]),
        int(cfg["environment"]["action_dim"]),
        cfg,
        device,
    )
    checkpoint = load_checkpoint(args.checkpoint, agent, device)
    if checkpoint["config_hash"] != content_hash(cfg):
        raise ValueError("checkpoint was trained with a different resolved configuration")
    # Support/query rollouts restart from the checkpoint RNG state so the
    # paired trajectories are reproducible across repeated audit invocations.
    rng_state = checkpoint["rng_state"]
    torch.set_rng_state(torch.as_tensor(rng_state["torch"], dtype=torch.uint8, device="cpu").clone())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state, dtype=torch.uint8, device="cpu").clone() for state in rng_state["cuda"]]
        )
    manifest = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
    checkpoint_hash = json.loads(
        Path(args.checkpoint).with_suffix(".manifest.json").read_text(encoding="utf-8")
    )["checkpoint_hash"]
    taskbook = load_taskbook(args.taskbook)
    tasks = _select_pair(taskbook, args.task_id)
    books = {
        task.task_id: load_casebook(task, args.casebook_root, required_schema=MECHANISM_CASEBOOK_SCHEMA)
        for task in tasks
    }
    shots = sorted(set(int(shot) for shot in args.shots))
    if not shots or min(shots) < 1:
        raise ValueError("Gate 3 causal shots must be positive support counts")
    provenance = {
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": {task_id: content_hash(book) for task_id, book in books.items()},
        "config_hash": content_hash(cfg),
        "critical_threshold_hash": manifest["calibration_hash"],
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_training_seed": checkpoint["training_seed"],
        "not_a_benchmark_or_holdout_result": True,
    }
    results = {}
    for target in tasks:
        wrong = next(task for task in tasks if task.task_id != target.task_id)
        results[target.task_id] = audit_task_context_interventions(
            agent, cfg, target, wrong, books[target.task_id], books[wrong.task_id],
            split="meta_validation", shots=shots,
        )
        results[target.task_id]["wrong_evidence_task_id"] = wrong.task_id
    root = Path(args.output)
    suite = {
        "schema": "gate3_vanilla_pearl_mechanism_causal_audit_suite_v1",
        "critical_metric_schema": manifest["critical_metric_schema"],
        "calibration_hash": manifest["calibration_hash"],
        "shots": shots,
        "provenance": provenance,
        "tasks": results,
    }
    write_json(root / "gate3_causal_audit.json", suite)
    gate = gate3_causal_chain_verdict(suite)
    gate["inputs"] = provenance
    write_json(root / "gate3_causal_chain_gate.json", gate)
    print(f"Gate 3 causal audit: {gate['status']} "
          f"(passed {gate['passed_stages'] or 'none'}) -> {root / 'gate3_causal_audit.json'}")


if __name__ == "__main__":
    main()
