"""Gate 3 Stage-A localization audit (zero training steps).

The v3 Stage-A rule requires a task-discriminative posterior
(D_cw >= 0.5 AND R_sep >= 0.25).  Round 1/2 fail it: D_cw 3.6-7.7 but
R_sep 0.068-0.141, so the correct/wrong shift is a shared common-context
direction.  Before touching the Encoder, this audit rules out the input-side
confounders on the frozen checkpoint:

  A. exact-PEARL-input probe      -- Gate 2 saw full trajectory summaries with
                                     raw reward; PEARL training feeds 8 random
                                     transitions/episode with r/200 and done
                                     flags, and the old causal audit used
                                     linspace blocks.  A tiny logistic probe
                                     measures whether the exact PEARL input
                                     still separates the two tasks.
  B. context sampling ablation     -- the same collected support episodes are
                                     re-cut as random / terminal-inclusive /
                                     conflict-window and re-encoded by the
                                     frozen Context Encoder (D_cw, R_sep, cos).
  C. channel ablation              -- does task information live in the reward
                                     channel or the dynamics (o, a, o')?

No parameter changes, no replay writes, no gradient steps; the only
environment rollouts are the support episodes collected once with the frozen
prior policy, exactly as training collection does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pearl_learning.src.benchmark_calibration import resolve_calibration
from pearl_learning.src.casebook import MECHANISM_CASEBOOK_SCHEMA, load_casebook
from pearl_learning.src.causal_audit import stage_a_context_posterior_diagnostics
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload

from pearl_learning.scripts.audit_gate3_vanilla_pearl_mechanism import _select_pair

# Training-irrelevant sections that legitimately differ between the audit
# configuration and the checkpoint's resolved configuration.
_PROVENANCE_SECTIONS = {"project", "experiment", "cases", "method_flow_pilot", "mechanism"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--split", default="meta_validation")
    parser.add_argument("--support-cases", type=int, default=4)
    parser.add_argument(
        "--gate2-probe-metrics",
        help="Gate 2 context_probe_metrics.json recorded for the same tasks; "
        "its held-out accuracy is embedded as the input-side reference",
    )
    args = parser.parse_args()
    cfg = resolve_calibration(read_config(args.config), args.critical_thresholds)
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
    checkpoint_dir = Path(args.checkpoint).parent
    resolved_path = checkpoint_dir / "config_resolved.json"
    if not resolved_path.exists():
        raise ValueError("checkpoint directory lacks config_resolved.json for configuration verification")
    resolved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    if checkpoint["config_hash"] != content_hash(resolved_config):
        raise ValueError("checkpoint config hash does not match its saved resolved configuration")
    training_cfg = {key: value for key, value in cfg.items() if key not in _PROVENANCE_SECTIONS}
    training_resolved = {key: value for key, value in resolved_config.items() if key not in _PROVENANCE_SECTIONS}
    if content_hash(training_cfg) != content_hash(training_resolved):
        raise ValueError("audit configuration differs from the checkpoint in training-relevant sections")
    # Restore the checkpoint RNG so support collection is reproducible across
    # repeated audit invocations.
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
    provenance = {
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": {task_id: content_hash(book) for task_id, book in books.items()},
        "config_hash": content_hash(cfg),
        "critical_threshold_hash": manifest["calibration_hash"],
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_training_seed": checkpoint["training_seed"],
        "not_a_benchmark_or_holdout_result": True,
        "frozen_checkpoint_reference": "gate3_vanilla_pearl (dense) round checkpoint; encoder weights frozen",
    }
    gate2_reference = None
    if args.gate2_probe_metrics:
        gate2 = json.loads(Path(args.gate2_probe_metrics).read_text(encoding="utf-8"))
        gate2_reference = {
            "held_out_accuracy": gate2.get("held_out_accuracy"),
            "probe": gate2.get("probe"),
            "provenance_hash": content_hash(gate2.get("provenance", {})),
            "note": "Gate 2 used full trajectory summaries with raw reward under the fixed "
            "P6 probing policy; it did not see PEARL preprocessing or the training sampler",
        }
    results = {}
    for target in tasks:
        wrong = next(task for task in tasks if task.task_id != target.task_id)
        results[target.task_id] = stage_a_context_posterior_diagnostics(
            agent, cfg, target, wrong, books[target.task_id], books[wrong.task_id],
            split=args.split, support_cases=args.support_cases,
        )
        results[target.task_id]["wrong_evidence_task_id"] = wrong.task_id
    root = Path(args.output)
    payload = {
        "schema": "gate3_stage_a_diagnostics_suite_v1",
        "split": args.split,
        "decision_context": {
            "stage_a_v3_rule": "D_cw >= 0.5 AND R_sep >= 0.25 for both tasks at K=4",
            "gate2_identifiability_reference": gate2_reference,
        },
        "provenance": provenance,
        "tasks": results,
    }
    write_json(root / "gate3_stage_a_diagnostics.json", payload)
    print(f"Stage-A diagnostics written -> {root / 'gate3_stage_a_diagnostics.json'}")


if __name__ == "__main__":
    main()
