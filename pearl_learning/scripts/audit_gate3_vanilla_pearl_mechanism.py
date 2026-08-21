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
from typing import Any, Mapping

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
GATE3_DECISION_SHOT = 4
STAGE_A_MIN_CORRECT_WRONG_LATENT_L2 = 0.5
# Stage A now requires the correct/wrong shift to be a substantial fraction of
# the total posterior displacement away from the prior, not merely a large
# absolute L2.  Round 2 showed L2 ~ 3.6-7.7 while the two posteriors stayed
# nearly collinear (cos ~ 0.9997, R_sep 0.068-0.141): the shift was a shared
# "has-context" direction, not a task-discriminative one.
STAGE_A_MIN_PRIOR_RELATIVE_SEPARATION_RATIO = 0.25
STAGE_B_MIN_CORRECT_WRONG_ACTION_L2 = 0.1
# Gate v4 makes the Critic action preference an explicit causal link.  The
# audit grid has 41 points on [-1, 1], so 0.10 corresponds to two grid steps
# and cannot be satisfied by a single-bin boundary fluctuation.
STAGE_B_Q_MIN_CRITIC_ARGMAX_ACTION_DISTANCE = 0.1
STAGE_B_PI_MIN_CORRECT_WRONG_ACTION_L2 = 0.1


def _shot_values(suite: dict) -> dict[str, dict[str, float]]:
    tasks = suite["tasks"]
    shot = str(GATE3_DECISION_SHOT)
    if not tasks or any(shot not in task["shots"] for task in tasks.values()):
        raise ValueError(f"Gate 3 decision shot K={GATE3_DECISION_SHOT} is missing from the audit suite")
    values: dict[str, dict[str, float]] = {}
    for task_id, task in tasks.items():
        row = task["shots"][shot]
        values[task_id] = {
            "correct_wrong_latent_l2": float(row["latent_l2"]["correct_wrong_l2"]),
            "correct_wrong_action_l2": float(row["action_adaptation"]["correct_wrong"]["action_l2"]["mean"]),
            "correct_wrong_vcsr_advantage": (
                float(row["trajectory_summaries"]["correct"]["valid_critical_strict_rate"])
                - float(row["trajectory_summaries"]["wrong"]["valid_critical_strict_rate"])
            ),
            "correct_vcsr": float(row["trajectory_summaries"]["correct"]["valid_critical_strict_rate"]),
            "correct_wrong_return_advantage": float(row["paired_gain_means"]["correct_minus_wrong_return"]),
            # The Stage-B latent-geometry separation ratio is D_cw / (0.5 *
            # (||mu_c|| + ||mu_w||) + eps); under the unit-normal prior
            # (mu_prior = 0) this equals the prior-relative R_sep used by the
            # v3 Stage-A criterion.  The cosine is recorded but never gates.
            "latent_separation_ratio": float(row["stage_b_diagnostics"]["latent_geometry"]["separation_ratio"]),
            "latent_cosine": float(row["stage_b_diagnostics"]["latent_geometry"]["correct_wrong_cosine"]),
            "critic_argmax_action_distance": float(
                row["stage_b_diagnostics"]["critic_q_grid"]["argmax_action_distance_mean"]
            ),
            "actor_regret_mean_correct": float(
                row["stage_b_diagnostics"]["critic_q_grid"]["actor_regret_mean"]["correct"]
            ),
        }
    return values


def policy_separation_ratio(suite: dict, oracle: Mapping[str, Any]) -> dict[str, float]:
    """PEARL correct/wrong policy distance as a fraction of the single-task SAC policy distance."""
    shot = str(GATE3_DECISION_SHOT)
    ratio: dict[str, float] = {}
    for task_id, row in oracle["tasks"].items():
        denominator = float(row["single_task_action_l2_mean"]) + 1e-8
        numerator = float(
            suite["tasks"][task_id]["shots"][shot]["action_adaptation"]["correct_wrong"]["action_l2"]["mean"]
        )
        ratio[task_id] = numerator / denominator
    return ratio


def gate3_causal_chain_verdict(suite: dict, oracle: Mapping[str, Any] | None = None) -> dict:
    """Judge the support -> posterior -> policy -> outcome chain sequentially.

    A -> B -> C gating at the fixed decision shot: Stage B is only judged once
    Stage A passes; Stage C is only judged once Stage B passes and the query
    cases are proven reachable by the oracle audit.  K=1/2 remain reported as
    an adaptation curve and never decide the gate, so there is no implicit
    best-K selection.

    Stage A ("context -> task-discriminative posterior") requires, for both
    tasks at the decision shot:

    * D_cw  = ||mu_c - mu_w||_2 >= STAGE_A_MIN_CORRECT_WRONG_LATENT_L2, and
    * R_sep = D_cw / (0.5 * (||mu_c - mu_p||_2 + ||mu_w - mu_p||_2) + eps)
            >= STAGE_A_MIN_PRIOR_RELATIVE_SEPARATION_RATIO, with mu_p the
            prior mean (0 under the unit-normal prior).

    The absolute L2 alone was disproved by Round 2 (L2 3.6-7.7 but cos ~
    0.9997): the posterior moved along a shared common-context direction, so
    the correct/wrong shift was not task-discriminative.  The cosine stays a
    reported diagnostic and never gates, because tasks may legitimately be
    encoded by latent amplitude.
    """
    values = _shot_values(suite)
    stage_a_ok = all(
        row["correct_wrong_latent_l2"] >= STAGE_A_MIN_CORRECT_WRONG_LATENT_L2
        and row["latent_separation_ratio"] >= STAGE_A_MIN_PRIOR_RELATIVE_SEPARATION_RATIO
        for row in values.values()
    )
    stage_a = {
        "status": "pass" if stage_a_ok else "fail",
        "decision_shot": GATE3_DECISION_SHOT,
        "criterion": (
            f"task-discriminative posterior for both tasks at K={GATE3_DECISION_SHOT}: "
            f"correct/wrong posterior mean L2 >= {STAGE_A_MIN_CORRECT_WRONG_LATENT_L2} AND "
            f"prior-relative separation ratio R_sep >= {STAGE_A_MIN_PRIOR_RELATIVE_SEPARATION_RATIO}"
        ),
        "minimum_latent_l2": STAGE_A_MIN_CORRECT_WRONG_LATENT_L2,
        "minimum_prior_relative_separation_ratio": STAGE_A_MIN_PRIOR_RELATIVE_SEPARATION_RATIO,
        "separation_ratio_definition": (
            "D_cw / (0.5 * (||mu_c - mu_prior||_2 + ||mu_w - mu_prior||_2) + eps) with mu_prior = 0 "
            "under the unit-normal prior; the Stage-B latent-geometry separation ratio equals R_sep "
            "exactly for Vanilla PEARL and the definition extends to a non-zero Structure-Aware prior"
        ),
        "cosine_diagnostic_only": "correct/wrong cosine similarity is reported but never gates Stage A",
    }
    if stage_a_ok:
        stage_b_ok = all(
            row["correct_wrong_action_l2"] >= STAGE_B_MIN_CORRECT_WRONG_ACTION_L2 for row in values.values()
        )
        stage_b = {
            "status": "pass" if stage_b_ok else "fail",
            "decision_shot": GATE3_DECISION_SHOT,
            "criterion": f"correct/wrong deterministic action L2 >= {STAGE_B_MIN_CORRECT_WRONG_ACTION_L2} for both tasks at K={GATE3_DECISION_SHOT}",
            "minimum_action_l2": STAGE_B_MIN_CORRECT_WRONG_ACTION_L2,
        }
    else:
        stage_b_ok = False
        stage_b = {"status": "blocked_by_stage_a", "decision_shot": GATE3_DECISION_SHOT}
    if stage_b_ok:
        if oracle is not None and str(oracle.get("feasibility", {}).get("status")) != "pass":
            stage_c = {
                "status": "blocked_by_query_feasibility",
                "decision_shot": GATE3_DECISION_SHOT,
                "criterion": "query cases must first pass the single-task SAC oracle feasibility audit",
            }
        else:
            stage_c_ok = any(
                row["correct_wrong_vcsr_advantage"] > 0.0 and row["correct_vcsr"] > 0.0
                for row in values.values()
            )
            stage_c = {
                "status": "pass" if stage_c_ok else "fail",
                "decision_shot": GATE3_DECISION_SHOT,
                "criterion": f"correct context strictly improves VCSR over wrong on at least one task at K={GATE3_DECISION_SHOT}",
            }
    else:
        stage_c = {
            "status": "blocked_by_stage_a" if not stage_a_ok else "blocked_by_stage_b",
            "decision_shot": GATE3_DECISION_SHOT,
        }
    all_ok = stage_a_ok and stage_b_ok and stage_c["status"] == "pass"
    passed = [
        name
        for name, ok in (
            ("stage_a", stage_a_ok),
            ("stage_b", stage_b_ok),
            ("stage_c", stage_c["status"] == "pass"),
        )
        if ok
    ]
    gate = {
        # v3 upgrades Stage A from the absolute L2 floor to the conjoint
        # task-discriminative criterion (L2 + prior-relative R_sep).  Existing
        # v2 gate files on disk are preserved; v3 verdicts are written to a
        # new file so the historical judgement is never overwritten.
        "schema": "gate3_vanilla_pearl_causal_chain_gate_v3",
        "gate_name": "gate3_vanilla_pearl_causal_chain",
        "status": "pass" if all_ok else "fail",
        "next_allowed_stage": "gate4_structure_aware_transfer" if all_ok else None,
        "decision_shot": GATE3_DECISION_SHOT,
        "stages": {
            "stage_a_context_to_posterior": stage_a,
            "stage_b_posterior_to_action": stage_b,
            "stage_c_action_to_outcome": stage_c,
        },
        "stage_values": values,
        "passed_stages": passed,
    }
    if oracle is not None:
        gate["policy_separation_ratio"] = policy_separation_ratio(suite, oracle)
        gate["oracle_feasibility_status"] = str(oracle.get("feasibility", {}).get("status"))
    return gate


def gate3_causal_chain_verdict_v5(suite: dict, oracle: Mapping[str, Any] | None = None) -> dict:
    """Judge Context -> posterior -> Actor -> outcome; report Critic only.

    SAC does not require its stochastic Actor to equal a hard Q-grid argmax.
    The v4 Critic metric therefore remains visible, but cannot block B_pi or C.
    """
    causal = gate3_causal_chain_verdict(suite, oracle)
    values = causal["stage_values"]
    critic_threshold_ok = all(
        row["critic_argmax_action_distance"]
        >= STAGE_B_Q_MIN_CRITIC_ARGMAX_ACTION_DISTANCE
        for row in values.values()
    )
    critic_diagnostic = {
        "status": "diagnostic_only",
        "observed_threshold_status": "pass" if critic_threshold_ok else "fail",
        "decision_shot": GATE3_DECISION_SHOT,
        "criterion": (
            "correct/wrong Critic argmax-action distance "
            f">= {STAGE_B_Q_MIN_CRITIC_ARGMAX_ACTION_DISTANCE} for both tasks at "
            f"K={GATE3_DECISION_SHOT}"
        ),
        "minimum_critic_argmax_action_distance": STAGE_B_Q_MIN_CRITIC_ARGMAX_ACTION_DISTANCE,
        "metric_path": "stage_b_diagnostics.critic_q_grid.argmax_action_distance_mean",
        "action_grid_contract": "fixed 41-point [-1, 1] grid and fixed audit state bank",
        "blocks_actor_or_outcome": False,
    }
    stages = causal["stages"]
    passed = []
    if stages["stage_a_context_to_posterior"]["status"] == "pass":
        passed.append("stage_a")
    if stages["stage_b_posterior_to_action"]["status"] == "pass":
        passed.append("stage_b_pi")
    if stages["stage_c_action_to_outcome"]["status"] == "pass":
        passed.append("stage_c")
    return {
        **causal,
        "schema": "gate3_vanilla_pearl_causal_chain_gate_v5",
        "next_allowed_stage": (
            "archive_mechanism_stress_test_and_run_physical_task_heterogeneity_gate"
            if causal["status"] == "pass" else None
        ),
        "stages": {
            "stage_a_context_to_posterior": stages["stage_a_context_to_posterior"],
            "stage_b_q_diagnostic_only": critic_diagnostic,
            "stage_b_pi_posterior_to_actor_action": stages["stage_b_posterior_to_action"],
            "stage_c_actor_to_outcome": stages["stage_c_action_to_outcome"],
        },
        "passed_stages": passed,
        "hard_gate_chain": ["stage_a", "stage_b_pi", "stage_c"],
    }


def gate3_causal_chain_verdict_v4(suite: dict, oracle: Mapping[str, Any] | None = None) -> dict:
    """Judge the explicit Context -> Critic -> Actor Gate 3 causal chain.

    This is intentionally separate from :func:`gate3_causal_chain_verdict`:
    that function is the frozen v3 contract used by historical verdict files
    and ``recompute_gate3_verdict_v3.py``.  v4 inserts a Critic-specific
    action-preference gate between posterior separation and actor adaptation.
    """
    values = _shot_values(suite)
    stage_a_ok = all(
        row["correct_wrong_latent_l2"] >= STAGE_A_MIN_CORRECT_WRONG_LATENT_L2
        and row["latent_separation_ratio"] >= STAGE_A_MIN_PRIOR_RELATIVE_SEPARATION_RATIO
        for row in values.values()
    )
    stage_a = {
        "status": "pass" if stage_a_ok else "fail",
        "decision_shot": GATE3_DECISION_SHOT,
        "criterion": (
            f"task-discriminative posterior for both tasks at K={GATE3_DECISION_SHOT}: "
            f"correct/wrong posterior mean L2 >= {STAGE_A_MIN_CORRECT_WRONG_LATENT_L2} AND "
            f"prior-relative separation ratio R_sep >= {STAGE_A_MIN_PRIOR_RELATIVE_SEPARATION_RATIO}"
        ),
        "minimum_latent_l2": STAGE_A_MIN_CORRECT_WRONG_LATENT_L2,
        "minimum_prior_relative_separation_ratio": STAGE_A_MIN_PRIOR_RELATIVE_SEPARATION_RATIO,
        "separation_ratio_definition": (
            "D_cw / (0.5 * (||mu_c - mu_prior||_2 + ||mu_w - mu_prior||_2) + eps) with mu_prior = 0 "
            "under the unit-normal prior; the Stage-B latent-geometry separation ratio equals R_sep "
            "exactly for Vanilla PEARL and the definition extends to a non-zero Structure-Aware prior"
        ),
        "cosine_diagnostic_only": "correct/wrong cosine similarity is reported but never gates Stage A",
    }
    if stage_a_ok:
        stage_b_q_ok = all(
            row["critic_argmax_action_distance"] >= STAGE_B_Q_MIN_CRITIC_ARGMAX_ACTION_DISTANCE
            for row in values.values()
        )
        stage_b_q = {
            "status": "pass" if stage_b_q_ok else "fail",
            "decision_shot": GATE3_DECISION_SHOT,
            "criterion": (
                "correct/wrong Critic argmax-action distance "
                f">= {STAGE_B_Q_MIN_CRITIC_ARGMAX_ACTION_DISTANCE} for both tasks at "
                f"K={GATE3_DECISION_SHOT}"
            ),
            "minimum_critic_argmax_action_distance": STAGE_B_Q_MIN_CRITIC_ARGMAX_ACTION_DISTANCE,
            "metric_path": "stage_b_diagnostics.critic_q_grid.argmax_action_distance_mean",
            "action_grid_contract": "fixed 41-point [-1, 1] grid and fixed audit state bank",
        }
    else:
        stage_b_q_ok = False
        stage_b_q = {"status": "blocked_by_stage_a", "decision_shot": GATE3_DECISION_SHOT}

    if stage_b_q_ok:
        stage_b_pi_ok = all(
            row["correct_wrong_action_l2"] >= STAGE_B_PI_MIN_CORRECT_WRONG_ACTION_L2
            for row in values.values()
        )
        stage_b_pi = {
            "status": "pass" if stage_b_pi_ok else "fail",
            "decision_shot": GATE3_DECISION_SHOT,
            "criterion": (
                "correct/wrong deterministic action L2 "
                f">= {STAGE_B_PI_MIN_CORRECT_WRONG_ACTION_L2} for both tasks at "
                f"K={GATE3_DECISION_SHOT}"
            ),
            "minimum_action_l2": STAGE_B_PI_MIN_CORRECT_WRONG_ACTION_L2,
        }
    else:
        stage_b_pi_ok = False
        stage_b_pi = {
            "status": "blocked_by_stage_a" if not stage_a_ok else "blocked_by_stage_b_q",
            "decision_shot": GATE3_DECISION_SHOT,
        }

    if stage_b_pi_ok:
        if oracle is not None and str(oracle.get("feasibility", {}).get("status")) != "pass":
            stage_c = {
                "status": "blocked_by_query_feasibility",
                "decision_shot": GATE3_DECISION_SHOT,
                "criterion": "query cases must first pass the single-task SAC oracle feasibility audit",
            }
        else:
            stage_c_ok = any(
                row["correct_wrong_vcsr_advantage"] > 0.0 and row["correct_vcsr"] > 0.0
                for row in values.values()
            )
            stage_c = {
                "status": "pass" if stage_c_ok else "fail",
                "decision_shot": GATE3_DECISION_SHOT,
                "criterion": (
                    "correct context strictly improves VCSR over wrong on at least one task at "
                    f"K={GATE3_DECISION_SHOT}"
                ),
                "paired_return_diagnostic_only": True,
            }
    else:
        stage_c = {
            "status": (
                "blocked_by_stage_a" if not stage_a_ok
                else "blocked_by_stage_b_q" if not stage_b_q_ok
                else "blocked_by_stage_b_pi"
            ),
            "decision_shot": GATE3_DECISION_SHOT,
        }

    all_ok = stage_a_ok and stage_b_q_ok and stage_b_pi_ok and stage_c["status"] == "pass"
    passed = [
        name
        for name, ok in (
            ("stage_a", stage_a_ok),
            ("stage_b_q", stage_b_q_ok),
            ("stage_b_pi", stage_b_pi_ok),
            ("stage_c", stage_c["status"] == "pass"),
        )
        if ok
    ]
    gate = {
        "schema": "gate3_vanilla_pearl_causal_chain_gate_v4",
        "gate_name": "gate3_vanilla_pearl_causal_chain",
        "status": "pass" if all_ok else "fail",
        "next_allowed_stage": "gate4_structure_aware_transfer" if all_ok else None,
        "decision_shot": GATE3_DECISION_SHOT,
        "stages": {
            "stage_a_context_to_posterior": stage_a,
            "stage_b_q_posterior_to_critic_action_preference": stage_b_q,
            "stage_b_pi_critic_to_actor_action": stage_b_pi,
            "stage_c_actor_to_outcome": stage_c,
        },
        "stage_values": values,
        "passed_stages": passed,
    }
    if oracle is not None:
        gate["policy_separation_ratio"] = policy_separation_ratio(suite, oracle)
        gate["oracle_feasibility_status"] = str(oracle.get("feasibility", {}).get("status"))
    return gate


def verify_casebook_split_provenance(
    checkpoint: Mapping[str, Any],
    books: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    screening_manifest: Mapping[str, Any] | None,
    oracle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Enforce split-level casebook provenance against the checkpoint.

    The checkpoint records per-split casebook hashes.  train_pool and
    validation_support are training-time support evidence and must be
    bit-identical between checkpoint and audit; validation_query may differ
    only when the revision is a legitimate oracle-screened query selection,
    which must carry the screening manifest and a passing oracle feasibility
    audit.  Checkpoints saved before split-level hashing fall back to the
    whole-book comparison and are marked as such.
    """
    audit_split_hashes = {
        task_id: {split: content_hash(rows) for split, rows in book.items()}
        for task_id, book in books.items()
    }
    provenance: dict[str, Any] = {"casebook_split_hashes": audit_split_hashes}
    saved = dict(checkpoint.get("casebook_split_hashes", {}))
    if not saved:
        provenance["casebook_split_provenance"] = "checkpoint predates split-level casebook hashing"
        return provenance
    provenance["checkpoint_casebook_split_hashes"] = saved
    if set(saved) != set(audit_split_hashes):
        raise ValueError("checkpoint casebook_split_hashes do not cover the audit tasks")
    query_changed = False
    for task_id, current in audit_split_hashes.items():
        checkpoint_splits = saved[task_id]
        for split in ("train_pool", "validation_support"):
            if checkpoint_splits.get(split) != current.get(split):
                raise ValueError(
                    f"{task_id} {split} casebook changed since the checkpoint was trained; "
                    "training support evidence is frozen provenance"
                )
        if checkpoint_splits.get("validation_query") != current.get("validation_query"):
            query_changed = True
    if not query_changed:
        return provenance
    if screening_manifest is None:
        raise ValueError(
            "validation_query casebook differs from the checkpoint; a revised query group "
            "requires --query-screening-manifest provenance"
        )
    if str(screening_manifest.get("schema")) != "gate3_query_candidate_screening_v1":
        raise ValueError("query screening manifest has an unexpected schema")
    if str(screening_manifest.get("provenance", {}).get("uses_test_or_ood")).lower() != "false":
        raise ValueError("query screening manifest must not use test or OOD data")
    feasibility = None if oracle is None else str(oracle.get("feasibility", {}).get("status"))
    if feasibility != "pass":
        raise ValueError(
            "a revised validation_query group requires oracle feasibility = pass; "
            "rerun the single-task SAC query-oracle audit first"
        )
    provenance["query_screening_manifest_hash"] = content_hash(screening_manifest)
    provenance["query_screening_manifest_schema"] = str(screening_manifest.get("schema"))
    provenance["query_screening_uses_test_or_ood"] = str(
        screening_manifest.get("provenance", {}).get("uses_test_or_ood")
    )
    provenance["query_revision_oracle_feasibility"] = feasibility
    return provenance


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
    parser.add_argument(
        "--oracle-audit",
        help="single-task SAC query-oracle audit JSON used for R_policy and Stage-C feasibility gating",
    )
    parser.add_argument(
        "--query-screening-manifest",
        help="query candidate screening manifest required when the audit validation_query "
        "differs from the checkpoint casebook",
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
    # Provenance-only config edits (training_seeds, cases.per_task, output
    # roots) legitimately change the resolved-config hash without touching any
    # training-relevant setting.  Verify against the checkpoint's own saved
    # resolved configuration on the training-relevant sections instead.
    checkpoint_dir = Path(args.checkpoint).parent
    resolved_path = checkpoint_dir / "config_resolved.json"
    if not resolved_path.exists():
        raise ValueError("checkpoint directory lacks config_resolved.json for configuration verification")
    resolved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    if checkpoint["config_hash"] != content_hash(resolved_config):
        raise ValueError("checkpoint config hash does not match its saved resolved configuration")
    _PROVENANCE_SECTIONS = {"project", "experiment", "cases", "method_flow_pilot", "mechanism"}
    training_cfg = {key: value for key, value in cfg.items() if key not in _PROVENANCE_SECTIONS}
    training_resolved = {key: value for key, value in resolved_config.items() if key not in _PROVENANCE_SECTIONS}
    if content_hash(training_cfg) != content_hash(training_resolved):
        raise ValueError("audit configuration differs from the checkpoint in training-relevant sections")
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
    oracle = None
    if args.oracle_audit:
        oracle = json.loads(Path(args.oracle_audit).read_text(encoding="utf-8"))
        if set(oracle.get("tasks", {})) != {task.task_id for task in tasks}:
            raise ValueError("oracle audit tasks do not match the causal audit tasks")
    screening_manifest = None
    if args.query_screening_manifest:
        screening_manifest = json.loads(Path(args.query_screening_manifest).read_text(encoding="utf-8"))
    provenance = {
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "casebook_hashes": {task_id: content_hash(book) for task_id, book in books.items()},
        "config_hash": content_hash(cfg),
        "critical_threshold_hash": manifest["calibration_hash"],
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_training_seed": checkpoint["training_seed"],
        "not_a_benchmark_or_holdout_result": True,
        # The support context is now cut by the same canonical selector the
        # training sampler and few-shot evaluation use; historical suites on
        # disk predate this and used linspace blocks.
        "context_sampling_scheme": str(cfg["pearl"].get("context_transition_sampling", "random")),
    }
    checkpoint_casebooks = dict(checkpoint.get("casebook_hashes", {}))
    if checkpoint_casebooks and any(
        checkpoint_casebooks.get(task_id) != content_hash(book) for task_id, book in books.items()
    ):
        # Expected when the query group was revised by the oracle screening;
        # the split-level check below proves that only validation_query moved.
        provenance["casebook_differs_from_checkpoint"] = {
            task_id: {
                "checkpoint_casebook_hash": checkpoint_casebooks.get(task_id),
                "audit_casebook_hash": content_hash(book),
            }
            for task_id, book in books.items()
            if checkpoint_casebooks.get(task_id) != content_hash(book)
        }
    provenance.update(
        verify_casebook_split_provenance(checkpoint, books, screening_manifest, oracle)
    )
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
        # The metric the environment actually evaluates under, and the v2
        # calibration schema that supplied its thresholds.  Recording both
        # keeps the provenance honest for v3 mechanism runs.
        "evaluation_metric_schema": str(cfg["critical_metric"]["schema"]),
        "threshold_source_metric_schema": manifest["critical_metric_schema"],
        "calibration_hash": manifest["calibration_hash"],
        "shots": shots,
        "provenance": provenance,
        "tasks": results,
    }
    write_json(root / "gate3_causal_audit.json", suite)
    diagnostics = {
        "schema": "gate3_stage_b_diagnostics_v1",
        "decision_shot": GATE3_DECISION_SHOT,
        "provenance": provenance,
        "tasks": {
            task_id: {
                shot: task["shots"][shot]["stage_b_diagnostics"]
                for shot in sorted(task["shots"], key=int)
            }
            for task_id, task in results.items()
        },
    }
    write_json(root / "gate3_stage_b_diagnostics.json", diagnostics)
    gate = gate3_causal_chain_verdict_v4(suite, oracle)
    gate["inputs"] = provenance
    # v4 is additive: the historical v2/v3 files remain immutable evidence.
    write_json(root / "gate3_causal_chain_gate_v4.json", gate)
    gate_v5 = gate3_causal_chain_verdict_v5(suite, oracle)
    gate_v5["inputs"] = provenance
    write_json(root / "gate3_causal_chain_gate_v5.json", gate_v5)
    print(f"Gate 3 causal audit v5: {gate_v5['status']} "
          f"(passed {gate_v5['passed_stages'] or 'none'}) -> {root / 'gate3_causal_chain_gate_v5.json'}")


if __name__ == "__main__":
    main()
