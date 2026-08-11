from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from pearl_learning.src.checkpoint import load_checkpoint, save_checkpoint
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.moe import DESCRIPTOR_FIELDS, physical_task_descriptor
from pearl_learning.src.pearl_agent import PEARLAgent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def state_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    if type(left) is type(right) and hasattr(left, "__dict__") and hasattr(right, "__dict__"):
        return state_equal(vars(left), vars(right))
    return left == right


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def checkpoint_audit(config: dict[str, Any], checkpoint_path: Path, sample_audit: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cpu")
    agent = PEARLAgent(37, 2, config, device)
    source = load_checkpoint(checkpoint_path, agent, device)
    physical = physical_task_descriptor(
        sample_audit["descriptor"]["raw_values"],
        schema=config["networks"]["moe"]["descriptor_schema"],
        normalization=config["networks"]["moe"]["descriptor_normalization"],
    )
    mu, log_var = agent.prior()
    route = agent.compute_route(physical, mu, log_var, 0)
    observation = torch.zeros((1, 37), dtype=torch.float32)
    action = agent.act(observation, mu, True, route)
    before_hash = agent.parameter_hash()
    with tempfile.TemporaryDirectory() as directory:
        roundtrip_path = Path(directory) / "moe_roundtrip.pt"
        save_checkpoint(
            roundtrip_path,
            agent,
            config,
            source["taskbook_hash"],
            int(source["step"]),
            casebook_hashes=source["casebook_hashes"],
            training_seed=int(source["training_seed"]),
            rng_state=source["rng_state"],
            trainer_state=source["trainer_state"],
        )
        restored = PEARLAgent(37, 2, config, device)
        replayed = load_checkpoint(roundtrip_path, restored, device)
        restored_route = restored.compute_route(physical, *restored.prior(), 0)
        restored_action = restored.act(observation, restored.prior()[0], True, restored_route)
        optimizer_equal = all(
            state_equal(getattr(agent, name).state_dict(), getattr(restored, name).state_dict())
            for name in ("actor_opt", "q_opt", "context_opt", "alpha_opt")
        )
        rng_equal = state_equal(source["rng_state"], replayed["rng_state"])
        trainer_state_equal = state_equal(source["trainer_state"], replayed["trainer_state"])

        dense_config = copy.deepcopy(config)
        dense_config["networks"]["actor_architecture"] = "dense"
        dense = PEARLAgent(37, 2, dense_config, device)
        dense_path = Path(directory) / "dense_roundtrip.pt"
        save_checkpoint(
            dense_path,
            dense,
            dense_config,
            "dense-taskbook",
            0,
            casebook_hashes={},
            training_seed=0,
            rng_state=source["rng_state"],
            trainer_state={"steps": 0},
        )
        dense_restored = PEARLAgent(37, 2, dense_config, device)
        load_checkpoint(dense_path, dense_restored, device)
        dense_pass = dense.parameter_hash() == dense_restored.parameter_hash()
        mismatch_rejected = False
        try:
            load_checkpoint(roundtrip_path, dense_restored, device)
        except ValueError as error:
            mismatch_rejected = "architecture" in str(error)

    checks = {
        "moe_parameter_hash_equal": before_hash == restored.parameter_hash(),
        "moe_route_equal": route == restored_route,
        "moe_deterministic_action_equal": bool(torch.equal(action, restored_action)),
        "optimizer_states_equal": optimizer_equal,
        "rng_state_equal": rng_equal,
        "trainer_state_equal": trainer_state_equal,
        "dense_parameter_hash_equal": dense_pass,
        "architecture_mismatch_rejected": mismatch_rejected,
    }
    return {
        "schema": "posterior_routed_moe_checkpoint_roundtrip_v1",
        "source_checkpoint": checkpoint_path.as_posix(),
        "source_checkpoint_hash": read_json(checkpoint_path.with_suffix(".manifest.json"))["checkpoint_hash"],
        "architecture_metadata": agent.architecture_metadata(),
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pearl_learning/configs/posterior_routed_moe.yaml")
    parser.add_argument("--output-root", default="results/pearl_learning/posterior_routed_moe")
    args = parser.parse_args()
    config = read_config(args.config)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config["routed_moe"]["training_seeds"]]
    run_roots = {seed: output / "runs" / "smoke" / f"routed_moe_seed_{seed}" for seed in seeds}

    smoke_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    for seed, root in run_roots.items():
        summary = read_json(root / "training_summary.json")
        rows = read_jsonl(root / "router_audit.jsonl")
        updates = [row for row in rows if row["phase"] == "actor_update"]
        router_rows.extend(rows)
        expert_gradients = {
            f"expert_{index}": min(
                float(row["gradient_norms"][f"expert_{index}_gradient_norm"])
                for row in updates
            )
            for index in range(int(config["networks"]["moe"]["num_experts"]))
        }
        all_weights = [weight for row in rows for weight in row["routing_weights"]]
        no_exclusive = all(0.01 < float(weight) < 0.99 for weight in all_weights)
        no_exclusive = no_exclusive and all(value > 0.0 for value in expert_gradients.values())
        smoke_rows.append({
            "schema": "posterior_routed_moe_smoke_seed_v1",
            "training_seed": seed,
            "environment_steps": int(summary["environment_steps"]),
            "gradient_updates": int(summary["gradient_updates"]),
            "router_audit_rows": len(rows),
            "actor_update_task_rows": len(updates),
            "min_entropy": min(float(row["entropy"]) for row in rows),
            "max_expert_load_cv": max(float(row["expert_load_cv"]) for row in updates),
            "min_router_gradient_norm": min(float(row["gradient_norms"]["router_gradient_norm"]) for row in updates),
            "min_expert_gradient_norms": expert_gradients,
            "routing_weight_range": [min(all_weights), max(all_weights)],
            "all_values_finite": all_finite(rows),
            "no_permanent_single_expert_exclusivity": no_exclusive,
            "status": "pass" if all_finite(rows) and no_exclusive and updates else "fail",
            "scope": "engineering_smoke_only_not_performance_evidence",
        })

    evaluation_path = output / "runs" / "smoke" / "meta_validation" / "routed_moe_seed_17_route_lifecycle" / "metrics.json"
    evaluation = read_json(evaluation_path)
    evaluation_rows = []
    query_consistency = []
    route_hashes = []
    route_weights = []
    descriptor_hashes = []
    for task_id, shots in evaluation["tasks"].items():
        task_ref = content_hash({"task_id": task_id})
        for shot, values in shots.items():
            audit = dict(values["router_audit"])
            evaluation_rows.append({
                "schema": "posterior_router_audit_v1",
                "training_seed": 17,
                "task_ref": task_ref,
                "phase": "deterministic_query",
                "support_episodes": int(shot),
                "posthoc_only": False,
                **audit,
                "query_route_hashes": values["query_route_hashes"],
                "query_route_consistent": bool(values["query_route_consistent"]),
            })
            query_consistency.append(bool(values["query_route_consistent"]))
            route_hashes.append(audit["route_hash"])
            route_weights.append(tuple(float(value) for value in audit["routing_weights"]))
            descriptor_hashes.append(audit["descriptor"]["content_hash"])
    router_rows.extend(evaluation_rows)
    (output / "router_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in router_rows),
        encoding="utf-8",
    )
    (output / "smoke_metrics_by_seed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in smoke_rows),
        encoding="utf-8",
    )

    architecture = PEARLAgent(37, 2, config, torch.device("cpu")).architecture_metadata()
    update_rows = [row for row in router_rows if row["phase"] == "actor_update"]
    architecture_contract = {
        "schema": "posterior_routed_moe_architecture_contract_v1",
        "architecture": architecture,
        "actor_only_moe": True,
        "critic_architecture_unchanged": "dense_twin_critics",
        "context_encoder_architecture_unchanged": True,
        "route_lifecycle": {
            "clock": "posterior_version",
            "fixed_within_episode": True,
            "recomputed_after_posterior_inference": True,
            "training_router_forward_is_differentiable": True,
            "collection_router_forward_uses_no_grad": True,
        },
        "gradient_boundary_evidence": {
            "all_router_gradients_nonzero": all(float(row["gradient_norms"]["router_gradient_norm"]) > 0 for row in update_rows),
            "all_soft_routed_expert_gradients_nonzero": all(
                float(row["gradient_norms"][f"expert_{index}_gradient_norm"]) > 0
                for row in update_rows
                for index in range(int(architecture["moe"]["num_experts"]))
            ),
            "context_encoder_actor_gradient_is_zero": all(float(row["gradient_norms"]["context_encoder_actor_gradient_norm"]) == 0 for row in update_rows),
            "critic_actor_gradient_is_zero": all(float(row["gradient_norms"]["critic_actor_gradient_norm"]) == 0 for row in update_rows),
        },
    }
    write_json(output / "architecture_contract.json", architecture_contract)

    first_descriptor = router_rows[0]["descriptor"]
    descriptor_contract = {
        "schema": first_descriptor["schema"],
        "fields": list(DESCRIPTOR_FIELDS),
        "normalization_scales": first_descriptor["normalization_scales"],
        "allowed_source": "initialized_map_static_topology_only",
        "computed_once_and_frozen_per_task": True,
        "forbidden_inputs": [
            "task_id", "geometry_id", "logical_type", "split", "hidden_contact_rule",
            "support_outcome_label", "query_data", "adversary_route_remaining", "sut_route_remaining",
        ],
        "same_physical_geometry_rule_pair_evidence": {
            "task_ref_count": len({row["task_ref"] for row in router_rows if row["phase"] == "collection" and row["training_seed"] == 17}),
            "descriptor_hash_count": len({row["descriptor"]["content_hash"] for row in router_rows if row["phase"] == "collection" and row["training_seed"] == 17}),
        },
    }
    write_json(output / "descriptor_schema.json", descriptor_contract)

    checkpoint = checkpoint_audit(
        config,
        run_roots[17] / "best_model.pt",
        router_rows[0],
    )
    write_json(output / "checkpoint_roundtrip.json", checkpoint)

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "pearl_learning/tests", "-q"],
        text=True,
        capture_output=True,
        check=False,
    )
    test_report = {
        "schema": "posterior_routed_moe_test_report_v1",
        "command": "python -m pytest pearl_learning/tests -q",
        "returncode": test.returncode,
        "stdout": test.stdout.strip(),
        "stderr": test.stderr.strip(),
        "status": "pass" if test.returncode == 0 else "fail",
    }
    write_json(output / "test_report.json", test_report)

    adaptation = read_json(Path("results/pearl_learning/posterior_adaptation/manifest.json"))
    route_log_text = (output / "router_audit.jsonl").read_text(encoding="utf-8").lower()
    leakage_free = not any(token in route_log_text for token in (
        "adversary_first", "sut_first", "task_id", "geometry_id", "logical_type", "priority_spec", "route_remaining"
    ))
    checks = {
        "automated_tests": test.returncode == 0,
        "checkpoint_roundtrip": checkpoint["status"] == "pass",
        "two_smoke_seeds": len(smoke_rows) == 2 and all(row["status"] == "pass" for row in smoke_rows),
        "evaluation_parameter_hash_unchanged": evaluation["parameter_hash_before"] == evaluation["parameter_hash_after"],
        "evaluation_module_hashes_unchanged": evaluation["module_hashes_before"] == evaluation["module_hashes_after"],
        "query_routes_consistent": all(query_consistency),
        "posterior_versions_have_distinct_route_hashes": len(set(route_hashes)) == len(route_hashes),
        "posterior_updates_change_route_weights": len(set(route_weights)) > 1,
        "descriptor_fixed_across_posterior_versions": len(set(descriptor_hashes)) == 1,
        "raw_router_log_has_no_hidden_rule_or_identifier_fields": leakage_free,
        "gradient_boundaries": all(architecture_contract["gradient_boundary_evidence"].values()),
    }
    manifest = {
        "schema": "posterior_routed_moe_manifest_v1",
        "experiment": "posterior_routed_moe_engineering",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "engineering_correctness_only_not_performance_superiority",
        "checks": checks,
        "architecture_hash": content_hash(architecture_contract),
        "descriptor_schema_hash": content_hash(descriptor_contract),
        "checkpoint_roundtrip_hash": content_hash(checkpoint),
        "test_report_hash": content_hash(test_report),
        "training_seeds": seeds,
        "conda_environment": "metadrive",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "posterior_adaptation_formal_status": adaptation.get("status"),
        "posterior_adaptation_allows_routed_moe": bool(adaptation.get("allows_routed_moe_engineering")),
        "posterior_adaptation_precondition_exception": "user explicitly authorized routed-MoE engineering after reviewing the physical controllability asymmetry",
        "legacy_dense_checkpoints_preserved_unmodified": True,
        "legacy_dense_automatic_loading": False,
        "legacy_dense_checkpoint_policy": "use the recorded pre-routed-MoE git commit or an explicit provenance-preserving migration; never infer missing architecture metadata",
        "allows_mechanism_pilot": all(checks.values()),
        "allows_formal_performance_or_transferability_claims": False,
        "artifacts": [
            "manifest.json", "architecture_contract.json", "descriptor_schema.json",
            "smoke_metrics_by_seed.jsonl", "router_audit.jsonl",
            "checkpoint_roundtrip.json", "test_report.json",
        ],
    }
    write_json(output / "manifest.json", manifest)
    if manifest["status"] != "PASS":
        raise RuntimeError(f"Routed-MoE finalization failed: {checks}")
    print(f"Posterior-routed MoE PASS: {output}")


if __name__ == "__main__":
    main()
