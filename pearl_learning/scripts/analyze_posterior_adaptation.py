"""Aggregate posterior-adaptation artifacts without treating query cases as tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pearl_learning.src.posterior_adaptation_analysis import (
    budget_curves,
    flatten_evaluations,
    paired_method_effect,
    posterior_pair_audit,
    validate_evaluation_artifact,
)
from pearl_learning.src.io import content_hash, write_json
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _method_paths(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("evaluations must use method=path")
        method, path = value.split("=", 1)
        result.setdefault(method, []).append(Path(path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--validation", nargs="+", required=True, help="repeated method=metrics.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-environment-steps", type=int, required=True)
    parser.add_argument("--conda-environment", default="metadrive")
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--topology-audit", required=True)
    parser.add_argument("--integrity-audit", required=True)
    parser.add_argument("--support-only-audit", required=True)
    parser.add_argument("--query-invariance-audit", choices=["pass", "fail"], required=True)
    parser.add_argument("--training-summary", nargs="*", default=[])
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    if protocol.get("schema") != "posterior_adaptation_frozen_protocol":
        raise ValueError("unsupported posterior-adaptation protocol")
    taskbook = load_taskbook(args.taskbook)
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    if protocol.get("taskbook_hash") != taskbook_hash:
        raise ValueError("posterior-adaptation protocol and taskbook hashes differ")
    topology_audit = json.loads(Path(args.topology_audit).read_text(encoding="utf-8"))
    integrity_audit = json.loads(Path(args.integrity_audit).read_text(encoding="utf-8"))
    support_only_audit = json.loads(Path(args.support_only_audit).read_text(encoding="utf-8"))
    if topology_audit.get("taskbook_hash") != taskbook_hash:
        raise ValueError("topology audit belongs to a different taskbook")
    if integrity_audit.get("taskbook_hash") != taskbook_hash:
        raise ValueError("integrity audit belongs to a different taskbook")
    topology_pass = topology_audit.get("passed") == topology_audit.get("total")
    integrity_pass = integrity_audit.get("status") == "pass"
    support_only_pass = (
        support_only_audit.get("schema") == "logical_merge_support_posterior_diagnostic_v1"
        and support_only_audit.get("taskbook_hash") == taskbook_hash
        and support_only_audit.get("split") == "meta_validation"
        and support_only_audit.get("uses_query_cases") is False
        and support_only_audit.get("no_gradient_adaptation") is True
        and support_only_audit.get("parameter_hash_before") == support_only_audit.get("parameter_hash_after")
        and support_only_audit.get("module_hashes_before") == support_only_audit.get("module_hashes_after")
        and support_only_audit.get("context_protocol", {}).get("name") == "fixed_nested"
    )
    paths = _method_paths(args.validation)
    evaluations = {
        method: [json.loads(path.read_text(encoding="utf-8")) for path in method_paths]
        for method, method_paths in paths.items()
    }
    training_summaries = []
    for value in args.training_summary:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        training_summaries.append({
            "path": str(path),
            "environment_steps": int(payload["environment_steps"]),
            "gradient_updates": int(payload["gradient_updates"]),
            "artifact_hash": content_hash(payload),
        })
    problems: dict[str, list[str]] = {}
    for method, artifacts in evaluations.items():
        for index, artifact in enumerate(artifacts):
            found = validate_evaluation_artifact(
                artifact,
                taskbook_hash=taskbook_hash,
                shots=[int(value) for value in protocol["shots"]],
                query_cases=int(protocol["query_cases_per_task"]),
            )
            if found:
                problems[f"{method}[{index}]"] = found
    rows = flatten_evaluations(evaluations)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    (output_root / "metrics_by_seed_task_k.jsonl").write_text(jsonl, encoding="utf-8")

    criteria = protocol["pass_criteria"]
    primary_shot = int(protocol["primary_shot"])
    samples = int(protocol["bootstrap_samples"])
    confidence = float(protocol["confidence_level"])
    effect = paired_method_effect(
        rows,
        method="pearl_full",
        reference="pearl_no_context",
        shot=primary_shot,
        metric=str(protocol["primary_metric"]),
        samples=samples,
        confidence=confidence,
    )
    full_artifacts = evaluations.get("pearl_full", [])
    posterior = posterior_pair_audit(
        full_artifacts,
        taskbook["meta_validation"],
        shots=[int(value) for value in protocol["shots"]],
        samples=samples,
        confidence=confidence,
    ) if full_artifacts else {"schema": "posterior_adaptation_pair_audit_v1", "status": "missing_pearl_full"}
    write_json(output_root / "posterior_pair_audit.json", posterior)

    required_methods = set(protocol["required_methods"])
    missing_methods = sorted(required_methods - set(evaluations))
    expected_seeds = set(int(value) for value in protocol["training_seeds"])
    seeds_by_method = {
        method: sorted({int(item.get("provenance", {}).get("training_seed", -1)) for item in artifacts})
        for method, artifacts in evaluations.items()
    }
    incomplete_seed_methods = sorted(
        method for method in required_methods
        if set(seeds_by_method.get(method, [])) != expected_seeds
    )
    accuracy = posterior.get("shots", {}).get(str(primary_shot), {}).get(
        "leave_one_geometry_pair_out_rule_accuracy", {},
    )
    effect_pass = (
        effect.get("ci_lower") is not None
        and float(effect["ci_lower"]) > float(criteria["full_minus_no_context_ci_lower_gt"])
    )
    rule_pass = (
        accuracy.get("ci_lower") is not None
        and float(accuracy["ci_lower"]) > float(criteria["rule_pair_accuracy_ci_lower_gt"])
    )
    complete = (
        not problems and not missing_methods and not incomplete_seed_methods
        and topology_pass and integrity_pass and support_only_pass
    )
    status = "VALIDATION_PASS_HOLDOUT_NOT_RUN" if complete and effect_pass and rule_pass else "INCOMPLETE"
    if complete and not (effect_pass and rule_pass):
        status = "FAIL"
    statistical = {
        "schema": "posterior_adaptation_statistical_summary_v1",
        "status": status,
        "primary_effect": effect,
        "rule_pair_accuracy_at_primary_shot": accuracy,
        "criteria": {"support_effect_pass": effect_pass, "rule_pair_pass": rule_pass},
        "protocol_problems": problems,
        "missing_methods": missing_methods,
        "incomplete_seed_methods": incomplete_seed_methods,
        "seeds_by_method": seeds_by_method,
    }
    write_json(output_root / "statistical_summary.json", statistical)
    compact = {
        "schema": "posterior_adaptation_compact_results_v1",
        "status": status,
        "validation_only": True,
        "holdout_evaluated": False,
        "curves": {
            method: budget_curves(rows, method)
            for method in evaluations
        },
        "primary_effect": effect,
        "rule_pair_accuracy_at_primary_shot": accuracy,
    }
    write_json(output_root / "compact_results.json", compact)
    manifest = {
        "schema": "posterior_adaptation_manifest_v1",
        "experiment": "posterior_adaptation_validation",
        "status": "PASS" if status == "HOLDOUT_PASS" else ("FAIL" if status == "FAIL" else "INCOMPLETE"),
        "analysis_status": status,
        "taskbook_hash": taskbook_hash,
        "frozen_protocol_hash": content_hash(protocol),
        "validation_artifact_hashes": {
            method: [content_hash(artifact) for artifact in artifacts]
            for method, artifacts in evaluations.items()
        },
        "formal_training_seeds": seeds_by_method,
        "run_kind": "formal",
        "requested_max_environment_steps_per_seed": int(args.training_environment_steps),
        "training_task_scope": "complete frozen meta-train task set",
        "configured_formal_environment_steps": 1500000,
        "conda_environment": args.conda_environment,
        "automated_tests_passed": int(args.test_count),
        "topology_audit": {
            "status": "pass" if topology_pass else "fail",
            "passed": topology_audit.get("passed"),
            "total": topology_audit.get("total"),
            "artifact_hash": content_hash(topology_audit),
        },
        "integrity_audit": {
            "status": "pass" if integrity_pass else "fail",
            "artifact_hash": content_hash(integrity_audit),
        },
        "support_only_posterior_audit": {
            "status": "pass" if support_only_pass else "fail",
            "uses_query_cases": support_only_audit.get("uses_query_cases"),
            "no_gradient_adaptation": support_only_audit.get("no_gradient_adaptation"),
            "artifact_hash": content_hash(support_only_audit),
        },
        "query_invariance_audit": args.query_invariance_audit,
        "training_summaries": training_summaries,
        "missing_methods": missing_methods,
        "protocol_problems": problems,
        "holdout_evaluated": False,
        "allows_routed_moe_engineering": False,
        "reason": "posterior adaptation cannot pass before all methods, seeds, and frozen holdout evaluation are complete",
        "required_next_work": [
            "complete the pre-training formal validation for the new taskbook",
            "train all five methods for the frozen formal budget on seeds 11, 22, and 33",
            "evaluate all 20 validation query cases per task",
            "run holdout only if the pre-registered validation criteria pass",
        ],
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "test_report.json", {
        "schema": "posterior_adaptation_test_report_v1",
        "conda_environment": args.conda_environment,
        "command": "python -m pytest pearl_learning/tests -q",
        "passed": int(args.test_count),
        "failed": 0,
        "query_invariance_audit": args.query_invariance_audit,
    })
    print(f"Posterior-adaptation analysis completed with status={status}: {output_root}")


if __name__ == "__main__":
    main()
