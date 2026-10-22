"""Independently rebuild post-selection labels from paid physical source traces."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from controllers.highway.catalog import BY_ID, SUTS

from .audit_collision_results import RawEvidence, compare, quantized_coordinate, registered_pool, require
from .protocol import DATA_ROOT, DEFAULT, RESULT_ROOT, ROOT, method_contract, validate_sut_freeze
from .scenarios import DOMAINS
from .studies import SCOPES, tasks


OUTPUT = ROOT / "results/benchmark/collision/certification_result_audit.json"


def coordinate(scene):
    return scene["kind"], tuple(sorted(scene["parameters"].items())), tuple(sorted(scene["categories"].items()))


class CertificationEvidence:
    def __init__(self, frozen, raw=None):
        self.frozen = frozen
        self.raw = RawEvidence() if raw is None else raw
        self.groups = {}
        self.paid_source_banks = {}

    def bank(self, path, sut_id, environment_seed):
        manifest, *_ = self.raw.bank(path)
        require(manifest["sut_id"] == sut_id and manifest["environment_seed"] == environment_seed,
                "Certification references another physical source or environment")
        compare(manifest["contract"], self.frozen["sut_contracts"][sut_id], "certification frozen SUT contract")
        require(manifest["candidate_count"] == len(manifest["scenarios"]), "Certification bank size differs")
        return manifest

    def group(self, condition, kind, environment_seed):
        key = condition, kind, environment_seed
        if key in self.groups:
            return self.groups[key]
        candidates = registered_pool(kind, condition, DEFAULT.candidate_count)
        targets, union = {}, set()
        for sut in SUTS:
            if sut.role != "target" or kind not in sut.supported_scenarios:
                continue
            root = DATA_ROOT / "targets" / sut.sut_id / condition / kind / f"env_{environment_seed}"
            bank = self.bank(root, sut.sut_id, environment_seed)
            compare(bank["scenarios"], candidates, "certification target candidate pool")
            failed = set()
            for index in range(len(candidates)):
                if self.raw.execution(root, index)["facts"]["collision"]:
                    failed.add(index)
            targets[sut.sut_id] = failed
            union.update(failed)
        require(bool(targets), "Certification has no registered target systems")
        selected = sorted(union)
        outcomes, paths, charges = {}, {}, {}
        for source in SUTS:
            if source.role != "history":
                continue
            if condition == "reuse":
                root = DATA_ROOT / "physical/history" / kind / source.sut_id
                bank = self.bank(root, source.sut_id, environment_seed)
                compare(bank["scenarios"], registered_pool(kind, "history", DEFAULT.candidate_count),
                        "real historical coordinate pool")
                actual = {coordinate(scene): index for index, scene in enumerate(bank["scenarios"])}
                require(len(actual) == len(bank["scenarios"]), "Historical physical coordinates repeat")
                source_indices = [actual[coordinate(candidates[index])] for index in selected]
                charges[source.sut_id] = (0, 0.0)
            elif selected:
                root = DATA_ROOT / "certification/shared" / condition / kind / f"env_{environment_seed}" / source.sut_id
                bank = self.bank(root, source.sut_id, environment_seed)
                compare(bank["scenarios"], [candidates[index] for index in selected],
                        "paid shared certification collision union")
                source_indices = list(range(len(selected)))
                charges[source.sut_id] = bank["attempted_calls"], bank["physical_execution_s"]
                self.paid_source_banks[str(root.resolve())] = {
                    "path": str(root.relative_to(DATA_ROOT)), "actual_physical_calls": bank["attempted_calls"],
                    "physical_execution_s": bank["physical_execution_s"],
                    "failed_or_interrupted_calls": bank["failed_or_interrupted_calls"],
                    "interrupted_call_duration_unknown": bank["interrupted_call_duration_unknown"],
                    "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
            else:
                outcomes[source.sut_id] = {}
                continue
            paths[source.sut_id] = str(root.relative_to(DATA_ROOT))
            outcomes[source.sut_id] = {
                target_index: self.raw.execution(root, source_index)["facts"]["collision"]
                for target_index, source_index in zip(selected, source_indices)}
        group = {"candidates": candidates, "targets": targets, "source_outcomes": outcomes,
                 "source_paths": paths, "source_charges": charges}
        self.groups[key] = group
        return group

    def certificate(self, sut_id, condition, kind, environment_seed):
        group = self.group(condition, kind, environment_seed)
        target = BY_ID[sut_id]
        require(target.role == "target" and target.reference_id, "Registered version reference is required")
        failed = group["targets"][sut_id]
        incurred = condition != "reuse" and bool(failed)
        return {"label_provenance": "real_execution", "target_id": sut_id, "reference_id": target.reference_id,
                "reference_lineage": BY_ID[target.reference_id].lineage,
                "change_type": "independent_training_run" if target.family == "learning" else "registered_configuration",
                "condition": condition, "scenario": kind, "environment_seed": environment_seed,
                "outcomes": {group["candidates"][index]["scenario_id"]: {
                    source: values[index] for source, values in group["source_outcomes"].items()}
                    for index in sorted(failed)},
                "actual_additional_source_calls": sum(c[0] for c in group["source_charges"].values()) if incurred else 0,
                "additional_source_execution_s": sum(c[1] for c in group["source_charges"].values()) if incurred else 0.0,
                "source_bank_paths": group["source_paths"] if condition == "reuse" or failed else {},
                "source_bank_scope": "historical_coordinates" if condition == "reuse" else "shared_across_targets",
                "use": "independent_evaluation_only"}


def validate_certificate(actual, expected):
    for field, value in expected.items():
        require(field in actual, f"Missing certificate field: {field}")
        compare(actual[field], value, f"independently rebuilt certificate.{field}")
    require("not additive" in actual["cost_accounting"], "Shared source-bank references must not be charged repeatedly")


def derive_metrics(candidates, queries, certificate, history_ids):
    """Rebuild both labels and their quantized prefix counts, without the producer."""
    require(bool(history_ids), "Certification needs the registered source set even for no-history selectors")
    reference = certificate["reference_id"]
    version, historical, found_ids = set(), set(), set()
    version_curve, history_curve = [], []
    for query in queries:
        label = query["feedback"]["collision"]
        require(type(label) is bool, "A predicted or numeric value cannot certify a collision")
        scene = candidates[query["index"]]
        expected_collision = scene["scenario_id"] in certificate["outcomes"]
        require(label == expected_collision, "Selected collision differs from the actual target trace")
        if label:
            source_labels = certificate["outcomes"][scene["scenario_id"]]
            require(all(type(source_labels[name]) is bool for name in {*history_ids, reference}),
                    "Actual Boolean source labels are required")
            found_ids.add(scene["scenario_id"])
            key = quantized_coordinate(scene)
            if source_labels[reference] is False:
                version.add(key)
            if not any(source_labels[name] for name in history_ids):
                historical.add(key)
        version_curve.append(len(version))
        history_curve.append(len(historical))
    return {"version_new_collisions": len(version), "historical_set_specific_collisions": len(historical),
            "version_new_curve": version_curve, "historical_set_specific_curve": history_curve,
            "certified_history_set": list(history_ids),
            "required_additional_source_calls_if_uncached": (
                0 if certificate["condition"] == "reuse" else len(found_ids) * len({*history_ids, reference})),
            "actual_shared_certification_calls": certificate["actual_additional_source_calls"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    frozen_path = DATA_ROOT / "frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    validate_sut_freeze(frozen)
    compare(frozen["source_contract"], method_contract(), "certification frozen source contract")
    plan = tasks()
    compare(frozen["study_tasks"], [asdict(task) for task in plan], "certification registered task matrix")
    evidence, certificates, pending, errors = CertificationEvidence(frozen), {}, [], []
    verified_certificates = []
    for condition in DEFAULT.conditions:
        for kind in DOMAINS:
            for env in DEFAULT.environment_seeds:
                for sut in SUTS:
                    if sut.role != "target" or kind not in sut.supported_scenarios:
                        continue
                    path = RESULT_ROOT / "certificates" / sut.sut_id / condition / kind / f"env_{env}.json"
                    if not path.exists():
                        pending.append(str(path))
                        continue
                    try:
                        expected = evidence.certificate(sut.sut_id, condition, kind, env)
                        validate_certificate(json.loads(path.read_text(encoding="utf-8")), expected)
                        certificates[(sut.sut_id, condition, kind, env)] = expected
                        verified_certificates.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                      "target_collision_coordinates": len(expected["outcomes"])})
                        print("verified real-source certificate", len(verified_certificates), "/ 96", flush=True)
                    except Exception as error:
                        errors.append({"path": str(path), "error": repr(error)})
    metrics_path, verified_runs = RESULT_ROOT / "new_collision_metrics.json", []
    if metrics_path.exists():
        try:
            from .evaluate_collision import task_path
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            require(saved["status"] == "complete" and len(saved["runs"]) == len(plan), "Incomplete certification metrics matrix")
            by_path = {str(Path(row["path"]).resolve()): row for row in saved["runs"]}
            require(len(by_path) == len(plan), "Duplicate certification metrics entry")
            expected_paths = {str(task_path(task).resolve()) for task in plan}
            require(set(by_path) == expected_paths, "Certification metrics reference another task matrix")
            for task in plan:
                path = task_path(task)
                run = json.loads(path.read_text(encoding="utf-8"))
                compare(run["specification"]["metadata"], asdict(task), "certification discovery task")
                queries = run["result"]["queries"]
                require(len(queries) == run["result"]["actual_queries"] == DEFAULT.budget, "Certification refers to an incomplete run")
                certificate = certificates[(task.target_id, task.condition, task.scenario, task.environment_seed)]
                group = evidence.group(task.condition, task.scenario, task.environment_seed)
                expected = derive_metrics(group["candidates"], queries, certificate, SCOPES[task.scope].history_ids)
                row = by_path[str(path.resolve())]
                compare({k: v for k, v in row.items() if k != "path"}, expected, "independently rebuilt new-collision metrics")
                verified_runs.append({"path": str(path), "version_new_collisions": expected["version_new_collisions"],
                                      "historical_set_specific_collisions": expected["historical_set_specific_collisions"]})
        except Exception as error:
            errors.append({"path": str(metrics_path), "error": repr(error)})
    else:
        pending.append(str(metrics_path))
    expected_paid = set(evidence.paid_source_banks)
    observed_paid = {str(path.parent.resolve()) for path in (DATA_ROOT / "certification/shared").glob("*/*/env_*/*/manifest.json")}
    if len(verified_certificates) == 96:
        try:
            require(expected_paid == observed_paid, "Unexpected or missing actual shared-source bank")
        except Exception as error:
            errors.append({"path": str(DATA_ROOT / "certification/shared"), "error": repr(error)})
    complete = not pending and len(verified_certificates) == 96 and len(verified_runs) == len(plan)
    report = {"status": "failed" if errors else "complete" if complete else "partial",
              "scope": "Independent real-source certification, both collision labels, all discovery prefixes and uniquely charged source banks.",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "additional_physical_calls": 0,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
              "expected_certificates": 96, "verified_certificates": len(verified_certificates),
              "expected_runs": len(plan), "verified_runs": len(verified_runs),
              "unique_raw_executions_checked": len(evidence.raw.executions),
              "actual_shared_source_calls": sum(row["actual_physical_calls"] for row in evidence.paid_source_banks.values()),
              "shared_source_execution_s": sum(row["physical_execution_s"] for row in evidence.paid_source_banks.values()),
              "physical_source_banks": list(evidence.paid_source_banks.values()),
              "certificates": verified_certificates, "runs": verified_runs, "pending": pending, "errors": errors}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(report["status"], len(verified_certificates), "/96 certificates;", len(verified_runs), "/10880 runs;", len(errors), "errors", flush=True)
    if errors or arguments.require_complete and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
