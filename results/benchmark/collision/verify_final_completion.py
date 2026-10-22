"""Final current-state acceptance, referencing completed independent raw audits.

This checks current immutable artifact bindings, original scope and review evidence.
It does not rerun a SUT, train anything, or replace the independent raw audits.
"""
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/benchmark/collision"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(test, message):
    if not test:
        raise AssertionError(message)


def binding(path, digest):
    require(sha(path) == digest, "Changed accepted artifact: " + str(path))


def main():
    now = datetime.now(timezone.utc).isoformat()
    audit = read(BASE / "completion_audit.json")
    original_ids = [r["id"] for r in audit["requirements"]]
    require(len(original_ids) == len(set(original_ids)) == 23, "Original 23 requirements")
    requirements = {r["id"]: r for r in audit["requirements"]}
    proofs = {}

    def proof(name, *, validator=None):
        path = BASE / name
        value = read(path)
        require(value.get("status") not in (None, "incomplete", "failed", "running"), "Incomplete evidence: " + name)
        for key in ("errors", "pending", "pending_runs"):
            require(not value.get(key), "Unresolved evidence: " + name + ":" + key)
        if validator is not None:
            binding(ROOT / validator, value["validator_sha256"])
        proofs[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "status": value["status"]}
        return value

    regression = proof("final_implementation_acceptance_verification.json")
    require(regression["native_exit_code"] == 0 and regression["tests"] == 102,
            "Final native regression failed")
    binding(ROOT / regression["junit"], regression["junit_sha256"])
    binding(ROOT / regression["command"], regression["command_sha256"])
    for path, digest in regression["current_source_contract"].items():
        binding(ROOT / path, digest)
    cases = ET.parse(ROOT / regression["junit"]).getroot().findall(".//testcase")
    require(len(cases) == len({(c.get("classname"), c.get("name")) for c in cases}) == 102,
            "Distinct regression cases")
    require(all(c.find("failure") is None and c.find("error") is None and c.find("skipped") is None for c in cases), "Regression result")

    from controllers.highway.catalog import SUTS
    from benchmark.protocol import method_contract, validate_sut_freeze
    from benchmark.studies import SCOPES, tasks, CONTINUOUS_TASKS
    from benchmark.evaluate_continuous import validate_frozen
    frozen_path = ROOT / "data/collision/frozen_protocol.json"
    continuous_freeze_path = ROOT / "data/collision/frozen_continuous.json"
    frozen, continuous_freeze = read(frozen_path), read(continuous_freeze_path)
    require(frozen["source_contract"] == method_contract() == audit["method_contract"], "Frozen method source")
    validate_sut_freeze(frozen)
    validate_frozen()
    catalog = json.loads(json.dumps([asdict(s) for s in SUTS]))
    require(len(catalog) == 20 and catalog == frozen["objects"] == requirements["I02"]["evidence"]["catalog"], "SUT catalog identity")
    require(Counter(s["role"] for s in catalog) == {"history": 8, "development": 4, "target": 8}, "SUT role counts")
    require(all(Counter(s["role"] for s in catalog if s["family"] == family) == {"history": 2, "development": 1, "target": 2}
                for family in ("rule", "optimization", "search", "learning")), "SUT family roles")
    require(len(SCOPES) == 29 and json.loads(json.dumps({k: asdict(v) for k, v in SCOPES.items()})) == frozen["model_scopes"], "All source scopes frozen")
    current_tasks = json.loads(json.dumps([asdict(t) for t in tasks()]))
    require(current_tasks == frozen["study_tasks"] and len(current_tasks) == 10880, "Registered finite matrix")
    study_counts = Counter(t["study"] for t in current_tasks)
    require(study_counts == {"overall": 3840, "strict_family": 3840, "source_swap": 2880, "mechanisms": 320}, "Complete study allocation")
    require([list(t) for t in CONTINUOUS_TASKS] == frozen["continuous_tasks"] == continuous_freeze["tasks"], "Separate continuous cases")
    print("verified current method/SUT/continuous freezes and all23 requirement identities", flush=True)

    ready = proof("policy_construction/all_sut_readiness_verification.json")
    require(len(ready["objects"]) == 20 and all(r["qualified"] for r in ready["objects"]), "All SUT capabilities")
    learning = []
    for row in ready["objects"]:
        qualification = BASE / "qualification" / row["sut_id"] / "qualification.json"
        binding(qualification, row["qualification_sha256"])
        q = read(qualification)
        require(q["qualified"] and len(q["courses"]) == 5, "Five capability courses")
        if row["family"] == "learning":
            directory = ROOT / "controllers/highway/checkpoints/collision" / row["training_run"]
            binding(directory / "best.zip", row["checkpoint_sha256"])
            training = read(directory / "training.json")
            require(training["status"] == "qualified" and training["actual_timesteps"] == row["actual_timesteps"] == 4194304,
                    "Full independent policy training")
            learning.append({"sut_id": row["sut_id"], "run": row["training_run"], "timesteps": training["actual_timesteps"],
                             "training_sha256": sha(directory / "training.json"), "checkpoint_sha256": row["checkpoint_sha256"]})
    require(len(learning) == 5, "Five formal learning objects")
    pre_target = proof("policy_construction/pre_target_freeze_verification.json")
    binding(ROOT / pre_target["frozen_protocol"], pre_target["frozen_protocol_sha256"])
    coordinates = proof("protocol_coordinate_audit.json")
    require(coordinates["canonical_history_equals_reuse"] and coordinates["four_role_pools_pairwise_disjoint"], "Separated coordinate roles")

    construction = proof("construction_data_audit.json", validator="benchmark/audit_collision_data.py")
    require(construction["verified_archives"] == 48 and construction["verified_responses"] == 24576, "All source responses raw-audited")
    for row in construction["banks"]:
        path = ROOT / "data/collision" / row["role"] / row["scenario"] / (row["source_id"] + ".npz")
        binding(path, row["archive_sha256"])
    target = proof("target_data_verification.json", validator="results/benchmark/collision/verify_target_data.py")
    require(target["verified_banks"] == 96 and target["verified_responses"] == 49152 and target["full_current_content_rehash_checked"], "Full target raw audit")
    binding(frozen_path, target["protocol_sha256"])
    for row in target["banks"]:
        binding(Path(row["bank"]) / "manifest.json", row["manifest_sha256"])
        binding(Path(row["bank"]) / "calls.jsonl", row["call_ledger_sha256"])
    model_audit = proof("full_model_verification.json", validator="results/benchmark/collision/verify_models.py")
    require(model_audit["verified_models"] == model_audit["expected_models"] == 156, "All model actual builds")
    frozen_models = {str((ROOT / m["path"]).resolve()): m["sha256"] for m in frozen["method"]["models"]}
    require(len(frozen_models) == 156, "Full frozen model set")
    for row in model_audit["models"]:
        path = Path(row["checkpoint"])
        binding(path, row["checkpoint_sha256"])
        binding(path.with_suffix(".json"), row["training_report_sha256"])
        require(row["completed_steps"] == 1500 and frozen_models[str(path.resolve())] == row["checkpoint_sha256"], "Completed frozen checkpoint")

    finite = proof("full_finite_raw_result_audit.json", validator="benchmark/audit_collision_results.py")
    continuous = proof("full_continuous_raw_result_audit.json", validator="benchmark/audit_continuous_results.py")
    development = proof("development_verification.json", validator="results/benchmark/collision/verify_development.py")
    continuous_development = proof("continuous_development_verification.json", validator="results/benchmark/collision/verify_continuous_development.py")
    require(finite["verified_runs"] == 10880 and finite["index_checked"] and finite["frozen_protocol_checked"], "Full finite actual execution")
    require(continuous["verified_runs"] == 100 and continuous["index_checked"] and continuous["frozen_protocol_checked"], "Full continuous actual execution")
    require(development["verified_runs"] == 288 and continuous_development["verified_runs"] == 40, "Both actual development matrices")
    for value, count, budget in ((finite, 10880, 128), (continuous, 100, 128), (development, 288, 32), (continuous_development, 40, 32)):
        require(len(value["runs"]) == len({r["path"] for r in value["runs"]}) == count, "Unique audited results")
        for row in value["runs"]:
            binding(row["path"], row["sha256"])
            require(row["queries"] == budget, "Paid initialization and budget")
    certification = proof("full_certification_result_audit.json", validator="benchmark/audit_collision_certification.py")
    require(certification["verified_certificates"] == 96 and certification["verified_runs"] == 10880
            and certification["actual_shared_source_calls"] == 15584, "Actual two-label certification")
    for row in certification["certificates"]:
        binding(row["path"], row["sha256"])
    print("verified current156 models,5 policies,48 source archives,96 target ledgers and11308 audited saved runs", flush=True)

    statistics = proof("full_statistical_report_audit.json", validator="benchmark/audit_collision_statistics.py")
    require(statistics["verified"]["finite"]["verified_statistical_contrasts"] == 456
            and statistics["verified"]["continuous"]["verified_statistical_contrasts"] == 16, "All472 inference checks")
    for part in statistics["verified"].values():
        for item in part["artifacts"]:
            binding(item["path"], item["sha256"])
    inventory = proof("construction_cost_inventory.json", validator="results/benchmark/collision/verify_cost_inventory.py")
    for item in inventory["physical_banks"]:
        binding(ROOT / item["path"], item["manifest_sha256"])
    for key in ("offline_model_builds", "policy_construction", "public_interface_qualification"):
        for item in inventory[key]:
            binding(ROOT / item["path"], item["sha256"])
    finite_cost = proof("finite_cost_report_verification.json", validator="results/benchmark/collision/verify_finite_cost_report.py")
    continuous_cost = proof("continuous_cost_report_verification.json", validator="results/benchmark/collision/verify_continuous_artifacts.py")
    require(finite_cost["unique_physical_calls"] == 89312 and continuous_cost["final_actual_calls"] == 12800
            and continuous_cost["development_actual_calls"] == 1280, "Nonduplicated real calls")
    binding(ROOT / "results/gcrt/collision/report/report.json", finite_cost["report_sha256"])
    binding(ROOT / "results/gcrt/collision/continuous/report/report.json", continuous_cost["report_sha256"])
    binding(BASE / "construction_cost_inventory.json", finite_cost["construction_inventory_sha256"])
    for item in finite_cost["certification_banks"]:
        binding(ROOT / item["path"], item["sha256"])
    for name, count, validator in (("finite_figure_verification.json", 19, "verify_finite_figures.py"),
                                    ("continuous_figure_verification.json", 8, "verify_continuous_artifacts.py")):
        figures = proof(name, validator="results/benchmark/collision/" + validator)
        require(figures["visual_review_complete"] and len(figures["figures"]) == count, "Actual visual review")
        for figure in figures["figures"]:
            require(len(figure["artifacts"]) == 3, "Three figure formats")
            for item in figure["artifacts"]:
                binding(item["path"], item["sha256"])
    annex = proof("continuous_reporting_annex_verification.json", validator="results/benchmark/collision/describe_continuous_results.py")
    for kind in ("coverage", "auxiliary"):
        binding(annex[kind + "_path"], annex[kind + "_sha256"])
    doc_sha = sha(ROOT / "docs/method.md")
    for name in ("finite_scientific_review.json", "continuous_scientific_review.json"):
        review = proof(name)
        require(review["canonical_design_sha256"] == doc_sha and review["frozen_protocol_retained"], "Current scientific narrative")
    for name in ("interface.json", "precollection_gate.json", "final_acquisition_gate.json"):
        proof(name)

    pipeline = read(ROOT / "results/gcrt/collision/pipeline/state.json")
    require(pipeline["status"] == "complete" and all(s["status"] == "complete" and s["native_exit_code"] == 0
            for s in pipeline["stages"].values()), "All native experiment stages exited0")
    archive = proof("archive_integrity_recheck.json")
    require(archive["crc_rechecked_all_entries"] and archive["includes_git"], "Original archive integrity")
    print("checking complete original12.7GB archive SHA against its full CRC-verified snapshot", flush=True)
    binding(archive["archive"], archive["archive_sha256"])
    require(Path(archive["archive"]).stat().st_size == archive["compressed_bytes"], "Original archive size")
    retired = read(BASE / "retired_sources.json")
    require(len(retired["files"]) == 174 and all(not (ROOT / path).exists() for path in retired["files"]), "Retired active files")
    require(not (ROOT / "method-overview.png").exists(), "Preserve user's original image deletion")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    require(head == archive["embedded_manifest"]["head"], "Preserve original commit HEAD")
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    require(diff.returncode == 0, "Whitespace errors: " + diff.stdout + diff.stderr)

    def receipt(identifier, names):
        row = requirements[identifier]
        row["evidence"]["final_current_state_evidence"] = [proofs[n] for n in names]
        if "expected_artifact" in row["evidence"]:
            row["evidence"]["exists"] = (ROOT / row["evidence"]["expected_artifact"]).exists()
            require(row["evidence"]["exists"], "Missing final required artifact")
        return {"id": identifier, "requirement": row["requirement"], "status": row["status"], "evidence": row["evidence"]["final_current_state_evidence"]}

    requirements["I03"]["evidence"]["note"] = "Coordinates are disjoint; all48 construction archives and96 final target banks have completed independent actual-data audits."
    requirements["I04"]["evidence"]["trained_models"] = "All156 actual models verified and frozen, including complete family exclusions."
    requirements["I05"]["evidence"]["note"] = "Registration and actual execution agree:10880 finite plus100 separate continuous runs; full raw audits are required evidence."
    requirements["E07"]["evidence"]["native_stage_status"] = "complete"
    requirements["R01"]["evidence"]["cost_review_scope"] = "Complete construction, actual shared certification, replay online and real continuous costs; unmeasured early cached qualification duration remains explicitly unknown."
    requirements["R02"]["status"] = "verified_result_review"
    requirements["R02"]["evidence"].update(continuous_review_pending=False, continuous_scientific_synthesis_pending=False,
        finite_scientific_review_sha256=sha(BASE / "finite_scientific_review.json"),
        continuous_scientific_review="results/benchmark/collision/continuous_scientific_review.json",
        continuous_scientific_review_sha256=sha(BASE / "continuous_scientific_review.json"))
    mapping = {
        "I01": ["final_implementation_acceptance_verification.json"],
        "I02": ["policy_construction/all_sut_readiness_verification.json"],
        "I03": ["protocol_coordinate_audit.json", "construction_data_audit.json", "target_data_verification.json"],
        "I04": ["full_model_verification.json"],
        "I05": ["full_finite_raw_result_audit.json", "full_continuous_raw_result_audit.json"],
        "I06": ["archive_integrity_recheck.json"], "I07": ["policy_construction/all_sut_readiness_verification.json"],
        "E01": ["policy_construction/all_sut_readiness_verification.json", "policy_construction/pre_target_freeze_verification.json"],
        "E02": ["construction_data_audit.json"], "E03": ["full_model_verification.json"],
        "E04": ["development_verification.json"], "E05": ["target_data_verification.json"],
        "E06": ["full_finite_raw_result_audit.json"], "E07": ["full_certification_result_audit.json"],
        "E08": ["full_statistical_report_audit.json", "finite_cost_report_verification.json", "finite_figure_verification.json"],
        "E09": ["continuous_development_verification.json"], "E10": ["full_continuous_raw_result_audit.json"],
        "E11": ["full_statistical_report_audit.json", "continuous_cost_report_verification.json", "continuous_figure_verification.json", "continuous_reporting_annex_verification.json"],
        "R01": ["full_finite_raw_result_audit.json", "full_certification_result_audit.json", "full_continuous_raw_result_audit.json", "full_statistical_report_audit.json", "finite_cost_report_verification.json", "continuous_cost_report_verification.json"],
        "R02": ["finite_scientific_review.json", "continuous_scientific_review.json", "finite_figure_verification.json", "continuous_figure_verification.json"],
        "I08": ["final_implementation_acceptance_verification.json"], "I09": ["precollection_gate.json"], "I10": ["final_acquisition_gate.json"]}
    require(set(mapping) == set(original_ids), "Complete original-scope acceptance mapping")
    checked = [receipt(identifier, mapping[identifier]) for identifier in original_ids]
    require(all(r["status"] in ("verified_implementation", "verified_execution", "verified_result_review") for r in checked), "No pending requirement")
    output = {"status": "verified_complete", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Full original method.md design: implementation, actual training/acquisition/development/finite/certification/continuous execution, independent raw/statistics/cost/figure/scientific review, and current immutable artifact bindings.",
              "validator_sha256": sha(__file__), "canonical_design": "docs/method.md", "canonical_design_sha256": doc_sha,
              "original_requirements": 23, "verified_requirements": 23, "requirements": checked,
              "method_source_files": len(frozen["source_contract"]), "continuous_source_files": len(continuous_freeze["source_contract"]),
              "formal_policies": learning, "model_builds": 156, "construction_banks": 48, "construction_responses": 24576,
              "target_banks": 96, "target_responses": 49152, "finite_development_runs": 288, "finite_runs": 10880,
              "certificates": 96, "extra_shared_certification_calls": 15584, "continuous_development_runs": 40,
              "continuous_final_runs": 100, "continuous_actual_calls": 14080, "actual_response_certification_continuous_calls": 103392,
              "statistical_contrasts": 472, "finite_equal_count_groups": 960, "continuous_equal_count_groups": 20,
              "figure_sets": 27, "figure_formats": ["png", "svg", "pdf"], "regression_cases": 102,
              "original_archive_sha256_rechecked": archive["archive_sha256"], "git_head_unchanged": head,
              "git_diff_check_exit_code": diff.returncode, "additional_physical_calls": 0, "errors": [], "pending": [],
              "accounting_limits": ["103392 excludes policy training and qualification, which are separately recorded.",
                                    "Independent raw audit execution counts overlap and are not added.",
                                    "Unmeasured early cached qualification duration is unknown, not zero.",
                                    "No claim that all proposed mechanisms or superiority over strong transfer baselines were established."]}
    final_path = BASE / "final_completion_verification.json"
    final_path.write_text(json.dumps(output, indent=2) + "\n", "utf-8")
    audit.update(status="complete", observed_at_utc=output["observed_at_utc"], canonical_design_sha256=doc_sha,
                 summary=dict(Counter(r["status"] for r in audit["requirements"])),
                 final_completion_verification=str(final_path.relative_to(ROOT)), final_completion_verification_sha256=sha(final_path))
    (BASE / "completion_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", "utf-8")
    observation = read(BASE / "execution_observation.json")
    observation.update(status="complete", observed_at_utc=output["observed_at_utc"], live_handles=[],
                       continuous_active_raw_progress=[], continuous_active_run_count=0, continuous_active_query_count_range=[],
                       next="Full original scope and scientific review complete; no pending experiment or review stage.",
                       continuous_scientific_review_complete=True, final_completion_verification=str(final_path.relative_to(ROOT)))
    (BASE / "execution_observation.json").write_text(json.dumps(observation, indent=2) + "\n", "utf-8")
    print("verified_complete:23/23 requirements;10880 finite+100 continuous;472 contrasts;27 figure sets;102 tests;0 pending", flush=True)


if __name__ == "__main__":
    main()
