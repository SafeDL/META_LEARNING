"""Reconcile the complete development matrix without additional SUT calls."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.audit_collision_results import RawEvidence, compare, reconstruct, require
from benchmark.build_collision_models import model_path
from benchmark.collision_runs import MAIN_METHODS
from benchmark.construction import load_construction
from benchmark.develop_collision import OUTPUT, REPRESENTATIVE_SCENARIOS, SETTINGS
from benchmark.initialization import initial_indices
from benchmark.protocol import DATA_ROOT, DEFAULT, ROOT, method_contract
from benchmark.scenarios import DOMAINS, sample
from controllers.highway.catalog import SUTS
from methods.baselines.response_baselines.collision import METHODS
from methods.common.response import archive_fingerprints


def verify(path, metadata, history, raw, core_hashes):
    content = path.read_bytes()
    saved = json.loads(content)
    specification, result, cost = saved["specification"], saved["result"], saved["cost"]
    compare(specification["metadata"], metadata, "registered development task")
    construction = specification["construction"]
    require(construction["budget"] == result["actual_queries"] == SETTINGS["budget"], "Development budget differs")
    require(construction["initial_count"] == SETTINGS["initial_count"] and construction["horizon"] == DEFAULT.horizon,
            "Development initialization or fixed horizon differs")
    method, kind = metadata["method"], metadata["scenario"]
    require(construction["method"] == method and construction["algorithm_seed"] == metadata["algorithm_seed"],
            "Development selector identity differs")
    sources = [] if method in ("random", "space_filling", "gp_collision") else history
    require(len(construction["history_ids"]) == len(sources) and set(construction["history_ids"]) == {a.source_id for a in sources},
            "Development historical source identity differs")
    compare(construction["history_fingerprints"], archive_fingerprints(sources), "development history fingerprints")
    by_id = {a.source_id: a for a in sources}
    compare(construction["history_contracts"], [by_id[name].provenance["contract"] for name in construction["history_ids"]],
            "development historical execution contracts")
    checkpoint = (model_path("history_response", "all", kind) if method in MAIN_METHODS else
                  model_path(method, "all", kind) if method in ("fsbo", "pooled_ft") else None)
    if checkpoint is not None:
        require(Path(construction["checkpoint"]["path"]).resolve() == checkpoint.resolve(), "Wrong development model path")
        require(construction["checkpoint"]["sha256"] == core_hashes[str(checkpoint.resolve())], "Wrong development model contents")
    else:
        require("checkpoint" not in construction, "Unexpected learned checkpoint")
    bank_root = DATA_ROOT / "physical/development" / kind / metadata["target_id"]
    require(Path(cost["physical_bank"]).resolve() == bank_root.resolve(), "Wrong development physical bank")
    manifest, *_ = raw.bank(bank_root)
    require(manifest["sut_id"] == metadata["target_id"] and manifest["environment_seed"] == metadata["environment_seed"],
            "Development execution identity differs")
    from dataclasses import asdict
    compare(manifest["scenarios"], [asdict(s) for s in sample(kind, DEFAULT.candidate_count, "development")],
            "registered development coordinates")
    compare(manifest["contract"], construction["bank_contract"], "development physical contract")
    candidates = manifest["scenarios"][:SETTINGS["candidate_count"]]
    compare(construction["candidate_ids"], [s["scenario_id"] for s in candidates], "development public candidate subset")
    queries, ledger = result["queries"], saved["query_ledger"]
    require(len(queries) == len(ledger) == SETTINGS["budget"], "Incomplete development paid ledger")
    selected = [q["index"] for q in queries]
    require(all(type(index) is int and 0 <= index < len(candidates) for index in selected)
            and len(set(selected)) == len(selected), "Invalid or repeated development query")
    compare(result["selected_indices"], selected, "development selected order")
    compare(selected[:SETTINGS["initial_count"]], initial_indices(len(candidates), SETTINGS["initial_count"], metadata["algorithm_seed"]),
            "development common initialization")
    scenes, facts = [], []
    for position, (query, record) in enumerate(zip(queries, ledger), 1):
        require(record["call"] == position and record["index"] == query["index"] and record["status"] == "complete"
                and record["account"] == "target", "Development paid ledger order differs")
        require(record["purpose"] == ("initial" if position <= SETTINGS["initial_count"] else "target"),
                "Development initialization was not charged")
        evidence = raw.execution(bank_root, query["index"])
        require(record["scenario_id"] == evidence["scenario"]["scenario_id"], "Development ledger coordinate differs")
        compare(query["feedback"], evidence["feedback"], "development selector feedback")
        compare(record["feedback"], evidence["feedback"], "development disclosed feedback")
        scenes.append(evidence["scenario"])
        facts.append(evidence["facts"])
    metrics = reconstruct(queries, scenes, facts)
    compare(saved["metrics"], metrics, "independent development metrics")
    require(cost["execution_mode"] == "replay" and cost["actual_new_physical_calls"] == 0, "Development replay cost differs")
    compare(cost["equivalent_selected_execution_s"], metrics["equivalent_execution_s"], "development selected physical cost")
    compare(cost["physical_bank_calls"], manifest["attempted_calls"], "development bank paid calls")
    compare(cost["physical_bank_execution_s"], manifest["physical_execution_s"], "development bank execution cost")
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), **metadata, **metrics}


def main():
    started = time.perf_counter()
    core_path = Path(__file__).with_name("core_model_verification.json")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    require(core["status"] == "complete" and core["verified_models"] == 12, "Core model verification is incomplete")
    core_hashes = {str(Path(row["checkpoint"]).resolve()): row["checkpoint_sha256"] for row in core["models"]}
    for path, digest in core_hashes.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "Core model changed after validation replay")
    source_contract = method_contract()
    rows, pending, errors, raw, expected = [], [], [], RawEvidence(), 0
    for kind in DOMAINS:
        history, _ = load_construction(kind)
        for sut in SUTS:
            if sut.role != "development":
                continue
            methods = ["history_response", *METHODS]
            if kind == REPRESENTATIVE_SCENARIOS[sut.family]:
                methods += ["no_response", "fixed_weights", "no_target", "greedy"]
            for seed in SETTINGS["seeds"]:
                for method in methods:
                    expected += 1
                    path = OUTPUT / sut.sut_id / kind / f"{method}_{seed}.json"
                    if not path.exists():
                        pending.append(str(path))
                        continue
                    metadata = {"target_id": sut.sut_id, "lineage": sut.lineage, "family": sut.family,
                                "scenario": kind, "condition": "development", "method": method,
                                "environment_seed": DEFAULT.environment_seeds[0], "algorithm_seed": seed}
                    try:
                        rows.append(verify(path, metadata, history, raw, core_hashes))
                    except Exception as error:
                        errors.append({"path": str(path), "error": repr(error)})
                    if (len(rows) + len(errors)) % 24 == 0:
                        print("reconciled development runs", len(rows), "/ 288", flush=True)
    require(expected == 288, "Development registered matrix changed")
    report_path, report_checked = OUTPUT / "report.json", False
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            require(report["status"] == "complete" and report["data_role"] == "development_only", "Incomplete development report")
            require(not pending and not errors, "Development report precedes successful full reconciliation")
            compare(report["settings"], SETTINGS, "development report settings")
            compare(report["source_contract"], source_contract, "development report code")
            expected_hashes = {str(Path(path).relative_to(ROOT)): digest for path, digest in core_hashes.items()}
            compare(report["model_hashes"], expected_hashes, "development report core models")
            require(len(report["runs"]) == len(rows) == 288, "Incomplete development result index")
            by_path = {str(Path(row["path"]).resolve()): row for row in report["runs"]}
            require(len(by_path) == 288, "Duplicate development result index entry")
            for row in rows:
                compare(by_path[str(Path(row["path"]).resolve())], {k: v for k, v in row.items() if k != "sha256"}, "development result index")
            report_checked = True
        except Exception as error:
            errors.append({"path": str(report_path), "error": repr(error)})
    require(method_contract() == source_contract, "Construction code changed during development reconciliation")
    result = {"status": "failed" if errors else "complete" if not pending and report_checked else "partial",
              "scope": "All small-development runs only; final method comparisons remain separate.",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "expected_runs": 288,
              "verified_runs": len(rows), "pending": pending, "errors": errors, "report_checked": report_checked,
              "additional_physical_calls": 0, "unique_raw_executions_checked": len(raw.executions),
              "verification_wall_s": time.perf_counter() - started,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "raw_metric_validator_sha256": hashlib.sha256((ROOT / "benchmark/audit_collision_results.py").read_bytes()).hexdigest(),
              "core_verification_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(), "runs": rows}
    output = Path(__file__).with_name("development_verification.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"], len(rows), "/ 288 runs;", len(errors), "errors", flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
