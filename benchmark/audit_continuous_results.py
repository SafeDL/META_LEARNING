"""Independent read-only reconciliation of actual continuous-search calls."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from .audit_collision_results import compare, reconstruct, require, trace_facts
from .continuous_runs import CONFIGURATIONS, source_contract
from .continuous_selection import METHODS
from .continuous_space import initial_scenarios
from .protocol import DEFAULT, RESULT_ROOT, ROOT
from .scenarios import difficult, validate
from .schema import Scenario
from .studies import CONTINUOUS_TASKS
from controllers.highway.catalog import BY_ID


OUTPUT = ROOT / "results/benchmark/collision/continuous_result_audit.json"


def verify_run(path, metadata, *, budget=None, initial_count=None, ordinary_only=False):
    budget = DEFAULT.budget if budget is None else budget
    initial_count = DEFAULT.initial_count if initial_count is None else initial_count
    content = Path(path).read_bytes()
    run = json.loads(content)
    identity, result = run["specification"], run["result"]
    compare(identity["metadata"], metadata, "continuous registered task")
    require(identity["method"] == metadata["method"] and identity["algorithm_seed"] == metadata["algorithm_seed"],
            "Continuous selector identity differs")
    require(identity["budget"] == result["actual_queries"] == budget
            and identity["initial_count"] == initial_count and identity["ordinary_only"] is ordinary_only,
            "Continuous registered protocol differs")
    compare(identity["configuration"], CONFIGURATIONS[metadata["method"]], "continuous search configuration")
    compare(identity["source_contract"], source_contract(), "continuous frozen code")
    initial_arguments = (metadata["scenario"], metadata["algorithm_seed"], initial_count)
    initial = [asdict(s) for s in (initial_scenarios(*initial_arguments, ordinary_only=True) if ordinary_only
                                 else initial_scenarios(*initial_arguments))]
    compare(identity["initial_scenarios"], initial, "continuous common initialization")
    candidates = run["candidates"]
    compare(candidates[:initial_count], initial, "continuous initial candidate order")
    coordinates, identifiers = set(), set()
    for data in candidates:
        scene = Scenario(**data)
        validate(scene)
        require(not ordinary_only or not difficult(scene), "Development proposed a held-out difficult combination")
        require(scene.kind == metadata["scenario"], "Continuous candidate changed functional scenario")
        require(scene.scenario_id not in identifiers and scene.coordinate() not in coordinates,
                "Continuous candidate identifiers or coordinates were duplicated")
        identifiers.add(scene.scenario_id)
        coordinates.add(scene.coordinate())
    queries, ledger = result["queries"], run["query_ledger"]
    require(len(queries) == len(ledger) == budget, "Incomplete continuous query budget")
    compare(result["selected_indices"], [q["index"] for q in queries], "continuous selected order")
    compare(result["selected_indices"][:initial_count], list(range(initial_count)),
            "continuous initialization calls")
    root = Path(run["cost"]["physical_log"])
    require(root.resolve() == Path(path).with_suffix("").resolve(), "Continuous result refers to another physical log")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    compare(manifest["identity"], identity, "continuous physical manifest")
    require(manifest["environment_seed"] == metadata["environment_seed"]
            and manifest["sut_contract"]["object"]["sut_id"] == metadata["target_id"],
            "Continuous calls used another SUT or seed")
    require(len(list(root.glob("attempt_*.json"))) == len(list(root.glob("query_*.json.gz"))) == budget,
            "Continuous physical attempt or response count differs from the paid budget")
    seen, facts, scenes, hashes = set(), [], [], []
    physical_s, initialization_s = 0.0, 0.0
    for position, (query, record) in enumerate(zip(queries, ledger), 1):
        index = query["index"]
        require(type(index) is int and 0 <= index < len(candidates), "Unknown continuous candidate")
        scenario = candidates[index]
        require(record["call"] == position and record["index"] == index and record["status"] == "complete"
                and record["account"] == "target" and record["scenario_id"] == scenario["scenario_id"],
                "Continuous paid ledger identity differs")
        purpose = "repeat" if index in seen else "initial" if position <= initial_count else "target"
        require(record["purpose"] == purpose, "Continuous initialization or repeat was not charged")
        seen.add(index)
        attempt_path = root / f"attempt_{position - 1:04d}.json"
        response_path = root / f"query_{position - 1:04d}.json.gz"
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        require(attempt["query"] == position and attempt["status"] == "complete", "Incomplete continuous physical attempt")
        compare(attempt["scenario"], scenario, "continuous actual-call coordinate")
        with gzip.open(response_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        compare(payload["scenario"], scenario, "continuous raw coordinate")
        compare(payload["feedback"], query["feedback"], "continuous disclosed feedback")
        compare(payload["feedback"], record["feedback"], "continuous ledger feedback")
        compare(payload["feedback"]["execution_s"], attempt["execution_s"], "continuous actual-call cost")
        fact = trace_facts(scenario, payload["trajectory"])
        compare(payload["feedback"]["collision"], fact["collision"], "continuous raw collision")
        physical_s += attempt["execution_s"]
        initialization_s += attempt["wall_s"] - attempt["execution_s"]
        facts.append(fact)
        scenes.append(scenario)
        hashes.append((hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
                       hashlib.sha256(response_path.read_bytes()).hexdigest()))
    metrics = reconstruct(queries, scenes, facts)
    compare(run["metrics"], metrics, "continuous reconstructed metrics")
    cost = run["cost"]
    require(cost["execution_mode"] == "actual_on_demand" and cost["actual_physical_calls"] == budget,
            "Continuous physical call accounting differs")
    compare(cost["physical_execution_s"], physical_s, "continuous physical cost")
    compare(cost["sut_initialization_s"], initialization_s, "continuous SUT initialization cost")
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), **metadata, **metrics,
            "raw_evidence_sha256": hashlib.sha256(json.dumps(hashes).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    expected, verified, pending, errors = 0, [], 0, []
    for sut_id, kind in CONTINUOUS_TASKS:
        sut = BY_ID[sut_id]
        for seed in DEFAULT.algorithm_seeds:
            for env in DEFAULT.environment_seeds:
                for method in METHODS:
                    expected += 1
                    path = RESULT_ROOT / "continuous" / sut_id / kind / f"env_{env}" / f"{method}_{seed}.json"
                    if not path.exists():
                        pending += 1
                        continue
                    metadata = {"target_id": sut_id, "lineage": sut.lineage, "family": sut.family,
                                "scenario": kind, "condition": "continuous", "method": method,
                                "algorithm_seed": seed, "environment_seed": env}
                    try:
                        verified.append(verify_run(path, metadata))
                    except Exception as error:
                        errors.append({"path": str(path), "error": repr(error)})
                    print("reconstructed continuous runs", len(verified), flush=True)
    index_path, index_checked = RESULT_ROOT / "continuous/run_index.json", False
    frozen_checked = False
    if verified:
        try:
            from .evaluate_continuous import validate_frozen
            validate_frozen()
            frozen_checked = True
        except Exception as error:
            errors.append({"path": "data/collision/frozen_continuous.json", "error": repr(error)})
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            require(index["status"] == "complete" and len(index["runs"]) == expected, "Incomplete continuous result index")
            require(not pending and not errors, "Continuous index claims completion before reconstruction succeeds")
            by_path = {str(Path(r["path"]).resolve()): r for r in index["runs"]}
            require(len(by_path) == expected, "Duplicate continuous index entry")
            for row in verified:
                compare(by_path[str(Path(row["path"]).resolve())],
                        {k: v for k, v in row.items() if k not in ("sha256", "raw_evidence_sha256")}, "continuous result index")
            index_checked = True
        except Exception as error:
            errors.append({"path": str(index_path), "error": repr(error)})
    report = {"status": "failed" if errors else "complete" if not pending and index_checked and frozen_checked else "partial",
              "scope": "Final continuous runs, actual raw trajectories, geometry safety, query metrics and costs only; statistical interpretation is separate.",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "additional_physical_calls": 0,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "metric_validator_sha256": hashlib.sha256((ROOT / "benchmark/audit_collision_results.py").read_bytes()).hexdigest(),
              "expected_runs": expected, "verified_runs": len(verified), "pending_runs": pending,
              "index_checked": index_checked, "frozen_protocol_checked": frozen_checked, "errors": errors, "runs": verified}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(report["status"], len(verified), "/", expected, "continuous runs;", len(errors), "errors", flush=True)
    if errors or arguments.require_complete and report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
