"""Evaluator-owned raw-pool, paid-feedback and causal acquisition verification."""

import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from testing_methods.history_transfer.domain_discovery import aggregate_history
from testing_methods.history_transfer.domain_training import load_model

from .collision_metrics import collision_recall, summarize
from .discovery_config import read_json, write_json
from .domain_main_validation import reconstruct
from .executions import ExecutionBank
from .independent_data_audit import verify_trace
from .independent_models import load_sources, model_root
from .independent_preparation import OUTPUT
from .independent_protocol import (
    CONTROLS,
    REUSED_METHODS,
    WORKERS,
    canonical_task,
    held_family,
    jobs,
    run_path,
    target_bank_path,
    validate_protocol,
)
from .independent_target_banks import raw_bank_sha256
from .independent_validation import bank_audit_path, first_jobs, public_task, selected_reuse_jobs
from .initialization import initial_indices
from .protocol import ROOT
from .query import PaidExecution
from .scenarios import sample
from .schema import QueryFeedback
from .source_registration import maintained_path


def check_attempts(bank):
    attempts = {int(path.stem): read_json(path) for path in (bank.root / "attempts").glob("*.json")}
    known = {s.scenario_id for s in bank.candidates}
    if any(
        r["call"] != call
        or r["sut_id"] != bank.manifest["sut_id"]
        or r["environment_seed"] != bank.manifest["environment_seed"]
        or r["scenario_id"] not in known
        or r["status"] not in ("running", "complete", "failed")
        for call, r in attempts.items()
    ):
        raise ValueError(
            "An independent physical attempt belongs to a different source or candidate"
        )
    completed = [r for r in attempts.values() if r["status"] == "complete"]
    failed = [r for r in attempts.values() if r["status"] == "failed"]
    running = [r for r in attempts.values() if r["status"] == "running"]
    manifest = bank.manifest
    if (
        manifest["attempted_calls"] != len(attempts)
        or manifest["successful_calls"] != len(completed)
        or manifest["failed_or_interrupted_calls"] != len(failed) + len(running)
        or manifest["interrupted_call_duration_unknown"] != len(running)
    ):
        raise ValueError("The independent physical cost ledger hides an attempted execution")
    for key, expected in (
        ("physical_execution_s", sum(r["execution_s"] for r in completed)),
        ("total_successful_call_wall_s", sum(r["wall_s"] for r in completed)),
        ("sut_initialization_s", sum(r["wall_s"] - r["execution_s"] for r in completed)),
        ("failed_call_wall_s", sum(r.get("wall_s", 0) for r in failed)),
    ):
        np.testing.assert_allclose(manifest[key], expected, atol=1e-8, rtol=1e-10)
    scheduled = [
        r
        for r in (json.loads(line) for line in (bank.root / "calls.jsonl").read_text().splitlines())
        if r["status"] == "scheduled"
    ]
    if len({r["call"] for r in scheduled}) != len(scheduled) or not set(attempts) <= {
        r["call"] for r in scheduled
    }:
        raise ValueError("An independent physical attempt was not uniquely scheduled")
    for row in scheduled:
        if (
            not 0 <= row["index"] < 512
            or row["scenario_id"] != bank.candidates[row["index"]].scenario_id
        ):
            raise ValueError("The independent physical schedule names another candidate")
    return attempts


def prepare_bank(target_id, kind):
    task = {"target_id": target_id, "scenario": kind}
    protocol = read_json(OUTPUT / "protocol.json")
    index = read_json(OUTPUT / "target_bank_index.json")
    expected = next(
        r for r in index["banks"] if r["target_id"] == target_id and r["scenario"] == kind
    )
    bank = ExecutionBank(target_bank_path(task))
    if (
        bank.candidates != sample(kind, 512, "new")
        or bank.manifest["candidate_count"] != 512
        or bank.manifest["environment_seed"] != 81001
        or bank.manifest["sut_id"] != target_id
        or bank.manifest["contract"] != protocol["sut_contracts"][target_id][kind]
        or hashlib.sha256((bank.root / "manifest.json").read_bytes()).hexdigest()
        != expected["manifest_sha256"]
        or raw_bank_sha256(bank.root) != expected["raw_execution_files_sha256"]
    ):
        raise ValueError("An independent raw bank differs from its fixed physical construction")
    attempts = check_attempts(bank)
    feedbacks, events, partners, retained, initial = [], [], [], set(), hashlib.sha256()
    for position, scenario in enumerate(bank.candidates):
        with gzip.open(
            bank.root / "executions" / f"{position:06d}.json.gz", "rt", encoding="utf-8"
        ) as stream:
            payload = json.load(stream)
        raw = PaidExecution(QueryFeedback.from_dict(payload["feedback"]), payload["trajectory"])
        attempt = attempts[payload["call"]]
        if (
            payload["scenario"] != asdict(scenario)
            or raw.feedback.scenario_id != scenario.scenario_id
            or attempt["status"] != "complete"
            or attempt["scenario_id"] != scenario.scenario_id
            or attempt["sut_id"] != target_id
            or attempt["environment_seed"] != 81001
            or attempt["execution_s"] != raw.feedback.execution_s
        ):
            raise ValueError("A retained independent response lacks its matching physical attempt")
        retained.add(payload["call"])
        frames = raw.trajectory
        times = np.asarray([f["time_s"] for f in frames])
        if (
            times[0] != 0
            or not np.allclose(np.diff(times), 0.05, atol=1e-9)
            or times[-1] > 10 + 1e-9
            or raw.feedback.collision != any(f["collision"] for f in frames)
        ):
            raise ValueError("The independent execution timebase or collision event differs")
        if (
            not raw.feedback.collision
            and not raw.feedback.safety.violations
            and abs(times[-1] - 10) > 1e-9
        ):
            raise ValueError("A safe independent execution ended before the common horizon")
        for number, frame in enumerate(frames):
            command = frame["requested_control"]
            if (
                not -8 - 1e-9 <= command["acceleration"] <= 3 + 1e-9
                or abs(command["steering"]) > 0.45 + 1e-9
            ):
                raise ValueError("An independent ego command exceeds the common actuator limits")
            if number and command != frames[number - 1]["requested_control"] and (number - 1) % 4:
                raise ValueError(
                    "An independent ego command changed outside the common decision ticks"
                )
        disclosed = verify_trace(scenario, raw, payload["events"])
        feedbacks.append(disclosed.feedback.to_dict())
        events.append(payload["events"])
        partners.append(next((f["collision_partner"] for f in frames if f["collision"]), None))
        initial.update(json.dumps(frames[0]["actors"], sort_keys=True).encode())
    if len(retained) != 512:
        raise ValueError("Independent coordinates reused a physical attempt")
    certificate = {
        "status": "passed",
        "target_id": target_id,
        "scenario": kind,
        "candidate_ids": [s.scenario_id for s in bank.candidates],
        "feedback": feedbacks,
        "events": events,
        "collision_partners": partners,
        "raw_execution_files_sha256": expected["raw_execution_files_sha256"],
        "initial_worlds_sha256": initial.hexdigest(),
        "physical_manifest": bank.manifest,
        "pool_unique_collisions": collision_recall(
            bank.candidates, [f["collision"] for f in feedbacks], []
        )["pool_unique_collisions"],
        "permission": "evaluator only; feedback row is disclosed after its metered query",
    }
    write_json(bank_audit_path(task), certificate)
    return {
        k: v
        for k, v in certificate.items()
        if k
        not in ("candidate_ids", "feedback", "events", "collision_partners", "physical_manifest")
    }


def prepare_banks():
    validate_protocol()
    index = read_json(OUTPUT / "target_bank_index.json")
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(prepare_bank, r["target_id"], r["scenario"]) for r in index["banks"]]
        for future in as_completed(futures):
            rows.append(future.result())
            print("independent_bank_audit", len(rows), "/ 40", flush=True)
    for kind in {r["scenario"] for r in rows}:
        if len({r["initial_worlds_sha256"] for r in rows if r["scenario"] == kind}) != 1:
            raise ValueError(
                "Independent objects saw different initial worlds for identical public coordinates"
            )
    write_json(
        OUTPUT / "evaluation_bank_audit.json",
        {
            "status": "passed",
            "banks": rows,
            "checked_executions": 512 * len(rows),
            "new_physical_calls": 0,
        },
    )
    return rows


def check_main(result, task, method, candidates):
    mode = "H1F1" if method == "gcrt" else method
    sources = load_sources(task["scenario"], held_family(task))
    if method.startswith("H0"):
        sources = aggregate_history(sources)
    if (
        result["method"] != mode
        or result["component_names"] != [*sources["names"], "independent_target", "physical_target"]
        or result["physical_mass"] != 0.1
        or result["beta"] != 1
        or result["online_feature_precision"] != "float64"
    ):
        raise ValueError("A final H-by-F run used a different model mechanism")
    scores = np.log(np.r_[0.75 * sources["weights"], 0.25])
    for step, row in enumerate(result["queries"]):
        expected = np.r_[0.9 * np.exp(scores - logsumexp(scores)), 0.1]
        np.testing.assert_allclose(row["prequery_component_weights"], expected, atol=1e-12)
        np.testing.assert_allclose(
            np.dot(expected, row["prequery_component_probabilities"]),
            row["prequery_probability"],
            atol=1e-12,
        )
        if row["conditioning_queries"] != (step if mode.endswith("F1") else 0):
            raise ValueError("An H-by-F run conditioned on disallowed feedback")
        if mode.endswith("F1"):
            p = np.asarray(row["prequery_component_probabilities"][:-1])
            likelihood = np.log(np.maximum(p if row["feedback"]["collision"] else 1 - p, 1e-15))
            np.testing.assert_allclose(row["prequery_log_likelihood"], likelihood, atol=1e-12)
            scores += likelihood
            if any(
                d["gradient_max"] > 1e-10 for d in (*row["conditional_modes"], row["physical_mode"])
            ):
                raise ValueError("A final conditional Bernoulli mode failed to converge")
        elif row["prequery_log_likelihood"] is not None:
            raise ValueError("A final F0 run updated its target reliability")
        np.testing.assert_allclose(
            row["component_weights_after_query"],
            np.r_[0.9 * np.exp(scores - logsumexp(scores)), 0.1],
            atol=1e-12,
        )
    base, _ = load_model(
        model_root("gcrt", task["scenario"], held_family(task)) / "target_kernel.pt"
    )
    for number in (9, 64, 128):
        row = result["queries"][number - 1]
        index, probability, weights, components = reconstruct(
            candidates,
            base,
            sources,
            result["queries"][: number - 1],
            mode,
            initial_indices(512, 8, task["algorithm_seed"]),
        )
        if index != row["index"]:
            raise ValueError("Independent full-pool acquisition differs in final validation")
        np.testing.assert_allclose(probability[index], row["prequery_probability"], atol=1e-7)
        np.testing.assert_allclose(
            components[:, index], row["prequery_component_probabilities"], atol=1e-7
        )
        np.testing.assert_allclose(weights, row["prequery_component_weights"], atol=1e-12)
    return 3


def check_record(saved, task, method, bank, executions, private, digest):
    result = saved["result"]
    queries, ledger = result["queries"], saved["query_ledger"]
    selected = [q["index"] for q in queries]
    initial = initial_indices(512, 8, task["algorithm_seed"])
    setting = saved["specification"]
    if (
        setting["task"] != task
        or setting["method"] != method
        or setting["protocol_sha256"] != digest
        or setting["bank_contract"] != bank.manifest["contract"]
        or setting["candidate_ids"] != private["candidate_ids"]
        or setting["bank_audit_sha256"]
        != hashlib.sha256(bank_audit_path(task).read_bytes()).hexdigest()
        or len(queries) != 128
        or len(ledger) != 128
        or len(set(selected)) != 128
        or selected[:8] != initial
        or saved["initial_indices"] != initial
        or result["selected_indices"] != selected
        or result["actual_queries"] != 128
        or result["maximum_queries"] != 128
        or result["termination"] != "budget_exhausted"
    ):
        raise ValueError(
            "A final run differs from its fixed independent task, initialization or paid budget"
        )
    for number, (query, record) in enumerate(zip(queries, ledger), 1):
        index = query["index"]
        expected = QueryFeedback.from_dict(private["feedback"][index]).to_dict()
        if (
            query["query"] != number
            or record["call"] != number
            or record["index"] != index
            or record["status"] != "complete"
            or record["account"] != "target"
            or record["scenario_id"] != bank.candidates[index].scenario_id
            or record["purpose"] != ("initial" if number <= 8 else "target")
            or record["feedback"] != query["feedback"]
            or QueryFeedback.from_dict(query["feedback"]).to_dict() != expected
        ):
            raise ValueError("A final run used a scalar not attached to its metered execution")
    metrics = {
        **summarize(bank.candidates, queries, [executions[i].trajectory for i in selected]),
        **collision_recall(bank.candidates, [f["collision"] for f in private["feedback"]], queries),
    }
    if saved["metrics"] != json.loads(json.dumps(metrics, allow_nan=False)):
        raise ValueError(
            "Final metrics differ from the complete collision denominator or paid trajectories"
        )
    if any(b < a for a, b in zip(metrics["discovery_curve"], metrics["discovery_curve"][1:])):
        raise ValueError("A final discovery curve decreased")
    if any(value is not None and not 0 <= value <= 1 for value in metrics["recall_curve"]):
        raise ValueError("A final recall escaped its complete-pool bounds")
    checks = (
        check_main(result, task, method, bank.candidates) if method in ("gcrt", *CONTROLS) else 0
    )
    if method == "random" and selected != initial_indices(512, 512, task["algorithm_seed"])[:128]:
        raise ValueError("Final random selection differs from its label-independent permutation")
    if method == "av_bo_rf_ei" and [r["paid_observations"] for r in result["tuning"]] != [50, 100]:
        raise ValueError("Final RF-EI tuned on another paid prefix")
    if method == "scenario_fuzz" and result["online_classifier_updates"] != 0:
        raise ValueError("Final ScenarioFuzz retrained before the declared threshold")
    return {"queries": 128, "independent_main_choices": checks}


def audit_group(target_id, kind, rows, digest):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        task_key = {"target_id": target_id, "scenario": kind}
        bank = ExecutionBank(target_bank_path(task_key))
        private = read_json(bank_audit_path(task_key))
        if raw_bank_sha256(bank.root) != private["raw_execution_files_sha256"]:
            raise ValueError("The independent raw pool changed after its full physics audit")
        executions = [bank(s) for s in bank.candidates]
        checks = []
        for row in rows:
            task = {k: row[k] for k in first_jobs()[0][0]}
            saved = read_json(run_path(task, row["method"]))
            checked = check_record(saved, task, row["method"], bank, executions, private, digest)
            if any(row[k] != v for k, v in saved["metrics"].items()):
                raise ValueError("The final run index differs from its actual saved metrics")
            checks.append({"path": row["path"], **checked})
        return checks


def run_key(task, method):
    return tuple(task.items()) + (("method", method),)


def check_reuse_references(index, expected, completed_paths, digest):
    completed_paths = {maintained_path(path) for path in completed_paths}
    expected_keys = {run_key(task, method) for task, method in expected}
    references = index["references"]
    actual = [run_key(row["task"], row["method"]) for row in references]
    if (
        index["status"] != "complete"
        or index["new_queries"] != 0
        or index["new_physical_calls"] != 0
        or len(actual) != len(expected_keys)
        or set(actual) != expected_keys
    ):
        raise ValueError("The source-agnostic reuse index has missing, repeated or charged records")
    for row in references:
        task, method = row["task"], row["method"]
        source = canonical_task(task, method)
        path = run_path(source, method)
        expected_path = str(path.relative_to(ROOT)).replace("\\", "/")
        if (
            method not in REUSED_METHODS
            or task["condition"] != "strict_leave_family"
            or row["source_task"] != source
            or maintained_path(row["source_path"]) != expected_path
            or expected_path not in completed_paths
            or row["source_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
            or row["protocol_sha256"] != digest
            or row["source_queries"] != 128
            or row["new_queries"] != 0
            or row["new_physical_calls"] != 0
        ):
            raise ValueError(
                "A reuse reference does not identify an exact, independently audited source run"
            )
        saved = read_json(path)
        if (
            saved["specification"]["task"] != source
            or saved["specification"]["method"] != method
            or saved["specification"]["protocol_sha256"] != digest
            or saved["specification"]["bank_audit_sha256"] != row["bank_audit_sha256"]
        ):
            raise ValueError(
                "A reuse reference changed target, feedback contract or source condition"
            )
        candidates = sample(task["scenario"], 512, "new")
        first, a = public_task(source, method, candidates, [])
        second, b = public_task(task, method, candidates, [])
        serialized = json.dumps(
            {**a, "candidates": [asdict(scene) for scene in candidates]},
            sort_keys=True,
            allow_nan=False,
        ).encode()
        if (
            first is not second
            or a != b
            or maintained_path(row["selector"].replace(".", "/")).replace("/", ".")
            != first.__module__ + "." + first.__name__
            or row["public_inputs_sha256"] != hashlib.sha256(serialized).hexdigest()
        ):
            raise ValueError(
                "A reused selector has different public inputs or uses the removed source family"
            )
    return len(references)


def audit(first_only=False):
    validate_protocol()
    index = read_json(OUTPUT / ("first.json" if first_only else "run_index.json"))
    expected = first_jobs() if first_only else jobs()
    expected_set = {run_key(t, m) for t, m in expected}
    actual = [run_key({k: r[k] for k in first_jobs()[0][0]}, r["method"]) for r in index["runs"]]
    if (
        index["status"] != "complete"
        or len(actual) != len(expected_set)
        or set(actual) != expected_set
    ):
        raise ValueError("The independent matrix lacks an exact registered counterpart")
    groups = sorted({(r["target_id"], r["scenario"]) for r in index["runs"]})
    digest = hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()
    checked = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(
                audit_group,
                target,
                kind,
                [r for r in index["runs"] if r["target_id"] == target and r["scenario"] == kind],
                digest,
            )
            for target, kind in groups
        ]
        for future in as_completed(futures):
            checked.extend(future.result())
            print("independent_result_audit", len(checked), "/", len(expected), flush=True)
    reuse = read_json(OUTPUT / ("first_reuse_index.json" if first_only else "reuse_index.json"))
    reuse_count = check_reuse_references(
        reuse, selected_reuse_jobs(first_only), {row["path"] for row in checked}, digest
    )
    report = {
        "status": "passed",
        "runs": len(checked),
        "queries": sum(r["queries"] for r in checked),
        "reused_references": reuse_count,
        "logical_comparison_rows": len(checked) + reuse_count,
        "independent_main_choices": sum(r["independent_main_choices"] for r in checked),
        "checks": checked,
        "new_physical_calls": 0,
        "source_conditions": ["all_sources", "strict_leave_family"],
    }
    write_json(OUTPUT / ("first_audit.json" if first_only else "audit.json"), report)
    return report
