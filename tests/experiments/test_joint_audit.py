import copy
import gzip
import json
import os
from dataclasses import asdict

import numpy as np
import pytest

from experiments.audit_joint import audit
from experiments.failure_metrics import summarize
from experiments.joint_validation import REGIMES
from experiments.rss import measurement_contract, trajectory_margin
from experiments.scenarios import candidate_pool
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.storage import read_json, write_json


def compressed(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)


def fixture(tmp_path, suite="pilot", *, failures=True, shared_initialization=True):
    root, truth = tmp_path / suite, tmp_path / "truth"
    kind, sut_id, seed = "car_following", "test_ads", 11
    replicate = 20 if suite == "sequential" else 30 if suite == "sensitivity" else 10
    scenes = candidate_pool(kind, 4, replicate=replicate)
    labels = [failures, failures, failures, False]
    bank = truth / (suite if suite in ("sequential", "sensitivity") else "fixed") / sut_id / kind
    contract = {"fixture": "unit-test records, no simulation"}
    write_json(
        bank / "manifest.json",
        {
            "status": "complete",
            "environment_seed": 81001,
            "contract": contract,
            "scenarios": [asdict(scene) for scene in scenes],
        },
    )
    records = []
    for index, scene in enumerate(scenes):
        actor = {"heading": 0, "speed": 0, "length": 5, "width": 2, "y": 0}
        trajectory = [
            {
                "time_s": 0,
                "collision": labels[index],
                "actors": [
                    {**actor, "actor_id": "ego", "x": 0},
                    {**actor, "actor_id": "lead", "x": 4 if labels[index] else 50},
                ],
            }
        ]
        margin = trajectory_margin(trajectory)["minimum_margin"]
        feedback = QueryFeedback(
            scene.scenario_id,
            labels[index],
            SafetyMeasurements(0, 0, rss_margin=margin),
            2.0,
            rss_contract=measurement_contract(),
        ).to_dict()
        record = {"scenario": asdict(scene), "feedback": feedback, "trajectory": trajectory}
        compressed(bank / "executions" / f"{index:06d}.json.gz", record)
        records.append(record)
    methods = list(REGIMES) if suite == "sequential" else ["joint"]
    sizes = [3, 4] if suite == "sensitivity" else [4]
    protocol = {
        "suite": suite,
        "config": {
            "candidate_count": 4,
            "scenarios": [kind],
            "algorithm_seeds": [seed],
            "budget": 3,
            "initial_count": 1,
            "environment_seed": 81001,
        },
        "systems": [{"sut_id": sut_id, "version": "fixture", "role": "independent"}],
        "main_methods": ["joint"],
        "controls": [],
        "candidate_counts": {kind: 4},
        "candidate_replicate": replicate,
    }
    if suite == "sensitivity":
        protocol["candidate_sizes"] = sizes
        protocol["initial_pool_size"] = min(sizes)
    write_json(root / "reports/protocol.json", protocol)
    rows = []
    for size in sizes:
        for method in methods:
            selected = np.random.default_rng(seed).permutation(size)[:3].tolist()
            if suite == "sensitivity" and shared_initialization:
                initial = np.random.default_rng(seed).permutation(min(sizes))[:1].tolist()
                selected = initial + [i for i in selected if i not in initial]
                selected = selected[:3]
            queries, ledger, executions = [], [], []
            for position, index in enumerate(selected, 1):
                record = records[index]
                feedback = record["feedback"]
                queries.append(
                    {
                        "query": position,
                        "index": index,
                        "feedback": feedback,
                        "prequery_probability": 0.5,
                        "selection_s": 0.2,
                        "adaptation_s": 0.3,
                    }
                )
                ledger.append(
                    {
                        "call": position,
                        "index": index,
                        "feedback": feedback,
                        "status": "complete",
                        "account": "target",
                        "purpose": "initial" if position == 1 else "target",
                        "rss_compute_s": 0.1,
                        "wall_s": 0.01,
                    }
                )
                executions.append({**record, "query": position, "record_id": f"paid:{position}"})
            metrics = summarize(
                scenes[:size],
                labels[:size],
                queries,
                [records[i]["trajectory"] for i in selected],
                initial_count=1,
                query_ledger=ledger,
            )
            spec = {
                "budget": 3,
                "initial_count": 1,
                "seed": seed,
                "environment_seed": 81001,
                "candidates": [asdict(scene) for scene in scenes[:size]],
                "bank_contract": contract,
                "target_model_version": "base",
                "encoder_version": "encoder",
                "prediction_snapshot_counts": [0, 1, 3],
            }
            if suite == "sensitivity":
                spec.update(initial_pool_size=min(sizes), initial_indices=selected[:1])
            result = {
                "selected_indices": selected,
                "queries": queries,
                "resources": {},
                "actual_queries": 3,
                "maximum_queries": 3,
                "termination": "budget_exhausted",
                "prediction_snapshots": [
                    {
                        "paid_queries": n,
                        "observed_indices": selected[:n],
                        "probabilities": [0.5] * size,
                    }
                    for n in (0, 1, 3)
                ],
            }
            destination = root / "methods" / method / str(size)
            archive_path = destination / "completed.json.gz"
            if suite in ("fixed", "mechanisms"):
                destination = tmp_path / "reused" / method / str(size)
                archive_path = destination / "completed.json.gz"
            if suite == "sequential":
                archive_path = (
                    root
                    / "methods"
                    / method
                    / str(seed)
                    / "completed"
                    / f"01_{sut_id}_{kind}.json.gz"
                )
            write_json(
                destination / "result.json",
                {
                    "specification": spec,
                    "result": result,
                    "metrics": metrics,
                    "query_ledger": ledger,
                    "evaluation": {"candidate_collisions": labels[:size]},
                },
            )
            compressed(
                archive_path,
                {
                    "completed": True,
                    "sut_id": sut_id,
                    "sut_version": "fixture",
                    "original_role": "independent",
                    "environment_seed": 81001,
                    "execution_contract": contract,
                    "rss_contract": measurement_contract(),
                    "model_snapshot": {
                        k: spec[k] for k in ("target_model_version", "encoder_version")
                    }
                    if suite == "sequential"
                    else spec,
                    "attempts": ledger,
                    "executions": executions,
                    "completed_order": 1,
                    "branch_id": f"{method}:{seed}",
                },
            )
            row = {
                "sut_id": sut_id,
                "scenario": kind,
                "seed": seed,
                "regime" if suite == "sequential" else "method": method,
                "metrics": metrics,
                "resources": {},
                "result_path": os.path.relpath(destination / "result.json", root),
            }
            if suite == "sequential":
                row.update(ordinal=1, branch_id=f"{method}:{seed}")
            if suite == "sensitivity":
                row["candidate_count"] = size
            rows.append(row)
    write_json(root / "reports/summary.json", {"status": "complete", "runs": rows})
    return root, truth, bank


@pytest.mark.parametrize("suite", ["pilot", "fixed", "mechanisms", "sequential", "sensitivity"])
def test_audit_handles_reuse_branches_and_nested_truth(tmp_path, suite):
    root, truth, _ = fixture(tmp_path, suite)
    report = audit(root, data_root=truth)
    expected = 3 if suite == "sequential" else 2 if suite == "sensitivity" else 1
    assert report["status"] == "complete"
    assert report["checked_runs"] == expected
    assert report["checked_paid_queries"] == 3 * expected


def test_zero_failure_metrics_remain_undefined(tmp_path):
    root, truth, _ = fixture(tmp_path, failures=False)
    assert audit(root, data_root=truth)["status"] == "complete"
    metrics = read_json(root / "reports/summary.json")["runs"][0]["metrics"]
    assert metrics["unique_collisions"] == 0
    assert metrics["collision_recall"] is None
    assert metrics["pair_distance"] is None


def test_audit_rejects_scale_specific_initial_queries(tmp_path):
    root, truth, _ = fixture(tmp_path, "sensitivity", shared_initialization=False)
    with pytest.raises(AssertionError, match="Saved-run verification failed"):
        audit(root, data_root=truth)


@pytest.mark.parametrize(
    "field",
    [
        "discovery_curve",
        "recall_curve",
        "distance_curves",
        "queries_to_recall",
        "collision_nll",
        "equivalent_testing_curve_s",
    ],
)
def test_audit_rejects_matching_but_wrong_summary_and_result_metrics(tmp_path, field):
    root, truth, _ = fixture(tmp_path)
    summary = read_json(root / "reports/summary.json")
    path = root / summary["runs"][0]["result_path"]
    saved = read_json(path)
    metrics = saved["metrics"]
    if field == "distance_curves":
        metrics[field]["pair_distance"][-1] += 0.1
    elif field == "queries_to_recall":
        metrics[field]["1.0"] = 1
    elif field == "collision_nll":
        metrics[field] += 0.1
    else:
        metrics[field][-1] += 0.1
    summary["runs"][0]["metrics"] = copy.deepcopy(metrics)
    write_json(path, saved)
    write_json(root / "reports/summary.json", summary)
    with pytest.raises(AssertionError, match="Saved-run verification failed"):
        audit(root, data_root=truth)


@pytest.mark.parametrize("fault", ["truth_event", "paid_rss", "unpaid_archive", "query_prefix"])
def test_audit_rejects_record_corruption(tmp_path, fault):
    root, truth, bank = fixture(tmp_path)
    row = read_json(root / "reports/summary.json")["runs"][0]
    path = root / row["result_path"]
    saved = read_json(path)
    archive_path = path.parent / "completed.json.gz"
    with gzip.open(archive_path, "rt", encoding="utf-8") as stream:
        archive = json.load(stream)
    if fault == "truth_event":
        # Candidate 2 is outside this seed's paid prefix: the truth denominator still matters.
        record_path = bank / "executions/000002.json.gz"
        with gzip.open(record_path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        record["trajectory"][0]["collision"] = False
        compressed(record_path, record)
    elif fault == "paid_rss":
        for feedback in (
            saved["result"]["queries"][0]["feedback"],
            saved["query_ledger"][0]["feedback"],
            archive["attempts"][0]["feedback"],
            archive["executions"][0]["feedback"],
        ):
            feedback["safety"]["rss_margin"] += 0.25
        write_json(path, saved)
        compressed(archive_path, archive)
    elif fault == "unpaid_archive":
        archive["executions"].append(copy.deepcopy(archive["executions"][0]))
        compressed(archive_path, archive)
    else:
        saved["result"]["prediction_snapshots"][-1]["observed_indices"].reverse()
        write_json(path, saved)
    with pytest.raises(AssertionError, match="Saved-run verification failed"):
        audit(root, data_root=truth)


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_complete_summary_requires_each_registered_task_once(tmp_path, fault):
    root, truth, _ = fixture(tmp_path, "sequential")
    path = root / "reports/summary.json"
    summary = read_json(path)
    if fault == "missing":
        summary["runs"].pop()
    else:
        summary["runs"][-1] = copy.deepcopy(summary["runs"][0])
    write_json(path, summary)
    with pytest.raises(AssertionError):
        audit(root, data_root=truth)
