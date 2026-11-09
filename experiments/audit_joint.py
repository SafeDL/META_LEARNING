"""Independently verify saved paid records and finite-pool metrics without running selectors."""

import gzip
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .joint_validation import OUTPUT, REGIMES
from .prepare_joint import DATASET_ROOT
from .rss import measurement_contract, trajectory_margin
from .scenarios import DOMAINS, candidate_pool
from .storage import read_json, read_trajectory, write_json


def read_execution(bank, index):
    with gzip.open(bank / "executions" / f"{index:06d}.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def verify_metrics(
    candidates, labels, selected, queries, ledger, metrics, initial_count, *, domain=None
):
    """Recompute formulas directly; do not call the production metric summarizer."""
    kind = candidates[0].kind
    assert metrics["scenario"] == kind
    assert metrics["candidate_count"] == len(candidates)
    assert metrics["queries"] == len(queries)
    assert metrics["initial_count"] == initial_count
    assert metrics["budget_fraction"] == len(queries) / len(candidates)
    counts = np.cumsum([labels[index] for index in selected]).tolist()
    assert metrics["discovery_curve"] == counts
    assert metrics["initial_collisions"] == counts[initial_count - 1]
    assert metrics["unique_collisions"] == counts[-1]
    assert metrics["collision_executions"] == counts[-1]
    assert metrics["post_initial_collisions"] == counts[-1] - counts[initial_count - 1]
    assert metrics["pool_unique_collisions"] == sum(labels)
    assert metrics["collision_recall"] == (counts[-1] / sum(labels) if sum(labels) else None)
    assert metrics["discovery_auc"] == sum(counts) / len(counts)
    found = [candidates[index] for index in selected if labels[index]]
    assert metrics["discovered_ids"] == [s.scenario_id for s in found]
    if len(found) < 2:
        assert metrics["pair_distance"] is None and metrics["nearest_neighbor_distance"] is None
    else:
        domain = DOMAINS[kind] if domain is None else domain
        width = len(domain.bounds) + len(domain.categories)
        matrix = np.array(
            [
                [
                    (
                        sum(
                            abs(a.parameters[name] - b.parameters[name]) / (upper - lower)
                            for name, (lower, upper) in domain.bounds.items()
                        )
                        + sum(
                            a.categories[name] != b.categories[name] for name in domain.categories
                        )
                    )
                    / width
                    for b in found
                ]
                for a in found
            ]
        )
        pair = sum(matrix[i, j] for i in range(len(found)) for j in range(i)) / (
            len(found) * (len(found) - 1) / 2
        )
        np.fill_diagonal(matrix, np.inf)
        nearest = float(matrix.min(axis=1).mean())
        assert math.isclose(metrics["pair_distance"], pair, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(
            metrics["nearest_neighbor_distance"], nearest, rel_tol=1e-12, abs_tol=1e-12
        )
    distance_prefixes = {"pair_distance": [], "nearest_neighbor_distance": []}
    for n in range(1, len(found) + 1):
        distance_prefixes["pair_distance"].append(
            None if n < 2 else float(np.triu(matrix[:n, :n], 1).sum() / (n * (n - 1) / 2))
        )
        distance_prefixes["nearest_neighbor_distance"].append(
            None if n < 2 else float(matrix[:n, :n].min(axis=1).mean())
        )
    for name, values in distance_prefixes.items():
        np.testing.assert_allclose(
            np.asarray(metrics["distances_by_discovery_count"][name], dtype=float),
            np.asarray(values, dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        expected_curve = [values[n - 1] if n else None for n in counts]
        np.testing.assert_allclose(
            np.asarray(metrics["distance_curves"][name], dtype=float),
            np.asarray(expected_curve, dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    failure_count = sum(labels)
    assert metrics["pool_failure_rate"] == failure_count / len(candidates)
    expected_recall = [n / failure_count if failure_count else None for n in counts]
    np.testing.assert_allclose(
        np.asarray(metrics["recall_curve"], dtype=float),
        np.asarray(expected_recall, dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    if failure_count:
        assert math.isclose(
            metrics["recall_auc"],
            sum(expected_recall) / len(expected_recall),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    else:
        assert metrics["recall_auc"] is None
    first = next((i for i, n in enumerate(counts, 1) if n), None)
    assert metrics["first_collision_query"] == first
    assert metrics["restricted_first_collision_queries"] == (first or len(queries))
    assert metrics["no_collision_within_budget"] == (first is None)
    assert metrics["queries_to_recall"] == {
        str(level): next(
            (
                i
                for i, value in enumerate(expected_recall, 1)
                if value is not None and value >= level
            ),
            None,
        )
        for level in (0.25, 0.5, 0.75, 1.0)
    }
    assert metrics["post_initial_curve"] == [
        n - counts[initial_count - 1] if i >= initial_count else None
        for i, n in enumerate(counts, 1)
    ]
    nll, brier = [], []
    for index, query in zip(selected, queries, strict=True):
        p = query.get("prequery_probability")
        if p is not None:
            assert math.isfinite(p) and 0 <= p <= 1
            brier.append((p - int(labels[index])) ** 2)
            clipped = min(1 - 1e-12, max(1e-12, p))
            nll.append(-math.log(clipped if labels[index] else 1 - clipped))
    assert metrics["prediction_count"] == len(nll)
    for name, values in (("collision_nll", nll), ("brier", brier)):
        if values:
            assert math.isclose(
                metrics[name], sum(values) / len(values), rel_tol=1e-12, abs_tol=1e-12
            )
        else:
            assert metrics[name] is None
    assert metrics["negative_rss_executions"] == sum(
        q["feedback"]["safety"]["rss_margin"] < 0 for q in queries
    )
    timing_values = {
        "execution": [q["feedback"]["execution_s"] for q in queries],
        "selection": [q.get("selection_s", 0.0) for q in queries],
        "adaptation": [q.get("adaptation_s", 0.0) for q in queries],
        "rss": [r.get("rss_compute_s", 0.0) for r in ledger],
        "replay": [r["wall_s"] for r in ledger],
    }
    for name, values in timing_values.items():
        assert all(math.isfinite(v) and v >= 0 for v in values)
        np.testing.assert_allclose(
            metrics["cumulative_costs_s"][name], np.cumsum(values), rtol=1e-12, atol=1e-12
        )
        total_key = {
            "execution": "equivalent_execution_s",
            "selection": "selection_s",
            "adaptation": "adaptation_s",
            "rss": "rss_compute_s",
            "replay": "replay_wall_s",
        }[name]
        assert math.isclose(metrics[total_key], sum(values), rel_tol=1e-12, abs_tol=1e-12)
    equivalent = np.cumsum(
        [
            sum(v)
            for v in zip(
                *(timing_values[name] for name in ("execution", "selection", "adaptation", "rss")),
                strict=True,
            )
        ]
    )
    np.testing.assert_allclose(
        metrics["equivalent_testing_curve_s"], equivalent, rtol=1e-12, atol=1e-12
    )


def verify_run(root, protocol, row, truth_root, truth_cache):
    config = protocol["config"]
    sequential = protocol["suite"] == "sequential"
    method = row["regime"] if sequential else row["method"]
    path = (root / row["result_path"]).resolve()
    saved = read_json(path)
    spec, result, metrics = (saved[k] for k in ("specification", "result", "metrics"))
    assert row["metrics"] == metrics
    assert row["resources"] == result["resources"]
    kind = row["scenario"]
    count = row.get("candidate_count", config["candidate_count"])
    replicate = 20 if sequential else protocol["candidate_replicate"]
    candidates = candidate_pool(kind, count, replicate=replicate)
    assert spec["candidates"] == [asdict(scene) for scene in candidates]
    budget = min(config["budget"], len(candidates))
    initial_count = min(config["initial_count"], budget)
    assert spec["budget"] == budget and spec["initial_count"] == initial_count
    assert spec["seed"] == row["seed"]
    assert spec["environment_seed"] == config["environment_seed"]
    bank = truth_root / row["sut_id"] / kind
    manifest = read_json(bank / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["environment_seed"] == spec["environment_seed"]
    assert manifest["contract"] == spec["bank_contract"]
    assert manifest["scenarios"][: len(candidates)] == spec["candidates"]
    truth_key = row["sut_id"], kind, len(candidates)
    if truth_key not in truth_cache:
        labels = []
        for index, scene in enumerate(candidates):
            record = read_execution(bank, index)
            assert record["scenario"] == asdict(scene)
            assert type(record["feedback"]["collision"]) is bool
            assert record["feedback"]["collision"] == any(
                frame["collision"] for frame in record["trajectory"]
            )
            labels.append(record["feedback"]["collision"])
        truth_cache[truth_key] = labels
    labels = truth_cache[truth_key]
    assert saved["evaluation"]["candidate_collisions"] == labels
    selected, queries, ledger = result["selected_indices"], result["queries"], saved["query_ledger"]
    assert len(selected) == len(set(selected)) == len(queries) == len(ledger) == budget
    assert all(0 <= index < len(candidates) for index in selected)
    initial_pool_size = protocol.get("initial_pool_size", len(candidates))
    assert initial_count <= initial_pool_size <= len(candidates)
    expected_initial = (
        np.random.default_rng(row["seed"]).permutation(initial_pool_size)[:initial_count].tolist()
    )
    assert selected[:initial_count] == expected_initial
    if protocol["suite"] == "sensitivity":
        assert initial_pool_size == min(protocol.get("candidate_sizes", [initial_pool_size]))
        assert spec["initial_pool_size"] == initial_pool_size
        assert spec["initial_indices"] == expected_initial
    assert result["actual_queries"] == result["maximum_queries"] == budget
    assert result["termination"] == "budget_exhausted"
    if sequential:
        ordinal = row["ordinal"]
        assert 1 <= ordinal <= len(protocol["systems"])
        branch = root / "methods" / method / str(row["seed"])
        archive_path = branch / "completed" / f"{ordinal:02d}_{row['sut_id']}_{kind}.json.gz"
        expected_snapshot = {
            name: spec[name] for name in ("target_model_version", "encoder_version")
        }
        order = np.random.default_rng(row["seed"]).permutation(len(protocol["systems"]))
        assert protocol["systems"][order[ordinal - 1]]["sut_id"] == row["sut_id"]
    else:
        archive_path = path.parent / "completed.json.gz"
        expected_snapshot = spec
    with gzip.open(archive_path, "rt", encoding="utf-8") as stream:
        archive = json.load(stream)
    assert archive["completed"] and archive["sut_id"] == row["sut_id"]
    sut = next(s for s in protocol["systems"] if s["sut_id"] == row["sut_id"])
    assert archive["sut_version"] == sut["version"]
    assert archive["original_role"] == sut["role"]
    assert archive["environment_seed"] == config["environment_seed"]
    assert archive["execution_contract"] == spec["bank_contract"]
    assert archive["rss_contract"] == measurement_contract()
    if sequential:
        assert archive["completed_order"] == ordinal
        assert archive["branch_id"] == row["branch_id"] == f"{method}:{row['seed']}"
    assert archive["model_snapshot"] == expected_snapshot
    assert archive["attempts"] == ledger
    assert len(archive["executions"]) == budget
    assert len({entry["record_id"] for entry in archive["executions"]}) == budget
    for position, (index, query, entry, executed) in enumerate(
        zip(selected, queries, ledger, archive["executions"], strict=True), 1
    ):
        assert index == query["index"] == entry["index"]
        assert query["query"] == entry["call"] == executed["query"] == position
        assert query["feedback"] == entry["feedback"] == executed["feedback"]
        assert query["feedback"]["scenario_id"] == candidates[index].scenario_id
        assert query["feedback"]["collision"] == labels[index]
        assert executed["scenario"] == asdict(candidates[index])
        actual = read_execution(bank, index)
        trajectory = actual["trajectory"]
        assert read_trajectory(executed) == trajectory
        assert query["feedback"]["execution_s"] == actual["feedback"]["execution_s"]
        assert any(frame["collision"] for frame in trajectory) == labels[index]
        assert entry["status"] == "complete" and entry["account"] == "target"
        assert entry["purpose"] == ("initial" if position <= initial_count else "target")
        assert query["feedback"]["rss_contract"] == measurement_contract()
        margin = query["feedback"]["safety"]["rss_margin"]
        assert math.isfinite(margin)
        assert math.isclose(
            margin, trajectory_margin(trajectory)["minimum_margin"], rel_tol=1e-12, abs_tol=1e-12
        )
        if "conditioning_queries" in query:
            assert query["conditioning_queries"] == position - 1
    if "prediction_snapshot_counts" in spec:
        assert [s["paid_queries"] for s in result["prediction_snapshots"]] == [
            n for n in spec["prediction_snapshot_counts"] if n <= budget
        ]
    for snapshot in result.get("prediction_snapshots", []):
        assert snapshot["observed_indices"] == selected[: snapshot["paid_queries"]]
        assert len(snapshot["probabilities"]) == len(candidates)
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in snapshot["probabilities"])
    verify_metrics(candidates, labels, selected, queries, ledger, metrics, initial_count)
    return {
        "sut_id": row["sut_id"],
        "scenario": kind,
        "seed": row["seed"],
        ("regime" if sequential else "method"): method,
        "candidate_count": len(candidates),
        "result_path": row["result_path"],
        "queries": budget,
        "shared_initialization_verified": True,
        "paid_feedback_and_actual_trajectories_verified": True,
        "rss_recalculated": True,
        "only_paid_records_archived": True,
        "independent_metrics_verified": True,
        "full_distance_and_recall_curves_verified": True,
        "prequery_calibration_verified": True,
        "timing_accounting_verified": True,
    }


def audit(root=OUTPUT / "pilot", *, data_root=DATASET_ROOT / "targets"):
    """Verify every currently saved run, including reused results and nested pools."""
    root, data_root = Path(root), Path(data_root)
    summary = read_json(root / "reports/summary.json")
    protocol = read_json(root / "reports/protocol.json")
    config, suite = protocol["config"], protocol["suite"]
    sequential = suite == "sequential"
    methods = REGIMES if sequential else protocol["main_methods"] + protocol["controls"]
    truth_root = data_root / (suite if suite in ("sequential", "sensitivity") else "fixed")
    sizes = protocol.get("candidate_sizes")
    expected = {
        (sut["sut_id"], kind, seed, method, size)
        for sut in protocol["systems"]
        for kind in config["scenarios"]
        for seed in config["algorithm_seeds"]
        for method in methods
        for size in (sizes or [protocol["candidate_counts"][kind]])
    }
    rows, seen, truth_cache = [], set(), {}
    for row in summary["runs"]:
        method = row["regime"] if sequential else row["method"]
        key = row["sut_id"], row["scenario"], row["seed"], method, row["metrics"]["candidate_count"]
        assert key in expected and key not in seen, key
        seen.add(key)
        try:
            checked = verify_run(root, protocol, row, truth_root, truth_cache)
        except AssertionError as error:
            raise AssertionError(
                f"Saved-run verification failed: {key}, {row['result_path']}"
            ) from error
        rows.append(checked)
        print(
            json.dumps(
                {
                    "verified_run": len(rows),
                    "method": method,
                    "scenario": row["scenario"],
                    "sut_id": row["sut_id"],
                }
            ),
            flush=True,
        )
    if summary["status"] == "complete":
        assert seen == expected
    report = {
        "format": "completed-formal-run-check",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete" if seen == expected and summary["status"] == "complete" else "partial",
        "expected_runs": len(expected),
        "checked_runs": len(rows),
        "checked_paid_queries": sum(row["queries"] for row in rows),
        "runs": rows,
        "trajectory_reading": "stream full-library labels and actual collision events; load paid trajectories per query without retaining all banks",
        "verification_scope": "paid records, actual trajectory feedback, archive identity, metrics and timing; source/model admission is checked separately",
    }
    write_json(root / "reports/completed_run_check.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}), flush=True)
    return report


if __name__ == "__main__":
    audit()
