"""Read-only audit of approved prediction/query records against native executions."""

import json
import math
from dataclasses import asdict, replace

import numpy as np

from simulator.highway.timing import PHYSICS_DT

from .domain_risk import disclose_driving_quality, disclose_impact
from .initialization import initial_indices
from .k_shot import split_candidates
from .rss import measurement_contract, trajectory_margin
from .schema import QueryFeedback
from .scenarios import DOMAINS


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


def _canonical_feedback(value):
    value = dict(value)
    value.pop("trajectory", None)
    return json.loads(json.dumps(QueryFeedback.from_dict(value).to_dict(), allow_nan=False))


def _compare(actual, saved, path="metric"):
    if isinstance(actual, dict):
        for key, value in actual.items():
            if key not in saved:
                raise ValueError(f"Missing {path}.{key}")
            _compare(value, saved[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if not isinstance(saved, list) or len(actual) != len(saved):
            raise ValueError(f"Different length at {path}")
        for i, (left, right) in enumerate(zip(actual, saved)):
            _compare(left, right, f"{path}[{i}]")
    elif isinstance(actual, float):
        if (
            not isinstance(saved, (int, float))
            or not math.isfinite(saved)
            or not math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"Independent recomputation differs at {path}")
    elif actual != saved:
        raise ValueError(f"Independent recomputation differs at {path}")


class BankTruth:
    """Only the evaluator holds this object; trajectories are read, not re-executed."""

    def __init__(self, bank):
        self.bank = bank
        self.candidates = bank.candidates
        if not self.candidates or len({s.kind for s in self.candidates}) != 1:
            raise ValueError("Audit one nonempty functional scenario at a time")
        duration = bank.manifest["contract"]["episode_duration_s"]
        labels, margins = [], []
        self._feedback = {}
        profile = {
            "records": len(self.candidates),
            "collisions": 0,
            "road_departures": 0,
            "noncollision_early_ends": 0,
            "background_crash_without_ego": 0,
        }
        from simulator.highway.collision_env import default_lane_centers

        for scene in self.candidates:
            centers = default_lane_centers(scene.kind)
            if scene.kind == "side_encroachment":
                centers = (0.0, 4.0) if scene.categories["side"] == "left" else (4.0, 8.0)
            lower, upper = min(centers) - 2, max(centers) + 2
            paid = bank(scene)
            trace = paid.trajectory
            if any(type(frame["collision"]) is not bool for frame in trace):
                raise ValueError("Native collision indicators must be measured Booleans")
            times = np.asarray([f["time_s"] for f in trace])
            if (
                not len(times)
                or times[0] != 0
                or times[-1] > duration + 1e-9
                or not np.allclose(np.diff(times), PHYSICS_DT, rtol=0, atol=1e-9)
            ):
                raise ValueError(
                    "Native trace clock differs from its registered physics or duration"
                )
            collision = any(bool(f["collision"]) for f in trace)
            departure = False
            for frame in trace:
                ego = next(a for a in frame["actors"] if a["actor_id"] == "ego")
                half = (
                    abs(math.sin(ego["heading"])) * ego["length"]
                    + abs(math.cos(ego["heading"])) * ego["width"]
                ) / 2
                departure |= ego["y"] - half < lower or ego["y"] + half > upper
            if collision != paid.feedback.collision or departure != (
                "road_departure" in paid.feedback.safety.violations
            ):
                raise ValueError(
                    "Native collision/road-departure feedback differs from its trajectory"
                )
            early = times[-1] < duration - 1e-9 and not collision
            if early and not departure:
                raise ValueError(
                    "Unexplained early noncollision termination is not a safe response"
                )
            labels.append(collision)
            margins.append(trajectory_margin(trace)["minimum_margin"])
            profile["collisions"] += int(collision)
            profile["road_departures"] += int(departure)
            profile["noncollision_early_ends"] += int(early)
            profile["background_crash_without_ego"] += int(
                not collision
                and any(not c["scripted"] for f in trace for c in f["background_controls"].values())
            )
        self.labels, self.margins = np.asarray(labels), np.asarray(margins)
        self.profile = profile

    def paid_feedback(self, index, native):
        key = index, native
        if key not in self._feedback:
            paid = self.bank(self.candidates[index])
            if native:
                paid = disclose_driving_quality(disclose_impact(paid))
            feedback = replace(
                paid.feedback,
                safety=replace(paid.feedback.safety, rss_margin=float(self.margins[index])),
                rss_contract=measurement_contract(),
                trajectory=None,
            )
            self._feedback[key] = _canonical_feedback(feedback.to_dict())
        return self._feedback[key]


def prediction_scores(probability, labels, candidate_ids, evaluation):
    """Direct NLL/Brier, tied-rank AUROC/AP, ID-tied recall and ten-bin calibration."""
    probability, raw_labels = np.asarray(probability, dtype=float), np.asarray(labels)
    evaluation = np.asarray(evaluation, dtype=int)
    if (
        probability.shape != raw_labels.shape
        or probability.shape != (len(candidate_ids),)
        or not np.isin(raw_labels, (0, 1)).all()
        or len(set(candidate_ids)) != len(candidate_ids)
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Invalid complete candidate prediction vector")
    if (
        evaluation.ndim != 1
        or not len(evaluation)
        or len(set(evaluation)) != len(evaluation)
        or ((evaluation < 0) | (evaluation >= len(probability))).any()
    ):
        raise ValueError("Invalid fixed evaluation indices")
    labels = raw_labels.astype(bool)
    p, y = probability[evaluation], labels[evaluation].astype(float)
    ids = np.asarray(candidate_ids)[evaluation]
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    unique, groups, counts = np.unique(p, return_inverse=True, return_counts=True)
    positive = np.bincount(groups, weights=y, minlength=len(unique))
    negative = counts - positive
    auc = (
        float(
            np.sum(positive * (np.cumsum(negative) - negative + negative / 2))
            / (positives * negatives)
        )
        if positives and negatives
        else None
    )
    precision = np.cumsum(positive[::-1]) / np.cumsum(counts[::-1])
    ap = float(np.sum(positive[::-1] * precision) / positives) if positives else None
    result = {
        "evaluation_count": len(y),
        "evaluation_collisions": positives,
        "nll": float(-np.mean(y * np.log(bounded) + (1 - y) * np.log1p(-bounded))),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc": auc,
        "auprc_ap": ap,
        "auroc_undefined": None if positives and negatives else "single_class",
        "auprc_undefined": None if positives else "no_positive_labels",
        "recall_undefined": None if positives else "no_positive_labels",
    }
    order = np.lexsort((ids, -p))
    for limit in (50, 100, 200):
        count = min(limit, len(y))
        hits = int(y[order[:count]].sum())
        result.update(
            {
                f"top_{limit}_count": count,
                f"top_{limit}_hits": hits,
                f"recall_{limit}": hits / positives if positives else None,
            }
        )
    bins = np.minimum((10 * p).astype(int), 9)
    calibration = []
    for index in range(10):
        mask = bins == index
        count = int(mask.sum())
        calibration.append(
            {
                "bin": index,
                "count": count,
                "mean_probability": float(p[mask].mean()) if count else None,
                "collision_fraction": float(y[mask].mean()) if count else None,
            }
        )
    result["ece"] = sum(
        r["count"] / len(y) * abs(r["mean_probability"] - r["collision_fraction"])
        for r in calibration
        if r["count"]
    )
    return result, calibration


def _check_specification(saved, expected, truth):
    if saved["specification"] != expected:
        raise ValueError("Record differs from its independently approved task specification")
    bank = truth.bank
    snapshot = expected.get("bank_snapshot", expected.get("task_snapshot", {}).get("bank_snapshot"))
    if (
        snapshot != bank.snapshot_fingerprint
        or expected["bank_contract"] != bank.manifest["contract"]
        or expected["candidates"] != [asdict(s) for s in truth.candidates]
    ):
        raise ValueError("Record differs from the approved physical bank or public coordinates")


def _queries(saved, truth, order, initial_count, native):
    result, ledger = saved["result"], saved["query_ledger"]
    indices = result["selected_indices"]
    if (
        indices != order
        or len(set(indices)) != len(indices)
        or len(result["queries"]) != len(order)
        or len(ledger) != len(order)
    ):
        raise ValueError(
            "Query result is incomplete, duplicated or differs from the registered order"
        )
    for call, (index, query, record) in enumerate(zip(indices, result["queries"], ledger), 1):
        if type(index) is not int or not 0 <= index < len(truth.candidates):
            raise ValueError("Query index is outside its public candidate library")
        if (
            record["status"] != "complete"
            or record["call"] != call
            or record["index"] != index
            or query["index"] != index
            or query["query"] != call
            or record["account"] != "target"
            or record["purpose"] != ("initial" if call <= initial_count else "target")
            or record["scenario_id"] != truth.candidates[index].scenario_id
        ):
            raise ValueError("Invalid paid query account, purpose, sequence or candidate")
        for feedback in (query["feedback"], record["feedback"]):
            if (
                feedback.get("trajectory") is not None
                and feedback["trajectory"] != truth.bank(truth.candidates[index]).trajectory
            ):
                raise ValueError("Inline paid trace differs from the native execution")
            if _canonical_feedback(feedback) != truth.paid_feedback(index, native):
                raise ValueError(
                    "Paid feedback differs from independently measured native feedback"
                )


def prediction_record(saved, expected, truth, *, method, seed):
    """Approved specifications come from the run registry, never from metric rows."""
    _check_specification(saved, expected, truth)
    if expected["method"] != method or expected["feedback_seed"] != seed:
        raise ValueError("Prediction record has a different method or feedback seed")
    split = split_candidates(truth.candidates, seed, budget=max(expected["checkpoints"]))
    if expected["split"] != split:
        raise ValueError("Prediction feedback and fixed evaluation partition changed")
    _queries(saved, truth, split["query_order"], 1, False)
    snapshots = saved["result"]["prediction_snapshots"]
    if [s["paid_queries"] for s in snapshots] != expected["checkpoints"] or [
        r["K"] for r in saved["metrics"]
    ] != expected["checkpoints"]:
        raise ValueError("Missing or extra prediction checkpoints")
    rows = []
    for snapshot, recorded in zip(snapshots, saved["metrics"]):
        count = snapshot["paid_queries"]
        if snapshot["observed_indices"] != split["query_order"][:count]:
            raise ValueError("Prediction checkpoint conditions on different feedback")
        metrics, calibration = prediction_scores(
            snapshot["probabilities"],
            truth.labels,
            split["candidate_ids"],
            split["evaluation_indices"],
        )
        _compare({**metrics, "calibration": calibration}, recorded)
        rows.append(
            {
                "sut_id": truth.bank.manifest["sut_id"],
                "scenario": truth.candidates[0].kind,
                "seed": seed,
                "method": method,
                "K": count,
                "status": "complete",
                "candidate_count": len(truth.candidates),
                "pool_collisions": int(truth.labels.sum()),
                **metrics,
                "calibration": calibration,
            }
        )
    return rows


def budget_record(saved, expected, truth, *, method, seed):
    _check_specification(saved, expected, truth)
    if expected["seed"] != seed or expected["task_snapshot"]["method_contract"]["method"] != method:
        raise ValueError("Budget record has a different method or initialization seed")
    budget, initial_count = expected["budget"], expected["initial_count"]
    selected = saved["result"]["selected_indices"]
    if len(selected) != budget or selected[:initial_count] != initial_indices(
        len(truth.candidates), initial_count, seed
    ):
        raise ValueError("Budget result did not use the exact common initialization candidates")
    _queries(saved, truth, selected, initial_count, True)
    labels = truth.labels.tolist()
    if saved["evaluation"]["candidate_collisions"] != labels:
        raise ValueError("Stored evaluation labels differ from native trajectories")
    if not __debug__:
        raise ValueError("Independent metric auditing requires Python assertion checks enabled")
    try:
        verify_metrics(
            truth.candidates,
            labels,
            selected,
            saved["result"]["queries"],
            saved["query_ledger"],
            saved["metrics"],
            initial_count,
            domain=truth.bank.domain,
        )
    except AssertionError as error:
        raise ValueError(
            "Independent discovery, coverage, distance or cost recomputation failed"
        ) from error
    return {
        "sut_id": truth.bank.manifest["sut_id"],
        "scenario": truth.candidates[0].kind,
        "seed": seed,
        "method": method,
        "status": "complete",
        "candidate_count": len(truth.candidates),
        "pool_collisions": int(truth.labels.sum()),
        "metrics": saved["metrics"],
    }
