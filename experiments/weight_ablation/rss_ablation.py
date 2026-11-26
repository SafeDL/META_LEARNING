"""Evaluate E03 RSS discovery on sealed queries and native trajectories."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from controllers.highway.catalog import FORMAL_TARGET_SUTS

from experiments.weight_ablation import run as study
from experiments.collect_transfer_records import ApprovedCase
from experiments.executions import ExecutionBank
from experiments.initialization import initial_indices
from experiments.paired_statistics import json_ready, paired_summary
from experiments.parallel import METHOD_WORKERS, initialize_worker, prepare_processes
from experiments.record_audit import BankTruth, _check_specification, _queries
from experiments.budget.report import write_csv
from experiments.rss import rss_values
from experiments.storage import compact_feedback, read_json, write_json
from experiments.transfer_execution import _paths

REPORTS = study.OUTPUT / "reports"
BUDGET = study.protocol.BUDGET
INITIAL = study.protocol.INITIAL
CHECKPOINTS = study.protocol.CHECKPOINTS
CANDIDATES = 2000
TAIL_RANK = 100
METHODS = ("full", "frozen")


def read_sealed(path, checksum):
    if study.sha(path) != checksum:
        raise ValueError(f"Sealed E03 input changed: {path}")
    return read_json(path)


def query_order(root, scene, target, update_weights, truth, files):
    case = ApprovedCase("budget", target, scene, study.SEED, "full", None)
    path, paid_path = _paths(root, case)
    saved = read_sealed(path, files[study.relative(path)])
    paid = read_sealed(paid_path, files[study.relative(paid_path)])
    spec = saved["specification"]
    result = saved["result"]
    order = result["selected_indices"]
    initial = initial_indices(CANDIDATES, INITIAL, study.SEED)
    if (
        spec["budget"] != BUDGET or spec["initial_count"] != INITIAL or spec["seed"] != study.SEED
        or spec["initial_indices"] != initial or order[:INITIAL] != initial or len(order) != BUDGET
        or spec["update_source_weights"] is not update_weights
        or result["source_weight_updates"] is not update_weights
        or result["target_feedback_updates"] is not True or result["feedback_used"] != "joint"
        or paid["status"] != "complete" or paid["successful_paid_responses"] != BUDGET
        or paid["query_ledger"] != compact_feedback(saved["query_ledger"])
    ):
        raise ValueError(f"Invalid E03 query protocol or paid ledger: {path}")
    _check_specification(saved, paid["specification"]["task"], truth)
    _queries(saved, truth, order, INITIAL, True)
    return np.asarray(order, dtype=int)


def first_hit(margins, running_minimum, threshold):
    hits = np.flatnonzero(margins <= threshold)
    crossing = np.flatnonzero(running_minimum <= threshold)
    if not np.array_equal(running_minimum, np.minimum.accumulate(margins)):
        raise ValueError("Invalid cumulative minimum RSS curve")
    first = int(hits[0]) + 1 if hits.size else None
    if first != (int(crossing[0]) + 1 if crossing.size else None):
        raise ValueError("The first observed hit does not match the curve crossing")
    return first


def evaluate_pair(scene, target, audit_checksum, frozen_root, prediction_root, budget_root):
    audit = read_sealed(frozen_root / "audits" / scene / f"{target}.json", audit_checksum)
    bank = ExecutionBank(prediction_root / "physical/target" / scene / target)
    if (
        audit["scene"] != scene or audit["target"] != target or audit["seed"] != study.SEED
        or audit["budget"]["status"] != "complete" or audit["budget"]["shared_feedback_verified"] != INITIAL
        or audit["native_records"] != CANDIDATES or len(bank.candidates) != CANDIDATES
        or len(bank.indices) != CANDIDATES or audit["bank_snapshot"] != bank.snapshot_fingerprint
        or bank.manifest["sut_id"] != target
    ):
        raise ValueError(f"Native bank differs from the E03 audit: {scene}/{target}")
    truth = BankTruth(bank)
    margins = rss_values(truth.margins)
    threshold = float(np.sort(margins)[TAIL_RANK - 1])
    tail_count = int(np.count_nonzero(margins <= threshold))
    files = audit["budget"]["files"]
    selections = np.stack([
        query_order(root, scene, target, update, truth, files)
        for root, update in ((budget_root, True), (frozen_root, False))
    ])
    observed = margins[selections]
    minima = np.minimum.accumulate(observed, axis=1)
    if (
        not np.array_equal(selections[0, :INITIAL], selections[1, :INITIAL])
        or not np.array_equal(minima[0, :INITIAL], minima[1, :INITIAL])
        or np.any(np.diff(minima, axis=1) > 0)
    ):
        raise ValueError(f"Invalid E03 common prefix or RSS curve: {scene}/{target}")

    base = {"scene": scene, "target": target, "seed": study.SEED}
    pool_minimum = float(margins.min())
    pair = {**base, "threshold": threshold, "tail_count": tail_count, "tail_fraction": tail_count / CANDIDATES,
            "pool_minimum": pool_minimum, "initial_minimum": float(minima[0, INITIAL - 1]),
            "initial_at_pool_minimum": bool(minima[0, INITIAL - 1] == pool_minimum)}
    curves = []
    for method, indices, rss, running in zip(METHODS, selections, observed, minima, strict=True):
        first = first_hit(rss, running, threshold)
        pair.update({
            f"{method}_first_hit": first,
            f"{method}_hit": int(first is not None),
            f"{method}_right_censored": first is None,
            f"{method}_wait": BUDGET if first is None else first,
        })
        curves.extend(
            {**base, "method": method, "query": call, "candidate_index": int(index),
             "candidate_id": bank.candidates[index].scenario_id, "rss_margin": float(value),
             "minimum_rss_margin": float(low)}
            for call, (index, value, low) in enumerate(zip(indices, rss, running, strict=True), 1)
        )
    pair.update(
        initial_hit=pair["initial_minimum"] <= threshold,
        delta_hit=pair["full_hit"] - pair["frozen_hit"],
        delta_wait=pair["full_wait"] - pair["frozen_wait"],
    )
    return pair, curves, minima


def difference_summary(values, scenes):
    # paired_summary uses the supplied direction to compute a signed difference.
    stats = paired_summary(
        values, ads_ids=study.TARGETS, ads_families=[s.family for s in FORMAL_TARGET_SUTS],
        methods=METHODS, baseline="frozen", lower_better=False, replicates=10000, seed=1301,
    )
    paired = stats["paired"]
    return dict(
        scenes=scenes, pairs=values.shape[0] * values.shape[1], valid_ads=paired["valid_ads"][0],
        full=paired["method_mean"][0], frozen=paired["reference_mean"][0], delta=paired["gain"][0],
        low=paired["gain_ci_low"][0], high=paired["gain_ci_high"][0],
    )


def summarize(data):
    scenes, targets = study.SCENES, study.TARGETS
    minima = np.asarray([[data[s, t][2] for s in scenes] for t in targets])
    hit_data = np.asarray([[[[data[s, t][0][f"{m}_{k}"] for k in ("hit", "wait")]
                             for m in METHODS] for s in scenes] for t in targets])
    scopes = {}
    for scope, selection in [("overall", slice(None)), *[
        (scene, slice(j, j + 1)) for j, scene in enumerate(scenes)
    ]]:
        subset = scenes[selection]
        pairs = [data[s, t][0] for s in subset for t in targets]
        delta = minima[:, selection, 0] - minima[:, selection, 1]
        scopes[scope] = {
            "curve": difference_summary(minima[:, selection, None], subset),
            "hit": difference_summary(hit_data[:, selection, None], subset),
            "counts": {
                "initial_hit": sum(p["initial_hit"] for p in pairs),
                "initial_at_pool_minimum": sum(p["initial_at_pool_minimum"] for p in pairs),
                "full_hits": sum(p["full_hit"] for p in pairs),
                "frozen_hits": sum(p["frozen_hit"] for p in pairs),
                "full_earlier_wait": sum(p["delta_wait"] < 0 for p in pairs),
                "same_wait": sum(p["delta_wait"] == 0 for p in pairs),
                "frozen_earlier_wait": sum(p["delta_wait"] > 0 for p in pairs),
                "tail_ties": sum(p["tail_count"] > TAIL_RANK for p in pairs),
                "full_lower_minimum": (delta < 0).sum(axis=(0, 1)),
                "same_minimum": (delta == 0).sum(axis=(0, 1)),
                "frozen_lower_minimum": (delta > 0).sum(axis=(0, 1)),
            },
        }
    return json_ready({
        "difference": "dynamic minus frozen", "hit_metrics": ["hit_rate", "restricted_wait_queries"],
        "budget": BUDGET, "common_queries": INITIAL, "checkpoints": CHECKPOINTS,
        "tail_order_statistic": TAIL_RANK, "candidate_count": CANDIDATES,
        "bootstrap": {"unit": "ADS", "replicates": 10000, "seed": 1301,
                      "interval": "pointwise percentile 95%", "pooling": "scenes within ADS, then ADS"},
        "scopes": scopes,
    })


def report(report_root=REPORTS, target_roots=None):
    source = report_root / "completion.json"
    output = report_root / "rss_discovery"
    source_sha = study.sha(source)
    original = read_sealed(source, source_sha)
    pairs = [(s, t) for s in study.SCENES for t in study.TARGETS]
    if (
        original["status"] != "complete" or original["paired_targets"] != len(pairs)
        or set(original["audit_files"]) != {f"{s}/{t}.json" for s, t in pairs}
    ):
        raise ValueError("Requires the complete E03 paired audit")
    prepare_processes()
    data = {}
    with ProcessPoolExecutor(max_workers=METHOD_WORKERS, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        defaults = (study.OUTPUT, study.INPUT, study.protocol.OUTPUT)
        roots = target_roots or {}
        futures = {pool.submit(evaluate_pair, s, t, original["audit_files"][f"{s}/{t}.json"],
                               *roots.get(t, defaults)): (s, t)
                   for s, t in pairs}
        for future in as_completed(futures):
            key = futures[future]
            data[key] = future.result()
            print(f"Audited RSS for {key[0]}/{key[1]} ({len(data)}/{len(pairs)})", flush=True)
    summary = summarize(data)
    tables = {"paired_first_hit.csv": [data[key][0] for key in pairs],
              "rss_curves.csv": [row for key in pairs for row in data[key][1]]}
    for name, rows in tables.items():
        path = output / name
        temporary = path.with_suffix(".tmp")
        write_csv(temporary, rows)
        temporary.replace(path)
    write_json(output / "paired_summary.json", summary)
    receipt = {
        "status": "complete", "paired_tasks": len(pairs), "query_sequences": len(pairs) * len(METHODS),
        "native_trajectories_recomputed": len(pairs) * CANDIDATES, "query_rows": len(tables["rss_curves.csv"]),
        "new_logical_queries": 0, "new_physical_executions": 0,
        "original_completion_sha256": source_sha, "report_source_sha256": study.sha(Path(__file__)),
        "files": {name: study.sha(output / name) for name in (*tables, "paired_summary.json")},
    }
    write_json(output / "completion.json", receipt)
    print(receipt, flush=True)
    return data


if __name__ == "__main__":
    report()
