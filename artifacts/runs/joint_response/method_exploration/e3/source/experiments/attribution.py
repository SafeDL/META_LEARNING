"""Replay registered HighwayEnv evidence without collecting any new executions."""

import copy
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.training import load_model

from .executions import ExecutionBank
from .failure_metrics import discovery_distances, summarize
from .history import build_source, load_training_archives, model_version
from .protocol import ROOT
from .query import QueryLedger, run_isolated
from .schema import Scenario
from .storage import read_json, write_json

OLD = ROOT / "artifacts/runs/joint_response"
OUTPUT = ROOT / "artifacts/runs/public_ads/attribution"
DATA = ROOT / "artifacts/data/joint_response"
CHECKPOINTS = (20, 50, 100, 200)
VARIANTS = ("joint", "collision", "fixed_weights")


def registered_runs():
    runs = {}
    for group in ("fixed", "mechanisms"):
        summary = read_json(OLD / group / "reports/summary.json")
        if summary["status"] != "complete":
            raise ValueError("Attribution requires completed registered comparisons")
        for row in summary["runs"]:
            key = row["sut_id"], row["scenario"], row["method"]
            runs.setdefault(key, {**row, "path": str((OLD / group / row["result_path"]).resolve())})
    return runs


def models(kind, historical_rss=True):
    root = OLD / "models"
    if not historical_rss:
        root /= "historical_collision"
    return tuple(
        load_model(root / kind / "all_sources" / name)[0]
        for name in ("target_kernel.pt", "historical_encoder.pt")
    )


def source_path(kind, historical_rss):
    return OUTPUT / "sources" / kind / ("joint.npz" if historical_rss else "collision.npz")


def read_source(path):
    import json

    with np.load(path, allow_pickle=False) as data:
        return {
            **{k: data[k].copy() for k in ("means", "blocks", "weights")},
            "names": data["names"].tolist(),
            "specification": json.loads(str(data["specification"])),
        }


def prepare_sources(runs):
    """Take record IDs from the executed full method, not a freshly sampled control."""
    changes = []
    for kind in sorted({key[1] for key in runs}):
        row = next(r for key, r in runs.items() if key[1:] == (kind, "joint"))
        joint = read_json(row["path"])
        reference = joint["specification"]["source_snapshot"]
        candidates = [Scenario(**s) for s in joint["specification"]["candidates"]]
        archives = load_training_archives(DATA / "manifest.json", kind)
        for use_rss in (True, False):
            _, encoder = models(kind, use_rss)
            built = build_source(
                encoder,
                archives,
                candidates,
                view_record_ids=reference["view_record_ids"],
                use_rss=use_rss,
                cache_path=source_path(kind, use_rss),
            )
            if use_rss and built["specification"] != reference:
                raise ValueError(
                    "Full-method source reconstruction changed its registered snapshot"
                )
        control = runs[row["sut_id"], kind, "historical_collision"]
        previous = read_json(control["path"])["specification"]["source_snapshot"]
        replacements = {
            name: len(set(ids) - set(previous["view_record_ids"][name]))
            for name, ids in reference["view_record_ids"].items()
        }
        changes.append(
            {
                "scenario": kind,
                "replaced_records": replacements,
                "affected": any(replacements.values()),
                "fixed_view_record_ids": reference["view_record_ids"],
                "old_control_model": previous["model_version"],
            }
        )
    write_json(OUTPUT / "reports/source_views.json", changes)
    return {r["scenario"] for r in changes if r["affected"]}


def prediction_metrics(snapshot, labels):
    """The selector has returned before any remaining labels enter this calculation."""
    remaining = np.ones(len(labels), dtype=bool)
    remaining[snapshot["observed_indices"]] = False
    y = np.asarray(labels, dtype=float)[remaining]
    p = np.asarray(snapshot["probabilities"])[remaining]
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    k = min(200, len(y))
    ranking = np.argsort(-p, kind="stable")[:k]
    return {
        "unqueried_count": len(y),
        "unqueried_failures": int(y.sum()),
        "nll": float(-(y * np.log(bounded) + (1 - y) * np.log(1 - bounded)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "top_200_failures": int(y[ranking].sum()),
        "top_200_recall": float(y[ranking].sum() / y.sum()) if y.sum() else None,
    }


def compact_result(result, bank_root):
    compact = copy.deepcopy(result)
    for query in compact["queries"]:
        query["feedback"].pop("trajectory", None)
        query["execution_ref"] = str(
            (bank_root / "executions" / f"{query['index']:06d}.json.gz").relative_to(ROOT)
        )
    return compact


def replay_task(sut_id, kind, affected, paths):
    torch.set_num_threads(1)
    started = time.perf_counter()
    joint = read_json(paths["joint"])
    random = read_json(paths["random"])
    bank = ExecutionBank(DATA / "targets/fixed" / sut_id / kind)
    labels = joint["evaluation"]["candidate_collisions"]
    if [asdict(s) for s in bank.candidates] != joint["specification"][
        "candidates"
    ] or labels != random["evaluation"]["candidate_collisions"]:
        raise ValueError("Saved comparisons must use identical candidate truth")
    order = random["result"]["selected_indices"]
    initial = joint["result"]["selected_indices"][:20]
    if order[:20] != initial or len(order) != 200:
        raise ValueError("Random replay must preserve all 20 common initial queries")
    base, _ = models(kind)
    if model_version(base) != joint["specification"]["target_model_version"]:
        raise ValueError("Target model differs from the original full-method checkpoint")
    source = read_source(source_path(kind, True))
    if source["specification"] != joint["specification"]["source_snapshot"]:
        raise ValueError("Task source snapshot differs from shared replay source")
    jobs = [(v, True) for v in VARIANTS]
    if affected:
        jobs.append(("historical_collision_fixed", False))
    outputs = []
    for variant, common in jobs:
        path = (
            OUTPUT
            / ("common_feedback" if common else "corrected_control")
            / sut_id
            / kind
            / f"{variant}.json.gz"
        )
        if common:
            active_base, active_source = base, source
        else:
            active_base, _ = models(kind, False)
            active_source = read_source(source_path(kind, False))
            old_spec = read_json(paths["historical_collision"])["specification"]
            if model_version(active_base) != old_spec["target_model_version"]:
                raise ValueError("Historical control target model must stay fixed")
            old = old_spec["source_snapshot"]
            if old["model_version"] != active_source["specification"]["model_version"]:
                raise ValueError("Historical control training must stay fixed")
        specification = {
            "format": "old-platform-attribution",
            "platform": "HighwayEnv",
            "sut_id": sut_id,
            "scenario": kind,
            "variant": variant,
            "original_joint": str(Path(paths["joint"]).relative_to(ROOT)),
            "original_random": str(Path(paths["random"]).relative_to(ROOT)),
            "source_ref": str(source_path(kind, common).relative_to(ROOT)),
            "target_model_version": model_version(active_base),
            "encoder_version": active_source["specification"]["model_version"],
            "query_order": order if common else None,
            "budget": 200,
            "initial_count": 20,
            "seed": 11,
            "feedback_mode": "collision" if variant == "collision" else "joint",
            "update_source_weights": variant != "fixed_weights",
            "new_physical_calls": 0,
        }
        if path.exists():
            saved = read_json(path)
            if saved["specification"] != specification:
                raise ValueError("Attribution replay differs from its recorded protocol")
        else:
            ledger = QueryLedger(bank.candidates, 200, bank, joint_feedback=True)
            public = dict(
                candidates=bank.candidates,
                pretrained=active_base,
                sources=active_source,
                initial_indices=initial,
                budget=200,
                snapshot_counts=CHECKPOINTS,
                query_order=order if common else None,
                feedback_mode=specification["feedback_mode"],
                update_source_weights=specification["update_source_weights"],
            )
            try:
                result = run_isolated(select, public, ledger)
                metrics = summarize(
                    bank.candidates,
                    labels,
                    result["queries"],
                    ledger.trajectories,
                    initial_count=20,
                    query_ledger=ledger.records,
                )
                snapshots = [
                    {**s, "evaluation": prediction_metrics(s, labels)}
                    for s in result["prediction_snapshots"]
                ]
                result["prediction_snapshots"] = snapshots
                saved = {
                    "specification": specification,
                    "result": compact_result(result, bank.root),
                    "metrics": metrics,
                }
                write_json(path, saved)
            except Exception as error:
                write_json(
                    path.with_name("failure.json"),
                    {
                        "specification": specification,
                        "error": repr(error),
                        "completed_queries": len(ledger.records),
                    },
                )
                raise
        outputs.append(
            {"variant": variant, "path": str(path.relative_to(OUTPUT)), "metrics": saved["metrics"]}
        )
    return {
        "sut_id": sut_id,
        "scenario": kind,
        "runs": outputs,
        "wall_s": time.perf_counter() - started,
    }


def csv_report(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_saved_metrics(saved):
    candidates = [Scenario(**s) for s in saved["specification"]["candidates"]]
    labels = saved["evaluation"]["candidate_collisions"]
    discovered, counts, seen = [], [], set()
    for q in saved["result"]["queries"]:
        i = q["index"]
        if q["feedback"]["collision"] != labels[i]:
            raise ValueError("Recorded paid response differs from full-bank truth")
        if labels[i] and i not in seen:
            discovered.append(candidates[i])
            seen.add(i)
        counts.append(len(seen))
    metric = saved["metrics"]
    if (
        counts != metric["discovery_curve"]
        or len(counts) != 200
        or metric["pool_unique_collisions"] != sum(labels)
        or metric["discovered_ids"] != [s.scenario_id for s in discovered]
    ):
        raise ValueError("Discovery count or ID verification failed")
    np.testing.assert_allclose(metric["discovery_auc"], np.mean(counts))
    expected_recall = counts[-1] / sum(labels) if sum(labels) else None
    if metric["collision_recall"] != expected_recall:
        raise ValueError("Coverage does not match complete candidate truth")
    distances = discovery_distances(discovered)
    for key, values in distances.items():
        assert values == metric["distances_by_discovery_count"][key]


def existing_comparisons(runs):
    rows, comparisons = [], []
    methods = (
        "joint",
        "no_history",
        "collision",
        "fixed_weights",
        "random",
        "av_bo_rf_ei",
        "samota",
        "scenario_fuzz",
        "paired_system_active_evaluation",
        "behavexplor",
        "historical_collision",
        "embedding_only",
        "aggregate",
    )
    for (sut, kind, method), row in runs.items():
        if method not in methods:
            continue
        saved = read_json(row["path"])
        validate_saved_metrics(saved)
        m = row["metrics"]
        if m != saved["metrics"]:
            raise ValueError("Report metrics differ from the original method record")
        rows.append(
            {
                "sut_id": sut,
                "scenario": kind,
                "method": method,
                **{
                    k: m[k]
                    for k in (
                        "unique_collisions",
                        "initial_collisions",
                        "post_initial_collisions",
                        "discovery_auc",
                        "pool_unique_collisions",
                        "collision_recall",
                        "pair_distance",
                        "nearest_neighbor_distance",
                    )
                },
                **{f"found_{n}": m["discovery_curve"][n - 1] for n in CHECKPOINTS},
                "source": str(Path(row["path"]).relative_to(ROOT)),
            }
        )
        if method == "joint":
            continue
        full = runs[sut, kind, "joint"]["metrics"]
        k = min(m["unique_collisions"], full["unique_collisions"])
        comparisons.append(
            {
                "sut_id": sut,
                "scenario": kind,
                "control": method,
                "delta_failures": full["unique_collisions"] - m["unique_collisions"],
                "delta_auc": full["discovery_auc"] - m["discovery_auc"],
                "matched_discovery_count": k,
                **{
                    f"{which}_{key}": stats["distances_by_discovery_count"][key][k - 1]
                    if k >= 2
                    else None
                    for which, stats in (("joint", full), ("control", m))
                    for key in ("pair_distance", "nearest_neighbor_distance")
                },
            }
        )
    csv_report(OUTPUT / "reports/task_metrics.csv", rows)
    csv_report(OUTPUT / "reports/paired_differences.csv", comparisons)
    return rows, comparisons


def diagnose_task(sut, kind, joint_path):
    saved = read_json(joint_path)
    bank = ExecutionBank(DATA / "targets/fixed" / sut / kind)
    rows = []
    for i, collision in enumerate(saved["evaluation"]["candidate_collisions"]):
        if not collision:
            continue
        path = bank.root / "executions" / f"{i:06d}.json.gz"
        payload = read_json(path)
        frames = payload["trajectory"]
        hit = next(f for f in frames if f["collision"])
        actors = {a["actor_id"]: a for a in frames[0]["actors"]}
        ego, lead = actors["ego"], actors["lead"]
        gap = lead["x"] - ego["x"] - (lead["length"] + ego["length"]) / 2
        closure = max(0, ego["speed"] - lead["speed"])
        events = payload["events"]
        reveal = events.get("obstacle_visible_s")
        rows.append(
            {
                "sut_id": sut,
                "scenario": kind,
                "candidate_id": bank.candidates[i].scenario_id,
                "collision_time_s": hit["time_s"],
                "partner": hit["collision_partner"],
                "initial_lead_gap_m": gap,
                "initial_ego_speed_mps": ego["speed"],
                "constant_lead_max_brake_margin_m": gap - closure**2 / 16,
                "first_brake_s": events.get("first_ego_brake_s"),
                "event_onset_s": events.get("measured_background_onset_s"),
                "event_to_collision_s": None
                if events.get("measured_background_onset_s") is None
                else hit["time_s"] - events["measured_background_onset_s"],
                "obstacle_visible_s": reveal,
                "visibility_to_collision_s": None if reveal is None else hit["time_s"] - reveal,
                "lane_change_onset_s": events.get("ego_lane_change_onset_s"),
                "minimum_applied_acceleration_mps2": min(
                    f["control"]["acceleration"] for f in frames
                ),
                "ego_lateral_displacement_m": max(
                    abs(next(a["y"] for a in f["actors"] if a["actor_id"] == "ego") - ego["y"])
                    for f in frames
                ),
                "execution_ref": str(path.relative_to(ROOT)),
            }
        )
    return rows


def main():
    torch.set_num_threads(1)
    runs = registered_runs()
    existing_comparisons(runs)
    affected = prepare_sources(runs)
    tasks = sorted({key[:2] for key in runs})
    source_files = (
        Path(__file__),
        ROOT / "experiments/history.py",
        ROOT / "testing_methods/history_transfer/selection.py",
    )
    write_json(
        OUTPUT / "reports/protocol.json",
        {
            "platform": "HighwayEnv",
            "purpose": "existing-benefit attribution",
            "common_feedback_tasks": len(tasks) * 3,
            "corrected_control_tasks": sum(kind in affected for _, kind in tasks),
            "checkpoints": CHECKPOINTS,
            "new_physical_calls": 0,
            "workers": 4,
            "source_files": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_files
            },
        },
    )
    results, diagnostics = [], []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                replay_task,
                sut,
                kind,
                kind in affected,
                {
                    m: runs[sut, kind, m]["path"]
                    for m in ("joint", "random", "historical_collision")
                },
            ): (sut, kind)
            for sut, kind in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(
                OUTPUT / "reports/progress.json",
                {
                    "status": "running",
                    "completed_tasks": len(results),
                    "expected_tasks": len(tasks),
                    "tasks": results,
                },
            )
            print(
                "attribution",
                len(results),
                "/",
                len(tasks),
                result["sut_id"],
                result["scenario"],
                flush=True,
            )
        futures = [
            pool.submit(diagnose_task, sut, kind, runs[sut, kind, "joint"]["path"])
            for sut, kind in tasks
        ]
        for future in as_completed(futures):
            diagnostics.extend(future.result())
    csv_report(OUTPUT / "reports/collision_diagnostics.csv", diagnostics)
    results.sort(key=lambda r: (r["sut_id"], r["scenario"]))
    write_json(
        OUTPUT / "reports/progress.json",
        {
            "status": "complete",
            "completed_tasks": len(tasks),
            "expected_tasks": len(tasks),
            "tasks": results,
            "new_physical_calls": 0,
        },
    )
    return results


if __name__ == "__main__":
    main()
