"""Recompute the registered public ADS driving cases and low-RSS slices."""

import csv
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.budget.protocol import BUDGET, INITIAL, SEED, sha
from experiments.budget.report import write_csv
from experiments.collect_transfer_records import ApprovedCase
from experiments.executions import ExecutionBank
from experiments.initialization import initial_indices
from experiments.parallel import initialize_worker, prepare_processes
from experiments.protocol import ROOT
from experiments.public_ads.evaluate_e01 import ORIGINAL, OUTPUT, TARGETS
from experiments.record_audit import BankTruth, _check_specification, _queries
from experiments.resume import paid_feedback_hash, trajectory_hash
from experiments.rss import rss_values, trajectory_margin
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.storage import read_json, write_json
from experiments.transfer_execution import _paths
from experiments.weight_ablation.rss_ablation import TAIL_RANK, first_hit

BUDGET_ROOT = OUTPUT / "budget_discovery"
WEIGHT_ROOT = OUTPUT / "weight_adaptation"


def sealed(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Public result no longer matches its published audit: {path}")
    return read_json(path)


def audited_result(kind, target, method, dynamic=True):
    destination = BUDGET_ROOT if dynamic else WEIGHT_ROOT
    summary = read_json(destination / "reports/summary.json")
    audit_path = destination / "audits" / kind / f"{target}.json"
    audit = sealed(audit_path, summary["files"][str(audit_path.relative_to(ROOT))])
    case = ApprovedCase("budget", target, kind, SEED, method, None)
    result, paid = _paths(destination, case)
    if dynamic:
        checksum = audit["result_sha256"][method]
    else:
        checksum = audit["frozen_budget"]["result_sha256"]
        if sha(paid) != audit["frozen_budget"]["paid_sha256"]:
            raise ValueError("Frozen RSS ledger changed after publication")
    saved, receipt = sealed(result, checksum), read_json(paid)
    ledger = saved["query_ledger"]
    if (receipt["status"] != "complete" or receipt["query_ledger"] != ledger
            or receipt["paid_feedback_hash"] != paid_feedback_hash(ledger)
            or receipt["specification"]["task"] != saved["specification"]):
        raise ValueError("Published public result differs from its paid ledger")
    return saved, receipt


def rss_pair(key):
    kind, target = key
    bank = ExecutionBank(OUTPUT / "physical/target" / kind / target)
    truth = BankTruth(bank)
    base = dict(scene=kind, target=target, seed=SEED)
    margins = rss_values(truth.margins)
    threshold = float(np.sort(margins)[TAIL_RANK - 1])
    tail = margins <= threshold
    tail_count = int(tail.sum())
    pair = {**base, "threshold": threshold, "tail_count": tail_count,
            "pool_minimum": float(margins.min())}
    curves = []
    for method, dynamic in (("dynamic", True), ("frozen", False)):
        saved, receipt = audited_result(kind, target, "full", dynamic=dynamic)
        spec, result = saved["specification"], saved["result"]
        order = result["selected_indices"]
        if (len(order) != BUDGET or len(set(order)) != BUDGET or
                order[:INITIAL] != initial_indices(len(margins), INITIAL, SEED) or
                spec["update_source_weights"] is not dynamic or
                result["source_weight_updates"] is not dynamic or
                not result["target_feedback_updates"] or result["feedback_used"] != "joint"):
            raise ValueError("Public RSS slice requires a matched E03 protocol")
        _check_specification(saved, receipt["specification"]["task"], truth)
        _queries(saved, truth, order, INITIAL, True)
        if receipt["trajectory_hashes"] != [trajectory_hash(bank(bank.candidates[i]).trajectory) for i in order]:
            raise ValueError("Public RSS slice must use the original paid trajectories")
        observed = margins[order]
        minima, hits = np.minimum.accumulate(observed), np.cumsum(tail[order])
        first = first_hit(observed, minima, threshold)
        pair.update({f"{method}_first_hit": first, f"{method}_censored": first is None,
                     f"{method}_wait": BUDGET if first is None else first,
                     f"{method}_tail_hits": int(hits[-1]), f"{method}_tail_recall": float(hits[-1] / tail_count)})
        curves.extend({**base, "method": method, "query": q, "candidate_index": i,
                       "candidate_id": bank.candidates[i].scenario_id, "rss_margin": float(value),
                       "minimum_rss_margin": float(low), "tail_hits": int(count),
                       "tail_recall": float(count / tail_count)}
                      for q, (i, value, low, count) in enumerate(zip(order, observed, minima, hits, strict=True), 1))
    if [r["candidate_index"] for r in curves[:INITIAL]] != [
        r["candidate_index"] for r in curves[BUDGET:BUDGET + INITIAL]
    ]:
        raise ValueError("Public RSS slices do not share the common initialization")
    pair.update(initial_hit=curves[INITIAL - 1]["tail_hits"] > 0,
                delta_tail_hits=pair["dynamic_tail_hits"] - pair["frozen_tail_hits"],
                delta_wait=pair["dynamic_wait"] - pair["frozen_wait"])
    return pair, curves


def rss_report():
    prepare_processes()
    keys = [(kind, target) for kind in FORMAL_SCENARIOS for target in TARGETS]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        data = list(pool.map(rss_pair, keys))
    pairs = [p for p, _ in data]
    root = WEIGHT_ROOT / "reports/rss_discovery"
    write_csv(root / "paired_first_hit.csv", pairs)
    write_csv(root / "rss_curves.csv", [r for _, rows in data for r in rows])
    summaries = {}
    for field, values in (("overall", (None,)), ("scene", FORMAL_SCENARIOS), ("target", TARGETS)):
        for value in values:
            group = [r for r in pairs if value is None or r[field] == value]
            summaries[value or field] = {
                "pairs": len(group), "initial_hit_pairs": sum(r["initial_hit"] for r in group),
                **{metric: {m: float(np.mean([r[f"{m}_{metric}"] for r in group]))
                            for m in ("dynamic", "frozen")} for metric in ("tail_hits", "tail_recall", "wait")},
                "hit_pairs": {m: sum(not r[f"{m}_censored"] for r in group) for m in ("dynamic", "frozen")},
                "earlier_tie_later": [sum(sign * r["delta_wait"] > 0 if sign else r["delta_wait"] == 0
                                          for r in group) for sign in (-1, 0, 1)],
                "more_tie_fewer": [sum(sign * r["delta_tail_hits"] > 0 if sign else r["delta_tail_hits"] == 0
                                       for r in group) for sign in (1, 0, -1)],
            }
    write_json(root / "paired_summary.json", {
        "status": "complete", "tail_rank": TAIL_RANK, "ties": "include all tied at the threshold",
        "wait_if_censored": BUDGET, "new_physical_executions": 0, "logical_queries_added": 0,
        "native_trajectories": 24000, "scopes": summaries,
        "files": {name: sha(root / name) for name in ("paired_first_hit.csv", "rss_curves.csv")},
    })


def replay_decisions(kind, target, trajectory):
    """Run the frozen policy only on recorded states and compare every requested control."""
    from controllers.highway.public_agents import create
    from controllers.highway.decision_interface import public_world
    from simulator.highway.collision_env import default_lane_centers
    from simulator.highway.observation import VehicleState, VisibleWorld
    from simulator.highway.timing import DECISION_DT, PHYSICS_PER_DECISION

    policy = create(target)
    decisions = []
    for frame_index in range(0, len(trajectory) - 1, PHYSICS_PER_DECISION):
        frame, successor = trajectory[frame_index:frame_index + 2]
        if not math.isclose(frame["time_s"], len(decisions) * DECISION_DT, abs_tol=1e-9):
            raise ValueError("Native decision frames do not match the registered frequency")
        actors = {row["actor_id"]: VehicleState(**row) for row in frame["actors"]}
        world = VisibleWorld(frame["time_s"], actors["ego"], tuple(actors[i] for i in frame["visible_ids"]),
                             default_lane_centers(kind))
        extra = {}
        if target == "online_mc_ppo":
            driver = policy.policy.driver
            driver.update(public_world(world))
            available = driver.get_available_actions()
            extra["available_actions"] = [int(a) for a in available]
        command = policy.act(world).limited()
        if not all(math.isclose(v, successor["requested_control"][k], abs_tol=1e-10)
                   for k, v in vars(command).items()):
            raise ValueError(f"Frozen policy command disagrees with archived native frame {frame_index}")
        if extra:
            rejected = policy.policy.last_decision["shield_rejected"]
            extra["maude_rejected"] = [a for a in rejected if a in available]
            extra["unavailable_rejected"] = [a for a in rejected if a not in available]
        decisions.append(dict(time_s=world.time_s, **policy.policy.last_decision,
                              policy_control=vars(command), **extra))
    return decisions


def original_response(kind, scene):
    from controllers.highway.catalog import FORMAL_TARGET_SUTS

    responses = {}
    for sut in FORMAL_TARGET_SUTS:
        root = ORIGINAL if sut.sut_id != "target_dqn" else ROOT / "results/dqn_continual_learning/k_shot_adaptation"
        bank = ExecutionBank(root / "physical/target" / kind / sut.sut_id)
        responses[sut.sut_id] = bank(scene).feedback.collision
    return responses


def case_report():
    report = BUDGET_ROOT / "reports"
    summary = read_json(report / "summary.json")
    summary_path = report / "task_results.csv"
    if sha(summary_path) != summary["tables_sha256"]["task_results.csv"]:
        raise ValueError("Public E02 comparison changed after publication")
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    collisions = {(r["scenario"], r["target"], r["method"]): int(r["C200"]) for r in table}
    tasks = sorted((s, t) for s in FORMAL_SCENARIOS for t in TARGETS)
    deltas = {key: collisions[(*key, "full")] - collisions[(*key, "target_only")] for key in tasks}
    decisions = []
    for kind, target in (min(tasks, key=deltas.get), max(tasks, key=deltas.get)):
        delta = deltas[kind, target]
        winner, loser = ("full", "target_only") if delta > 0 else ("target_only", "full")
        a, receipt = audited_result(kind, target, winner)
        b, _ = audited_result(kind, target, loser)
        other = set(b["result"]["selected_indices"])
        hit = next(r for r in a["query_ledger"] if r["index"] not in other and r["feedback"]["collision"])
        bank = ExecutionBank(OUTPUT / "physical/target" / kind / target)
        path = bank.execution_path(hit["index"])
        raw = read_json(path)
        actor_frames = raw["trajectory"]
        if trajectory_hash(actor_frames) != receipt["trajectory_hashes"][hit["call"] - 1]:
            raise ValueError("Driving case differs from its audited paid execution")
        query = a["result"]["queries"][hit["call"] - 1]
        weight_audit_path = WEIGHT_ROOT / "audits" / kind / f"{target}.json"
        weight_summary = read_json(WEIGHT_ROOT / "reports/summary.json")
        weight_audit = sealed(weight_audit_path, weight_summary["files"][str(weight_audit_path.relative_to(ROOT))])
        e03 = sealed(OUTPUT / "k_shot" / target / kind / "11/full.json.gz",
                     weight_audit["dynamic_prediction"]["result_sha256"])
        weights = [{"query": q, "history": sum(row["component_weights_after_query"][:-2]),
                    "learned_target": row["component_weights_after_query"][-2],
                    "raw_target": row["component_weights_after_query"][-1]}
                   for q, row in enumerate(e03["result"]["queries"], 1)]
        decisions.append({
            "scene": kind, "target": target, "delta_c200": delta, "finding_method": winner,
            "query": hit["call"], "candidate_index": hit["index"], "scenario": raw["scenario"],
            "original_eight_collisions": original_response(kind, bank.candidates[hit["index"]]),
            "native_path": path.relative_to(ROOT).as_posix(), "native_sha256": sha(path),
            "native_events": raw["events"], "trajectory_margin": trajectory_margin(actor_frames),
            "collision_time": actor_frames[-1]["time_s"], "partner": actor_frames[-1]["collision_partner"],
            "prequery_probability": query["prequery_probability"],
            "other_method_snapshots": [{"paid": snap["paid_queries"], "probability": snap["probabilities"][hit["index"]]}
                                       for snap in b["result"]["prediction_snapshots"]],
            "decision_source": "frozen policy inference with exact archived requested-control verification",
            "decisions": replay_decisions(kind, target, actor_frames), "e03_common_feedback_weights": weights,
        })
    write_json(report / "driving_cases.json", {"new_physical_executions": 0, "cases": decisions})


if __name__ == "__main__":
    case_report()
    rss_report()
