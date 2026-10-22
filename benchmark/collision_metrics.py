"""Evaluator-only collision counting, observable behaviors and paired statistics."""

import math
from collections import Counter

import numpy as np

from .scenarios import dedup_key


def behavior_signature(scenario, trajectory):
    collided = [(i, row) for i, row in enumerate(trajectory) if row["collision"]]
    if not collided:
        raise ValueError("Behavior signatures require an actual ego collision")
    index, frame = collided[0]
    actors = {a["actor_id"]: a for a in frame["actors"]}
    ego, partner = actors["ego"], actors[frame["collision_partner"]]
    dx, dy = partner["x"] - ego["x"], partner["y"] - ego["y"]
    c, s = math.cos(ego["heading"]), math.sin(ego["heading"])
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy
    normalized_x = local_x / ((ego["length"] + partner["length"]) / 2)
    normalized_y = local_y / ((ego["width"] + partner["width"]) / 2)
    if abs(normalized_x) >= abs(normalized_y):
        contact = "front" if local_x >= 0 else "rear"
    else:
        contact = "right" if local_y >= 0 else "left"
    previous = next((r for r in reversed(trajectory[:index + 1])
                     if frame["time_s"] - r["time_s"] >= 0.4), trajectory[0])
    previous_ego = next(a for a in previous["actors"] if a["actor_id"] == "ego")
    displacement_y = ego["y"] - previous_ego["y"]
    lateral = ("right" if displacement_y > 0.25 else
               "left" if displacement_y < -0.25 else "lane_keep")
    acceleration = frame["control"]["acceleration"]
    longitudinal = "brake" if acceleration < -0.5 else "accelerate" if acceleration > 0.5 else "coast"
    p, time_s = scenario.parameters, frame["time_s"]
    if scenario.kind == "lane_change_rear":
        changing = any(abs(r["actors"][0]["y"] - trajectory[0]["actors"][0]["y"]) > 0.25
                       for r in trajectory[:index + 1])
        settled = changing and abs(ego["y"]) < 0.3 and abs(ego["heading"]) < 0.1
        phase = "post_maneuver" if settled else "active_maneuver" if changing else "pre_maneuver"
    else:
        onset = p["event_time_s"]
        if scenario.kind == "front_brake":
            stopped = [r["time_s"] for r in trajectory if r["time_s"] >= onset
                       and any(a["actor_id"] == "lead" and a["speed"] < 0.02 for a in r["actors"])]
            end = min(stopped, default=math.inf)
        else:
            end = onset + p["lateral_duration_s"]
        phase = "pre_event" if time_s < onset else "active_event" if time_s < end else "post_event"
    return partner["actor_id"], contact, (lateral, longitudinal), phase


def summarize(candidates, queries, trajectories):
    if not queries or len(queries) != len(trajectories):
        raise ValueError("Each paid query requires its matching evaluator trajectory")
    unique, behaviors = set(), set()
    behavior_counts = {name: Counter() for name in ("collision_object", "contact_direction", "ego_action", "event_phase")}
    curve, coverage, actual = [], [], []
    first = None
    probability_errors = []
    for position, (query, trajectory) in enumerate(zip(queries, trajectories), 1):
        scenario = candidates[query["index"]]
        feedback = query["feedback"]
        if feedback["scenario_id"] != scenario.scenario_id:
            raise ValueError("Metric query/candidate identity mismatch")
        label = feedback["collision"]
        if type(label) is not bool or bool(any(r["collision"] for r in trajectory)) != label:
            raise ValueError("Collision feedback must agree with the independently logged execution")
        if label:
            if first is None:
                first = position
            unique.add(dedup_key(scenario))
            signature = behavior_signature(scenario, trajectory)
            behaviors.add(signature)
            for name, value in zip(behavior_counts, signature):
                behavior_counts[name]["/".join(value) if isinstance(value, tuple) else value] += 1
        curve.append(len(unique))
        coverage.append(len(behaviors))
        actual.append(int(label))
        probability = query.get("prequery_probability")
        if probability is not None:
            probability_errors.append((probability - int(label))**2)
    violation_counts = Counter(v for q in queries for v in q["feedback"]["safety"]["violations"])
    margins = [q["feedback"]["safety"]["min_clearance_m"] for q in queries
               if q["feedback"]["safety"]["min_clearance_m"] is not None]
    ttcs = [q["feedback"]["safety"]["min_ttc_s"] for q in queries
            if q["feedback"]["safety"]["min_ttc_s"] is not None]
    return {"queries": len(queries), "unique_collisions": len(unique),
            "collision_executions": sum(actual), "discovery_curve": curve,
            "discovery_auc": float(np.mean(curve)),
            "behavior_coverage": len(behaviors), "behavior_curve": coverage,
            "behavior_signatures": sorted(behaviors),
            "behavior_counts": {name: dict(counts) for name, counts in behavior_counts.items()},
            "first_collision_query": first,
            "first_collision": {"time": first if first is not None else len(queries),
                                "observed": first is not None},
            "brier": float(np.mean(probability_errors)) if probability_errors else None,
            "violation_counts": dict(violation_counts),
            "minimum_clearance_m": min(margins, default=None),
            "minimum_ttc_s": min(ttcs, default=None),
            "equivalent_execution_s": sum(q["feedback"]["execution_s"] for q in queries),
            "selection_s": sum(q.get("selection_s", 0) for q in queries),
            "adaptation_s": sum(q.get("adaptation_s", 0) for q in queries)}


def equal_count_coverage(summaries):
    common = min(s["unique_collisions"] for s in summaries)
    return {"common_maximum": common, "counts": list(range(1, common + 1)),
            "coverage": [[s["behavior_curve"][next(i for i, n in enumerate(s["discovery_curve"]) if n >= k)]
                          for k in range(1, common + 1)] for s in summaries]}


def kaplan_meier(first_collisions, *, restriction=None):
    if not first_collisions:
        raise ValueError("Survival analysis requires all runs, including censored ones")
    cutoff = max(r["time"] for r in first_collisions) if restriction is None else restriction
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("Survival restriction must be finite and nonnegative")
    times = sorted({r["time"] for r in first_collisions if r["time"] <= cutoff})
    at_risk, survival, previous, restricted_mean = len(first_collisions), 1.0, 0, 0.0
    curve = [{"time": 0, "survival": 1.0, "at_risk": at_risk}]
    for time in times:
        restricted_mean += survival * (time - previous)
        events = sum(r["time"] == time and r["observed"] for r in first_collisions)
        censored = sum(r["time"] == time and not r["observed"] for r in first_collisions)
        survival *= 1 - events / at_risk
        curve.append({"time": time, "survival": survival, "at_risk": at_risk,
                      "events": events, "censored": censored})
        at_risk -= events + censored
        previous = time
    if cutoff > previous:
        if at_risk == 0 and survival > 0:
            raise ValueError("Follow-up cannot identify survival beyond the last censored observation")
        restricted_mean += survival * (cutoff - previous)
        curve.append({"time": cutoff, "survival": survival, "at_risk": at_risk,
                      "events": 0, "censored": 0})
    return {"curve": curve, "restricted_mean_queries": restricted_mean,
            "restriction": cutoff, "censored_runs": sum(not r["observed"] for r in first_collisions)}


def lineage_survival(records, *, restriction):
    """Average implementation/training lineages equally, preserving within-lineage censoring."""
    lineages = sorted({r["lineage"] for r in records})
    if not lineages:
        raise ValueError("Lineage survival requires completed experimental runs")
    grouped = {name: [r["first_collision"] for r in records if r["lineage"] == name] for name in lineages}
    summaries = {name: kaplan_meier(group, restriction=restriction) for name, group in grouped.items()}
    times = sorted({point["time"] for summary in summaries.values() for point in summary["curve"]})
    values = []
    for summary in summaries.values():
        curve = summary["curve"]
        positions = np.searchsorted([p["time"] for p in curve], times, side="right") - 1
        values.append([curve[i]["survival"] for i in positions])
    return {"aggregation": "equal_weight_lineage_kaplan_meier", "restriction": restriction,
            "independent_lineages": len(lineages), "runs": len(records),
            "curve": [{"time": time, "survival": float(value)}
                      for time, value in zip(times, np.mean(values, axis=0))],
            "restricted_mean_queries": float(np.mean([s["restricted_mean_queries"] for s in summaries.values()])),
            "censored_runs": sum(s["censored_runs"] for s in summaries.values()),
            "by_lineage": {name: {"runs": len(grouped[name]), "censored_runs": s["censored_runs"],
                                  "restricted_mean_queries": s["restricted_mean_queries"]}
                           for name, s in summaries.items()}}


def paired_lineage_bootstrap(records, method, baseline, metric, *, repetitions=2000, seed=1501):
    """Pair before aggregation; implementation lineages are the outer sampling units."""
    pair_keys = ("lineage", "target_id", "scenario", "condition", "environment_seed", "algorithm_seed")
    paired = {}
    for row in records:
        if row["method"] in (method, baseline):
            key = tuple(row[k] for k in pair_keys)
            group = paired.setdefault(key, {})
            if row["method"] in group:
                raise ValueError("Duplicate paired experimental run")
            group[row["method"]] = row[metric]
    if not paired or any(set(row) != {method, baseline} for row in paired.values()):
        raise ValueError("Every compared run must have its exact paid-budget counterpart")
    lineages = sorted({k[0] for k in paired})
    grouped = {lineage: np.asarray([r[method] - r[baseline] for k, r in paired.items()
                                   if k[0] == lineage], dtype=float) for lineage in lineages}
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        selected = generator.choice(lineages, len(lineages), replace=True)
        estimates.append(np.mean([generator.choice(grouped[lineage], len(grouped[lineage]),
                                                  replace=True).mean() for lineage in selected]))
    return {"method": method, "baseline": baseline, "metric": metric,
            "difference": float(np.mean([grouped[lineage].mean() for lineage in lineages])),
            "interval_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "independent_lineages": len(lineages), "paired_runs": len(paired),
            "lineage_differences": {lineage: float(values.mean()) for lineage, values in grouped.items()}}
