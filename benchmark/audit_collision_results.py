"""Read-only reconstruction of final finite-pool metrics from paid raw traces.

This validator does not call the production metric summarizer or behavior labeler.
It checks completed runs only and cannot mark an unfinished study matrix complete.
No SUT, encoder, selector, or simulator is executed.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import Polygon

from .initialization import initial_indices
from .protocol import DATA_ROOT, DEFAULT, RESULT_ROOT, ROOT, method_contract, validate_sut_freeze
from .scenarios import DOMAINS, sample
from .studies import SCOPES, tasks


OUTPUT = ROOT / "results/benchmark/collision/final_result_audit.json"


@lru_cache(maxsize=None)
def registered_pool(kind, condition, count):
    return [asdict(scenario) for scenario in sample(kind, count, condition)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, location="value"):
    """Compare JSON-shaped evidence without confusing Boolean labels and numbers."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{location}: keys differ")
        for key, value in expected.items():
            compare(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{location}: lengths differ")
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{location}[{index}]")
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
                f"{location}: numerical values differ ({actual}, {expected})")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{location}: values differ")


def polygon(actor):
    c, s = math.cos(actor["heading"]), math.sin(actor["heading"])
    return Polygon([(actor["x"] + c * x - s * y, actor["y"] + s * x + c * y)
                    for x, y in ((-actor["length"] / 2, -actor["width"] / 2),
                                 (actor["length"] / 2, -actor["width"] / 2),
                                 (actor["length"] / 2, actor["width"] / 2),
                                 (-actor["length"] / 2, actor["width"] / 2))])


def time_to_overlap(ego, other):
    """Scalar separating-axis intersection of the two moving rectangle intervals."""
    earliest, latest = 0.0, math.inf
    dx, dy = other["x"] - ego["x"], other["y"] - ego["y"]
    vx = other["speed"] * math.cos(other["heading"]) - ego["speed"] * math.cos(ego["heading"])
    vy = other["speed"] * math.sin(other["heading"]) - ego["speed"] * math.sin(ego["heading"])
    for heading in (ego["heading"], other["heading"]):
        for ux, uy in ((math.cos(heading), math.sin(heading)), (-math.sin(heading), math.cos(heading))):
            radius = 0.0
            for actor in (ego, other):
                c, s = math.cos(actor["heading"]), math.sin(actor["heading"])
                radius += abs(ux * c + uy * s) * actor["length"] / 2
                radius += abs(-ux * s + uy * c) * actor["width"] / 2
            position, velocity = ux * dx + uy * dy, ux * vx + uy * vy
            if abs(velocity) < 1e-12:
                if abs(position) > radius:
                    return math.inf
            else:
                a, b = (-radius - position) / velocity, (radius - position) / velocity
                earliest, latest = max(earliest, min(a, b)), min(latest, max(a, b))
                if earliest > latest:
                    return math.inf
    return earliest if latest >= 0 else math.inf


def trace_facts(scenario, trajectory):
    """Reconstruct continuous safety and the registered observable collision label."""
    require(bool(trajectory), "Missing paid trajectory")
    clearance, ttc, violations, signature = math.inf, math.inf, set(), None
    for frame_index, frame in enumerate(trajectory):
        require(type(frame["collision"]) is bool, "Raw collision event is not Boolean")
        actors = {actor["actor_id"]: actor for actor in frame["actors"]}
        ego = actors["ego"]
        body = polygon(ego)
        for name, other in actors.items():
            if name != "ego":
                clearance = min(clearance, body.distance(polygon(other)))
                ttc = min(ttc, time_to_overlap(ego, other))
        upper_lane = 4 if scenario["kind"] == "lane_change_rear" else 8
        if not -1 <= ego["y"] <= upper_lane + 1:
            violations.add("road_departure")
        if not frame["collision"] or signature is not None:
            continue
        other = actors[frame["collision_partner"]]
        c, s = math.cos(ego["heading"]), math.sin(ego["heading"])
        dx, dy = other["x"] - ego["x"], other["y"] - ego["y"]
        longitudinal, lateral = c * dx + s * dy, -s * dx + c * dy
        if abs(longitudinal) / (ego["length"] + other["length"]) >= abs(lateral) / (ego["width"] + other["width"]):
            contact = "front" if longitudinal >= 0 else "rear"
        else:
            contact = "right" if lateral >= 0 else "left"
        earlier = [row for row in trajectory[:frame_index + 1] if frame["time_s"] - row["time_s"] >= 0.4]
        previous = earlier[-1] if earlier else trajectory[0]
        old_ego = next(actor for actor in previous["actors"] if actor["actor_id"] == "ego")
        shift = ego["y"] - old_ego["y"]
        action_y = "right" if shift > 0.25 else "left" if shift < -0.25 else "lane_keep"
        acceleration = frame["control"]["acceleration"]
        action_x = "brake" if acceleration < -0.5 else "accelerate" if acceleration > 0.5 else "coast"
        if scenario["kind"] == "lane_change_rear":
            initial_y = next(actor["y"] for actor in trajectory[0]["actors"] if actor["actor_id"] == "ego")
            moved = any(abs(next(a["y"] for a in row["actors"] if a["actor_id"] == "ego") - initial_y) > 0.25
                        for row in trajectory[:frame_index + 1])
            settled = moved and abs(ego["y"]) < 0.3 and abs(ego["heading"]) < 0.1
            phase = "post_maneuver" if settled else "active_maneuver" if moved else "pre_maneuver"
        else:
            parameters = scenario["parameters"]
            onset = parameters["event_time_s"]
            if scenario["kind"] == "front_brake":
                stops = [row["time_s"] for row in trajectory if row["time_s"] >= onset
                         and any(a["actor_id"] == "lead" and a["speed"] < 0.02 for a in row["actors"])]
                end = min(stops, default=math.inf)
            else:
                end = onset + parameters["lateral_duration_s"]
            phase = "pre_event" if frame["time_s"] < onset else "active_event" if frame["time_s"] < end else "post_event"
        signature = [other["actor_id"], contact, [action_y, action_x], phase]
    return {"collision": signature is not None, "signature": signature,
            "safety": {"min_clearance_m": clearance if math.isfinite(clearance) else None,
                       "min_ttc_s": ttc if math.isfinite(ttc) else None, "violations": sorted(violations)}}


def quantized_coordinate(scenario):
    domain = DOMAINS[scenario["kind"]]
    bins = tuple(math.floor((scenario["parameters"][name] - domain.bounds[name][0]) / step)
                 for name, step in domain.quantization.items())
    return scenario["kind"], bins, tuple(sorted(scenario["categories"].items()))


def reconstruct(queries, scenarios, facts):
    unique, behaviors, curve, coverage, errors = set(), set(), [], [], []
    counts = {name: Counter() for name in ("collision_object", "contact_direction", "ego_action", "event_phase")}
    collisions, first, violations, margins, ttcs = 0, None, Counter(), [], []
    for position, (query, scenario, fact) in enumerate(zip(queries, scenarios, facts), 1):
        label = query["feedback"]["collision"]
        require(type(label) is bool and label == fact["collision"], "Raw collision event differs from query label")
        if label:
            collisions += 1
            first = position if first is None else first
            unique.add(quantized_coordinate(scenario))
            signature = fact["signature"]
            behaviors.add((signature[0], signature[1], tuple(signature[2]), signature[3]))
            for name, value in zip(counts, signature):
                counts[name]["/".join(value) if isinstance(value, list) else value] += 1
        curve.append(len(unique))
        coverage.append(len(behaviors))
        probability = query.get("prequery_probability")
        if probability is not None:
            require(type(probability) in (int, float) and math.isfinite(probability) and 0 <= probability <= 1,
                    "Invalid prequery collision probability")
            errors.append((probability - label) ** 2)
        compare(query["feedback"]["safety"], fact["safety"], "raw safety")
        safety = fact["safety"]
        violations.update(safety["violations"])
        if safety["min_clearance_m"] is not None:
            margins.append(safety["min_clearance_m"])
        if safety["min_ttc_s"] is not None:
            ttcs.append(safety["min_ttc_s"])
    require(len(queries) == len(scenarios) == len(facts) and bool(queries), "Missing query reconstruction input")
    return {"queries": len(queries), "unique_collisions": len(unique), "collision_executions": collisions,
            "discovery_curve": curve, "discovery_auc": sum(curve) / len(curve),
            "behavior_coverage": len(behaviors), "behavior_curve": coverage,
            "behavior_signatures": [[a, b, list(c), d] for a, b, c, d in sorted(behaviors)],
            "behavior_counts": {name: dict(values) for name, values in counts.items()},
            "first_collision_query": first,
            "first_collision": {"time": first if first is not None else len(queries), "observed": first is not None},
            "brier": sum(errors) / len(errors) if errors else None, "violation_counts": dict(violations),
            "minimum_clearance_m": min(margins, default=None), "minimum_ttc_s": min(ttcs, default=None),
            "equivalent_execution_s": sum(q["feedback"]["execution_s"] for q in queries),
            "selection_s": sum(q.get("selection_s", 0) for q in queries),
            "adaptation_s": sum(q.get("adaptation_s", 0) for q in queries)}


class RawEvidence:
    def __init__(self):
        self.banks, self.executions = {}, {}

    def bank(self, root):
        root = Path(root).resolve()
        if root not in self.banks:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            require(manifest["status"] == "complete", "Final run references an unfinished physical bank")
            scheduled, completed = {}, {}
            for row in map(json.loads, (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()):
                if row["status"] == "scheduled":
                    require(row["call"] not in scheduled, "Duplicate physical call number")
                    scheduled[row["call"]] = row
                elif row["status"] == "complete":
                    require(row["call"] not in completed, "Duplicate physical completion")
                    completed[row["call"]] = row
            attempts = {row["call"]: row for row in (json.loads(p.read_text(encoding="utf-8"))
                        for p in (root / "attempts").glob("*.json"))}
            require(len(attempts) == manifest["attempted_calls"], "Physical attempt count differs from manifest")
            require(all(call in scheduled for call in attempts), "Physical attempt has no scheduled call")
            successes = [row for row in attempts.values() if row["status"] == "complete"]
            total = sum(row["execution_s"] for row in successes)
            compare(manifest["physical_execution_s"], total, "physical bank execution cost")
            compare(manifest["successful_calls"], len(successes), "physical successful calls")
            compare(manifest["failed_or_interrupted_calls"], len(attempts) - len(successes), "physical failed attempts")
            compare(manifest["failed_call_wall_s"], sum(row.get("wall_s", 0) for row in attempts.values()
                    if row["status"] == "failed"), "physical failed-call cost")
            compare(manifest["interrupted_call_duration_unknown"], sum(row["status"] == "running"
                    for row in attempts.values()), "unknown interrupted duration")
            self.banks[root] = (manifest, scheduled, completed, attempts)
        return self.banks[root]

    def execution(self, root, index):
        root = Path(root).resolve()
        key = root, index
        if key not in self.executions:
            manifest, scheduled, completed, attempts = self.bank(root)
            path = root / "executions" / f"{index:06d}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                raw = json.load(stream)
            compare(raw["scenario"], manifest["scenarios"][index], "raw coordinate")
            call = raw["call"]
            require(scheduled[call]["index"] == completed[call]["index"] == index,
                    "Raw response does not match the paid call coordinate")
            require(attempts[call]["status"] == "complete", "Raw response has no completed physical attempt")
            compare(raw["feedback"]["execution_s"], attempts[call]["execution_s"], "raw execution cost")
            compare(completed[call]["execution_s"], attempts[call]["execution_s"], "completed ledger execution cost")
            facts = trace_facts(raw["scenario"], raw["trajectory"])
            compare(raw["feedback"]["collision"], facts["collision"], "raw collision")
            compare(raw["feedback"]["safety"], facts["safety"], "raw geometry safety")
            self.executions[key] = {"scenario": raw["scenario"], "feedback": raw["feedback"], "facts": facts,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return self.executions[key]


def verify_run(path, task, raw):
    content = Path(path).read_bytes()
    run = json.loads(content)
    expected = asdict(task)
    specification, result = run["specification"], run["result"]
    compare(specification["metadata"], expected, "registered study task")
    construction = specification["construction"]
    require(construction["budget"] == result["actual_queries"] == DEFAULT.budget, "Final paid budget differs")
    require(construction["initial_count"] == DEFAULT.initial_count and construction["horizon"] == DEFAULT.horizon,
            "Initialization or lookahead differs from the protocol")
    require(construction["method"] == task.method and construction["algorithm_seed"] == task.algorithm_seed,
            "Selector identity differs from the registered task")
    allowed = set(SCOPES[task.scope].history_ids)
    used = set(construction["history_ids"])
    require(used == (set() if task.method in ("random", "space_filling", "gp_collision") else allowed),
            "Final selector used disallowed historical sources")
    root = Path(run["cost"]["physical_bank"])
    manifest, *_ = raw.bank(root)
    require(manifest["sut_id"] == task.target_id and manifest["environment_seed"] == task.environment_seed,
            "Final run references another SUT or environment seed")
    require(manifest["candidate_count"] == DEFAULT.candidate_count, "Final physical pool size differs")
    compare(manifest["scenarios"], registered_pool(task.scenario, task.condition, DEFAULT.candidate_count),
            "registered role/condition coordinates")
    compare(manifest["contract"], construction["bank_contract"], "physical bank contract")
    compare(construction["candidate_ids"], [s["scenario_id"] for s in manifest["scenarios"]], "candidate pool")
    queries, ledger = result["queries"], run["query_ledger"]
    require(len(queries) == len(ledger) == DEFAULT.budget, "Missing paid query")
    selected = [q["index"] for q in queries]
    compare(result["selected_indices"], selected, "selector query order")
    compare(selected[:DEFAULT.initial_count], list(initial_indices(len(manifest["scenarios"]), DEFAULT.initial_count,
                                                                 task.algorithm_seed)), "common initialization")
    require(len(set(selected)) == len(selected), "Finite-pool selector repeated a coordinate")
    scenes, facts = [], []
    for position, (query, record) in enumerate(zip(queries, ledger), 1):
        require(record["status"] == "complete" and record["call"] == position and record["index"] == query["index"],
                "Query ledger order differs from selection")
        require(record["purpose"] == ("initial" if position <= DEFAULT.initial_count else "target")
                and record["account"] == "target", "Initialization or target call was not charged")
        evidence = raw.execution(root, query["index"])
        require(record["scenario_id"] == evidence["scenario"]["scenario_id"], "Paid ledger scenario identity differs")
        compare(record["feedback"], evidence["feedback"], "disclosed physical feedback")
        compare(query["feedback"], evidence["feedback"], "selector feedback")
        scenes.append(evidence["scenario"])
        facts.append(evidence["facts"])
    metrics = reconstruct(queries, scenes, facts)
    compare(run["metrics"], metrics, "reconstructed run metrics")
    cost = run["cost"]
    require(cost["execution_mode"] == "replay" and cost["actual_new_physical_calls"] == 0,
            "Replay run cost claims new physical calls")
    compare(cost["equivalent_selected_execution_s"], metrics["equivalent_execution_s"], "selected execution cost")
    compare(cost["physical_bank_calls"], manifest["attempted_calls"], "physical bank calls")
    compare(cost["physical_bank_execution_s"], manifest["physical_execution_s"], "physical bank cost")
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), **expected, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    # Import paths only; no evaluator or selector is started.
    from .evaluate_collision import task_path
    plan, verified, pending, errors, raw = tasks(), [], 0, [], RawEvidence()
    for task in plan:
        path = task_path(task)
        if not path.exists():
            pending += 1
            continue
        try:
            verified.append(verify_run(path, task, raw))
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
        if (len(verified) + len(errors)) % 128 == 0:
            print("reconstructed final runs", len(verified), "/", len(plan), flush=True)
    index_path = RESULT_ROOT / "run_index.json"
    index_checked = False
    frozen_checked = False
    if verified:
        try:
            frozen = json.loads((DATA_ROOT / "frozen_protocol.json").read_text(encoding="utf-8"))
            validate_sut_freeze(frozen)
            compare(frozen["source_contract"], method_contract(), "final frozen source contract")
            compare(frozen["protocol"], asdict(DEFAULT), "final frozen protocol")
            compare(frozen["study_tasks"], [asdict(task) for task in plan], "final frozen matrix")
            for artifact in frozen["method"]["models"]:
                require(hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"],
                        "A frozen final model changed")
            frozen_checked = True
        except Exception as error:
            errors.append({"path": str(DATA_ROOT / "frozen_protocol.json"), "error": repr(error)})
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            require(index["status"] == "complete" and len(index["runs"]) == len(plan), "Incomplete final run index")
            require(not pending and not errors, "Run index claims completion before reconstruction succeeds")
            by_path = {str(Path(row["path"]).resolve()): row for row in index["runs"]}
            require(len(by_path) == len(plan), "Duplicate run index entry")
            for row in verified:
                compare(by_path[str(Path(row["path"]).resolve())], {k: v for k, v in row.items() if k != "sha256"}, "run index")
            index_checked = True
        except Exception as error:
            errors.append({"path": str(index_path), "error": repr(error)})
    report = {"status": "failed" if errors else "complete" if not pending and index_checked and frozen_checked else "partial",
              "scope": "Final finite-pool runs only; raw geometry safety, collision events, deduplication, behavior, query order and costs. Continuous runs and statistical interpretation are separate.",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "additional_physical_calls": 0,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "method_contract": method_contract(), "expected_runs": len(plan), "verified_runs": len(verified),
              "pending_runs": pending, "index_checked": index_checked, "frozen_protocol_checked": frozen_checked, "errors": errors,
              "unique_raw_executions_checked": len(raw.executions),
              "raw_evidence_sha256": hashlib.sha256(json.dumps(sorted((str(root), index, evidence["sha256"])
                   for (root, index), evidence in raw.executions.items())).encode()).hexdigest(),
              "runs": verified}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(report["status"], len(verified), "/", len(plan), "final runs;", len(errors), "errors", flush=True)
    if errors or arguments.require_complete and report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
