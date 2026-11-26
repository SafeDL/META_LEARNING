"""Bounded obstacle-removal controls for the approved cut-out interaction scope."""

import hashlib
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

from controllers.highway.catalog import FORMAL_HISTORY_SUTS, FORMAL_TARGET_SUTS
from controllers.highway.factory import create
from simulator.highway.collision_env import CollisionWorld

from experiments.scene_design.audit_interactions import mechanism_event
from experiments.scene_design.design_interactions import OUTPUT, auxiliary_executions, inspect, reference, run_chunk
from experiments.executions import execution_contract
from experiments.parallel import initialize_worker
from experiments.protocol import ROOT
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from experiments.storage import read_json, write_json

DIRECTORY = OUTPUT / "cut_out_rear_gap" / "initial"
TARGET_ORDER = {"target_uct": [115, 35, 71, 118], "target_beam": [2, 9, 91, 97]}
REFERENCE_MODES = ("coast_lane", "reveal", "brake", "offset", "brake_escape")


class NoObstacleWorld(CollisionWorld):
    """Remove only the obstacle before native road population and initial recording."""

    def _create_traffic(self):
        super()._create_traffic()
        del self.actors["obstacle"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(directory):
    path = directory / "secondary_protocol.json"
    specification = {
        "scope": "user-approved reveal-related evasive secondary collisions",
        "base_protocol_sha256": digest(directory / "extended_protocol.json"),
        "original_summary_sha256": digest(directory / "summary.json"),
        "original_audit_sha256": digest(directory / "audit.json"),
        "implementation_sha256": digest(ROOT / "experiments/scene_design/verify_secondary.py"),
        "intervention": "remove obstacle at creation; preserve all vehicles, intentions, ADS and seed",
        "maximum_counterfactual_executions": 8,
        "maximum_auxiliary_executions_total": 64,
        "maximum_regular_ads_executions_total": 2048,
        "target_candidate_order": TARGET_ORDER,
        "history_candidate_order": "valid secondary collisions with existing safe references first, then ordinal",
        "requirements": [
            "real collision with lead or rear guard after native obstacle reveal and lead cut-out",
            "identical states, visibility excluding obstacle and controls through first responsive decision",
            "requested controls diverge after reveal and before factual impact",
            "no-obstacle control completes 10 seconds with no collision or road departure",
            "same-coordinate legal reference with obstacle and all vehicles retained",
        ],
        "formal_executions": 0,
        "training_steps": 0,
    }
    # JSON converts integer keys/tuples, so compare in its stored representation.
    if path.exists():
        previous = read_json(path)
        assert previous == specification, "Approved secondary protocol changed"
    else:
        write_json(path, specification)
    return read_json(path)


def compare_pair(factual, counterfactual):
    trace, control = factual["trajectory"], counterfactual["trajectory"]
    _, onset, reveal = mechanism_event(factual)
    crash = next((f for f in trace if f["collision"]), None)
    responsive_time = None if reveal is None else round(math.ceil(reveal * 5 - 1e-9) / 5, 6)
    lookup = {f["time_s"]: f for f in control}

    def without_obstacle(frame):
        return {
            **frame,
            "actors": [a for a in frame["actors"] if a["actor_id"] != "obstacle"],
            "visible_ids": [name for name in frame["visible_ids"] if name != "obstacle"],
        }

    prefix = [f for f in trace if responsive_time is not None and f["time_s"] <= responsive_time]
    prefix_matches = bool(prefix) and all(
        f["time_s"] in lookup and without_obstacle(f) == lookup[f["time_s"]] for f in prefix
    )
    difference = next((
        round(f["time_s"] - 0.05, 6) for f in trace[1:]
        if f["time_s"] in lookup and f["requested_control"] != lookup[f["time_s"]]["requested_control"]
    ), None)
    result = inspect(Scenario(**counterfactual["scenario"]), control, counterfactual["violations"])
    no_obstacle = all(all(a["actor_id"] != "obstacle" for a in f["actors"]) for f in control)
    safe = not result["collision"] and not result["violations"] and result["end_time_s"] == 10 and result["native_valid"]
    eligible = bool(
        crash and crash["collision_partner"] in ("lead", "guard_rear")
        and onset is not None and reveal is not None and 0 < reveal < crash["time_s"]
        and onset <= crash["time_s"] and factual["summary"]["native_valid"]
        and not factual["summary"]["violations"]
    )
    changed_after_reveal = bool(difference is not None and responsive_time is not None and responsive_time <= difference < crash["time_s"]) if crash else False
    return {
        "confirmed": bool(eligible and no_obstacle and prefix_matches and changed_after_reveal and safe),
        "collision_partner": crash["collision_partner"] if crash else None,
        "factual_collision_time_s": crash["time_s"] if crash else None,
        "obstacle_first_visible_s": reveal,
        "first_responsive_decision_s": responsive_time,
        "prefix_frames_matching": len(prefix) if prefix_matches else 0,
        "prefix_matches": prefix_matches,
        "first_control_difference_s": difference,
        "counterfactual_safe": safe,
        "counterfactual_collision": result["collision"],
        "counterfactual_collision_partner": result["collision_partner"],
        "counterfactual_end_time_s": result["end_time_s"],
        "counterfactual_violations": result["violations"],
        "counterfactual_minimum_body_clearance_m": result["minimum_body_clearance_m"],
        "counterfactual_minimum_road_margin_m": result["minimum_road_margin_m"],
    }


def counterfactual(sut, ordinal, directory):
    key = f"{sut.sut_id}_{ordinal:03d}"
    path = directory / "counterfactual" / f"{key}.json.gz"
    factual_path = directory / sut.sut_id / f"{ordinal:03d}.json.gz"
    factual = read_json(factual_path)
    if path.exists():
        payload = read_json(path)
        assert payload["factual_sha256"] == digest(factual_path)
        return compare_pair(factual, payload)
    attempts = directory / "counterfactual" / "attempts"
    attempt = attempts / f"{key}.json"
    if attempt.exists():
        raise ValueError("An interrupted control cannot silently spend another execution")
    if len(list(attempts.glob("*.json"))) >= 8 or auxiliary_executions(directory) >= 64:
        return None
    protocol = read_json(directory / "extended_protocol.json")
    assert execution_contract(sut, kind="cut_out_rear_gap") == protocol["executions"][sut.sut_id]
    write_json(attempt, {"status": "running", "sut_id": sut.sut_id, "ordinal": ordinal})
    policy = create(sut.sut_id)
    scene = Scenario(**factual["scenario"])
    started = time.perf_counter()
    world = NoObstacleWorld(scene, environment_seed=81001, duration_s=10)
    try:
        world.reset(seed=81001)
        while not world.terminal:
            world.advance(policy.act(world.visible_world()))
        feedback = QueryFeedback(
            scene.scenario_id, bool(world.ego.crashed),
            SafetyMeasurements(world.min_clearance, world.min_ttc, tuple(sorted(world.violations))),
            time.perf_counter() - started,
        )
        payload = {
            "scenario": asdict(scene), "sut_id": sut.sut_id,
            "intervention": "remove_obstacle_at_creation", "environment_seed": 81001,
            "factual_sha256": digest(factual_path),
            "execution_contract": protocol["executions"][sut.sut_id],
            "secondary_protocol_sha256": digest(directory / "secondary_protocol.json"),
            "feedback": feedback.to_dict(), "violations": sorted(world.violations),
            "trajectory": world.trace,
        }
        write_json(path, payload)
        write_json(attempt, {"status": "complete", "sut_id": sut.sut_id, "ordinal": ordinal})
    finally:
        world.close()
    result = compare_pair(factual, payload)
    print(sut.sut_id, ordinal, result, flush=True)
    return result


def existing_reference(directory, ordinal):
    for mode in REFERENCE_MODES:
        path = directory / "reference" / f"{ordinal:03d}_{mode}.json.gz"
        if path.exists() and read_json(path)["summary"]["reference_passed"]:
            return mode
    return None


def find_reference(directory, scene):
    ordinal = int(scene.scenario_id.rsplit(":", 1)[-1])
    mode = existing_reference(directory, ordinal)
    if mode:
        return mode
    for mode in REFERENCE_MODES:
        path = directory / "reference" / f"{ordinal:03d}_{mode}.json.gz"
        if not path.exists() and auxiliary_executions(directory) >= 64:
            break
        if reference(scene, directory, mode)["reference_passed"]:
            return mode
    return None


def evaluate(sut, group, directory):
    payloads = [read_json(directory / sut.sut_id / f"{i:03d}.json.gz") for i in range(128)]
    results = [p["summary"] for p in payloads]
    direct = [i for i, r in enumerate(results) if r["mechanism_collision"] and r["native_valid"] and not r["violations"]]
    safe = sum(not r["collision"] and not r["violations"] and r["native_valid"] and r["end_time_s"] == 10 for r in results)
    certificate = None
    for i in sorted(direct, key=lambda i: (existing_reference(directory, i) is None, i)):
        mode = find_reference(directory, group[i])
        if mode:
            certificate = {"ordinal": i, "mode": mode, "mechanism": "direct_obstacle"}
            break
    checks = []
    if not certificate:
        candidates = [i for i, r in enumerate(results) if r["collision"] and r["collision_partner"] in ("lead", "guard_rear") and r["native_valid"] and not r["violations"] and r["obstacle_first_visible_s"] is not None and 0 < r["obstacle_first_visible_s"] < r["collision_time_s"]]
        preferred = TARGET_ORDER.get(sut.sut_id, [])
        candidates.sort(key=lambda i: (preferred.index(i) if i in preferred else len(preferred), existing_reference(directory, i) is None, i))
        for i in candidates:
            mode = find_reference(directory, group[i])
            if not mode:
                continue
            check = counterfactual(sut, i, directory)
            if check is None:
                break
            checks.append({"ordinal": i, "reference_mode": mode, **check})
            if check["confirmed"]:
                certificate = {"ordinal": i, "mode": mode, "mechanism": "verified_secondary"}
                break
    return {
        "sut_id": sut.sut_id, "records": 128,
        "collisions": sum(r["collision"] for r in results),
        "direct_obstacle_collisions": len(direct), "safe": safe,
        "verified_secondary_collisions": sum(c["confirmed"] for c in checks),
        "secondary_checks": checks, "avoidable_representative": certificate,
        "coverage_passed": bool(safe and certificate),
    }


def run(directory=DIRECTORY):
    register(directory)
    protocol = read_json(directory / "extended_protocol.json")
    group = [Scenario(**s) for s in protocol["scenes"]]
    rows = []
    for role, systems in (("target", FORMAL_TARGET_SUTS), ("history", FORMAL_HISTORY_SUTS)):
        if role == "history":
            tasks = [(sut, group[i:i+8], directory) for sut in systems for i in range(0, 128, 8)]
            with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context("spawn"), initializer=initialize_worker) as pool:
                futures = [pool.submit(run_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    sut_id, completed = future.result()
                    print("history", sut_id, completed[-1]["scenario_id"], flush=True)
        for sut in systems:
            row = evaluate(sut, group, directory)
            rows.append(row)
            print("coverage", {k: v for k, v in row.items() if k != "secondary_checks"}, flush=True)
        passed = all(r["coverage_passed"] for r in rows)
        result = {
            "status": "development_coverage_passed" if passed and role == "history" else "target_passed" if passed else "coverage_insufficient",
            "criterion": "direct obstacle or counterfactually verified evasive secondary collision",
            "protocol_file": "secondary_protocol.json", "count_per_ads": 128,
            "ads_executions": len(rows) * 128,
            "reference_executions": len(list((directory / "reference").glob("*.json.gz"))),
            "counterfactual_executions": len(list((directory / "counterfactual" / "attempts").glob("*.json"))),
            "auxiliary_executions": auxiliary_executions(directory),
            "rows": rows, "formal_executions": 0, "training_steps": 0,
        }
        write_json(directory / "secondary_summary.json", result)
        if not passed:
            break
    return result


if __name__ == "__main__":
    run()
