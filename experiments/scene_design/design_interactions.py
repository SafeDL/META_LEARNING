"""Bounded, resumable native development of the three remaining interactions."""

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product

from scipy.stats import qmc

from controllers.highway.catalog import FORMAL_SUTS, FORMAL_HISTORY_SUTS, FORMAL_TARGET_SUTS
from controllers.highway.control import Control, lane_steering
from controllers.highway.factory import create
from simulator.highway.collision_env import CollisionWorld, execute
from simulator.highway.observation import VehicleState
from simulator.highway.timing import timing_contract

from experiments.executions import execution_contract
from experiments.parallel import initialize_worker
from experiments.protocol import ROOT
from experiments.scenarios import DEVELOPMENT_BOUNDS
from experiments.schema import Scenario
from experiments.storage import read_json, write_json

OUTPUT = ROOT / "results/scene_design/interaction"
DEVELOPMENT_SCENARIOS = ("cut_in_onset", "cut_out_rear_gap", "side_encroachment")
TEMPLATES = {}


def scenes(kind, variant="initial", *, count=None):
    count = (32 if kind == "cut_out_rear_gap" else 128) if count is None else count
    if kind not in DEVELOPMENT_SCENARIOS or variant != "initial":
        raise ValueError("Only the three retained initial development domains are supported")
    bounds = DEVELOPMENT_BOUNDS[kind]
    points = [*product((0, 1), repeat=4), (0.5,) * 4]
    points.extend(qmc.Sobol(4, scramble=True, seed=41011).random_base2(10))
    result, seen = [], set()
    sides = (None,) if kind.startswith("cut_out") else ("left", "right")
    for point in points:
        parameters = {n: lo + min(int(v * (hi - lo + 1)), hi - lo) for (n, (lo, hi)), v in zip(bounds.items(), point)}
        for side in sides:
            scene = Scenario(f"interaction-design:{kind}:{variant}:{len(result):03d}", kind, parameters, {} if side is None else {"side": side})
            if scene.coordinate() in seen:
                continue
            seen.add(scene.coordinate())
            result.append(scene)
            if len(result) == count:
                return result
    raise ValueError("Insufficient distinct development coordinates")


class Reference:
    """Causal visible-state controls, with no event schedule or hidden actors."""

    def __init__(self, mode):
        self.mode = mode
        self.target_y = None

    def act(self, world):
        ego, actors = world.ego, world.vehicles
        if self.mode == "reveal":
            brake = any(v.speed < 0.01 and v.x > ego.x for v in actors)
            return Control(-8 if brake else 0, lane_steering(ego, 4))
        if self.mode == "accelerate":
            return Control(3, lane_steering(ego, 4))
        if self.mode == "offset":
            if self.target_y is None:
                adjacent = [v for v in actors if abs(v.y - ego.y) > 2]
                actor = min(adjacent, key=lambda v: abs(v.x - ego.x), default=None)
                self.target_y = 4 + (0.8 if actor is not None and actor.y < ego.y else -0.8)
            return Control(-8, lane_steering(ego, self.target_y))
        return Control(-8 if any(v.x > ego.x or abs(v.y-ego.y)>1 for v in actors) else 0, lane_steering(ego, 4))


def inspect(scene, trajectory, violations):
    partner = {"cut_in_onset": "cut_in", "cut_out_rear_gap": "obstacle", "side_encroachment": "intruder"}[scene.kind]
    collision = next((f for f in trajectory if f["collision"]), None)
    minimum, road_margin, background_margin = math.inf, math.inf, math.inf
    reveal = next((f["time_s"] for f in trajectory if "obstacle" in f["visible_ids"]), None)
    lateral = []
    first = {a["actor_id"]: a for a in trajectory[0]["actors"]}
    road = (-2, 6) if scene.kind.startswith("cut_out") or (scene.kind == "side_encroachment" and scene.categories["side"] == "left") else ((2, 10) if scene.kind == "side_encroachment" else (-2, 10))
    non_ego_crash = False
    for frame in trajectory:
        actors = [VehicleState(**a) for a in frame["actors"]]
        bodies = [a.polygon() for a in actors]
        road_margin = min(road_margin, bodies[0].bounds[1]-road[0], road[1]-bodies[0].bounds[3])
        for a, b in zip(actors[1:], bodies[1:]):
            background_margin = min(background_margin, b.bounds[1]-road[0], road[1]-b.bounds[3])
            minimum = min(minimum, bodies[0].distance(b))
        for i in range(1, len(bodies)):
            for j in range(i+1, len(bodies)):
                if bodies[i].intersects(bodies[j]) and not frame["collision"]:
                    non_ego_crash = True
        role = "lead" if scene.kind.startswith("cut_out") else partner
        actor = next(a for a in actors if a.actor_id == role)
        lateral.append(abs(actor.y-first[role]["y"]))
    onset = next((f["time_s"] for f, d in zip(trajectory, lateral) if d > 0.01), None)
    intended = bool(collision and collision["collision_partner"] == partner and onset is not None and collision["time_s"] >= onset)
    if scene.kind.startswith("cut_out"):
        intended = intended and reveal is not None and reveal > 0 and reveal < collision["time_s"]
    return {
        "collision": collision is not None, "collision_partner": collision["collision_partner"] if collision else None,
        "collision_time_s": collision["time_s"] if collision else None,
        "mechanism_collision": intended, "minimum_body_clearance_m": minimum,
        "minimum_road_margin_m": road_margin, "minimum_background_road_margin_m": background_margin,
        "background_overlap_before_ego_collision": non_ego_crash,
        "event_onset_s": onset, "obstacle_first_visible_s": reveal,
        "initial_obstacle_hidden": "obstacle" not in trajectory[0]["visible_ids"],
        "maximum_background_lateral_displacement_m": max(lateral),
        "end_time_s": trajectory[-1]["time_s"], "violations": list(violations),
        "native_valid": background_margin >= -1e-8 and not non_ego_crash,
        "reference_passed": collision is None and not violations and minimum >= 0.1 and road_margin >= 0.05-1e-8 and background_margin >= -1e-8 and not non_ego_crash and trajectory[-1]["time_s"] == 10,
    }


def reference(scene, directory, mode):
    path = directory / "reference" / f"{scene.scenario_id.rsplit(':', 1)[-1]}_{mode}.json.gz"
    if path.exists():
        return read_json(path)["summary"]
    if auxiliary_executions(directory) >= 64:
        raise ValueError("Registered 64-reference budget exhausted")
    world = CollisionWorld(scene)
    if mode == "coast_lane":
        from experiments.scene_design.references import CoastLaneReference

        policy = CoastLaneReference()
        write_json(directory / "coast_lane_reference_protocol.json", {
            "policy": "coast to other visible lane on visible front; no hidden obstacle",
            "source_sha256": hashlib.sha256((ROOT / "experiments/scene_design/references.py").read_bytes()).hexdigest(),
            "minimum_body_clearance_m": 0.1, "minimum_road_margin_m": 0.05,
            "maximum_reference_executions_total": 64,
        })
    elif mode == "brake_escape":
        from experiments.scene_design.references import BrakeLaneReference

        policy = BrakeLaneReference()
        write_json(directory / "brake_lane_reference_protocol.json", {
            "policy": "maximum braking and other-lane tracking on visible front; no hidden obstacle",
            "source_sha256": hashlib.sha256((ROOT / "experiments/scene_design/references.py").read_bytes()).hexdigest(),
            "minimum_body_clearance_m": 0.1, "minimum_road_margin_m": 0.05,
            "maximum_reference_executions_total": 64,
        })
    elif mode == "escape":
        from experiments.scene_design.references import EscapeReference

        policy = EscapeReference()
        write_json(directory / "escape_reference_protocol.json", {
            "policy": "coast to visible free-side lane; no schedule or labels",
            "source_sha256": hashlib.sha256((ROOT / "experiments/scene_design/references.py").read_bytes()).hexdigest(),
            "minimum_body_clearance_m": 0.1, "minimum_road_margin_m": 0.05,
            "maximum_reference_executions_total": 64,
        })
    else:
        policy = Reference(mode)
    try:
        while not world.terminal:
            world.advance(policy.act(world.visible_world()))
        result = inspect(scene, world.trace, world.violations)
        write_json(path, {"scenario": asdict(scene), "mode": mode, "summary": result, "trajectory": world.trace})
        return result
    finally:
        world.close()


def auxiliary_executions(directory):
    return len(list((directory / "reference").glob("*.json.gz"))) + len(
        list((directory / "counterfactual" / "attempts").glob("*.json"))
    )


def run_chunk(task):
    sut, group, directory = task
    rows = []
    for scene in group:
        path = directory / sut.sut_id / f"{scene.scenario_id.rsplit(':', 1)[-1]}.json.gz"
        if path.exists():
            rows.append(read_json(path)["summary"])
            continue
        if sut.family == "learning":
            if sut.sut_id not in TEMPLATES:
                TEMPLATES[sut.sut_id] = create(sut.sut_id)
            template = TEMPLATES[sut.sut_id]
            policy = copy.deepcopy(template, {id(template.policy.model): template.policy.model})
        else:
            policy = create(sut.sut_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            attempt = directory / "attempts" / sut.sut_id / f"{scene.scenario_id.rsplit(':', 1)[-1]}.json"
            write_json(attempt, {"status": "running", "scenario_id": scene.scenario_id})
            paid = execute(policy, scene)
            write_json(attempt, {"status": "complete", "scenario_id": scene.scenario_id, "feedback": paid.feedback.to_dict()})
        result = inspect(scene, paid.trajectory, paid.feedback.safety.violations)
        result["scenario_id"] = scene.scenario_id
        write_json(path, {"scenario": asdict(scene), "sut_id": sut.sut_id, "summary": result, "feedback": paid.feedback.to_dict(), "trajectory": paid.trajectory})
        rows.append(result)
    return sut.sut_id, rows


def register(kind, variant, count=32):
    directory = OUTPUT / kind / variant
    specification = {
        "kind": kind, "variant": variant, "bounds": DEVELOPMENT_BOUNDS[kind],
        "event_time_s": 0, "duration_s": 10, "timing": timing_contract(),
        "scenes": [asdict(s) for s in scenes(kind, variant)],
        "executions": {s.sut_id: execution_contract(s, kind=kind) for s in FORMAL_SUTS},
        "reference_modes": ["brake", "reveal", "offset", "accelerate"],
        "limits": {"ads_executions": 512 if kind == "cut_out_rear_gap" else 2048, "reference_executions": 64},
        "scope": "scene development only; no formal collection or risk-model training",
        "implementation_sha256": hashlib.sha256((ROOT / "experiments/scene_design/design_interactions.py").read_bytes()).hexdigest(),
    }
    specification = json.loads(json.dumps(specification))
    path = directory / "protocol.json"
    if path.exists():
        previous = read_json(path)
        # Paid traces bind the unchanged simulator/ADS contracts. Orchestration
        # and additional reference checks are recorded separately, not relabelled.
        if {k:v for k,v in previous.items() if k != "implementation_sha256"} != {k:v for k,v in specification.items() if k != "implementation_sha256"}:
            raise ValueError("Frozen physical development protocol differs; do not reuse executions")
    else:
        write_json(path, specification)
    if kind == "cut_out_rear_gap" and count == 128:
        base = read_json(path)
        extended = {**specification,
            "scenes": [asdict(s) for s in scenes(kind, variant, count=128)],
            "limits": {"ads_executions": 2048, "reference_executions": 64},
            "base_protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reused_prefix_count": 32,
            "maximum_additional_ads_executions": 1792,
            "maximum_additional_reference_executions": 38,
        }
        assert extended["scenes"][:32] == base["scenes"]
        extension_path = directory / "extended_protocol.json"
        if extension_path.exists():
            previous = read_json(extension_path)
            if {k:v for k,v in previous.items() if k != "implementation_sha256"} != {k:v for k,v in extended.items() if k != "implementation_sha256"}:
                raise ValueError("Registered extension changed")
        else:
            for name in ("summary", "audit"):
                original = directory / f"{name}.json"
                if original.exists():
                    saved = read_json(original)
                    assert saved.get("count_per_ads", saved.get("ads_records") // 8 if "ads_records" in saved else None) == 32
                    (directory / f"first_stage_{name}.json").write_bytes(original.read_bytes())
            write_json(extension_path, extended)
    history_path = directory / "driver_invocations.json"
    history = read_json(history_path) if history_path.exists() else []
    source = (ROOT / "experiments/scene_design/design_interactions.py").read_text(encoding="utf-8")
    if not history or history[-1]["source"] != source:
        history.append({"source": source, "sha256": specification["implementation_sha256"]})
        write_json(history_path, history)
    return directory


def run(kind, variant, count):
    if kind == "cut_out_rear_gap" and (variant != "initial" or count not in (32,128)):
        raise ValueError("The rear-gap extension retains the initial range and at most 128 points")
    directory = register(kind, variant, count)
    if (directory / "summary.json").exists() and read_json(directory / "summary.json").get("count_per_ads",0) > count:
        print("The completed larger cohort is retained; no physical executions started",flush=True)
        return
    group = scenes(kind, variant, count=count)
    corners = group[:17] if kind.startswith("cut_out") else group[:34:2]
    checks = [reference(s, directory, "brake") for s in corners]
    if not all(r["native_valid"] for r in checks):
        write_json(directory / "summary.json", {"status": "invalid_traffic_geometry", "reference": checks})
        return
    rows = []
    for role in ("target", "history"):
        systems = FORMAL_TARGET_SUTS if role == "target" else FORMAL_HISTORY_SUTS
        if len(systems) != 8:
            raise ValueError("Expected the complete eight-ADS cohort")
        tasks = [(sut, group[i:i+8], directory) for sut in systems for i in range(0, count, 8)]
        records = {s.sut_id: [] for s in systems}
        with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context("spawn"), initializer=initialize_worker) as pool:
            futures = [pool.submit(run_chunk, t) for t in tasks]
            for future in as_completed(futures):
                sut_id, chunk = future.result()
                records[sut_id].extend(chunk)
                print(kind, variant, sut_id, len(records[sut_id]), flush=True)
        for sut in systems:
            results = sorted(records[sut.sut_id], key=lambda r:r["scenario_id"])
            collisions = [r for r in results if r["mechanism_collision"] and not r["violations"] and r["native_valid"]]
            safe = [r for r in results if not r["collision"] and not r["violations"] and r["end_time_s"] == 10 and r["native_valid"]]
            representative = None
            def room(record):
                p = next(s.parameters for s in group if s.scenario_id == record["scenario_id"])
                if kind.startswith("cut_out"):
                    return p["obstacle_distance_m"] - p["ego_speed_mps"]**2/16
                if kind == "cut_in_onset":
                    return p["gap_m"] - p["closing_speed_mps"]**2/16
                return abs(p["longitudinal_offset_m"])
            for record in sorted(collisions, key=room, reverse=True):
                scene = next(s for s in group if s.scenario_id == record["scenario_id"])
                modes = (("reveal", "brake", "offset", "brake_escape", "coast_lane") if kind == "cut_out_rear_gap" else ("accelerate", "brake", "offset") if kind == "side_encroachment" else ("brake", "offset", "escape"))
                for mode in modes:
                    if reference(scene, directory, mode)["reference_passed"]:
                        representative = {"scenario_id": scene.scenario_id, "mode": mode}
                        break
                if representative:
                    break
            rows.append({"sut_id": sut.sut_id, "records": len(results), "collisions": sum(r["collision"] for r in results), "mechanism_collisions": len(collisions), "safe": len(safe), "avoidable_representative": representative, "rows": results})
        passed = all(r["mechanism_collisions"] and r["safe"] and r["avoidable_representative"] for r in rows)
        write_json(directory / "summary.json", {"status": "development_coverage_passed" if passed and role == "history" else "target_passed" if passed else "coverage_insufficient", "protocol_file": "extended_protocol.json" if kind == "cut_out_rear_gap" and count == 128 else "protocol.json", "count_per_ads": count, "ads_executions": len(rows)*count, "reference_executions": len(list((directory/"reference").glob("*.json.gz"))), "rows": rows, "formal_executions": 0, "training_steps": 0})
        if not passed:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=DEVELOPMENT_SCENARIOS)
    parser.add_argument("variant", choices=("initial",), default="initial", nargs="?")
    parser.add_argument("count", type=int, choices=(32,128), default=32, nargs="?")
    args = parser.parse_args()
    run(args.kind, args.variant, args.count)
