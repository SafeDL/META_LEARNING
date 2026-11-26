"""Independent frame/identity audit of completed native development records."""

import argparse
import copy
import hashlib
import math

from controllers.highway.catalog import FORMAL_SUTS
from controllers.highway.control import steering_limit
from simulator.highway.observation import VehicleState

from experiments.scene_design.design_interactions import OUTPUT
from experiments.executions import execution_contract
from experiments.storage import read_json, write_json


def mechanism_event(payload):
    """Reconstruct the registered interaction from raw frames, not summary labels."""
    kind = payload["scenario"]["kind"]
    role = "lead" if kind.startswith("cut_out") else "cut_in" if kind == "cut_in_onset" else "intruder"
    partner = "obstacle" if kind.startswith("cut_out") else role
    trace = payload["trajectory"]
    initial_y = next(a["y"] for a in trace[0]["actors"] if a["actor_id"] == role)
    onset = next((f["time_s"] for f in trace if abs(next(a["y"] for a in f["actors"] if a["actor_id"] == role)-initial_y)>0.01),None)
    crash = next((f for f in trace if f["collision"]),None)
    reveal = next((f["time_s"] for f in trace if "obstacle" in f["visible_ids"]),None)
    valid = bool(crash and crash["collision_partner"] == partner and onset is not None and onset <= crash["time_s"])
    if kind.startswith("cut_out"):
        valid = bool(valid and reveal is not None and 0 < reveal < crash["time_s"])
    return valid,onset,reveal


def check_dynamics(payload):
    trace = payload["trajectory"]
    assert trace[0]["time_s"] == 0
    for previous, current in zip(trace, trace[1:]):
        assert abs(current["time_s"]-previous["time_s"]-0.05) < 1e-8
        old = {v["actor_id"]:v for v in previous["actors"]}
        new = {v["actor_id"]:v for v in current["actors"]}
        assert old.keys() == new.keys()
        if current["collision"]:
            continue
        for name, before in old.items():
            if name == "obstacle":
                assert before == new[name]
                continue
            if name == "ego":
                acceleration = current["control"]["acceleration"]
                steering = current["control"]["steering"]
            else:
                control = current["background_controls"][name]
                assert control["scripted"]
                acceleration, steering = control["acceleration_mps2"],control["steering_rad"]
            assert -8-1e-8 <= acceleration <= 3+1e-8
            assert abs(steering) <= steering_limit(before["speed"],before["length"])+1e-8
            beta = math.atan(math.tan(steering)/2)
            expected = {
                "x": before["x"] + before["speed"]*math.cos(before["heading"]+beta)*0.05,
                "y": before["y"] + before["speed"]*math.sin(before["heading"]+beta)*0.05,
                "heading": before["heading"] + 2*before["speed"]*math.sin(beta)/before["length"]*0.05,
                "speed": before["speed"]+acceleration*0.05,
            }
            assert all(abs(new[name][key]-value)<1e-7 for key,value in expected.items()), (payload["scenario"]["scenario_id"],name,current["time_s"])
    return len(trace)-1


def check_reference(reference):
    assert not any(f["collision"] for f in reference["trajectory"])
    assert reference["trajectory"][-1]["time_s"] == 10
    kind = reference["scenario"]["kind"]
    side = reference["scenario"]["categories"].get("side")
    lower, upper = (-2,6) if kind.startswith("cut_out") or (kind == "side_encroachment" and side == "left") else ((2,10) if kind == "side_encroachment" else (-2,10))
    clearance, road_margin = math.inf, math.inf
    for frame in reference["trajectory"]:
        bodies = [VehicleState(**a).polygon() for a in frame["actors"]]
        clearance = min(clearance,*(bodies[0].distance(b) for b in bodies[1:]))
        road_margin = min(road_margin,bodies[0].bounds[1]-lower,upper-bodies[0].bounds[3])
        assert all(b.bounds[1] >= lower-1e-8 and b.bounds[3] <= upper+1e-8 for b in bodies[1:])
        assert all(not a.intersects(b) for i,a in enumerate(bodies[1:]) for b in bodies[i+2:])
    assert clearance >= 0.1 and road_margin >= 0.05-1e-8
    return check_dynamics(reference)


def audit(directory):
    summary = read_json(directory / "summary.json")
    protocol = read_json(directory / summary.get("protocol_file", "protocol.json"))
    if "base_protocol_sha256" in protocol:
        base_path = directory / "protocol.json"
        assert hashlib.sha256(base_path.read_bytes()).hexdigest() == protocol["base_protocol_sha256"]
        assert protocol["scenes"][:32] == read_json(base_path)["scenes"]
    assert summary["status"] in ("development_coverage_passed","coverage_insufficient")
    covered = summary["status"] == "development_coverage_passed"
    count = summary["count_per_ads"]
    population = protocol["scenes"][:count]
    cohort = {s.sut_id:s for s in FORMAL_SUTS}
    expected = set(cohort) if covered else {name for name in cohort if name.startswith("target_")}
    assert len(summary["rows"]) == len(expected)
    assert {r["sut_id"] for r in summary["rows"]} == expected
    frames, events = 0, 0
    snapshot_sources = set()

    audit_rows = []
    for row in summary["rows"]:
        sut_id = row["sut_id"]
        original = protocol["executions"][sut_id]
        current = execution_contract(cohort[sut_id],kind=protocol["kind"])
        normalized = copy.deepcopy(original)
        for old_sources,new_sources in ((normalized["source_files"],current["source_files"]),(normalized["runtime_contract"]["source_files"],current["runtime_contract"]["source_files"])):
            for name,digest in list(old_sources.items()):
                if new_sources[name] != digest:
                    snapshot = directory/"execution_sources"/name
                    assert snapshot.is_file() and hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
                    snapshot_sources.add(name)
                    old_sources[name] = new_sources[name]
        assert normalized == current
        assert row["records"] == count
        if covered:
            assert row["safe"] > 0 and row["mechanism_collisions"] > 0
        labels, safe, mechanism = 0, 0, 0
        for ordinal, scene in enumerate(population):
            payload = read_json(directory/sut_id/f"{ordinal:03d}.json.gz")
            assert payload["scenario"] == scene
            result = payload["summary"]
            collision = any(f["collision"] for f in payload["trajectory"])
            assert collision == payload["feedback"]["collision"] == result["collision"]
            valid,onset,reveal = mechanism_event(payload)
            assert valid == result["mechanism_collision"]
            assert onset == result["event_onset_s"] and reveal == result["obstacle_first_visible_s"]
            assert set(result["violations"]) == set(payload["feedback"]["safety"]["violations"])
            frames += check_dynamics(payload)
            labels += collision
            mechanism += result["mechanism_collision"]
            safe += not collision and not result["violations"] and result["end_time_s"] == 10 and result["native_valid"]
        assert (labels,safe,mechanism)==(row["collisions"],row["safe"],row["mechanism_collisions"])
        certificate = row["avoidable_representative"]
        events += labels
        audit_rows.append({"sut_id":sut_id,"collisions":labels,"mechanism_collisions":mechanism,"safe":safe,"avoidable_representative":certificate})
        if certificate is None:
            assert not covered
            continue
        ordinal = int(certificate["scenario_id"].rsplit(":",1)[-1])
        reference = read_json(directory/"reference"/f"{ordinal:03d}_{certificate['mode']}.json.gz")
        assert reference["scenario"] == population[ordinal]
        assert reference["summary"]["reference_passed"]
        frames += check_reference(reference)
    result = {"passed":True,"coverage_passed":covered,"ads_records":len(expected)*count,"physics_frames_checked":frames,"collisions":events,"all_sut_execution_contracts_current":not snapshot_sources,"archived_source_files_verified":sorted(snapshot_sources),"new_physical_executions":0,"rows":audit_rows}
    write_json(directory/"audit.json",result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind")
    parser.add_argument("variant", default="initial", nargs="?")
    args = parser.parse_args()
    result = audit(OUTPUT/args.kind/args.variant)
    print({key:value for key,value in result.items() if key != "rows"})
