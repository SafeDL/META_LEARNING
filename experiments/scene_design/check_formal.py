"""Reuse accepted scene evidence and recheck native traces under the current stack."""

import hashlib
import math
from dataclasses import asdict

from controllers.highway.catalog import FORMAL_SUTS
from controllers.highway.contracts import runtime_contract
from controllers.highway.control import Control
from simulator.highway.collision_env import CollisionWorld
from simulator.highway.timing import timing_contract

from experiments.scene_design.check_braking import verify_development as verify_braking
from experiments.formal_scenarios import BRAKING_KIND, DEVELOPMENT_ROOT, INTERACTION_ROOT, FREE_BOUNDS
from experiments.protocol import ROOT, episode_duration
from experiments.schema import Scenario
from experiments.storage import read_json, write_json

REPLAY_PATH = ROOT / "results/k_shot_adaptation/reports/development_replay.json"
ATTEMPTS_PATH = REPLAY_PATH.with_name("development_replay_attempts.json")


def same_frame(actual, expected):
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_frame(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_frame(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)
    return actual == expected


def evidence():
    braking = verify_braking()
    result = {BRAKING_KIND: (DEVELOPMENT_ROOT, "protocol.json", braking)}
    for kind in FREE_BOUNDS:
        if kind == BRAKING_KIND:
            continue
        directory = INTERACTION_ROOT / kind / "initial"
        secondary = kind == "cut_out_rear_gap"
        summary = read_json(directory / ("secondary_summary.json" if secondary else "summary.json"))
        audit = read_json(directory / ("secondary_audit.json" if secondary else "audit.json"))
        protocol_name = "extended_protocol.json" if secondary else "protocol.json"
        protocol = read_json(directory / protocol_name)
        if (
            summary["status"] != "development_coverage_passed"
            or not audit["passed"] or not audit["coverage_passed"]
            or summary["ads_executions"] != audit["ads_records"]
            or protocol["bounds"] != {name: list(bounds) for name, bounds in FREE_BOUNDS[kind].items()}
            or protocol["timing"] != timing_contract()
            or protocol["event_time_s"] != 0 or protocol["duration_s"] != 10
            or {row["sut_id"] for row in summary["rows"]} != {s.sut_id for s in FORMAL_SUTS}
        ):
            raise ValueError(f"Incomplete accepted development evidence: {kind}")
        for sut in FORMAL_SUTS:
            previous = dict(protocol["executions"][sut.sut_id]["object"])
            current = asdict(sut)
            previous.pop("supported_scenarios")
            current.pop("supported_scenarios")
            identity = runtime_contract(sut)
            original = protocol["executions"][sut.sut_id]["runtime_contract"]
            if previous != current or any(
                identity.get(name) != value for name, value in original.items() if name != "source_files"
            ) or any(
                identity["source_files"][name] != digest
                for name, digest in original["source_files"].items()
                if name not in ("experiments/scenarios.py", "simulator/highway/collision_env.py")
            ):
                raise ValueError(f"Accepted scene uses another ADS or dynamics: {kind}/{sut.sut_id}")
        result[kind] = directory, protocol_name, summary
    return result


def replay_specification(records):
    files = {}
    for kind, (directory, protocol_name, _) in records.items():
        names = (protocol_name, "summary.json", "secondary_summary.json", "secondary_audit.json") if kind == "cut_out_rear_gap" else (protocol_name, "summary.json", "audit.json")
        if kind == BRAKING_KIND:
            names = (protocol_name, "summary.json")
        for name in names:
            path = directory / name
            files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "current_runtime_contracts": {s.sut_id: runtime_contract(s) for s in FORMAL_SUTS},
        "accepted_evidence_sha256": files,
        "check_source_sha256": hashlib.sha256((ROOT / "experiments/scene_design/check_formal.py").read_bytes()).hexdigest(),
        "selection": "first saved collision and first complete safe development record per ADS and scene",
        "new_formal_responses": 0,
    }


def verify_development():
    records = evidence()
    saved = read_json(REPLAY_PATH)
    if saved["specification"] != replay_specification(records) or saved["status"] != "complete" or len(saved["rows"]) != 2 * len(FORMAL_SUTS) * len(records):
        raise ValueError("Current-stack scene revalidation is incomplete")
    for row in saved["rows"]:
        path = ROOT / row["record"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("A revalidated scene record changed")
    return {
        "status": "development_coverage_passed",
        "development_design_passed": True,
        "ads_executions": sum(s[2]["ads_executions"] for s in records.values()),
        "native_replay_executions": saved["physical_replay_executions"],
    }


def revalidate():
    records = evidence()
    specification = replay_specification(records)
    if REPLAY_PATH.exists():
        return verify_development()
    rows = []
    attempts = read_json(ATTEMPTS_PATH) if ATTEMPTS_PATH.exists() else []
    for kind, (directory, _, summary) in records.items():
        count = summary.get("count_per_ads", summary.get("development_count_per_ads"))
        for sut in FORMAL_SUTS:
            chosen = {}
            for index in range(count):
                path = directory / sut.sut_id / f"{index:03d}.json.gz"
                payload = read_json(path)
                trace = payload["trajectory"]
                collision = any(frame["collision"] for frame in trace)
                if collision or trace[-1]["time_s"] == episode_duration(kind) and not payload["feedback"]["safety"]["violations"]:
                    chosen.setdefault(collision, (path, payload))
                if len(chosen) == 2:
                    break
            if len(chosen) != 2:
                raise ValueError(f"Missing accepted collision/safe evidence: {kind}/{sut.sut_id}")
            for collision, (path, payload) in chosen.items():
                trace = payload["trajectory"]
                attempt = {"record": path.relative_to(ROOT).as_posix(), "executions": 1, "status": "running"}
                attempts.append(attempt)
                write_json(ATTEMPTS_PATH, attempts)
                world = CollisionWorld(Scenario(**payload["scenario"]), duration_s=episode_duration(kind), environment_seed=81001)
                try:
                    if not same_frame(world.trace, trace[:1]):
                        raise ValueError(f"Changed native initial state: {path}")
                    while not world.terminal:
                        index = len(world.trace)
                        world.advance(Control(**trace[index]["requested_control"]))
                        if not same_frame(world.trace[index:], trace[index:len(world.trace)]):
                            raise ValueError(f"Changed native trace: {path}, frame {index}")
                    if len(world.trace) != len(trace):
                        raise ValueError(f"Changed native termination: {path}")
                    attempt["status"] = "complete"
                except Exception as error:
                    attempt.update(status="failed", error=repr(error))
                    raise
                finally:
                    world.close()
                    write_json(ATTEMPTS_PATH, attempts)
                rows.append({"scenario": kind, "sut_id": sut.sut_id, "collision": collision, "record": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frames": len(trace)})
            print({"scene_revalidation": kind, "sut_id": sut.sut_id, "passed": True}, flush=True)
    write_json(REPLAY_PATH, {"status": "complete", "specification": specification, "rows": rows, "physical_replay_executions": sum(row["executions"] for row in attempts), "new_formal_responses": 0})
    return verify_development()


if __name__ == "__main__":
    print(revalidate(), flush=True)
