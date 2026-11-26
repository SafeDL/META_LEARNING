"""Register public ADS interfaces, native capability courses and four scene replays."""

import argparse
import math
from dataclasses import asdict

import torch
from PIL import Image, ImageChops, ImageDraw, ImageSequence

from controllers.highway.control import Control
from controllers.highway.public_agents import (
    CHECKPOINTS,
    LOCAL_SOURCES,
    ROOT,
    RUNS,
    WEIGHTS,
    create,
    runtime_contract,
    sha256,
)
from simulator.highway.collision_env import CollisionWorld
from simulator.highway.observation import VehicleState, observe
from simulator.highway.timing import PHYSICS_HZ, PHYSICS_PER_DECISION

from experiments.public_ads import checks as public_agent_checks
from experiments.plotting.animation import save_gif
from experiments.scene_design.audit_interactions import check_dynamics
from experiments.execution_events import analyze_events
from experiments.qualification import COURSES, ENVIRONMENT_SEED, course_result, save_course_evidence, verify_course_evidence
from experiments.plotting.replay_safe import SCENES, SMALL, WIDTH, draw_frame
from experiments.schema import Scenario
from experiments.storage import read_json, write_json

IDS = (*WEIGHTS, *LOCAL_SOURCES)
FIGURES = ROOT / "figures/scene_replays/public_qualification"
DEVELOPMENT_SEED = 81001
DEVELOPMENT_IDS = (
    "critical_following:016", "cut_in_onset:initial:032",
    "cut_out_rear_gap:initial:016", "side_encroachment:initial:032",
)


def development_scenes():
    for directory, suffix in zip(SCENES, DEVELOPMENT_IDS, strict=True):
        path = directory / "protocol.json"
        protocol = read_json(path)

        def center_distance(row):
            return sum(
                ((row["parameters"][name] - low) / (high - low) - 0.5) ** 2
                for name, (low, high) in protocol["bounds"].items() if high > low
            ), row["scenario_id"]

        rows = protocol.get("development_scenes", protocol.get("scenes"))
        row = min(rows, key=center_distance)
        if not row["scenario_id"].endswith(suffix):
            raise ValueError("Registered development representative changed")
        yield Scenario(**row), protocol["duration_s"], path


def drive(sut, scenario, seed, duration, *, course=None):
    policy = create(sut)
    world = CollisionWorld(
        scenario, duration_s=duration, environment_seed=seed,
        lane_centers=None if course is None else course.lane_centers,
    )
    decisions = []
    try:
        while not world.terminal:
            visible = world.visible_world()
            command = policy.act(visible)
            decisions.append({
                "time_s": visible.time_s, **policy.policy.last_decision,
                "policy_control": asdict(command),
            })
            world.advance(command)
        return {
            "scenario": asdict(scenario), "environment_seed": seed, "duration_s": duration,
            "lane_centers": list(world.lane_centers), "trajectory": world.trace,
            "events": world.event_log, "decisions": decisions,
            "feedback": {"collision": bool(world.ego.crashed)},
            "violations": sorted(world.violations),
            **({"result": course_result(world, course)} if course is not None else {}),
        }
    finally:
        world.close()


def capabilities(sut, directory):
    rows = []
    for course in COURSES:
        args = sut, course.scenario, ENVIRONMENT_SEED, course.duration_s
        payload = drive(*args, course=course)
        replay = drive(*args, course=course)
        if payload != replay:
            raise ValueError(f"Controller decisions or native course replay changed: {course.name}")
        write_json(directory / f"{course.name}.json.gz", payload)
        rows.append(payload["result"])
    evidence = save_course_evidence(directory, rows, courses=COURSES, mode="controller_execution")
    return {
        "qualified": evidence["qualified"], "courses": rows,
        "independent_policy_rerun_identical": True,
    }


def inspect_frames(payload):
    trace = payload["trajectory"]
    check_dynamics(payload)
    collisions = [row for row in trace if row["collision"]]
    if bool(collisions) != payload["feedback"]["collision"] or (
        collisions and collisions != trace[-len(collisions):]
    ):
        raise ValueError("Native collision and terminal trajectory disagree")
    if collisions and not any(
        event["event"] == "native_ego_collision"
        and event["time_s"] == collisions[0]["time_s"]
        and event["partner"] == collisions[0]["collision_partner"]
        for event in payload["events"]
    ):
        raise ValueError("First collision frame differs from the recorded native event")
    decisions, clearance = payload["decisions"], []
    clipped = {"acceleration": 0, "steering": 0}
    for i, row in enumerate(trace):
        states = [VehicleState(**actor) for actor in row["actors"]]
        visible = observe(row["time_s"], states, lane_centers=payload["lane_centers"])
        if [actor.actor_id for actor in visible.vehicles] != row["visible_ids"]:
            raise ValueError("Recorded visibility differs from physical frame")
        ego = states[0].polygon()
        clearance.append(min(ego.distance(actor.polygon()) for actor in states[1:]))
        if i:
            decision = decisions[(i - 1) // PHYSICS_PER_DECISION]
            intended = decision["policy_control"]
            requested = asdict(Control(**intended).limited())
            if any(
                not math.isclose(requested[key], row["requested_control"][key], abs_tol=1e-12)
                for key in requested
            ):
                raise ValueError("Requested command differs from the policy's limited output")
            for key in clipped:
                clipped[key] += not math.isclose(
                    intended[key], row["control"][key], rel_tol=1e-9, abs_tol=1e-9
                )
        if i and (i - 1) % PHYSICS_PER_DECISION == 0:
            if not math.isclose(decision["time_s"], trace[i - 1]["time_s"], abs_tol=1e-9):
                raise ValueError("Native frame time is not aligned with the policy decision")
    if len(decisions) != math.ceil((len(trace) - 1) / PHYSICS_PER_DECISION):
        raise ValueError("Trajectory contains missing or extra policy decisions")
    if not collisions and not payload["violations"] and (
        len(trace) != round(payload["duration_s"] * PHYSICS_HZ) + 1
    ):
        raise ValueError("Nonterminal development trajectory was truncated")
    return clearance, clipped


def animate(sut, payload, clearance, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = payload["trajectory"]
    dx = [a["x"] - f["actors"][0]["x"] for f in frames for a in f["actors"]]
    extent = min(-30, min(dx) - 8), max(90, max(dx) + 8)
    images = []
    for i, frame in enumerate(frames):
        image = draw_frame(payload, sut, i, {"clearance_series_m": clearance}, extent)
        if i:
            d = payload["decisions"][(i - 1) // PHYSICS_PER_DECISION]
            label = str(d["action"])
            if isinstance(d["action"], list):
                label = ", ".join(f"{value:+.2f}" for value in d["action"])
            if d.get("shield_rejected"):
                label += f"  |  rejected {d['shield_rejected']}"
            ImageDraw.Draw(image).text(
                (WIDTH - 410, 237), f"Policy action {label}", fill=1, font=SMALL
            )
        if i == len(frames) - 1 and payload["violations"]:
            ImageDraw.Draw(image).text((WIDTH - 250, 9), "Road departure", fill=8, font=SMALL)
        images.append(image)
    save_gif(images, [frame["time_s"] for frame in frames], path)
    with Image.open(path) as gif:
        if gif.n_frames != len(frames):
            raise ValueError("GIF omitted physics frames")
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            diff = ImageChops.difference(frame.convert("RGB"), images[i].convert("RGB"))
            if diff.getbbox() or frame.info["duration"] != (500 if i == len(frames) - 1 else 50):
                raise ValueError("GIF pixels or timing differ from the native trajectory rendering")


def replay_development(sut, directory):
    rows = []
    for scene, duration, protocol in development_scenes():
        payload = drive(sut, scene, DEVELOPMENT_SEED, duration)
        clearance, clipped = inspect_frames(payload)
        payload["development_protocol"] = protocol.relative_to(ROOT).as_posix()
        payload["development_protocol_sha256"] = sha256(protocol.read_bytes())
        record = directory / f"{scene.kind}.json.gz"
        gif = FIGURES / sut / f"{scene.kind}.gif"
        write_json(record, payload)
        animate(sut, payload, clearance, gif)
        collision = payload["feedback"]["collision"]
        last = payload["trajectory"][-1]
        impact = next((frame for frame in payload["trajectory"] if frame["collision"]), None)
        rows.append({
            "scenario_id": scene.scenario_id, "kind": scene.kind, "collision": collision,
            "collision_time_s": impact["time_s"] if impact else None,
            "collision_partner": last["collision_partner"], "terminal_time_s": last["time_s"],
            "violations": payload["violations"], "physics_frames": len(clearance),
            "applied_limits_frames": clipped, "minimum_clearance_m": min(clearance),
            "events": analyze_events(scene, payload["trajectory"]),
            "trajectory": record.relative_to(ROOT).as_posix(),
            "trajectory_sha256": sha256(record.read_bytes()),
            "gif": gif.relative_to(ROOT).as_posix(), "gif_sha256": sha256(gif.read_bytes()),
        })
    return rows


def qualify_one(sut):
    contract = runtime_contract(sut)
    directory = RUNS / sut
    interface = (
        public_agent_checks.external_interface(sut)
        if sut in WEIGHTS else public_agent_checks.adapted_interface(sut)
    )
    result = {
        "sut_id": sut, "runtime_contract": contract, "interface": interface,
        "verification_source": verification_source(),
        **capabilities(sut, directory / "qualification"),
        "development_replays": replay_development(sut, directory / "development"),
    }
    write_json(directory / "acceptance.json", result)
    return result


def verification_source():
    return {
        name: sha256((ROOT / "experiments" / name).read_bytes())
        for name in ("public_ads/qualify.py", "public_ads/checks.py", "plotting/animation.py")
    }


def verify_admitted(sut):
    """Check all release evidence before returning the identity for formal collection."""
    if sut not in IDS:
        raise ValueError(f"Unknown public ADS: {sut}")
    directory = RUNS / sut
    acceptance = read_json(directory / "acceptance.json")
    if acceptance["sut_id"] != sut or acceptance["runtime_contract"] != runtime_contract(sut):
        raise ValueError(f"Public ADS runtime identity changed: {sut}")
    if acceptance["verification_source"] != verification_source():
        raise ValueError(f"Public ADS verification source changed: {sut}")
    if not acceptance["qualified"] or not acceptance["independent_policy_rerun_identical"]:
        raise ValueError(f"Public ADS failed qualification: {sut}")
    evidence = verify_course_evidence(directory / "qualification")
    if acceptance["courses"] != evidence["courses"]:
        raise ValueError("Capability course results changed")
    replays = acceptance["development_replays"]
    if [row["scenario_id"] for row in replays] != [
        scene.scenario_id for scene, _, _ in development_scenes()
    ]:
        raise ValueError("Incomplete public ADS development replay suite")
    for row in replays:
        for field in ("trajectory", "gif"):
            path = ROOT / row[field]
            if sha256(path.read_bytes()) != row[f"{field}_sha256"]:
                raise ValueError(f"Development replay hash changed: {path}")
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sut", nargs="*", metavar="ADS_ID", help=", ".join(IDS))
    parser.add_argument("--verify", action="store_true", help="Verify an admitted release")
    args = parser.parse_args()
    if set(args.sut) - set(IDS):
        parser.error(f"Unknown public ADS: {sorted(set(args.sut) - set(IDS))}")
    if args.verify and not args.sut:
        parser.error("Supply the admitted ADS IDs to verify")
    torch.set_num_threads(1)
    for sut in args.sut or IDS:
        if sut in LOCAL_SOURCES and not (CHECKPOINTS / sut / "policy.pt").exists():
            print(f"{sut}: training checkpoint unavailable", flush=True)
            continue
        result = verify_admitted(sut) if args.verify else qualify_one(sut)
        failures = [row["course"] for row in result["courses"] if not row["passed"]]
        print(f"{sut}: qualified={result['qualified']}; failed={failures}", flush=True)


if __name__ == "__main__":
    main()
