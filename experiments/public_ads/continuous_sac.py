"""Train and qualify the preregistered continuous-action policy."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version

import torch
from stable_baselines3 import SAC

from controllers.highway.continuous_sac import CHECKPOINT, SUT_ID, create
from controllers.highway import interval_robust_planner
from controllers.highway.public_agents import ROOT, RL_AGENTS_REV, sha256
from simulator.highway.collision_env import CollisionWorld
from simulator.highway.continuous_env import ContinuousDrivingEnv
from simulator.highway.timing import timing_contract

from experiments.history_transfer_experiment import OUTPUT as ORIGINAL, freeze, registered_populations
from experiments.public_ads import qualify as public
from experiments.qualification import (
    COURSES, DEVELOPMENT_COURSES, ENVIRONMENT_SEED, course_result, qualify,
    save_course_evidence, verify_course_evidence,
)
from experiments.storage import read_json, write_json

QUALIFICATION = ROOT / "results/lifelong_transfer/qualification"
RUN = QUALIFICATION / SUT_ID
FALLBACK = interval_robust_planner.SUT_ID
TRAINING_STEPS = 300_000
CHECKPOINT_INTERVAL = 25_000
CONFIGURATION = {
    "seed": 1301,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 256,
    "learning_starts": 5_000,
    "tau": 0.005,
    "ent_coef": "auto",
}
SOURCE_FILES = (
    "controllers/highway/adapted_agents.py",
    "controllers/highway/continuous_sac.py",
    "controllers/highway/control.py",
    "controllers/highway/decision_interface.py",
    "controllers/highway/learning.py",
    "simulator/highway/collision_env.py",
    "simulator/highway/observation.py",
    "simulator/highway/timing.py",
    "simulator/highway/continuous_env.py",
    "simulator/highway/training_env.py",
    "experiments/scenarios.py",
    "experiments/public_ads/continuous_sac.py",
    "experiments/public_ads/qualify.py",
    "experiments/qualification.py",
)


def selected_target():
    path = QUALIFICATION / "release.json"
    if not path.is_file():
        return SUT_ID
    release = read_json(path)
    sut = release["sut_id"]
    if sut not in (SUT_ID, FALLBACK) or release["acceptance_sha256"] != sha256(
        (QUALIFICATION / sut / "acceptance.json").read_bytes()
    ):
        raise ValueError("12th ADS selection changed after qualification")
    return sut


def controller(sut, *, digest=None):
    if sut == SUT_ID:
        return create(digest=digest)
    if sut == FALLBACK:
        return interval_robust_planner.create()
    raise ValueError(f"Unregistered 12th ADS: {sut}")


def runtime_contract(sut=SUT_ID):
    fallback = sut == FALLBACK
    files = (*SOURCE_FILES, "controllers/highway/interval_robust_planner.py") if fallback else SOURCE_FILES
    return {
        "sut_id": sut,
        "checkpoint_sha256": None if fallback else sha256(CHECKPOINT.read_bytes()),
        "upstream_revision": RL_AGENTS_REV if fallback else None,
        "upstream_assets": interval_robust_planner.upstream_assets() if fallback else {},
        "timing": timing_contract(),
        "dependencies": {
            package: version(package)
            for package in ("highway-env", "gymnasium", "numpy", "torch", "stable-baselines3")
        },
        "source_files": {name: sha256((ROOT / name).read_bytes()) for name in files},
    }


def _save_checkpoint(model, path):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        model.save(stream)
    temporary.replace(path)


def _freeze(checkpoint, digest):
    content = checkpoint.read_bytes()
    if sha256(content) != digest:
        raise ValueError("Selected development checkpoint changed")
    if CHECKPOINT.exists():
        if sha256(CHECKPOINT.read_bytes()) != digest:
            raise ValueError("Existing frozen SAC checkpoint has another identity")
        return
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("xb") as stream:
        stream.write(content)


def train(device):
    if version("stable-baselines3") != "2.3.0":
        raise ValueError("Continuous SAC training requires Stable-Baselines3 2.3.0")
    directory = RUN / "training"
    report_path = directory / "training.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["status"] != "complete":
            raise ValueError("A SAC training run already consumed part of the fixed budget")
        if report["selected_checkpoint"] is not None:
            row = report["selected_checkpoint"]
            _freeze(directory / row["checkpoint"], row["sha256"])
        return report
    directory.mkdir(parents=True, exist_ok=True)
    reserved = set()
    for (split, _), scenes in registered_populations(ORIGINAL).items():
        if split == "target":
            reserved.update(s.coordinate() for s in scenes)
    formal_courses = {c.scenario.coordinate() for c in COURSES}
    development = {c.scenario.coordinate() for c in DEVELOPMENT_COURSES}
    if reserved & development or formal_courses & development:
        raise ValueError("Development checkpoints would use reserved scenarios")
    reserved.update(formal_courses)
    reserved.update(development)
    report = {
        "status": "running",
        "configuration": CONFIGURATION,
        "maximum_steps": TRAINING_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "device": device,
        "namespace": "policy_train",
        "excluded_coordinates": len(reserved),
        "development_courses": [asdict(course) for course in DEVELOPMENT_COURSES],
        "checkpoints": [],
        "selected_checkpoint": None,
    }
    with report_path.open("x", encoding="utf-8") as stream:
        # Claim this fixed-seed run before consuming its first training step.
        json.dump(report, stream, allow_nan=False)
    env = ContinuousDrivingEnv(excluded_coordinates=reserved)
    try:
        model = SAC("MlpPolicy", env, device=device, verbose=1, **CONFIGURATION)
        for step in range(CHECKPOINT_INTERVAL, TRAINING_STEPS + 1, CHECKPOINT_INTERVAL):
            model.learn(
                total_timesteps=CHECKPOINT_INTERVAL,
                reset_num_timesteps=step == CHECKPOINT_INTERVAL,
                log_interval=100,
            )
            if model.num_timesteps != step:
                raise RuntimeError("Continuous SAC exceeded its registered checkpoint step")
            checkpoint = directory / f"step_{step}.pt"
            _save_checkpoint(model, checkpoint)
            digest = sha256(checkpoint.read_bytes())
            row = {"step": step, "checkpoint": checkpoint.name, "sha256": digest}
            if report["selected_checkpoint"] is None:
                row["development"] = qualify(
                    lambda: create(checkpoint, digest=digest), courses=DEVELOPMENT_COURSES
                )
                if row["development"]["qualified"]:
                    report["selected_checkpoint"] = row
            report["checkpoints"].append(row)
            report["training_episodes_by_kind"] = env.training_kinds
            write_json(report_path, report)
            print(SUT_ID, "training checkpoint", row, flush=True)
        report["status"] = "complete"
        write_json(report_path, report)
        if report["selected_checkpoint"] is not None:
            row = report["selected_checkpoint"]
            _freeze(directory / row["checkpoint"], row["sha256"])
        return report
    finally:
        env.close()


def drive(sut, scene, seed, duration, *, course=None):
    policy = controller(sut)
    world = CollisionWorld(
        scene, duration_s=duration, environment_seed=seed,
        lane_centers=None if course is None else course.lane_centers,
    )
    decisions = []
    try:
        while not world.terminal:
            visible = world.visible_world()
            command = policy.act(visible)
            decisions.append({
                "time_s": visible.time_s,
                **policy.policy.last_decision,
                "policy_control": asdict(command),
            })
            world.advance(command)
        return {
            "scenario": asdict(scene), "environment_seed": seed, "duration_s": duration,
            "lane_centers": list(world.lane_centers), "trajectory": world.trace,
            "events": world.event_log, "decisions": decisions,
            "feedback": {"collision": bool(world.ego.crashed)},
            "violations": sorted(world.violations),
            **({"result": course_result(world, course)} if course is not None else {}),
        }
    finally:
        world.close()


def qualify_release(sut=SUT_ID):
    root = QUALIFICATION / sut
    accept_path = root / "acceptance.json"
    if accept_path.exists():
        return verify_release(sut, require_qualified=False)
    contract = runtime_contract(sut)
    directory = root / "qualification"
    course_rows = []
    for course in COURSES:
        args = (sut, course.scenario, ENVIRONMENT_SEED, course.duration_s)
        result = drive(*args, course=course)
        if drive(*args, course=course) != result:
            raise ValueError(f"SAC course replay changed: {course.name}")
        write_json(directory / f"{course.name}.json.gz", result)
        course_rows.append(result["result"])
    evidence = save_course_evidence(
        directory, course_rows, courses=COURSES, mode="controller_execution"
    )
    replays = []
    if evidence["qualified"]:
        for scene, duration, protocol in public.development_scenes():
            payload = drive(sut, scene, public.DEVELOPMENT_SEED, duration)
            clearance, clipped = public.inspect_frames(payload)
            record = root / "development" / f"{scene.kind}.json.gz"
            gif = public.FIGURES / sut / f"{scene.kind}.gif"
            write_json(record, payload)
            public.animate(sut, payload, clearance, gif)
            replays.append({
                "scenario_id": scene.scenario_id,
                "protocol": protocol.relative_to(ROOT).as_posix(),
                "protocol_sha256": sha256(protocol.read_bytes()),
                "trajectory": record.relative_to(ROOT).as_posix(),
                "trajectory_sha256": sha256(record.read_bytes()),
                "gif": gif.relative_to(ROOT).as_posix(),
                "gif_sha256": sha256(gif.read_bytes()),
                "applied_limits_frames": clipped,
            })
    report = {
        "sut_id": sut, "runtime_contract": contract, "qualified": evidence["qualified"],
        "independent_policy_rerun_identical": True, "courses": course_rows,
        "development_replays": replays,
    }
    write_json(accept_path, report)
    return report


def verify_release(sut=None, *, require_qualified=True):
    sut = selected_target() if sut is None else sut
    root = QUALIFICATION / sut
    report = read_json(root / "acceptance.json")
    if report["runtime_contract"] != runtime_contract(sut):
        raise ValueError("Continuous SAC runtime contract changed after qualification")
    if not report["qualified"] and not require_qualified:
        return report
    evidence = verify_course_evidence(root / "qualification")
    if (
        report["courses"] != evidence["courses"]
        or not report["independent_policy_rerun_identical"]
        or [row["scenario_id"] for row in report["development_replays"]]
        != [scene.scenario_id for scene, _, _ in public.development_scenes()]
    ):
        raise ValueError("Incomplete Continuous SAC qualification")
    for row in report["development_replays"]:
        for field in ("protocol", "trajectory", "gif"):
            if sha256((ROOT / row[field]).read_bytes()) != row[f"{field}_sha256"]:
                raise ValueError(f"Continuous SAC qualification artifact changed: {row[field]}")
    return report


def select_release(device):
    release_path = QUALIFICATION / "release.json"
    if release_path.exists():
        return verify_release()
    report = train(device)
    if report["selected_checkpoint"] is None:
        reason = "no_checkpoint_passed_development_at_fixed_budget"
    else:
        result = qualify_release(SUT_ID)
        if result["qualified"]:
            reason, sut = None, SUT_ID
        else:
            reason = "frozen_sac_failed_formal_driving_capability"
    if reason is not None:
        sut = FALLBACK
        result = qualify_release(FALLBACK)
    if not result["qualified"]:
        raise ValueError(f"The fixed fallback failed independent qualification: {sut}")
    freeze(release_path, {
        "sut_id": sut, "fallback_reason": reason,
        "planned_sut": SUT_ID,
        "acceptance_sha256": sha256((QUALIFICATION / sut / "acceptance.json").read_bytes()),
    })
    return verify_release(sut)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "qualify", "verify", "release"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "train":
        result = train(args.device)
        selected = result["selected_checkpoint"]
        print(SUT_ID, "selected development checkpoint:", None if selected is None else selected["step"])
    elif args.action == "qualify":
        result = qualify_release()
        print(SUT_ID, "formal capability:", [row["passed"] for row in result["courses"]])
    elif args.action == "verify":
        verify_release()
        print(selected_target(), "admitted")
    else:
        report = select_release(args.device)
        print(report["sut_id"], "admitted")


if __name__ == "__main__":
    main()
