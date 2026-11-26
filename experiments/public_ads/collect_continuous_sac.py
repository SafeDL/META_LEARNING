"""Collect the admitted SAC's fixed 8,000-scenario execution library."""

import argparse
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

from simulator.highway.collision_env import execute

from experiments.execution_events import analyze_events
from experiments.executions import ExecutionBank
from experiments.history_transfer_experiment import OUTPUT as ORIGINAL, registered_populations
from experiments.parallel import initialize_worker, prepare_processes
from experiments.protocol import ROOT, episode_duration
from experiments.public_ads.continuous_sac import QUALIFICATION, controller, verify_release
from experiments.rss import measurement_contract, trajectory_margin
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.schema import QueryFeedback
from experiments.storage import read_json, write_json

PHYSICAL = ROOT / "results/public_ads/physical/target"
ENVIRONMENT_SEED = 81001


def _execute_one(scene, index, directory, sut, checkpoint_digest):
    attempt = directory / "attempts" / f"{index:06d}.json"
    row = {"call": index + 1, "scenario_id": scene.scenario_id, "started": time.time()}
    with attempt.open("x", encoding="utf-8") as stream:
        json.dump({**row, "status": "running"}, stream)
    try:
        execution = execute(
            controller(sut, digest=checkpoint_digest), scene,
            environment_seed=ENVIRONMENT_SEED, duration_s=episode_duration(scene.kind),
        )
        trajectory = execution.trajectory
        margin = trajectory_margin(trajectory)["minimum_margin"]
        feedback = replace(
            execution.feedback,
            safety=replace(execution.feedback.safety, rss_margin=margin),
            rss_contract=measurement_contract(),
        )
        destination = directory / "executions" / f"{index:06d}.json.gz"
        write_json(destination, {
            "call": index + 1, "scenario": asdict(scene), "feedback": feedback.to_dict(),
            "trajectory": trajectory, "events": analyze_events(scene, trajectory),
        })
    except Exception as error:
        write_json(attempt, {**row, "status": "failed", "error": repr(error)})
        raise
    write_json(attempt, {**row, "status": "complete", "execution_s": feedback.execution_s})


def _audit(directory, scenes):
    collisions = 0
    maximum_error = 0.0
    for index, scene in enumerate(scenes):
        path = directory / "executions" / f"{index:06d}.json.gz"
        record = read_json(path)
        attempt = read_json(directory / "attempts" / f"{index:06d}.json")
        feedback = QueryFeedback.from_dict(record["feedback"])
        trajectory, events = record["trajectory"], record["events"]
        observed_collision = any(frame["collision"] for frame in trajectory)
        margin = trajectory_margin(trajectory)["minimum_margin"]
        if (
            attempt["status"] != "complete"
            or attempt["call"] != record["call"] or record["call"] != index + 1
            or attempt["scenario_id"] != scene.scenario_id
            or feedback.scenario_id != scene.scenario_id or record["scenario"] != asdict(scene)
            or feedback.collision != observed_collision
            or events != analyze_events(scene, trajectory)
            or feedback.rss_contract != measurement_contract()
            or not math.isclose(feedback.safety.rss_margin, margin, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"Physical execution audit failed for {scene.scenario_id}")
        maximum_error = max(maximum_error, abs(feedback.safety.rss_margin - margin))
        collisions += observed_collision
    return {"executions": len(scenes), "collisions": collisions, "maximum_rss_error": maximum_error}


def collect_bank(scenes, acceptance, *, workers):
    sut = acceptance["sut_id"]
    kind = scenes[0].kind
    if len(scenes) != 2000 or any(scene.kind != kind for scene in scenes):
        raise ValueError("Continuous SAC requires one complete registered target population")
    directory = PHYSICAL / kind / sut
    manifest_path = directory / "manifest.json"
    specification = {
        "format": "collision-execution-bank",
        "sut_id": sut,
        "environment_seed": ENVIRONMENT_SEED,
        "qualification": (QUALIFICATION / sut / "acceptance.json").relative_to(ROOT).as_posix(),
        "contract": {
            "public_ads": acceptance["runtime_contract"], "scenario_kind": kind,
            "episode_duration_s": episode_duration(kind),
        },
        "scenarios": [asdict(scene) for scene in scenes],
    }
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if any(saved.get(key) != value for key, value in specification.items()):
            raise ValueError("Existing SAC execution bank uses another protocol")
        if saved["status"] == "complete":
            if _audit(directory, scenes) != saved["audit"]:
                raise ValueError("Completed SAC bank audit changed")
            return ExecutionBank(directory)
    for name in ("executions", "attempts"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    pending = []
    for index in range(len(scenes)):
        attempt_path = directory / "attempts" / f"{index:06d}.json"
        result_path = directory / "executions" / f"{index:06d}.json.gz"
        if not attempt_path.exists() and not result_path.exists():
            pending.append(index)
        elif not result_path.exists() or read_json(attempt_path)["status"] != "complete":
            raise ValueError(f"Resolve the incomplete physical attempt before continuing: {attempt_path}")
    write_json(manifest_path, {**specification, "status": "running"})
    digest = acceptance["runtime_contract"]["checkpoint_sha256"]
    prepare_processes()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker,
    ) as pool:
        futures = [pool.submit(_execute_one, scenes[i], i, directory, sut, digest) for i in pending]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 100 == 0 or completed == len(pending):
                print(sut, kind, completed, "/", len(pending), flush=True)
    audit = _audit(directory, scenes)
    write_json(manifest_path, {
        **specification, "status": "complete", "candidate_count": len(scenes), "audit": audit,
    })
    return ExecutionBank(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    acceptance = verify_release()
    populations = registered_populations(ORIGINAL)
    for kind in FORMAL_SCENARIOS:
        collect_bank(populations["target", kind], acceptance, workers=args.workers)


if __name__ == "__main__":
    main()
