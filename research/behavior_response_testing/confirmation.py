"""Freeze a behavioral candidate before measuring fresh SUTs and coordinates."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing as mp
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.stats import qmc

from methods.history_guided_testing.config import BOUNDS, TARGET
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene
from methods.history_guided_testing.scenarios import scene_library
from research.response_adaptive_testing.confirmation import CONFIRMATION as FIRST_CONFIRMATION, verify_lock as verify_first_lock

from .prediction import behavior_grid
from .session import BUDGET
from .train import OUTPUT, ROOT, SEEDS

CONFIRMATION = OUTPUT / "confirmation"
REPO = ROOT.parents[1]
PROFILE_COUNT = 48
REPLICATES = 2
ROUND = 2
ALPHA = 0.05 / 2**ROUND
WORKERS = 12
METHODS = ("candidate", "previous_best", "frozen_original", "ras_frt_uq",
           "matched_collision_gp", "fixed_behavior", "discrete_history")
PRIMARY_CONTROLS = ("previous_best", "frozen_original", "ras_frt_uq",
                    "matched_collision_gp")


def profiles():
    rng = np.random.default_rng(20261020)
    return [
        replace(TARGET,
                name=f"{controller.lower()}_{index:02d}",
                controller=controller,
                max_brake=float(rng.uniform(3.5, 9)),
                desired_gap=float(rng.uniform(6.5, 9.5)),
                perception_delay_s=float(rng.uniform(0.025, 0.5)))
        for controller in ("IDM", "FVDM")
        for index in range(PROFILE_COUNT // 2)
    ]


def pool_scenes(profile_index, replicate):
    scenes = scene_library("target")
    pool = profile_index * REPLICATES + replicate
    for family in (0, 1):
        seed = 3000001 + 2 * pool + family
        points = qmc.Sobol(4, scramble=True, seed=seed).random_base2(10)
        bounds = np.asarray(BOUNDS[family])
        for index, point in enumerate(points):
            scene = scenes[family * 1024 + index]
            scene["numeric_input"] = [*point.tolist(), family]
            scene["active_parameters"] = dict(
                zip(scene["active_parameters"],
                    (bounds[:, 0] + point *
                     (bounds[:, 1] - bounds[:, 0])).tolist()))
            scene[
                "scenario_id"] = f"behavior_confirmation:{pool}:{family}:{index:04d}"
            scene["sampling_seed"] = seed
    return scenes


def verify_lock(protocol):
    seal = read_json(CONFIRMATION / "lock.json")
    assert hashlib.sha256(
        (CONFIRMATION /
         "protocol.json").read_bytes()).hexdigest() == seal["protocol_sha256"]
    changed = [
        name for name, digest in protocol["sha256"].items()
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise ValueError(
            f"Prospective inputs changed after locking: {changed}")


def lock():
    path = CONFIRMATION / "protocol.json"
    if path.exists():
        protocol = read_json(path)
        verify_lock(protocol)
        return protocol
    first = read_json(FIRST_CONFIRMATION / "summary.json")
    assert first["success"] is False
    verify_first_lock(first["protocol"])
    training = read_json(OUTPUT / "protocol.json")
    development = read_json(OUTPUT / "development_protocol.json")
    grid = behavior_grid()
    candidate = {
        "method":
        "Continuous behavior posterior, historical likelihood-fitted risk discrepancy, stable Bernoulli probability ranking",
        "continuous_behaviors":
        grid.tolist(),
        "prior":
        "Uniform normalized brake/gap/delay and balanced IDM/FVDM; equal prior particle masses",
        "historical_behaviors":
        list(training["historical_behavior_descriptors"].values()),
        "risk_discrepancy": {
            str(seed):
            read_json(OUTPUT / "calibration" /
                      f"discrepancy_{seed}.json")["parameters"]
            for seed in SEEDS
        },
        "risk_calibrators":
        development["historical_risk_calibration"],
        "history_scope":
        "65536 response labels from sixteen profiles train the conditional predictor; 32768 labels and known configurations from eight additional historical profiles validate it and calibrate risk model error",
        "collision_model":
        "Conditional Bernoulli NN given latent behavior; independent of risk discrepancy under the working model",
        "selection":
        "Maximize posterior collision probability through stable log noncollision probability; no budget-dependent risk mixing",
        "discrete_control":
        "Same fitted model error and sixteen actual historical behavioral hypotheses",
        "fixed_control":
        "Same continuous prior and NN, fixed ranking without target-risk updates",
        "matched_control":
        "Same conditional response NN at sixteen historical states, same risk/collision calibration; previous best GP kernel with collision-score-only acquisition"
    }
    source_paths = [
        ROOT / name
        for name in ("__init__.py", "model.py", "session.py",
                     "log_probability.py", "prediction.py", "train.py",
                     "calibration.py", "statistics.py", "confirmation.py",
                     "confirm.py", "evaluate_confirmation.py",
                     "audit_confirmation.py", "verify_selector.py",
                     "test_session.py", "test_calibration.py",
                     "test_log_probability.py", "test_statistics.py")
    ]
    source_paths += [
        OUTPUT / "protocol.json", OUTPUT / "calibration" / "protocol.json",
        OUTPUT / "development_protocol.json",
        OUTPUT / "selector_verification.json"
    ]
    source_paths += [
        OUTPUT / "models" / f"predictor_{seed}{suffix}" for seed in SEEDS
        for suffix in (".pt", ".json")
    ]
    source_paths += [
        OUTPUT / "calibration" / f"discrepancy_{seed}.json" for seed in SEEDS
    ]
    source_paths += [
        FIRST_CONFIRMATION / "protocol.json", FIRST_CONFIRMATION / "lock.json"
    ]
    source_paths += list(FIRST_CONFIRMATION.glob("*/pool_*/responses.npz"))
    source_paths += [
        REPO / "research/response_adaptive_testing/audit_confirmation.py"
    ]
    source_paths += [
        REPO / "research/response_adaptive_testing/audit_statistics.py"
    ]
    hashes = dict(first["protocol"]["sha256"])
    hashes.update({
        str(p.relative_to(REPO)).replace("\\", "/"):
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    })
    protocol = {
        "stage":
        "Prospective confirmation; freeze before any new target outcome",
        "round":
        ROUND,
        "round_alpha":
        ALPHA,
        "methods":
        METHODS,
        "primary_controls":
        PRIMARY_CONTROLS,
        "candidate":
        candidate,
        "budget":
        BUDGET,
        "seeds":
        SEEDS,
        "profiles": [asdict(p) for p in profiles()],
        "replicates_per_profile":
        REPLICATES,
        "pool_size":
        2048,
        "total_physical_measurements":
        PROFILE_COUNT * REPLICATES * 2048,
        "statistical_unit":
        "SUT profile, averaging two pools and five predictor seeds",
        "primary_metrics":
        ["mean_cumulative_collisions/pool_collisions", "F200/pool_collisions"],
        "zero_failure_pool":
        "Area=0, recall=1, zero paired differences; retained",
        "testing":
        "Eight exact two-sided paired sign-flip tests with Holm correction, bounded-memory half-sum search",
        "test_assumption":
        "Exchangeable signs of paired profile differences under their nulls",
        "confidence_interval":
        "Paired bootstrap within controller kind, 20000 draws",
        "success_rule":
        "Positive means and Holm p below round_alpha for both metrics against all four primary controls; both bootstrap 95% lower limits positive against previous_best and matched_collision_gp",
        "multiple_rounds":
        "Spend 0.05/2**r; future failed rounds become development and use new profiles and coordinates",
        "selection_feedback":
        "Queried continuous risk only; RAS receives queried collision only",
        "target_parameters":
        "No actual target parameters or full response arrays enter selector payloads",
        "secondary_metrics": [
            "F10", "F30", "F50", "F100", "F150", "F200", "all_failures_found",
            "queries_to_all_failures", "budget_upper_bound_attainment",
            "missed_failures"
        ],
        "scope":
        "New IDM/FVDM parameter profiles within declared ranges, same highway-env simulator and two finite scenario families",
        "sha256":
        hashes
    }
    write_json(path, protocol)
    archive_path = CONFIRMATION / "locked_inputs.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(REPO / name, name)
        archive.write(path, str(path.relative_to(REPO)).replace("\\", "/"))
    write_json(
        CONFIRMATION / "lock.json", {
            "protocol_sha256":
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_archive_sha256":
            hashlib.sha256(archive_path.read_bytes()).hexdigest()
        })
    verify_lock(protocol)
    for index, profile in enumerate(profiles()):
        for replicate in range(REPLICATES):
            write_json(
                CONFIRMATION / profile.name / f"pool_{replicate}" /
                "scenarios.json", pool_scenes(index, replicate))
    print("BEHAVIOR PROSPECTIVE LOCK",
          PROFILE_COUNT,
          "profiles",
          ALPHA,
          flush=True)
    return protocol


def measure(protocol):
    verify_lock(protocol)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             mp_context=mp.get_context("spawn")) as executor:
        for profile in profiles():
            for replicate in range(REPLICATES):
                folder = CONFIRMATION / profile.name / f"pool_{replicate}"
                if (folder / "responses.npz").exists():
                    continue
                scenes = read_json(folder / "scenarios.json")
                journal = folder / "measurements.jsonl"
                cache = ({
                    row["index"]: row
                    for row in map(
                        json.loads,
                        journal.read_text(encoding="utf-8").splitlines())
                } if journal.exists() else {})
                missing = [i for i in range(len(scenes)) if i not in cache]
                with journal.open("a", encoding="utf-8") as handle:
                    for index, row in zip(
                            missing,
                            executor.map(measure_scene, [(scenes[i], profile)
                                                         for i in missing],
                                         chunksize=4)):
                        row["index"] = index
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        cache[index] = row
                        if len(cache) % 512 == 0:
                            print("BEHAVIOR PHYSICAL",
                                  profile.name,
                                  replicate,
                                  len(cache),
                                  flush=True)
                risk = np.array([cache[i]["risk"] for i in range(len(scenes))])
                assert np.isfinite(risk).all() and ((0 <= risk) &
                                                    (risk <= 1)).all()
                np.savez_compressed(folder / "responses.npz",
                                    x=[s["numeric_input"] for s in scenes],
                                    risk=risk,
                                    collision=[
                                        cache[i]["collision"]
                                        for i in range(len(scenes))
                                    ])
                write_json(
                    folder / "cost.json", {
                        "full_pool_measured": True,
                        "unique_physical_measurements": len(cache),
                        "profile": asdict(profile)
                    })


def main():
    protocol = lock()
    measure(protocol)
    from .confirm import main as select_all
    from .evaluate_confirmation import main as evaluate
    from .audit_confirmation import main as audit
    select_all()
    evaluate()
    audit()


if __name__ == "__main__":
    main()
