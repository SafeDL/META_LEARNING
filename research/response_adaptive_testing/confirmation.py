"""Lock one candidate and measure prospective complete target pools."""
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

from .config import BUDGET, OUTPUT, ROOT, SEEDS

CONFIRMATION = OUTPUT / "confirmation"
PROFILE_COUNT = 24
REPLICATES = 2
ROUND = 1
ALPHA = 0.05 / 2**ROUND
METHODS = ("candidate", "previous_best", "frozen_original", "ras_frt_uq",
           "class_rank")
PRIMARY_CONTROLS = ("previous_best", "frozen_original", "ras_frt_uq")
WORKERS = 12
REPO = ROOT.parents[1]


def profiles():
    rng = np.random.default_rng(20261006)
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
        seed = 2000001 + 2 * pool + family
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
                "scenario_id"] = f"response_confirmation:{pool}:{family}:{index:04d}"
            scene["sampling_seed"] = seed
    return scenes


def verify_lock(protocol):
    seal = read_json(CONFIRMATION / "lock.json")
    if hashlib.sha256(
        (CONFIRMATION /
         "protocol.json").read_bytes()).hexdigest() != seal["protocol_sha256"]:
        raise ValueError("The prospective protocol changed after locking")
    changed = [
        name for name, digest in protocol["sha256"].items()
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise ValueError(f"Locked experiment inputs changed: {changed}")


def lock():
    path = CONFIRMATION / "protocol.json"
    if path.exists():
        protocol = read_json(path)
        verify_lock(protocol)
        return protocol
    state = read_json(OUTPUT / "models/monotone_transfer/logit_600.json")
    options = {**state["options"], "lookahead": False}
    source_paths = [
        ROOT / name for name in ("model.py", "session.py", "config.py",
                                 "develop.py", "confirmation.py", "confirm.py",
                                 "evaluate_confirmation.py")
    ]
    source_paths.append(OUTPUT / "models/monotone_transfer/logit_600.json")
    previous = REPO / "research/history_response_testing"
    source_paths += [
        previous / name
        for name in ("config.py", "kernel.py", "session.py", "calibration.py",
                     "history_model.py", "scenarios.py")
    ]
    source_paths += [
        previous / "results/models" / f"emulator_{seed}.pt" for seed in SEEDS
    ]
    frozen = read_json(previous / "archives/baseline_manifest.json")["sha256"]
    hashes = {
        str(p.relative_to(REPO)).replace("\\", "/"):
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    }
    for name, digest in frozen.items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen original input changed: {name}")
        hashes[name] = digest
    protocol = {
        "stage":
        "prospective confirmation; locked before any new target outcome",
        "round":
        ROUND,
        "round_alpha":
        ALPHA,
        "candidate_options":
        options,
        "methods":
        METHODS,
        "primary_controls":
        PRIMARY_CONTROLS,
        "budget":
        BUDGET,
        "seeds":
        SEEDS,
        "replicates_per_profile":
        REPLICATES,
        "profiles": [asdict(profile) for profile in profiles()],
        "pool_size":
        2048,
        "total_physical_measurements":
        PROFILE_COUNT * REPLICATES * 2048,
        "statistical_unit":
        "SUT parameter profile; mean over two coordinate pools and five predictor seeds",
        "primary_metrics":
        ["mean_cumulative_collisions/pool_collisions", "F200/pool_collisions"],
        "zero_failure_pool":
        "normalized area=0 and recall=1 for every method; retained with zero paired differences",
        "testing":
        "six exact two-sided paired sign-flip tests with Holm correction",
        "test_assumption":
        "paired profile differences have exchangeable signs under their respective null",
        "confidence_interval":
        "paired profile bootstrap, resampled within controller type; 20000 replicates",
        "success_rule":
        "positive means and Holm p below round_alpha for both metrics versus all three primary controls; both bootstrap CI lower bounds above zero versus previous_best",
        "multiple_rounds":
        "round r spends 0.05/2**r; failed confirmation becomes development, next round uses new profiles and coordinates",
        "selection_feedback":
        "queried continuous risk only; RAS receives queried binary collision only",
        "secondary_metrics": [
            "F50", "F100", "F200", "missed_failures", "all_failures_found",
            "budget_upper_bound_attainment", "queries_to_all_failures"
        ],
        "scope":
        "new IDM/FVDM parameter profiles in declared ranges, same simulator and two finite scenario families",
        "sha256":
        hashes
    }
    write_json(path, protocol)
    archive = CONFIRMATION / "locked_inputs.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        for source in source_paths:
            handle.write(source,
                         str(source.relative_to(REPO)).replace("\\", "/"))
        handle.write(path, str(path.relative_to(REPO)).replace("\\", "/"))
    write_json(
        CONFIRMATION / "lock.json", {
            "protocol_sha256":
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_archive_sha256":
            hashlib.sha256(archive.read_bytes()).hexdigest()
        })
    for index, profile in enumerate(profiles()):
        for replicate in range(REPLICATES):
            folder = CONFIRMATION / profile.name / f"pool_{replicate}"
            write_json(folder / "scenarios.json",
                       pool_scenes(index, replicate))
    print("PROSPECTIVE LOCK",
          PROFILE_COUNT,
          "profiles",
          PROFILE_COUNT * REPLICATES,
          "complete pools",
          "alpha",
          ALPHA,
          flush=True)
    return protocol


def measure(protocol):
    verify_lock(protocol)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS,
                             mp_context=context) as executor:
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
                            print("CONFIRMATION MEASURE",
                                  profile.name,
                                  replicate,
                                  len(cache),
                                  flush=True)
                risk = np.array([cache[i]["risk"] for i in range(len(scenes))])
                if not np.isfinite(risk).all() or not ((0 <= risk) &
                                                       (risk <= 1)).all():
                    raise ValueError(
                        "Confirmation contains an invalid continuous risk")
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
    select_all()
    evaluate()


if __name__ == "__main__":
    main()
