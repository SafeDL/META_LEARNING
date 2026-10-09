"""Freeze and measure the third prospective candidate confirmation."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.stats import qmc

from methods.history_guided_testing.config import BOUNDS, TARGET
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene
from methods.history_guided_testing.scenarios import scene_library
from research.behavior_response_testing.confirmation import (
    CONFIRMATION as ROUND_TWO, verify_lock as verify_round_two_lock)
from research.response_adaptive_testing.audit_confirmation import (
    prior_coordinates)
from research.behavior_response_testing.train import SEEDS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


CONFIRMATION = RESULTS / "confirmation"
PREMEASUREMENT = RESULTS / "premeasurement_attempt"
REPO = ROOT.parents[1]
PROFILE_COUNT = 48
REPLICATES = 2
ROUND = 3
ALPHA = 0.05 / 2**ROUND
WORKERS = 12
BUDGET = 200
METHODS = (
    "risk_conditioned", "collision_only_ablation", "behavior_posterior",
    "previous_best", "frozen_original", "ras_frt_uq",
    "matched_collision_gp")
PRIMARY_CONTROLS = (
    "previous_best", "frozen_original", "ras_frt_uq",
    "matched_collision_gp")


def profiles():
    rng = np.random.default_rng(20261023)
    return [
        replace(TARGET,
                name=f"{controller.lower()}_r3_{index:02d}",
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
        seed = 5000001 + 2 * pool + family
        points = qmc.Sobol(4, scramble=True,
                           seed=seed).random_base2(10)
        bounds = np.asarray(BOUNDS[family])
        for index, point in enumerate(points):
            scene = scenes[family * 1024 + index]
            scene["numeric_input"] = [*point.tolist(), family]
            scene["active_parameters"] = dict(
                zip(scene["active_parameters"],
                    (bounds[:, 0] + point *
                     (bounds[:, 1] - bounds[:, 0])).tolist()))
            scene["scenario_id"] = (
                f"risk_conditioned_confirmation:{pool}:{family}:{index:04d}")
            scene["sampling_seed"] = seed
    return scenes


def _round_two_inputs():
    protocol = read_json(ROUND_TWO / "protocol.json")
    summary = read_json(ROUND_TWO / "summary.json")
    if summary.get("success") is not False:
        raise ValueError("Expected a failed, audited development confirmation")
    verify_round_two_lock(protocol)
    return protocol


def verify_freshness(protocol):
    old_profiles = []
    known = prior_coordinates()
    excluded_path = PREMEASUREMENT / "excluded_coordinates.npz"
    if excluded_path.exists():
        with np.load(excluded_path) as previous_attempt:
            known.update(map(tuple, previous_attempt["x"]))
    previous = [
        REPO / "research/response_adaptive_testing/results/confirmation",
        ROUND_TWO,
    ]
    for folder in previous:
        for profile_path in folder.glob("*/pool_*/responses.npz"):
            with np.load(profile_path) as bank:
                known.update(map(tuple, bank["x"]))
        source_protocol = folder / "protocol.json"
        if source_protocol.exists():
            old_profiles += read_json(source_protocol)["profiles"]
    describe = lambda p: (p["controller"], p["max_brake"],
                          p["desired_gap"], p["perception_delay_s"])
    old_states = {describe(profile) for profile in old_profiles}
    new_states = {describe(profile) for profile in protocol["profiles"]}
    if (len(new_states) != PROFILE_COUNT or old_states & new_states):
        raise ValueError("Round-three SUT profiles overlap earlier cohorts")
    new_coordinates = set()
    for index, profile in enumerate(profiles()):
        if asdict(profile) != protocol["profiles"][index]:
            raise ValueError("Frozen profiles differ from their generator")
        for replicate in range(REPLICATES):
            scenarios = read_json(
                CONFIRMATION / profile.name / f"pool_{replicate}" /
                "scenarios.json")
            if scenarios != pool_scenes(index, replicate):
                raise ValueError("Frozen scenario pool differs from generator")
            coordinates = {tuple(scene["numeric_input"])
                           for scene in scenarios}
            if (len(coordinates) != 2048 or coordinates & known
                    or coordinates & new_coordinates):
                raise ValueError("Scenario coordinates are repeated")
            new_coordinates.update(coordinates)
    return {
        "known_coordinates": len(known),
        "new_coordinates": len(new_coordinates),
        "all_coordinates_disjoint": True,
        "all_profiles_new": True,
    }


def _source_paths():
    paths = [
        ROOT / name for name in (
            "__init__.py", "design.md", "diagnose_risk_coupling.py",
            "risk_session.py", "test_risk_session.py", "train_decoder.py",
            "confirmation.py", "confirm.py", "evaluate_confirmation.py",
            "audit_confirmation.py", "report_confirmation.py")
    ]
    paths += [
        ROOT / "results" / name for name in (
            "risk_coupling_diagnostic.json", "holdout_replay.json",
            "decoder.json")
    ]
    if PREMEASUREMENT.exists():
        paths += list(PREMEASUREMENT.glob("*"))
    paths += [
        ROUND_TWO / name for name in (
            "protocol.json", "lock.json", "summary.json",
            "locked_inputs.zip")
    ]
    round_two_profiles = _round_two_inputs()["profiles"]
    paths += [
        ROUND_TWO / profile["name"] / f"pool_{replicate}" /
        "responses.npz"
        for profile in round_two_profiles for replicate in range(REPLICATES)
    ]
    paths += [
        ROUND_TWO.parent / "models" / f"predictor_{seed}{suffix}"
        for seed in SEEDS for suffix in (".pt", ".json")
    ]
    paths += [
        REPO / name for name in (
            "research/behavior_response_testing/model.py",
            "research/behavior_response_testing/prediction.py",
            "research/behavior_response_testing/session.py",
            "research/behavior_response_testing/log_probability.py",
            "research/behavior_response_testing/statistics.py",
            "research/response_adaptive_testing/confirm.py",
            "research/response_adaptive_testing/audit_confirmation.py",
            "research/response_adaptive_testing/audit_statistics.py",
            "research/response_adaptive_testing/develop.py",
            "research/history_response_testing/history_model.py",
            "research/history_response_testing/scenarios.py",
            "research/history_response_testing/kernel.py",
            "research/history_response_testing/session.py",
            "research/history_response_testing/calibration.py",
            "methods/history_guided_testing/history.py",
            "methods/history_guided_testing/kernel.py",
            "methods/history_guided_testing/experiment.py",
            "methods/history_guided_testing/config.py",
            "methods/history_guided_testing/prepare.py",
            "methods/history_guided_testing/scenarios.py")
    ]
    return paths


def verify_lock(protocol):
    seal = read_json(CONFIRMATION / "lock.json")
    digest = hashlib.sha256((CONFIRMATION / "protocol.json").read_bytes())
    if digest.hexdigest() != seal["protocol_sha256"]:
        raise ValueError("Round-three protocol changed after locking")
    changed = [
        name for name, expected in protocol["sha256"].items()
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected
    ]
    if changed:
        raise ValueError(f"Frozen round-three inputs changed: {changed}")


def lock():
    protocol_path = CONFIRMATION / "protocol.json"
    lock_revision = 1
    restart_metadata = None
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        try:
            verify_lock(protocol)
            return protocol
        except ValueError:
            measured = list(CONFIRMATION.glob("*/pool_*/responses.npz"))
            measured += list(CONFIRMATION.glob("*/pool_*/cost.json"))
            measured += [path for path in
                         CONFIRMATION.glob("*/pool_*/measurements.jsonl")
                         if path.stat().st_size]
            measured += [path for path in
                         CONFIRMATION.glob("*/pool_*/selection/*.json")]
            if measured:
                raise ValueError(
                    "Cannot revise a round-three lock after results were recorded")
            seal = read_json(CONFIRMATION / "lock.json")
            PREMEASUREMENT.mkdir(parents=True, exist_ok=True)
            for name in ("protocol.json", "lock.json", "locked_inputs.zip"):
                shutil.copy2(CONFIRMATION / name, PREMEASUREMENT / name)
            coordinates = np.asarray([
                scene["numeric_input"]
                for scenario_path in CONFIRMATION.glob("*/pool_*/scenarios.json")
                for scene in read_json(scenario_path)
            ], dtype=np.float64)
            excluded_path = PREMEASUREMENT / "excluded_coordinates.npz"
            np.savez_compressed(excluded_path, x=coordinates)
            restart_metadata = {
                "first_protocol_sha256": seal["protocol_sha256"],
                "first_source_archive_sha256": seal[
                    "source_archive_sha256"],
                "allocated_pools_excluded": int(
                    len(list(CONFIRMATION.glob("*/pool_*/scenarios.json")))),
                "excluded_scenario_coordinates": int(len(coordinates)),
                "physical_outcomes_recorded": False,
                "selector_runs_recorded": False,
                "reason": (
                    "The first measurement attempt exposed a missing scenario-index mapping in the journal writer. No complete response bank, non-empty journal, cost record or selector result was persisted. All initially allocated coordinates were quarantined because the process pool may already have executed work; the corrected confirmation uses disjoint scenario seeds."
                ),
            }
            write_json(PREMEASUREMENT / "restart_metadata.json",
                       restart_metadata)
            lock_revision = int(protocol.get("lock_revision", 1)) + 1
            print("RESTARTING PRE-MEASUREMENT LOCK", lock_revision,
                  "excluded coordinates", len(coordinates), flush=True)
    round_two = _round_two_inputs()
    decoder = read_json(ROOT / "results" / "decoder.json")
    candidate_options = round_two["candidate"]
    sha256 = dict(round_two["sha256"])
    for path in _source_paths():
        sha256[str(path.relative_to(REPO)).replace("\\", "/")] = (
            hashlib.sha256(path.read_bytes()).hexdigest())
    protocol = {
        "stage": "Prospective confirmation; sealed before new target outcomes",
        "round": ROUND,
        "lock_revision": lock_revision,
        "premeasurement_restart": restart_metadata,
        "round_alpha": ALPHA,
        "methods": METHODS,
        "candidate_method": "risk_conditioned",
        "primary_controls": PRIMARY_CONTROLS,
        "development_ablation": "collision_only_ablation",
        "secondary_mechanism_control": "behavior_posterior",
        "candidate": {
            "continuous_behaviors": candidate_options["continuous_behaviors"],
            "risk_discrepancy": candidate_options["risk_discrepancy"],
            "matched_collision_calibrators": candidate_options[
                "risk_calibrators"],
            "risk_conditioned_decoder": decoder["calibrators"],
            "decoder_fit_profiles": decoder["development_profiles"],
            "prior": "Uniform normalized physical behavior coordinates; balanced controller type",
            "selection": "Posterior mean collision probability after integrating the local risk-residual GP posterior",
            "online_feedback": "One queried continuous risk per selector step; collision labels remain hidden except to RAS",
            "uncertainty_integral": "Logistic-normal moment approximation sigma(mu/sqrt(1+pi*v/8))",
            "collision_only_ablation": "Separate collision-logit-only logistic decoder fit to the same 48 development profiles",
        },
        "budget": BUDGET,
        "seeds": list(SEEDS),
        "profiles": [asdict(profile) for profile in profiles()],
        "replicates_per_profile": REPLICATES,
        "pool_size": 2048,
        "total_physical_measurements": PROFILE_COUNT * REPLICATES * 2048,
        "total_selector_disclosures": PROFILE_COUNT * REPLICATES *
        len(SEEDS) * len(METHODS) * BUDGET,
        "statistical_unit": "SUT profile; average both pools and all five predictor seeds",
        "primary_metrics": ["normalized discovery area", "budget-end recall"],
        "zero_failure_pool": "Area=0; recall=1; retained with zero paired difference",
        "testing": "Eight exact two-sided paired sign-flip tests with Holm correction",
        "round_alpha": ALPHA,
        "confidence_interval": "Paired bootstrap within controller kind, 20000 draws",
        "success_rule": "Both co-primary metrics must be positive with Holm p below round alpha against all four primary controls; both bootstrap 95% lower limits must be positive against previous_best and matched_collision_gp",
        "multiple_rounds": "Spend 0.05/2**r; any failed round becomes development and requires new profiles and coordinates",
        "target_parameters": "Retained by measurement parent and never sent to selectors",
        "full_target_arrays": "Retained by measurement parent; each isolated selector receives one allowed response at a time",
        "scope": "New IDM/FVDM parameter profiles in declared ranges, same simulator and two finite scenario families",
        "sha256": sha256,
    }
    CONFIRMATION.mkdir(parents=True, exist_ok=True)
    for index, profile in enumerate(profiles()):
        for replicate in range(REPLICATES):
            write_json(CONFIRMATION / profile.name / f"pool_{replicate}" /
                       "scenarios.json", pool_scenes(index, replicate))
    write_json(protocol_path, protocol)
    archive_path = CONFIRMATION / "locked_inputs.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name in sha256:
            archive.write(REPO / name, name)
        archive.write(protocol_path,
                      str(protocol_path.relative_to(REPO)).replace("\\", "/"))
    write_json(CONFIRMATION / "lock.json", {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    })
    verify_lock(protocol)
    fresh = verify_freshness(protocol)
    write_json(CONFIRMATION / "freshness.json", fresh)
    print("RISK-CONDITIONED ROUND-THREE LOCK", PROFILE_COUNT,
          "profiles; alpha", ALPHA, "fresh coordinates", fresh["new_coordinates"],
          flush=True)
    return protocol


def measure(protocol):
    verify_lock(protocol)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             mp_context=mp.get_context("spawn")) as executor:
        for profile in profiles():
            for replicate in range(REPLICATES):
                folder = (CONFIRMATION / profile.name /
                          f"pool_{replicate}")
                if (folder / "responses.npz").exists():
                    continue
                scenes = read_json(folder / "scenarios.json")
                journal_path = folder / "measurements.jsonl"
                cache = ({
                    row["index"]: row for row in map(
                        json.loads,
                        journal_path.read_text(encoding="utf-8").splitlines())
                } if journal_path.exists() else {})
                missing = [i for i in range(len(scenes)) if i not in cache]
                with journal_path.open("a", encoding="utf-8") as journal:
                    measured = executor.map(
                        measure_scene,
                        [(scenes[i], profile) for i in missing])
                    for index, row in zip(missing, measured):
                        row["index"] = int(index)
                        cache[row["index"]] = row
                        journal.write(json.dumps(row) + "\n")
                        journal.flush()
                rows = [cache[i] for i in range(len(scenes))]
                x = np.asarray([scene["numeric_input"] for scene in scenes],
                               dtype=np.float64)
                risk = np.asarray([row["risk"] for row in rows],
                                  dtype=np.float64)
                collision = np.asarray([row["collision"] for row in rows],
                                       dtype=np.bool_)
                np.savez_compressed(folder / "responses.npz", x=x, risk=risk,
                                    collision=collision)
                write_json(folder / "cost.json", {
                    "full_pool_measured": True,
                    "unique_physical_measurements": len(scenes),
                    "profile": asdict(profile),
                })
                print("RISK-CONDITIONED PHYSICAL POOL", profile.name,
                      replicate, "fails", int(collision.sum()), flush=True)


def main():
    protocol = lock()
    measure(protocol)


if __name__ == "__main__":
    main()
