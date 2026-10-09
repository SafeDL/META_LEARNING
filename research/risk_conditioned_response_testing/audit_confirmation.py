"""Independently audit frozen inputs, physical banks and selector statistics."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import multiprocessing as mp
from zipfile import ZipFile

import numpy as np

from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene
from research.behavior_response_testing.statistics import exact_sign_flip
from research.response_adaptive_testing.audit_confirmation import verify_bank
from research.response_adaptive_testing.audit_statistics import (
    exact_p_meet_in_middle)

from .confirmation import (CONFIRMATION, REPO, pool_scenes, profiles,
                           verify_freshness, verify_lock)


AUDIT = CONFIRMATION.parent / "confirmation_audit"
REPLAY_INDICES = (0, 1023, 1024, 2047)


def verify_archives(protocol):
    seal = read_json(CONFIRMATION / "lock.json")
    archive_path = CONFIRMATION / "locked_inputs.zip"
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != seal[
            "source_archive_sha256"]:
        raise ValueError("Locked source archive digest differs")
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Locked source archive is corrupt")
        for name in archive.namelist():
            if archive.read(name) != (REPO / name).read_bytes():
                raise ValueError(f"Archived source differs: {name}")
    return {
        "locked_files": len(protocol["sha256"]),
        "source_archive_equal_current_inputs": True,
    }


def verify_selector_statistics(protocol):
    summary = read_json(CONFIRMATION / "summary.json")
    methods, profiles_list = protocol["methods"], protocol["profiles"]
    values = np.zeros((len(profiles_list), len(methods), 2))
    disclosures = 0
    for profile_index, profile in enumerate(profiles_list):
        grouped = {method: [] for method in methods}
        for replicate in range(protocol["replicates_per_profile"]):
            folder = (CONFIRMATION / profile["name"] /
                      f"pool_{replicate}")
            with np.load(folder / "responses.npz") as bank:
                for method in methods:
                    for seed in protocol["seeds"]:
                        result = read_json(folder / "selection" /
                                           f"{method}_{seed}.json")
                        selected = result["selected_indices"]
                        if (len(selected) != len(set(selected)) or
                                len(selected) != protocol["budget"]):
                            raise ValueError("Invalid or duplicate selections")
                        expected = ([{"index": index,
                                      "collision": bool(bank["collision"][index])}
                                     for index in selected]
                                    if method == "ras_frt_uq" else
                                    [{"index": index,
                                      "continuous_risk": float(bank["risk"][index])}
                                     for index in selected])
                        if result["observations"] != expected:
                            raise ValueError("Disclosure log is not exact")
                        curve = np.cumsum(bank["collision"][selected])
                        if curve.tolist() != result["curve"]:
                            raise ValueError("Cached discovery curve differs")
                        total = int(bank["collision"].sum())
                        grouped[method].append([
                            curve.mean() / total if total else 0.0,
                            curve[-1] / total if total else 1.0,
                        ])
                        disclosures += len(selected)
        for method_index, method in enumerate(methods):
            values[profile_index, method_index] = np.mean(grouped[method], 0)
            saved = summary["per_profile"][method][profile_index]
            np.testing.assert_allclose(
                values[profile_index, method_index],
                [saved["normalized_area"], saved["recall"]],
                atol=1e-12, rtol=0)

    keys = [(control, metric) for control in protocol["primary_controls"]
            for metric in ("normalized_area", "recall")]
    differences = np.column_stack([
        values[:, methods.index("risk_conditioned"), column] -
        values[:, methods.index(control), column]
        for control in protocol["primary_controls"] for column in (0, 1)
    ])
    probabilities = np.asarray([
        exact_p_meet_in_middle(differences[:, index:index + 1])[0]
        for index in range(differences.shape[1])
    ])
    adjusted = np.zeros(len(probabilities))
    running = 0.0
    for rank, index in enumerate(
            sorted(range(len(probabilities)), key=lambda i: probabilities[i])):
        running = max(running,
                      probabilities[index] * (len(probabilities) - rank))
        adjusted[index] = min(1.0, running)
    rng = np.random.default_rng(20261019)
    weights = np.zeros((20000, len(profiles_list)), dtype=np.int16)
    for controller in ("IDM", "FVDM"):
        indices = np.asarray([
            i for i, profile in enumerate(profiles_list)
            if profile["controller"] == controller
        ])
        draws = rng.integers(0, len(indices), (20000, len(indices)))
        for local, index in enumerate(indices):
            weights[:, index] = (draws == local).sum(1)
    intervals = np.quantile(weights @ differences / len(profiles_list),
                            [0.025, 0.975], axis=0).T
    for column, (control, metric) in enumerate(keys):
        saved = summary["comparisons"][control][metric]
        if (abs(saved["exact_two_sided_p"] - probabilities[column]) > 1e-12
                or abs(saved["holm_p"] - adjusted[column]) > 1e-12):
            raise ValueError("Independent exact-test result differs")
        np.testing.assert_allclose(saved["bootstrap_95_ci"],
                                   intervals[column], atol=1e-12, rtol=0)
    success = bool(np.all(differences.mean(0) > 0) and
                   np.all(adjusted < protocol["round_alpha"]))
    success = success and all(
        intervals[keys.index((control, metric)), 0] > 0
        for control in ("previous_best", "matched_collision_gp")
        for metric in ("normalized_area", "recall"))
    if (summary["success"] != success or
            summary["selector_disclosures_audited"] != disclosures):
        raise ValueError("Registered co-primary success gate differs")
    write_json(AUDIT / "statistics.json", {
        "primary_metrics_rebuilt": True,
        "exact_tests_independently_recomputed": True,
        "bootstrap_rebuilt_using_counts": True,
        "success_gate_verified": True,
        "selector_disclosures": disclosures,
        "success": success,
        "exact_p": probabilities.tolist(),
        "holm_p": adjusted.tolist(),
    })


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    archive = verify_archives(protocol)
    freshness = verify_freshness(protocol)
    AUDIT.mkdir(parents=True, exist_ok=True)
    banks, replays = [], []
    with ProcessPoolExecutor(max_workers=4,
                             mp_context=mp.get_context("spawn")) as executor:
        for profile_index, profile in enumerate(profiles()):
            if asdict(profile) != protocol["profiles"][profile_index]:
                raise ValueError("Audited profile differs from locked profile")
            for replicate in range(protocol["replicates_per_profile"]):
                folder = (CONFIRMATION / profile.name /
                          f"pool_{replicate}")
                scenes = read_json(folder / "scenarios.json")
                if scenes != pool_scenes(profile_index, replicate):
                    raise ValueError("Scenario bank differs from frozen generator")
                rows = verify_bank(folder, scenes, profile)
                replay_path = AUDIT / profile.name / f"pool_{replicate}.json"
                if replay_path.exists():
                    checks = read_json(replay_path)
                else:
                    arguments = [(scenes[index], profile)
                                 for index in REPLAY_INDICES]
                    checks = [{"index": index, "replay": replay}
                              for index, replay in zip(
                                  REPLAY_INDICES,
                                  executor.map(measure_scene, arguments))]
                    write_json(replay_path, checks)
                for check in checks:
                    index, replay = check["index"], check["replay"]
                    if (replay["scenario_id"] != rows[index]["scenario_id"]
                            or abs(replay["risk"] - rows[index]["risk"]) > 1e-12
                            or replay["collision"] != rows[index]["collision"]):
                        raise ValueError("Independent physical replay differs")
                    replays.append({
                        "profile": profile.name,
                        "replicate": replicate,
                        "index": index,
                        "risk_matches": True,
                        "collision_matches": True,
                    })
                banks.append({"profile": profile.name,
                              "replicate": replicate,
                              "rows": len(rows)})
                print("R3 AUDITED PHYSICAL POOL", profile.name, replicate,
                      flush=True)
    verify_selector_statistics(protocol)
    result = {
        **archive,
        **freshness,
        "all_physical_banks_audited": len(banks) == 96,
        "physical_banks": banks,
        "physical_replays": len(replays),
        "all_replays_match": len(replays) == 384,
        "statistical_audit_passed": True,
        "success": read_json(AUDIT / "statistics.json")["success"],
    }
    write_json(AUDIT / "summary.json", result)
    print("ROUND-THREE AUDIT PASSED", result["success"], flush=True)


if __name__ == "__main__":
    main()
