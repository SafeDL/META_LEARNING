"""Check frozen inputs, disjoint pools, physical replays and profile statistics."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import multiprocessing as mp
from zipfile import ZipFile

import numpy as np

from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene
from research.response_adaptive_testing.audit_confirmation import prior_coordinates, verify_bank, verify_archives as verify_previous_archives
from research.response_adaptive_testing.audit_statistics import exact_p_meet_in_middle
from research.response_adaptive_testing.confirmation import CONFIRMATION as FIRST_CONFIRMATION

from .confirmation import CONFIRMATION, REPO, pool_scenes, profiles, verify_lock
from .train import OUTPUT

AUDIT = OUTPUT / "confirmation_audit"
REPLAY_INDICES = (0, 1023, 1024, 2047)


def verify_archives(protocol):
    seal = read_json(CONFIRMATION / "lock.json")
    source = CONFIRMATION / "locked_inputs.zip"
    assert hashlib.sha256(
        source.read_bytes()).hexdigest() == seal["source_archive_sha256"]
    with ZipFile(source) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert archive.read(name) == (REPO / name).read_bytes(), name
    previous = verify_previous_archives(
        read_json(FIRST_CONFIRMATION / "protocol.json"))
    return {
        "locked_files": len(protocol["sha256"]),
        "frozen_original_files": previous["frozen_files"],
        "first_confirmation_files": previous["locked_inputs"],
        "archives_equal_current_inputs": True
    }


def verify_coordinates(protocol):
    known = prior_coordinates()
    for path in FIRST_CONFIRMATION.glob("*/pool_*/responses.npz"):
        with np.load(path) as bank:
            known.update(map(tuple, bank["x"]))
    new = set()
    old_profiles = read_json(FIRST_CONFIRMATION / "protocol.json")["profiles"]
    descriptors = lambda p: (p["controller"], p["max_brake"], p["desired_gap"],
                             p["perception_delay_s"])
    old_states = {descriptors(p) for p in old_profiles}
    new_states = {descriptors(p) for p in protocol["profiles"]}
    assert len(new_states) == len(
        protocol["profiles"]) and not old_states & new_states
    for index, profile in enumerate(profiles()):
        assert asdict(profile) == protocol["profiles"][index]
        for replicate in range(protocol["replicates_per_profile"]):
            scenes = read_json(CONFIRMATION / profile.name /
                               f"pool_{replicate}" / "scenarios.json")
            assert scenes == pool_scenes(index, replicate)
            coordinates = {tuple(scene["numeric_input"]) for scene in scenes}
            assert len(coordinates) == 2048
            assert not coordinates & known and not coordinates & new
            new.update(coordinates)
    return {
        "known_coordinates": len(known),
        "new_coordinates": len(new),
        "all_coordinates_disjoint": True,
        "all_profiles_new": True
    }


def audit_statistics(protocol):
    summary = read_json(CONFIRMATION / "summary.json")
    methods, profiles = protocol["methods"], protocol["profiles"]
    values = np.zeros((len(profiles), len(methods), 2))
    disclosures = 0
    for p, profile in enumerate(profiles):
        rows = {method: [] for method in methods}
        for replicate in range(protocol["replicates_per_profile"]):
            folder = CONFIRMATION / profile["name"] / f"pool_{replicate}"
            with np.load(folder / "responses.npz") as bank:
                total = int(np.count_nonzero(bank["collision"]))
                for method in methods:
                    for seed in protocol["seeds"]:
                        result = read_json(folder / "selection" /
                                           f"{method}_{seed}.json")
                        selected = result["selected_indices"]
                        assert len(selected) == len(
                            set(selected)) == protocol["budget"]
                        expected = [{
                            "index": i,
                            "collision": bool(bank["collision"][i])
                        } if method == "ras_frt_uq" else {
                            "index": i,
                            "continuous_risk": float(bank["risk"][i])
                        } for i in selected]
                        assert result["observations"] == expected
                        curve = np.cumsum(bank["collision"][selected])
                        assert curve.tolist() == result["curve"]
                        rows[method].append([
                            curve.mean() / total if total else 0,
                            curve[-1] / total if total else 1
                        ])
                        disclosures += len(selected)
        for m, method in enumerate(methods):
            values[p, m] = np.mean(rows[method], axis=0)
            saved = summary["per_profile"][method][p]
            np.testing.assert_allclose(
                values[p, m], [saved["normalized_area"], saved["recall"]],
                atol=1e-12,
                rtol=0)
    keys = [(control, metric) for control in protocol["primary_controls"]
            for metric in ("normalized_area", "recall")]
    differences = np.column_stack([
        values[:, methods.index("candidate"), column] -
        values[:, methods.index(control), column]
        for control in protocol["primary_controls"] for column in (0, 1)
    ])
    probabilities = np.array([
        exact_p_meet_in_middle(differences[:, i:i + 1])[0]
        for i in range(differences.shape[1])
    ])
    adjusted = np.zeros(len(probabilities))
    running = 0.
    for rank, index in enumerate(
            sorted(range(len(probabilities)), key=lambda i: probabilities[i])):
        running = max(running,
                      probabilities[index] * (len(probabilities) - rank))
        adjusted[index] = min(1., running)
    rng = np.random.default_rng(20261019)
    weights = np.zeros((20000, len(profiles)), dtype=np.int16)
    for controller in ("IDM", "FVDM"):
        indices = np.array([
            i for i, p in enumerate(profiles) if p["controller"] == controller
        ])
        draws = rng.integers(0, len(indices), (20000, len(indices)))
        for local, index in enumerate(indices):
            weights[:, index] = (draws == local).sum(1)
    intervals = np.quantile(weights @ differences / len(profiles),
                            [0.025, 0.975],
                            axis=0).T
    for column, (control, metric) in enumerate(keys):
        saved = summary["comparisons"][control][metric]
        assert abs(saved["exact_two_sided_p"] - probabilities[column]) < 1e-12
        assert abs(saved["holm_p"] - adjusted[column]) < 1e-12
        np.testing.assert_allclose(saved["bootstrap_95_ci"],
                                   intervals[column],
                                   atol=1e-12,
                                   rtol=0)
    success = bool(
        np.all(differences.mean(0) > 0)
        and np.all(adjusted < protocol["round_alpha"]))
    success = success and all(
        intervals[keys.index((control, metric)), 0] > 0
        for control in ("previous_best", "matched_collision_gp")
        for metric in ("normalized_area", "recall"))
    assert summary["success"] == success and summary[
        "selector_disclosures_audited"] == disclosures
    write_json(
        AUDIT / "statistics.json", {
            "primary_metrics_rebuilt": True,
            "exact_tests_independently_recomputed": True,
            "bootstrap_rebuilt_using_counts": True,
            "success_gate_verified": True,
            "disclosures": disclosures,
            "success": success,
            "exact_p": probabilities.tolist(),
            "holm_p": adjusted.tolist()
        })


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    inputs = verify_archives(protocol)
    coordinates = verify_coordinates(protocol)
    complete, replays = [], []
    with ProcessPoolExecutor(max_workers=4,
                             mp_context=mp.get_context("spawn")) as executor:
        for profile in profiles():
            for replicate in range(protocol["replicates_per_profile"]):
                folder = CONFIRMATION / profile.name / f"pool_{replicate}"
                scenes = read_json(folder / "scenarios.json")
                rows = verify_bank(folder, scenes, profile)
                destination = AUDIT / profile.name / f"pool_{replicate}.json"
                if destination.exists():
                    checks = read_json(destination)
                else:
                    checks = [{
                        "index": index,
                        "replay": replay
                    } for index, replay in zip(
                        REPLAY_INDICES,
                        executor.map(measure_scene, [(scenes[i], profile)
                                                     for i in REPLAY_INDICES]))
                              ]
                    write_json(destination, checks)
                assert [r["index"] for r in checks] == list(REPLAY_INDICES)
                for check in checks:
                    replay, original = check["replay"], rows[check["index"]]
                    assert replay["scenario_id"] == original["scenario_id"]
                    assert abs(replay["risk"] - original["risk"]) < 1e-12
                    assert replay["collision"] == original["collision"]
                replays.extend(checks)
                complete.append({
                    "profile": profile.name,
                    "replicate": replicate,
                    "rows": len(rows)
                })
                print("BEHAVIOR AUDITED BANK",
                      profile.name,
                      replicate,
                      flush=True)
    audit_statistics(protocol)
    write_json(
        AUDIT / "summary.json", {
            **inputs,
            **coordinates, "all_physical_banks_audited":
            len(complete) == len(profiles()) * 2,
            "physical_banks":
            complete,
            "physical_replays":
            len(replays),
            "additional_physical_calls":
            len(replays),
            "statistical_audit_passed":
            True
        })


if __name__ == "__main__":
    main()
