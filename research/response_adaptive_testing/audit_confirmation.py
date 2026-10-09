"""Verify frozen inputs, declared coordinates, full banks and physical replays."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from zipfile import ZipFile

import numpy as np

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene
from research.history_response_testing.config import CONFIRMATION as PREVIOUS_CONFIRMATION

from .config import OUTPUT
from .confirmation import CONFIRMATION, REPO, pool_scenes, profiles, verify_lock

REPLAY_INDICES = (0, 1023, 1024, 2047)
REPLAY_WORKERS = 4
AUDIT = OUTPUT / "confirmation_audit"
PREVIOUS = REPO / "research/history_response_testing"


def verify_archives(protocol):
    seal = read_json(CONFIRMATION / "lock.json")
    source_archive = CONFIRMATION / "locked_inputs.zip"
    assert hashlib.sha256(source_archive.read_bytes()).hexdigest(
    ) == seal["source_archive_sha256"]
    with ZipFile(source_archive) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert archive.read(name) == (REPO / name).read_bytes(), name
    frozen = read_json(PREVIOUS / "archives/baseline_manifest.json")["sha256"]
    with ZipFile(PREVIOUS / "archives/baseline.zip") as archive:
        for name in frozen:
            assert archive.read(name) == (REPO / name).read_bytes(), name
    return {
        "locked_inputs": len(protocol["sha256"]),
        "frozen_files": len(frozen),
        "archives_equal_current_inputs": True
    }


def prior_coordinates():
    paths = [
        BASELINE / "history/responses.npz", BASELINE / "target/responses.npz"
    ]
    paths += list((OUTPUT / "development_pools").glob("*/responses.npz"))
    points = set()
    for path in paths:
        with np.load(path) as bank:
            points.update(map(tuple, bank["x"]))
    for path in PREVIOUS_CONFIRMATION.glob("pool_*/scenarios.json"):
        points.update(
            tuple(scene["numeric_input"]) for scene in read_json(path))
    with ZipFile(PREVIOUS / "archives/development.zip") as archive:
        for name in archive.namelist():
            if "/results/confirmation/pool_" in name and name.endswith(
                    "/scenarios.json"):
                scenes = json.loads(archive.read(name))
                points.update(
                    tuple(scene["numeric_input"]) for scene in scenes)
    return points


def verify_bank(folder, scenes, profile):
    rows = [
        json.loads(line) for line in (folder / "measurements.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(scenes) == 2048
    assert [row["index"] for row in rows] == list(range(len(scenes)))
    assert [row["scenario_id"]
            for row in rows] == [scene["scenario_id"] for scene in scenes]
    with np.load(folder / "responses.npz") as bank:
        assert set(bank.files) == {"x", "risk", "collision"}
        assert bank["collision"].dtype == np.bool_
        assert np.isfinite(bank["risk"]).all()
        assert ((0 <= bank["risk"]) & (bank["risk"] <= 1)).all()
        assert np.array_equal(bank["x"], [s["numeric_input"] for s in scenes])
        assert np.array_equal(bank["risk"], [row["risk"] for row in rows])
        assert np.array_equal(bank["collision"],
                              [row["collision"] for row in rows])
    assert all(
        np.isfinite(row["elapsed_s"]) and row["elapsed_s"] > 0 for row in rows)
    cost = read_json(folder / "cost.json")
    assert cost == {
        "full_pool_measured": True,
        "unique_physical_measurements": 2048,
        "profile": asdict(profile)
    }
    return rows


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    inputs = verify_archives(protocol)
    old_points = prior_coordinates()
    new_points = set()
    completed, physical = [], []
    with ProcessPoolExecutor(max_workers=REPLAY_WORKERS,
                             mp_context=mp.get_context("spawn")) as executor:
        for profile_index, profile in enumerate(profiles()):
            assert asdict(profile) == protocol["profiles"][profile_index]
            for replicate in range(protocol["replicates_per_profile"]):
                folder = CONFIRMATION / profile.name / f"pool_{replicate}"
                scenes = read_json(folder / "scenarios.json")
                assert scenes == pool_scenes(profile_index, replicate)
                points = {tuple(scene["numeric_input"]) for scene in scenes}
                assert len(points) == 2048
                assert not points & old_points
                assert not points & new_points
                new_points.update(points)
                if not (folder / "cost.json").exists():
                    continue
                rows = verify_bank(folder, scenes, profile)
                saved = AUDIT / profile.name / f"pool_{replicate}.json"
                if saved.exists():
                    checks = read_json(saved)
                    assert [row["index"]
                            for row in checks] == list(REPLAY_INDICES)
                else:
                    checks = []
                    arguments = [(scenes[index], profile)
                                 for index in REPLAY_INDICES]
                    for index, replay in zip(
                            REPLAY_INDICES,
                            executor.map(measure_scene, arguments)):
                        checks.append({"index": index, "replay": replay})
                    write_json(saved, checks)
                for check in checks:
                    index, replay = check["index"], check["replay"]
                    assert replay["scenario_id"] == rows[index]["scenario_id"]
                    assert abs(replay["risk"] - rows[index]["risk"]) < 1e-12
                    assert replay["collision"] == rows[index]["collision"]
                physical.extend({
                    "profile": profile.name,
                    "replicate": replicate,
                    "index": check["index"],
                    "risk_matches": True,
                    "collision_matches": True
                } for check in checks)
                completed.append({
                    "profile":
                    profile.name,
                    "replicate":
                    replicate,
                    "rows":
                    len(rows),
                    "simulator_seconds":
                    sum(r["elapsed_s"] for r in rows)
                })
                print("AUDITED PHYSICAL BANK",
                      profile.name,
                      replicate,
                      flush=True)
    value = {
        **inputs, "known_coordinates": len(old_points),
        "declared_new_coordinates": len(new_points),
        "all_new_coordinates_disjoint": True,
        "complete_banks_audited": completed,
        "physical_replays": physical,
        "all_physical_banks_audited": len(completed) == 48,
        "additional_physical_verification_calls": len(physical),
        "stage":
        "measurement audit; no method tuning from prospective outcomes"
    }
    write_json(AUDIT / "summary.json", value)
    print("AUDIT COMPLETE",
          len(completed),
          "banks",
          len(physical),
          "replays",
          flush=True)


if __name__ == "__main__":
    main()
