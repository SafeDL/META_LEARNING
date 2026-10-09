"""Check preserved evidence and replay the retained method after code cleanup."""
import hashlib
import json
import zipfile
from io import BytesIO

import numpy as np
import torch

from methods.history_guided_testing.config import ROOT as BASELINE, TARGET
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene

from .config import ARCHIVES, CONFIRMATION, OUTPUT, ROOT, SEEDS, SOURCE_WEIGHTS
from .history_model import predict
from .kernel import covariance
from .scenarios import confirmation_scenes
from .session import RiskTestingSession

ORIGINAL_PREFIX = "research/response_hypothesis_testing/"


def main():
    torch.set_num_threads(1)
    manifest = read_json(ARCHIVES / "baseline_manifest.json")
    with zipfile.ZipFile(ARCHIVES / "baseline.zip") as archive:
        for name, digest in manifest["sha256"].items():
            current = (ROOT.parents[1] / name).read_bytes()
            assert hashlib.sha256(current).hexdigest() == digest, name
            assert archive.read(name) == current, name
    protocol = read_json(CONFIRMATION / "protocol.json")
    with zipfile.ZipFile(ARCHIVES / "development.zip") as archive:
        first_protocol = json.loads(
            archive.read(ORIGINAL_PREFIX +
                         "results/confirmation/protocol.json"))
        for locked, filename in (
            (first_protocol, "development_source.zip"),
            (protocol, "research_source.zip"),
        ):
            with zipfile.ZipFile(
                    BytesIO(archive.read(ORIGINAL_PREFIX +
                                         filename))) as sources:
                for name, digest in locked["source_sha256"].items():
                    source = sources.read(ORIGINAL_PREFIX + name)
                    assert hashlib.sha256(source).hexdigest() == digest, name
        previous = [
            np.load(BASELINE / "target/responses.npz")["x"],
            np.load(BASELINE / "history/responses.npz")["x"],
        ]
        for pool in range(10):
            scenes = json.loads(
                archive.read(
                    ORIGINAL_PREFIX +
                    f"results/confirmation/pool_{pool}/scenarios.json"))
            previous.append(
                np.asarray([scene["numeric_input"] for scene in scenes]))
        previous_points = set(map(tuple, np.vstack(previous)))
        old_summary = json.loads(
            archive.read(ORIGINAL_PREFIX +
                         "results/fresh_confirmation/summary.json"))
        current_summary = read_json(CONFIRMATION / "summary.json")
        for method, old_metrics in old_summary["aggregate"].items():
            for metric, value in old_metrics.items():
                name = "mean_cumulative_collisions" if metric == "discovery_auc" else metric
                assert current_summary["aggregate"][method][name] == value
        for method, old_metrics in old_summary["comparisons"].items():
            for metric, value in old_metrics.items():
                name = "mean_cumulative_collisions" if metric == "discovery_auc" else metric
                assert current_summary["comparisons"][method][name] == value
        assert current_summary["components"] == old_summary["components"]
        for seed in SEEDS:
            original = json.loads(
                archive.read(
                    ORIGINAL_PREFIX +
                    f"results/final_development/candidate_{seed}.json"))
            original["mean_cumulative_collisions"] = original.pop(
                "discovery_auc")
            assert read_json(OUTPUT / "original_pool" /
                             f"candidate_{seed}.json") == original
    for name, digest in protocol["model_sha256"].items():
        assert hashlib.sha256(
            (OUTPUT / "models" / name).read_bytes()).hexdigest() == digest
    calibration = {
        int(key): value
        for key, value in protocol["calibration"].items()
    }
    replayed, physical = [], []
    for pool in range(protocol["pool_count"]):
        folder = CONFIRMATION / f"pool_{pool}"
        scenes = read_json(folder / "scenarios.json")
        assert confirmation_scenes(pool) == scenes
        x = np.asarray([scene["numeric_input"] for scene in scenes])
        points = set(map(tuple, x))
        assert not points & previous_points
        previous_points.update(points)
        rows = {
            row["index"]: row
            for row in map(json.loads, (folder /
                                        "measurements.jsonl").read_text(
                                            encoding="utf-8").splitlines())
        }
        risk, collision = predict(x, SEEDS[pool % len(SEEDS)])
        kernel = covariance(x, risk, device="cpu")
        for method, mode in (
            ("candidate", "dual"),
            ("risk_only", "risk"),
            ("calibrated_only", "calibrated"),
        ):
            expected = read_json(folder / f"{method}.json")["selected_indices"]
            session = RiskTestingSession(
                x,
                kernel,
                risk,
                collision,
                np.asarray(SOURCE_WEIGHTS),
                calibration,
                mode=mode,
                device="cpu",
            )
            selected = []
            while (index := session.next_index()) is not None:
                assert index == expected[len(selected)], (pool, method,
                                                          len(selected))
                session.observe(float(rows[index]["risk"]))
                selected.append(index)
            assert selected == expected
            replayed.append({
                "pool": pool,
                "method": method,
                "queries": len(selected)
            })
        if pool in (0, 10, 19):
            for family in (0, 1):
                index = next(i for i in expected if x[i, 4] == family)
                measured = measure_scene((scenes[index], TARGET))
                assert abs(measured["risk"] - rows[index]["risk"]) < 1e-12
                assert measured["collision"] == rows[index]["collision"]
                physical.append({
                    "pool": pool,
                    "index": index,
                    "risk_matches": True,
                    "collision_matches": True,
                })
        print("Verified saved sequences:", pool, flush=True)
    write_json(
        OUTPUT / "verification.json", {
            "frozen_files_unchanged": len(manifest["sha256"]),
            "archived_locked_sources_match": True,
            "model_hashes_match": True,
            "metrics_and_statistics_unchanged": True,
            "confirmation_coordinates_disjoint": True,
            "saved_sequences_replayed": replayed,
            "identical_selected_queries": sum(row["queries"]
                                              for row in replayed),
            "physical_replays": physical,
        })
    print("Evidence and numerical behavior verified", flush=True)


if __name__ == "__main__":
    main()
