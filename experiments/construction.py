"""Derive one globally defined ADS-centred RSS response from construction traces."""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

from controllers.highway.catalog import SUTS
from testing_methods.common.response import archive_fingerprints

from .discovery_config import write_json
from .executions import execution_contract, require_qualified
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .rss import measurement_contract, road_projection, trajectory_margin
from .scenarios import sample
from .source_registration import specification_matches
from .storage import load_archive

OUTPUT = ROOT / "experiment_outputs/benchmark/collision/rss_response"


def load_archives(kind, held_family=None):
    groups = {}
    for role in ("history", "development"):
        coordinates = sample(kind, DEFAULT.candidate_count, role)
        archives = [
            load_archive(p) for p in sorted((TEST_RECORDS_ROOT / role / kind).glob("*.npz"))
        ]
        expected = {s.sut_id: s for s in SUTS if s.role == role}
        if {a.source_id for a in archives} != set(expected) or len(archives) != len(expected):
            raise ValueError(
                "Construction requires exactly the registered 8 historical/4 development objects"
            )
        for archive in archives:
            sut = expected[archive.source_id]
            if (archive.role, archive.family, archive.lineage) != (role, sut.family, sut.lineage):
                raise ValueError("Archive family/lineage/role differs from the registered object")
            if not archive.provenance.get("qualified"):
                raise ValueError("Only independently qualified responses may build formal models")
            require_qualified(sut)
            contract = json.loads(json.dumps(execution_contract(sut, kind=kind)))
            if not specification_matches(archive.provenance["contract"], contract):
                raise ValueError(
                    "Archive controller version or policy weights differ from the registered object"
                )
            if archive.scenarios != coordinates:
                raise ValueError(
                    "Archive coordinates/count differ from the registered role-specific pool"
                )
        groups[role] = [a for a in archives if a.family != held_family]
    return groups["history"], groups["development"]


def response_path(archive):
    return OUTPUT / archive.role / archive.scenarios[0].kind / f"{archive.source_id}.npz"


def response_specification(archive):
    return {
        "measure": measurement_contract(),
        "recorded_archive_fingerprint": archive_fingerprints([archive])[archive.source_id],
        "scenario_ids": [scene.scenario_id for scene in archive.scenarios],
        "evaluated_actor_id": "ego",
    }


def attach_rss(archive):
    path = response_path(archive)
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved["specification"])) != response_specification(archive):
            raise ValueError("Derived RSS response differs from its global measure or raw archive")
        return replace(
            archive,
            rss_margin=saved["margin"].copy(),
            rss_contract=measurement_contract(),
        )


def load_training_data(kind, held_family=None):
    history, development = load_archives(kind, held_family)
    return [attach_rss(a) for a in history], [attach_rss(a) for a in development]


def sampled_lateral_acceleration(trajectory):
    maximum = {"ego": 0.0, "npc": 0.0}
    previous = {}
    previous_time = None
    for frame in trajectory:
        if frame["collision"]:
            break
        current = {a["actor_id"]: road_projection(a)[1] for a in frame["actors"]}
        if previous_time is not None:
            dt = frame["time_s"] - previous_time
            for name in previous.keys() & current.keys():
                role = "ego" if name == "ego" else "npc"
                maximum[role] = max(maximum[role], abs(current[name] - previous[name]) / dt)
        previous, previous_time = current, frame["time_s"]
    return maximum


def derive_archive(path):
    archive = load_archive(path)
    destination = response_path(archive)
    bank = TEST_RECORDS_ROOT / archive.provenance["bank"]
    margins, initial, critical_actors, critical_times = [], [], [], []
    max_acceleration = {"ego": 0.0, "npc": 0.0}
    for index, scene in enumerate(archive.scenarios):
        payload = json.loads(
            gzip.decompress((bank / f"executions/{index:06d}.json.gz").read_bytes())
        )
        if payload["scenario"]["scenario_id"] != scene.scenario_id:
            raise ValueError("Trajectory order differs from the recorded response archive")
        if payload["feedback"]["collision"] != bool(archive.collision[index]):
            raise ValueError("Derived RSS must preserve the measured collision label")
        summary = trajectory_margin(payload["trajectory"])
        margins.append(summary["minimum_margin"])
        initial.append(summary["initial_margin"])
        critical_actors.append(summary["critical_actor"])
        critical_times.append(summary["critical_time_s"])
        acceleration = sampled_lateral_acceleration(payload["trajectory"])
        for role in max_acceleration:
            max_acceleration[role] = max(max_acceleration[role], acceleration[role])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            margin=np.asarray(margins),
            initial_margin=np.asarray(initial),
            critical_time_s=np.asarray(critical_times),
            critical_actor=np.asarray(critical_actors),
            specification=json.dumps(response_specification(archive)),
        )
    temporary.replace(destination)
    margins = np.asarray(margins)
    report = {
        "source_id": archive.source_id,
        "role": archive.role,
        "scenario": archive.scenarios[0].kind,
        "executions": len(margins),
        "collisions": int(archive.collision.sum()),
        "negative": int((margins < 0).sum()),
        "initial_negative": int((np.asarray(initial) < 0).sum()),
        "margin_quantiles": np.quantile(margins, (0, 0.25, 0.5, 0.75, 1)).tolist(),
        "preimpact_sampled_lateral_acceleration_max_mps2": max_acceleration,
        "new_simulator_calls": 0,
    }
    write_json(destination.with_suffix(".json"), report)
    return report


def build_responses():
    paths = [
        p
        for role in ("history", "development")
        for p in sorted((TEST_RECORDS_ROOT / role).glob("*/*.npz"))
    ]
    if len(paths) != 60:
        raise ValueError("RSS construction requires all 60 history/development archives")
    reports = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(derive_archive, path) for path in paths]
        for future in as_completed(futures):
            reports.append(future.result())
            print(f"RSS construction responses {len(reports)}/60", flush=True)
    result = {
        "status": "complete",
        "measure": measurement_contract(),
        "archives": sorted(
            reports, key=lambda row: (row["role"], row["scenario"], row["source_id"])
        ),
        "executions": sum(row["executions"] for row in reports),
        "independent_target_records_used": False,
        "new_simulator_calls": 0,
        "interpretation": "Fixed reference-parameter response derivation, not statistical parameter fitting or a safety guarantee.",
    }
    write_json(OUTPUT / "construction.json", result)
    return result


if __name__ == "__main__":
    build_responses()
