"""Verify recorded histories, collect a separate supplement, and assemble sources."""

import gzip
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from controllers.highway.catalog import CURRENT_SUTS
from testing_methods.common.response import ResponseArchive, archive_fingerprints

from .domain import Domain
from .executions import collect_bank, require_qualified
from .protocol import DATA_ROOT
from .rss import measurement_contract, trajectory_margin
from .scenarios import DOMAINS, TEST_SCENARIOS, validate
from .schema import QueryFeedback, Scenario
from .storage import load_archive, read_json, save_archive, write_json

DATASET_ROOT = DATA_ROOT / "joint_response"
ADDITIONS = {"car_following": 512, "front_brake": 256, "cut_out": 256, "lane_change_rear": 256}
SUPPLEMENT_SEED = 31013
ENVIRONMENT_SEED = 81001


def supplement_coordinates(kind, count, *, excluded=(), seed=SUPPLEMENT_SEED):
    """Accept physical feasibility only, excluding coordinates already present in any source."""
    if kind not in TEST_SCENARIOS or count < 1:
        raise ValueError("Supplement needs a current scenario and positive record count")
    domain = DOMAINS[kind]
    sampler = qmc.Sobol(len(domain.bounds), scramble=True, seed=seed)
    seen = set(excluded)
    scenes = []
    ordinal = 0
    while len(scenes) < count:
        for point in sampler.random(1024):
            scene = Scenario(
                f"{kind}:history_supplement:{seed}:{ordinal:06d}",
                kind,
                {
                    name: float(low + value * (high - low))
                    for (name, (low, high)), value in zip(domain.bounds.items(), point)
                },
            )
            ordinal += 1
            try:
                validate(scene)
            except ValueError:
                continue
            if scene.coordinate() in seen:
                continue
            scenes.append(scene)
            seen.add(scene.coordinate())
            if len(scenes) == count:
                break
    return scenes


def response_from_bank(bank, *, batch, count=None, recorded=None):
    bank = Path(bank)
    manifest = read_json(bank / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Historical responses require a complete executed bank")
    sut = manifest["contract"]["object"]
    if sut["role"] != "history" or sut["sut_id"] != manifest["sut_id"]:
        raise ValueError("Historical preparation accepts original historical ADS only")
    scenes = [Scenario(**s) for s in manifest["scenarios"]]
    count = len(scenes) if count is None else min(count, len(scenes))
    scenes = scenes[:count]
    if recorded is not None and (
        recorded.scenarios[:count] != scenes
        or recorded.source_id != sut["sut_id"]
        or recorded.provenance["contract"] != manifest["contract"]
        or recorded.role != "history"
    ):
        raise ValueError("Recorded archive differs from its actual execution contract")
    calls = [
        json.loads(line) for line in (bank / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    completed = {(r["call"], r["index"]) for r in calls if r["status"] == "complete"}
    labels, safety, margins, records = [], [], [], {}
    for index, scene in enumerate(scenes):
        relative = f"executions/{index:06d}.json.gz"
        payload = json.loads(gzip.decompress((bank / relative).read_bytes()))
        feedback = QueryFeedback.from_dict(payload["feedback"])
        if (
            payload["scenario"] != asdict(scene)
            or feedback.scenario_id != scene.scenario_id
            or (payload["call"], index) not in completed
            or not payload["trajectory"]
            or bool(any(f["collision"] for f in payload["trajectory"])) != feedback.collision
        ):
            raise ValueError("Historical response lacks matching paid execution evidence")
        measured = [feedback.safety.min_clearance_m, feedback.safety.min_ttc_s]
        if recorded is not None and (
            feedback.collision != bool(recorded.collision[index])
            or not np.array_equal(measured, recorded.safety[index])
        ):
            raise ValueError("Recorded response differs from measured collision or safety")
        started = time.perf_counter()
        margin = trajectory_margin(payload["trajectory"])["minimum_margin"]
        derived_s = time.perf_counter() - started
        record_id = f"{sut['sut_id']}:{batch}:{scene.kind}:{payload['call']}"
        records[record_id] = {
            "bank": str(bank.resolve()),
            "execution": relative,
            "call": payload["call"],
            "scenario_id": scene.scenario_id,
            "environment_seed": manifest["environment_seed"],
            "original_role": sut["role"],
            "sut_version": sut["version"],
            "execution_s": feedback.execution_s,
            "rss_compute_s": derived_s,
        }
        labels.append(feedback.collision)
        safety.append(measured)
        margins.append(margin)
    return ResponseArchive(
        sut["sut_id"],
        sut["family"],
        sut["lineage"],
        "history",
        scenes,
        np.array(labels, dtype=bool),
        np.array(safety),
        provenance={
            "qualified": True,
            "batch": batch,
            "bank": str(bank.resolve()),
            "contract": manifest["contract"],
            "environment_seed": manifest["environment_seed"],
            "original_role": "history",
            "record_ids": list(records),
            "executions": records,
            "physical_calls": count,
        },
        rss_margin=np.array(margins),
        rss_contract=measurement_contract(),
        domain=Domain.from_dict(manifest["parameter_definition"])
        if "parameter_definition" in manifest
        else None,
    )


def recorded_response(path, *, count=None):
    archive = load_archive(path)
    return response_from_bank(
        DATA_ROOT / archive.provenance["bank"], batch="recorded", count=count, recorded=archive
    )


def save_verified(path, archive):
    if path.exists():
        saved = load_archive(path)
        if archive_fingerprints([saved]) != archive_fingerprints([archive]):
            raise ValueError("Saved historical responses differ from their execution evidence")
        return saved
    save_archive(path, archive, provenance=archive.provenance)
    return archive


def convert_history(destination=DATASET_ROOT, *, kinds=TEST_SCENARIOS, systems=None, count=512):
    """Derive RSS from existing evidence without invoking a simulator."""
    systems = historical_systems(systems)
    converted = {}
    for kind in kinds:
        if kind == "car_following":
            continue
        for sut in systems:
            path = DATA_ROOT / "history" / kind / f"{sut.sut_id}.npz"
            archive = recorded_response(path, count=count)
            if len(archive.scenarios) != count:
                raise ValueError(
                    "Recorded history does not contain the scheduled number of records"
                )
            output = Path(destination) / "batches/recorded" / kind / f"{sut.sut_id}.npz"
            converted[(kind, sut.sut_id)] = save_verified(output, archive)
    return converted


def historical_systems(systems):
    systems = (
        tuple(s for s in CURRENT_SUTS if s.role == "history") if systems is None else tuple(systems)
    )
    if (
        not systems
        or any(s.role != "history" for s in systems)
        or len({s.sut_id for s in systems}) != len(systems)
    ):
        raise ValueError("Historical preparation needs distinct original historical ADS")
    return systems


def collect_history(
    destination=DATASET_ROOT,
    *,
    recorded,
    kinds=TEST_SCENARIOS,
    systems=None,
    additions=None,
    workers=12,
):
    systems = historical_systems(systems)
    additions = ADDITIONS if additions is None else additions
    batches = {}
    for kind in kinds:
        excluded = {
            s.coordinate()
            for (scenario, _), archive in recorded.items()
            if scenario == kind
            for s in archive.scenarios
        }
        scenes = supplement_coordinates(kind, additions[kind], excluded=excluded)
        for sut in systems:
            require_qualified(sut)
            bank = collect_bank(
                sut.sut_id,
                scenes,
                Path(destination) / "physical/history" / kind / sut.sut_id / "supplement",
                environment_seed=ENVIRONMENT_SEED,
                workers=workers,
                current_registration=True,
            )
            archive = response_from_bank(bank.root, batch="supplement")
            output = Path(destination) / "batches/supplement" / kind / f"{sut.sut_id}.npz"
            batches[(kind, sut.sut_id)] = save_verified(output, archive)
    return batches


def merge_batches(batches):
    if not batches:
        raise ValueError("Historical source needs at least one executed batch")
    first = batches[0]
    identity = first.source_id, first.family, first.lineage, first.role, first.scenarios[0].kind
    records, duplicates = {}, []
    for archive in batches:
        if (
            archive.source_id,
            archive.family,
            archive.lineage,
            archive.role,
            archive.scenarios[0].kind,
        ) != identity:
            raise ValueError(
                "Batches must describe the same historical ADS and functional scenario"
            )
        for index, scene in enumerate(archive.scenarios):
            record_id = archive.provenance["record_ids"][index]
            if scene.coordinate() in records:
                duplicates.append(record_id)
                continue
            records[scene.coordinate()] = (archive, index, record_id)
    chosen = list(records.values())
    return ResponseArchive(
        first.source_id,
        first.family,
        first.lineage,
        first.role,
        [a.scenarios[i] for a, i, _ in chosen],
        np.array([a.collision[i] for a, i, _ in chosen]),
        np.array([a.safety[i] for a, i, _ in chosen]),
        provenance={
            "qualified": True,
            "original_role": "history",
            "contract": {
                "batches": [
                    {
                        "batch": a.provenance["batch"],
                        "contract": a.provenance["contract"],
                        "bank": a.provenance["bank"],
                        "environment_seed": a.provenance["environment_seed"],
                    }
                    for a in batches
                ]
            },
            "record_ids": [record_id for _, _, record_id in chosen],
            "executions": {
                record_id: a.provenance["executions"][record_id] for a, _, record_id in chosen
            },
            "raw_records": sum(len(a.scenarios) for a in batches),
            "repeated_execution_ids": duplicates,
        },
        rss_margin=np.array([a.continuous_margin[i] for a, i, _ in chosen]),
        rss_contract=measurement_contract(),
    )


def merge_scenario(
    kind,
    destination=DATASET_ROOT,
    *,
    systems=None,
    recorded_count=512,
    supplement_count=None,
):
    """Publish a complete same-scenario source set from already verified batches."""
    destination = Path(destination)
    systems = historical_systems(systems)
    supplement_count = ADDITIONS[kind] if supplement_count is None else supplement_count
    entries = []
    for sut in systems:
        batches = []
        sizes = [("supplement", supplement_count)]
        if kind != "car_following":
            sizes.insert(0, ("recorded", recorded_count))
        for batch, count in sizes:
            archive = load_archive(destination / "batches" / batch / kind / f"{sut.sut_id}.npz")
            if archive.source_id != sut.sut_id or len(archive.scenarios) != count:
                raise ValueError("Historical batch does not match the complete source schedule")
            batches.append(archive)
        path = destination / "history" / kind / f"{sut.sut_id}.npz"
        archive = save_verified(path, merge_batches(batches))
        entries.append(
            {
                "scenario": kind,
                "role": "history",
                "source_id": sut.sut_id,
                "path": str(path.relative_to(destination)),
                "records": len(archive.scenarios),
                "fingerprint": archive_fingerprints([archive])[archive.source_id],
            }
        )
        print("joint historical source", kind, sut.sut_id, len(archive.scenarios), flush=True)
    write_json(
        destination / "history" / kind / "manifest.json",
        {
            "format": "joint-historical-batches",
            "scenarios": [kind],
            "archives": [{**entry, "path": Path(entry["path"]).name} for entry in entries],
            "rss_contract": measurement_contract(),
            "total_records": sum(e["records"] for e in entries),
            "development_ads_used": False,
        },
    )
    return entries


def prepare_dataset(
    destination=DATASET_ROOT,
    *,
    workers=12,
    kinds=TEST_SCENARIOS,
    systems=None,
    additions=None,
    recorded_count=512,
):
    destination = Path(destination)
    kinds = tuple(kinds)
    if not kinds or any(kind not in TEST_SCENARIOS for kind in kinds):
        raise ValueError("Prepare the registered current functional scenarios")
    systems = historical_systems(systems)
    additions = ADDITIONS if additions is None else additions
    recorded = convert_history(destination, kinds=kinds, systems=systems, count=recorded_count)
    supplement = collect_history(
        destination,
        recorded=recorded,
        kinds=kinds,
        systems=systems,
        additions=additions,
        workers=workers,
    )
    entries = []
    for kind in kinds:
        entries.extend(
            merge_scenario(
                kind,
                destination,
                systems=systems,
                recorded_count=recorded_count,
                supplement_count=additions[kind],
            )
        )
    result = {
        "format": "joint-historical-batches",
        "archives": entries,
        "rss_contract": measurement_contract(),
        "scenarios": list(kinds),
        "recorded_records": sum(len(a.scenarios) for a in recorded.values()),
        "supplement_records": sum(len(a.scenarios) for a in supplement.values()),
        "total_records": sum(e["records"] for e in entries),
        "supplement_seed": SUPPLEMENT_SEED,
        "environment_seed": ENVIRONMENT_SEED,
        "additions_per_source": {k: additions[k] for k in kinds},
        "policy_weights_retrained": False,
        "development_ads_used": False,
    }
    write_json(destination / "manifest.json", result)
    return result


def main():
    prepare_dataset()


if __name__ == "__main__":
    main()
