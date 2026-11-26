"""Build historical responses from complete paid execution banks."""

import gzip
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from testing_methods.common.response import ResponseArchive, archive_fingerprints

from .domain import Domain
from .rss import measurement_contract, trajectory_margin
from .schema import QueryFeedback, Scenario
from .storage import load_archive, read_json, save_archive


def response_from_bank(bank, *, batch):
    bank = Path(bank)
    manifest = read_json(bank / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Historical responses require a complete executed bank")
    sut = manifest["contract"]["object"]
    if sut["role"] != "history" or sut["sut_id"] != manifest["sut_id"]:
        raise ValueError("Historical preparation accepts original historical ADS only")
    scenes = [Scenario(**s) for s in manifest["scenarios"]]
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
            "physical_calls": len(scenes),
        },
        rss_margin=np.array(margins),
        rss_contract=measurement_contract(),
        domain=(
            Domain.from_dict(manifest["parameter_definition"])
            if "parameter_definition" in manifest
            else None
        ),
    )


def save_verified(path, archive):
    if path.exists():
        saved = load_archive(path)
        if archive_fingerprints([saved]) != archive_fingerprints([archive]):
            raise ValueError("Saved historical responses differ from their execution evidence")
        return saved
    save_archive(path, archive, provenance=archive.provenance)
    return archive
