"""Acquire and inspect the retained PPO checkpoint without executing it."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ASSETS = (
    {
        "id": "ppo_ece",
        "repo": "paranjaa/ece-rl-highway-driving",
        "ref": "17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e",
        "path": "models/PPO/vd_1_5_trial_1.zip",
        "size": 1_820_517,
        "blob_sha": "7b5da3c0b400a42f949054c904d27822416668fb",
    },
)


def fetch(output_directory: Path) -> list[dict]:
    output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for asset in ASSETS:
        url = (
            f"https://raw.githubusercontent.com/{asset['repo']}/"
            f"{asset['ref']}/{asset['path']}"
        )
        record = {**asset, "url": url}
        try:
            request = Request(
                url,
                headers={"User-Agent": "Highway-SUT-Reproduction"},
            )
            payload = urlopen(request, timeout=60).read(asset["size"] + 1)
            git_header = f"blob {len(payload)}\0".encode()
            git_sha = hashlib.sha1(git_header + payload).hexdigest()
            if len(payload) != asset["size"] or git_sha != asset["blob_sha"]:
                raise ValueError("checkpoint size or Git blob SHA mismatch")

            destination = (
                output_directory / asset["id"] / Path(asset["path"]).name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            record.update(
                weights_status="download_hash_verified",
                sha256=hashlib.sha256(payload).hexdigest(),
                local_path=str(destination.resolve()),
            )
        except Exception as error:
            record.update(
                weights_status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        records.append(record)
    return records


def inspect(output_directory: Path) -> list[dict]:
    records = []
    for asset in ASSETS:
        checkpoint = output_directory / asset["id"] / Path(asset["path"]).name
        record = {"id": asset["id"], "local_path": str(checkpoint)}
        try:
            payload = checkpoint.read_bytes()
            record["sha256"] = hashlib.sha256(payload).hexdigest()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                record["archive_members"] = archive.namelist()
                version_file = "_stable_baselines3_version"
                record["sb3_version"] = (
                    archive.read(version_file).decode()
                    if version_file in archive.namelist()
                    else None
                )
            record["inspection_status"] = "static_metadata_verified"
        except Exception as error:
            record.update(
                inspection_status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        records.append(record)
    return records
