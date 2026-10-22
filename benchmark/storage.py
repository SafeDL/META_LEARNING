"""New collision archives have explicit provenance and never read legacy banks."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from benchmark.schema import Scenario
from methods.common.response import ResponseArchive


FORMAT = "historical-response-collision"


def save_archive(path, archive, *, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"format": FORMAT, "source_id": archive.source_id,
                "family": archive.family, "lineage": archive.lineage, "role": archive.role,
                "scenarios": [asdict(s) for s in archive.scenarios], "provenance": provenance}
    with path.open("xb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, allow_nan=False),
                            collision=archive.collision, safety=archive.safety)


def load_archive(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.pop("format") != FORMAT:
            raise ValueError("Legacy risk banks cannot be used as collision archives")
        metadata["scenarios"] = [Scenario(**s) for s in metadata["scenarios"]]
        return ResponseArchive(collision=data["collision"].copy(),
                               safety=data["safety"].copy(), **metadata)
