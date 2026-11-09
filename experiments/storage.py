"""JSON records and provenance-preserving response archives."""

import gzip
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.domain import Domain
from experiments.protocol import ROOT
from experiments.schema import Scenario
from testing_methods.common.response import ResponseArchive

FORMAT = "historical-response-collision"


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def read_trajectory(execution):
    """Resolve a paid archive's shared execution, or its directly executed trajectory."""
    if "execution_ref" in execution:
        physical = read_json(ROOT / execution["execution_ref"])
        if physical["scenario"] != execution["scenario"]:
            raise ValueError("Paid execution reference belongs to another scenario")
        return physical["trajectory"]
    return execution["trajectory"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    if path.suffix == ".gz":
        content = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
            stream.write(content)
    else:
        content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
        temporary.write_text(content, encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            # Windows readers can briefly prevent replacement during progress polling.
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 19:
                raise
            time.sleep(0.05)


def compact_feedback(records):
    """Keep scalar feedback; the completed execution owns the actual trajectory."""
    return [
        {**record, "feedback": {k: v for k, v in record["feedback"].items() if k != "trajectory"}}
        if "feedback" in record
        else dict(record)
        for record in records
    ]


def compact_result(result, bank_root=None):
    compact = {**result, "queries": compact_feedback(result["queries"])}
    if bank_root is not None:
        for query in compact["queries"]:
            query["execution_ref"] = (
                (Path(bank_root) / "executions" / f"{query['index']:06d}.json.gz")
                .relative_to(ROOT)
                .as_posix()
            )
    return compact


def save_archive(path, archive, *, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": FORMAT,
        "source_id": archive.source_id,
        "family": archive.family,
        "lineage": archive.lineage,
        "role": archive.role,
        "scenarios": [asdict(s) for s in archive.scenarios],
        "provenance": provenance,
    }
    optional = {}
    if archive.domain is not None:
        metadata["domain"] = archive.domain.to_dict()
    if archive.rss_margin is not None:
        metadata["rss_contract"] = archive.rss_contract
        optional["rss_margin"] = archive.rss_margin
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            metadata=json.dumps(metadata, allow_nan=False),
            collision=archive.collision,
            safety=archive.safety,
            **optional,
        )


def load_archive(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.pop("format") != FORMAT:
            raise ValueError("Legacy risk banks cannot be used as collision archives")
        metadata["scenarios"] = [Scenario(**s) for s in metadata["scenarios"]]
        if "domain" in metadata:
            metadata["domain"] = Domain.from_dict(metadata["domain"])
        return ResponseArchive(
            collision=data["collision"].copy(),
            safety=data["safety"].copy(),
            rss_margin=data["rss_margin"].copy() if "rss_margin" in data else None,
            **metadata,
        )
