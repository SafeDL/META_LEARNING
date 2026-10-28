"""Historical response archives, fingerprints and frozen source predictions."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import cached_property

import numpy as np

from experiments.rss import measurement_contract, rss_values
from experiments.scenarios import difficult, features


@dataclass
class ResponseArchive:
    source_id: str
    family: str
    lineage: str
    role: str
    scenarios: list
    collision: np.ndarray
    safety: np.ndarray
    provenance: dict = field(default_factory=dict)
    rss_margin: np.ndarray | None = None
    rss_contract: dict = field(default_factory=dict)

    def __post_init__(self):
        self.collision = np.asarray(self.collision)
        self.safety = np.asarray(self.safety, dtype=np.float64)
        if self.collision.dtype != bool or self.collision.shape != (len(self.scenarios),):
            raise ValueError("Archives require actual Boolean collision labels")
        if self.safety.shape != (len(self.scenarios), 2) or np.isnan(self.safety).any():
            raise ValueError("Archive safety columns are clearance and TTC")
        if self.role not in ("history", "development"):
            raise ValueError("Final targets cannot enter representation/model construction")
        if any(difficult(s) for s in self.scenarios):
            raise ValueError("Held difficult combinations cannot enter method archives")
        if len({s.coordinate() for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("Repeated coordinates must be stored as environment repetitions")
        features(self.scenarios)
        if self.rss_margin is not None:
            self.rss_margin = rss_values(self.rss_margin)
            if self.rss_margin.shape != (len(self.scenarios),):
                raise ValueError("RSS responses must align with the archive scenarios")
            if self.rss_contract != measurement_contract():
                raise ValueError("All construction archives must use the same global RSS contract")

    @property
    def continuous_margin(self):
        if self.rss_margin is None:
            raise ValueError("RSS construction requires derived trajectory margins, not clearance")
        return self.rss_margin

    @cached_property
    def x(self):
        return features(self.scenarios)

    @cached_property
    def auxiliary(self):
        return np.column_stack(
            (np.clip(self.safety[:, 0], -5, 100) / 100, np.minimum(self.safety[:, 1], 10) / 10)
        )


def archive_fingerprints(archives):
    """A changed response bank cannot silently reuse a model with identical SUT IDs."""
    fingerprints = {}
    for archive in archives:
        metadata = {
            "source_id": archive.source_id,
            "family": archive.family,
            "lineage": archive.lineage,
            "role": archive.role,
            "execution_contract": archive.provenance.get("contract"),
            "scenarios": [asdict(s) for s in archive.scenarios],
        }
        if archive.rss_margin is not None:
            metadata["rss_contract"] = archive.rss_contract
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        digest.update(archive.collision.tobytes())
        digest.update(archive.safety.astype("<f8").tobytes())
        if archive.rss_margin is not None:
            digest.update(archive.rss_margin.astype("<f8").tobytes())
        fingerprints[archive.source_id] = digest.hexdigest()
    return fingerprints


def source_weights(archives):
    """Initialize allowed historical sources equally, independent of ADS family."""
    return np.ones(len(archives), dtype=np.float64) / len(archives)


class HistoricalPredictor:
    def __init__(self, archives, bandwidth=0.25):
        if not archives or any(a.role != "history" for a in archives):
            raise ValueError("Historical prediction accepts historical sources only")
        self.archives = tuple(archives)
        self.bandwidth = bandwidth

    def predict(self, x):
        predictions = []
        for archive in self.archives:
            distance = ((x[:, None] - archive.x[None]) ** 2).sum(-1)
            weight = np.exp(-distance / (2 * self.bandwidth**2))
            predictions.append((weight @ archive.collision + 0.5) / (weight.sum(1) + 1))
        return np.asarray(predictions)
