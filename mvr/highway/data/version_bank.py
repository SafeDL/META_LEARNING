"""Response-bank metadata and validity checks for controller version lineages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.sut.version_lineage import VersionLineages, VersionMeta, _parse_version, validate_lineages


@dataclass(frozen=True)
class VersionedBank:
    """Complete offline bank paired with frozen version and seed metadata."""

    responses: ResponseBank
    versions: tuple[VersionMeta, ...]
    observed_mask: np.ndarray
    valid_mask: np.ndarray
    episode_seeds: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        expected = self.responses.vulnerability.shape
        if self.observed_mask.shape != expected or self.valid_mask.shape != expected:
            raise ValueError("version masks must match response matrix shape")
        if self.episode_seeds.shape != (len(self.responses.anchors),):
            raise ValueError("one deterministic seed is required per anchor")
        if len(self.versions) != len(self.responses.sut_names):
            raise ValueError("one version record is required per response row")
        if tuple(item.version_id for item in self.versions) != self.responses.sut_names:
            raise ValueError("response rows must follow manifest version order")
        if not np.isfinite(self.responses.vulnerability[self.valid_mask]).all():
            raise ValueError("valid responses require finite vulnerability")

    def lineage_rows(self, lineage_id: str) -> tuple[np.ndarray, tuple[VersionMeta, ...]]:
        pairs = sorted(
            ((i, version) for i, version in enumerate(self.versions)
             if version.lineage_id == lineage_id),
            key=lambda pair: pair[1].version_order,
        )
        return np.asarray([pair[0] for pair in pairs], dtype=int), tuple(
            pair[1] for pair in pairs
        )


def make_versioned_bank(
    responses: ResponseBank, lineages: VersionLineages, seed: int, metadata: dict
) -> VersionedBank:
    """Pair fully observed deterministic responses with their frozen lineage."""
    expected_names = tuple(item.version_id for item in lineages.versions)
    if responses.sut_names != expected_names:
        raise ValueError("response bank names must match frozen manifest order")
    shape = responses.vulnerability.shape
    return VersionedBank(
        responses=responses,
        versions=lineages.versions,
        observed_mask=np.ones(shape, dtype=bool),
        valid_mask=np.ones(shape, dtype=bool),
        episode_seeds=np.arange(len(responses.anchors), dtype=int) + seed,
        metadata=dict(metadata),
    )


def save_version_metadata(bank: VersionedBank, path: Path) -> None:
    """Persist non-array version provenance alongside the NPZ response bank."""
    payload = {
        "versions": [
            {**asdict(version), "profile": asdict(version.profile)}
            for version in bank.versions
        ],
        "observed_mask": bank.observed_mask.astype(int).tolist(),
        "valid_mask": bank.valid_mask.astype(int).tolist(),
        "episode_seeds": bank.episode_seeds.tolist(),
        "metadata": bank.metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_versioned_bank(npz_path: Path, metadata_path: Path) -> VersionedBank:
    """Reload a VersionedBank without pickled objects or implicit defaults."""
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    versions = tuple(_parse_version(item) for item in payload["versions"])
    validate_lineages(versions)
    return VersionedBank(
        responses=ResponseBank.load(npz_path),
        versions=versions,
        observed_mask=np.asarray(payload["observed_mask"], dtype=bool),
        valid_mask=np.asarray(payload["valid_mask"], dtype=bool),
        episode_seeds=np.asarray(payload["episode_seeds"], dtype=int),
        metadata=dict(payload["metadata"]),
    )
