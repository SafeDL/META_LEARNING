"""Frozen, validated controller-version lineages for regression testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mvr.highway.sut.idm_profiles import SUTProfile


@dataclass(frozen=True)
class VersionMeta:
    """One version in a controlled controller-development lineage."""

    lineage_id: str
    version_id: str
    version_order: int
    predecessor_id: str | None
    controller: str
    profile: SUTProfile
    change_note: str
    origin: str


@dataclass(frozen=True)
class VersionLineages:
    """Validated, immutable collection of independent version lineages."""

    versions: tuple[VersionMeta, ...]

    @property
    def lineage_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.lineage_id for item in self.versions}))

    def by_lineage(self, lineage_id: str) -> tuple[VersionMeta, ...]:
        return tuple(
            sorted(
                (item for item in self.versions if item.lineage_id == lineage_id),
                key=lambda item: item.version_order,
            )
        )

    def profile_tuple(self) -> tuple[SUTProfile, ...]:
        return tuple(item.profile for item in self.versions)


def _parse_version(item: dict) -> VersionMeta:
    profile = SUTProfile(**dict(item["profile"]))
    if profile.name != item["version_id"]:
        raise ValueError("profile.name must equal version_id")
    if profile.controller != item["controller"]:
        raise ValueError("profile.controller must equal controller")
    return VersionMeta(
        lineage_id=str(item["lineage_id"]),
        version_id=str(item["version_id"]),
        version_order=int(item["version_order"]),
        predecessor_id=(
            str(item["predecessor_id"])
            if item.get("predecessor_id") is not None
            else None
        ),
        controller=str(item["controller"]),
        profile=profile,
        change_note=str(item.get("change_note", "")),
        origin=str(item.get("origin", "controlled_parameter_evolution")),
    )


def validate_lineages(versions: tuple[VersionMeta, ...]) -> None:
    """Validate order, predecessor, controller family, and lineage isolation."""
    if not versions:
        raise ValueError("at least one version is required")
    ids = [item.version_id for item in versions]
    if len(ids) != len(set(ids)):
        raise ValueError("version IDs must be globally unique")
    for lineage_id in {item.lineage_id for item in versions}:
        ordered = sorted(
            (item for item in versions if item.lineage_id == lineage_id),
            key=lambda item: item.version_order,
        )
        if [item.version_order for item in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise ValueError("version orders must be contiguous starting at 1")
        for index, item in enumerate(ordered):
            expected = None if index == 0 else ordered[index - 1].version_id
            if item.predecessor_id != expected:
                raise ValueError("predecessor must be the direct prior version")
            if item.controller != ordered[0].controller:
                raise ValueError("a lineage cannot cross controller families")


def load_version_lineages(path: Path) -> VersionLineages:
    """Read the JSON manifest (also valid YAML) and return validated lineages."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    versions = tuple(_parse_version(item) for item in data["versions"])
    validate_lineages(versions)
    return VersionLineages(versions)
