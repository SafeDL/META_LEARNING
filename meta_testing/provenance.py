"""Small, dependency-free provenance helpers shared by all MVR runners."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunManifest:
    schema: str
    run_name: str
    config_hash: str
    seed: int
    checkpoint_schema: str
    taskbook_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def build(cls, run_name: str, config: Mapping[str, Any], seed: int, taskbook: Any) -> "RunManifest":
        return cls(
            schema="meta_testing_run_manifest_v1",
            run_name=str(run_name),
            config_hash=content_hash(config),
            seed=int(seed),
            checkpoint_schema="hierarchical_checkpoint_v1",
            taskbook_hash=content_hash(taskbook),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
