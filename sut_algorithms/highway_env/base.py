"""Common contract for Highway-env systems under test."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from highway_env_benchmark.envs.external_cutin import ExternalCutInEnv


class Policy(Protocol):
    name: str
    ego_kind: str

    def reset(self) -> None: ...

    def act(self, env: ExternalCutInEnv) -> int: ...

