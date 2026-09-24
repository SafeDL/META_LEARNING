"""Highway-env native IDM longitudinal control with MOBIL lane selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from highway_env_benchmark.envs.external_cutin import ExternalCutInEnv


@dataclass
class IDMMobilPolicy:
    name: str = "idm_mobil"
    ego_kind: str = "idm_mobil"

    def reset(self) -> None:
        return None

    def act(self, env: ExternalCutInEnv) -> int:
        del env
        return 1

