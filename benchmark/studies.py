"""Preregistered task scopes for discovery, transfer, mechanisms and continuous work."""

from dataclasses import dataclass

from controllers.highway.catalog import SUTS

from methods.baselines.response_baselines.collision import METHODS
from .protocol import DEFAULT
from .scenarios import DOMAINS


FAMILIES = ("rule", "optimization", "search", "learning")
REPRESENTATIVES = ("target_idm_mobil", "target_linear_mpc", "target_uct", "target_ppo")
CONTINUOUS_TASKS = (("target_idm_mobil", "front_brake"),
                    ("target_linear_mpc", "cut_out"),
                    ("target_uct", "cut_in"),
                    ("target_ppo", "lane_change_rear"))


@dataclass(frozen=True)
class ModelScope:
    name: str
    history_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    held_family: str | None = None


def scopes():
    history = tuple(s for s in SUTS if s.role == "history")
    development = tuple(s for s in SUTS if s.role == "development")
    output = [ModelScope("all", tuple(s.sut_id for s in history),
                         tuple(s.sut_id for s in development))]
    for family in FAMILIES:
        output.append(ModelScope(family, tuple(s.sut_id for s in history if s.family != family),
                                 tuple(s.sut_id for s in development if s.family != family), family))
    for target_family in FAMILIES:
        for alternative in FAMILIES:
            if target_family == alternative:
                continue
            background = {f for f in FAMILIES if f not in (target_family, alternative)}
            for source in (target_family, alternative):
                allowed = background | {source}
                output.append(ModelScope(f"swap_{target_family}_{alternative}_{source}",
                                         tuple(s.sut_id for s in history if s.family in allowed),
                                         tuple(s.sut_id for s in development if s.family in background)))
    return tuple(output)


SCOPES = {scope.name: scope for scope in scopes()}


@dataclass(frozen=True)
class Task:
    study: str
    target_id: str
    family: str
    lineage: str
    scenario: str
    condition: str
    environment_seed: int
    algorithm_seed: int
    method: str
    scope: str


def tasks():
    result = []
    for sut in SUTS:
        if sut.role != "target":
            continue
        for kind in DOMAINS:
            for condition in DEFAULT.conditions:
                for env_seed in DEFAULT.environment_seeds:
                    for seed in DEFAULT.algorithm_seeds:
                        for study, scope in (("overall", "all"), ("strict_family", sut.family)):
                            for method in ("history_response", *METHODS):
                                result.append(Task(study, sut.sut_id, sut.family, sut.lineage, kind,
                                                   condition, env_seed, seed, method, scope))
                        if condition == "new" and sut.sut_id in REPRESENTATIVES:
                            for method in ("no_response", "fixed_weights", "no_target", "greedy"):
                                result.append(Task("mechanisms", sut.sut_id, sut.family, sut.lineage,
                                                   kind, condition, env_seed, seed, method, "all"))
                        if condition == "new":
                            for alternative in FAMILIES:
                                if alternative == sut.family:
                                    continue
                                for source in (sut.family, alternative):
                                    scope = f"swap_{sut.family}_{alternative}_{source}"
                                    for method in ("history_response", "fixed_weights", "history_ranking"):
                                        result.append(Task("source_swap", sut.sut_id, sut.family, sut.lineage,
                                                           kind, condition, env_seed, seed, method, scope))
    return tuple(result)
