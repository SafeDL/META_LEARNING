"""One outer action maps to exactly one physical simulator rollout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .parameter_space import NormalizedScenarioAction, ParameterSpace
from .task_spec import MetaTestTaskSpec


class ScenarioAdapter(Protocol):
    family: str

    def build_env(self, task: MetaTestTaskSpec, config: Mapping[str, float | str]) -> Any: ...
    def reset(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str], seed: int) -> tuple[Any, Mapping[str, Any]]: ...
    def validate_runtime(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str]) -> None: ...


@dataclass
class ScenarioExecutor:
    adapters: Mapping[str, ScenarioAdapter]
    spaces: Mapping[str, ParameterSpace]

    def reset(self, task: MetaTestTaskSpec, action: NormalizedScenarioAction) -> tuple[Any, Mapping[str, Any], dict[str, float | str]]:
        task.validate()
        try:
            adapter, space = self.adapters[task.scenario_family], self.spaces[task.parameter_space_id]
        except KeyError as error:
            raise ValueError(f"no executable contract for task {task.task_id}") from error
        config = space.decode(action)
        env = adapter.build_env(task, config)
        observation, info = adapter.reset(env, task, config, task.seed)
        adapter.validate_runtime(env, task, config)
        return observation, info, config
