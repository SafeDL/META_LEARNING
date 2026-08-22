"""One outer action maps to one verified physical simulator episode."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from ..map.metadrive_tokenizer import tokenize_road_network
from ..sut.registry import SUTRegistry, default_registry
from .applied import AppliedScenario, ExecutableEpisode
from .layout import ScenarioLayout
from .parameter_space import NormalizedScenarioAction, ParameterSpace
from .roles import spawn_sut
from .task_spec import MetaTestTaskSpec


class ScenarioAdapter(Protocol):
    family: str
    def build_env(self, task: MetaTestTaskSpec, config: Mapping[str, float | str], layout: ScenarioLayout | None = None) -> Any: ...
    def reset(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str], seed: int) -> tuple[Any, Mapping[str, Any]]: ...
    def resolve_layout(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str], candidates: tuple[str, ...]) -> ScenarioLayout: ...
    def validate_runtime(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str]) -> None: ...


@dataclass
class ScenarioExecutor:
    adapters: Mapping[str, ScenarioAdapter]
    spaces: Mapping[str, ParameterSpace]
    sut_registry: SUTRegistry = field(default_factory=default_registry)

    @staticmethod
    def _adversary(env: Any) -> Any:
        agents = env.engine.agent_manager.active_agents
        if "default_agent" not in agents:
            raise RuntimeError("MetaDrive episode has no default adversary agent")
        return agents["default_agent"]

    @staticmethod
    def _speed_mps(vehicle: Any) -> float:
        return float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6

    @staticmethod
    def _assert_vehicle_applied(vehicle: Any, lane_index: tuple[Any, Any, int], spawn_m: float, speed_mps: float, destination: Any) -> None:
        config = vehicle.config
        if tuple(config["spawn_lane_index"]) != tuple(lane_index):
            raise RuntimeError("runtime vehicle spawn lane differs from resolved scenario layout")
        if not np.isclose(float(config["spawn_longitude"]), float(spawn_m), atol=1e-5):
            raise RuntimeError("runtime vehicle spawn longitudinal position differs from outer output")
        if config.get("destination") != destination:
            raise RuntimeError("runtime vehicle destination differs from resolved candidate route")
        if not np.isclose(ScenarioExecutor._speed_mps(vehicle), float(speed_mps), atol=0.25):
            raise RuntimeError("runtime vehicle initial speed differs from outer output")

    def _resolve_layout(self, adapter: ScenarioAdapter, task: MetaTestTaskSpec, config: Mapping[str, float | str], candidates: tuple[str, ...]) -> ScenarioLayout:
        layout_env = adapter.build_env(task, config)
        try:
            adapter.reset(layout_env, task, config, task.seed)
            layout_tokens = tokenize_road_network(layout_env.current_map.road_network)
            if layout_tokens.map_hash != task.map_hash:
                raise RuntimeError(f"runtime map hash mismatch for {task.task_id}: expected {task.map_hash}, got {layout_tokens.map_hash}")
            return adapter.resolve_layout(layout_env, task, config, candidates)
        finally:
            layout_env.close()

    def reset(self, task: MetaTestTaskSpec, action: NormalizedScenarioAction) -> ExecutableEpisode:
        task.validate()
        try:
            adapter, space = self.adapters[task.scenario_family], self.spaces[task.parameter_space_id]
        except KeyError as error:
            raise ValueError(f"no executable contract for task {task.task_id}") from error
        config = space.decode(action)
        layout = self._resolve_layout(adapter, task, config, space.candidates)
        env = adapter.build_env(task, config, layout)
        try:
            observation, _ = adapter.reset(env, task, config, task.seed)
            adapter.validate_runtime(env, task, config)
            map_tokens = tokenize_road_network(env.current_map.road_network)
            if map_tokens.map_hash != task.map_hash:
                raise RuntimeError(f"runtime map hash changed between layout and execution: expected {task.map_hash}, got {map_tokens.map_hash}")
            adversary = self._adversary(env)
            sut_adapter, sut_profile = self.sut_registry.create(task.sut_ref)
            sut_adapter.reset(env, task, config, task.seed)
            sut = spawn_sut(env, lane_index=layout.sut_lane, longitudinal_m=float(config["sut_spawn_m"]), speed_mps=float(config["sut_initial_speed_mps"]), destination=layout.sut_destination, adapter=sut_adapter, profile=sut_profile, seed=task.seed)
            self._assert_vehicle_applied(adversary, layout.adversary_lane, float(config["adversary_spawn_m"]), float(config["adversary_initial_speed_mps"]), layout.adversary_destination)
            self._assert_vehicle_applied(sut, layout.sut_lane, float(config["sut_spawn_m"]), float(config["sut_initial_speed_mps"]), layout.sut_destination)
            applied = AppliedScenario(str(adversary.id), str(sut.id), layout.adversary_lane, layout.sut_lane, float(config["adversary_spawn_m"]), float(config["sut_spawn_m"]), float(config["adversary_initial_speed_mps"]), float(config["sut_initial_speed_mps"]), layout.candidate, layout.conflict_zone_id, str(config["option"]), layout.adversary_route, layout.sut_route)
            setattr(env, "_meta_testing_episode", applied)
            return ExecutableEpisode(env, observation, adversary, sut, sut_adapter, sut_profile, applied, map_tokens, layout)
        except Exception:
            env.close()
            raise
