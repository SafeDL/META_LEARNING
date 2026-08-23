"""One outer action maps to one verified physical simulator episode."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from ..map.metadrive_tokenizer import tokenize_road_network
from ..sut.registry import SUTRegistry, default_registry
from .applied import AppliedScenario, ExecutableEpisode
from .layout import ScenarioLayout
from .interaction import InteractionCandidate
from .parameter_space import NormalizedScenarioAction, ParameterSpace
from .roles import spawn_sut
from .route_geometry import RoutePolyline
from .task_spec import ScenarioMiningTaskSpec


class ScenarioAdapter(Protocol):
    family: str
    def build_env(self, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], layout: ScenarioLayout | None = None) -> Any: ...
    def reset(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], seed: int) -> tuple[Any, Mapping[str, Any]]: ...
    def resolve_layout(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], candidates: tuple[str, ...]) -> ScenarioLayout: ...
    def validate_runtime(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str]) -> None: ...
    def spawn_from_conflict_distance(self, route: RoutePolyline, conflict_xy: tuple[float, float], distance_to_conflict_m: float) -> float: ...


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

    def _resolve_layout(
        self,
        adapter: ScenarioAdapter,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        candidates: tuple[str, ...],
    ) -> tuple[ScenarioLayout, dict[str, float | str]]:
        layout_env = adapter.build_env(task, config)
        try:
            adapter.reset(layout_env, task, config, task.geometry_seed)
            layout_tokens = tokenize_road_network(layout_env.current_map.road_network)
            if layout_tokens.map_hash != task.map_hash:
                raise RuntimeError(f"runtime map hash mismatch for {task.task_id}: expected {task.map_hash}, got {layout_tokens.map_hash}")
            layout = adapter.resolve_layout(layout_env, task, config, candidates)
            adversary_route = RoutePolyline.from_env(
                layout_env, {"route_id": "adversary", "lane_sequence": layout.adversary_route}
            )
            sut_route = RoutePolyline.from_env(
                layout_env, {"route_id": "sut", "lane_sequence": layout.sut_route}
            )
            resolved = dict(config)
            resolved["adversary_spawn_m"] = adapter.spawn_from_conflict_distance(
                adversary_route,
                layout.conflict_xy,
                float(config["adversary_distance_to_conflict_m"]),
            )
            resolved["sut_spawn_m"] = adapter.spawn_from_conflict_distance(
                sut_route,
                layout.conflict_xy,
                float(config["sut_distance_to_conflict_m"]),
            )
            for name, route in (("adversary", adversary_route), ("sut", sut_route)):
                if float(resolved[f"{name}_spawn_m"]) > route.lane_end_s_m[0]:
                    raise ValueError(f"{name} spawn must remain on its selected start lane")
            return layout, resolved
        finally:
            layout_env.close()

    def enumerate_interactions(
        self, task: ScenarioMiningTaskSpec
    ) -> tuple[Any, tuple[InteractionCandidate, ...]]:
        """Inspect the runtime map before policy selection without exposing labels."""
        task.validate()
        try:
            adapter, space = self.adapters[task.adapter_id], self.spaces[task.functional_scenario + "_v1"]
        except KeyError as error:
            raise ValueError(f"no executable contract for task {task.task_id}") from error
        base = space.decode(NormalizedScenarioAction(0, (0.0,) * space.continuous_dim, space.options[0]))
        env = adapter.build_env(task, base)
        try:
            adapter.reset(env, task, base, task.geometry_seed)
            tokens = tokenize_road_network(env.current_map.road_network)
            if tokens.map_hash != task.geometry_hash:
                raise RuntimeError(f"runtime map hash mismatch for {task.task_id}")
            candidates = []
            for index in range(len(space.candidates)):
                config = space.decode(
                    NormalizedScenarioAction(index, (0.0,) * space.continuous_dim, space.options[0])
                )
                layout = adapter.resolve_layout(env, task, config, space.candidates)
                candidates.append(InteractionCandidate.from_layout(env, layout))
            return tokens, tuple(candidates)
        finally:
            env.close()

    def reset(
        self,
        task: ScenarioMiningTaskSpec,
        action: NormalizedScenarioAction,
        *,
        episode_seed: int | None = None,
    ) -> ExecutableEpisode:
        task.validate()
        run_seed = task.geometry_seed if episode_seed is None else int(episode_seed)
        try:
            adapter, space = self.adapters[task.adapter_id], self.spaces[task.functional_scenario + "_v1"]
        except KeyError as error:
            raise ValueError(f"no executable contract for task {task.task_id}") from error
        config = space.decode(action)
        layout, config = self._resolve_layout(adapter, task, config, space.candidates)
        env = adapter.build_env(task, config, layout)
        try:
            observation, _ = adapter.reset(env, task, config, task.geometry_seed)
            adapter.validate_runtime(env, task, config)
            map_tokens = tokenize_road_network(env.current_map.road_network)
            if map_tokens.map_hash != task.map_hash:
                raise RuntimeError(f"runtime map hash changed between layout and execution: expected {task.map_hash}, got {map_tokens.map_hash}")
            adversary = self._adversary(env)
            sut_adapter, sut_profile = self.sut_registry.create(task.sut_ref)
            sut_adapter.reset(env, task, config, run_seed)
            sut = spawn_sut(env, lane_index=layout.sut_lane, longitudinal_m=float(config["sut_spawn_m"]), speed_mps=float(config["sut_initial_speed_mps"]), destination=layout.sut_destination, adapter=sut_adapter, profile=sut_profile, seed=run_seed)
            self._assert_vehicle_applied(adversary, layout.adversary_lane, float(config["adversary_spawn_m"]), float(config["adversary_initial_speed_mps"]), layout.adversary_destination)
            self._assert_vehicle_applied(sut, layout.sut_lane, float(config["sut_spawn_m"]), float(config["sut_initial_speed_mps"]), layout.sut_destination)
            applied = AppliedScenario(
                str(adversary.id), str(sut.id), layout.adversary_lane, layout.sut_lane,
                float(config["adversary_spawn_m"]), float(config["sut_spawn_m"]),
                float(config["adversary_distance_to_conflict_m"]),
                float(config["sut_distance_to_conflict_m"]),
                float(config["adversary_initial_speed_mps"]), float(config["sut_initial_speed_mps"]),
                layout.candidate, layout.conflict_zone_id, str(config["option"]),
                layout.adversary_route, layout.sut_route,
            )
            setattr(env, "_mvr_episode", applied)
            return ExecutableEpisode(
                env, observation, adversary, sut, sut_adapter, sut_profile,
                applied, map_tokens, layout, run_seed,
            )
        except Exception:
            env.close()
            raise
