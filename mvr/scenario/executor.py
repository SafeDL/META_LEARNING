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
    def build_env(
        self,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        layout: ScenarioLayout | None = None,
        environment_overrides: Mapping[str, Any] | None = None,
    ) -> Any: ...
    def reset(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], seed: int) -> tuple[Any, Mapping[str, Any]]: ...
    def resolve_layout(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], candidates: tuple[str, ...]) -> ScenarioLayout: ...
    def validate_runtime(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str]) -> None: ...
    def spawn_from_conflict_distance(self, route: RoutePolyline, conflict_xy: tuple[float, float], distance_to_conflict_m: float) -> float: ...


@dataclass(frozen=True)
class _CachedLayout:
    layout: ScenarioLayout
    adversary_route: RoutePolyline
    sut_route: RoutePolyline


@dataclass(frozen=True)
class _StaticScene:
    map_tokens: Any
    candidates: tuple[InteractionCandidate, ...]
    layouts: Mapping[str, _CachedLayout]


@dataclass
class ScenarioExecutor:
    adapters: Mapping[str, ScenarioAdapter]
    spaces: Mapping[str, ParameterSpace]
    sut_registry: SUTRegistry = field(default_factory=default_registry)
    _static_scenes: dict[tuple[str, str], _StaticScene] = field(default_factory=dict, init=False, repr=False)

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

    @staticmethod
    def _assert_vehicle_route(vehicle: Any, route: tuple[Any, ...]) -> None:
        expected = tuple([route[0][0], *(lane_index[1] for lane_index in route)])
        actual = tuple(vehicle.navigation.checkpoints)
        if actual != expected:
            raise RuntimeError("runtime vehicle navigation differs from resolved candidate route")

    @staticmethod
    def _static_key(task: ScenarioMiningTaskSpec) -> tuple[str, str]:
        return task.adapter_id, task.geometry_hash

    @staticmethod
    def _resolve_spawn_config(
        adapter: ScenarioAdapter,
        cached: _CachedLayout,
        config: Mapping[str, float | str],
    ) -> dict[str, float | str]:
        resolved = dict(config)
        for name, route in (("adversary", cached.adversary_route), ("sut", cached.sut_route)):
            distance_key = f"{name}_distance_to_conflict_m"
            conflict_s = route.conflict_s(cached.layout.conflict_xy)
            lower = max(0.0, conflict_s - route.lane_end_s_m[0] + 1e-3)
            upper = max(lower, conflict_s)
            applied_distance = float(np.clip(float(config[distance_key]), lower, upper))
            resolved[distance_key] = applied_distance
            resolved[f"{name}_spawn_m"] = adapter.spawn_from_conflict_distance(
                route, cached.layout.conflict_xy, applied_distance
            )
        return resolved

    def _static_scene(self, task: ScenarioMiningTaskSpec) -> _StaticScene:
        key = self._static_key(task)
        if key not in self._static_scenes:
            self.enumerate_interactions(task)
        return self._static_scenes[key]

    def enumerate_interactions(
        self, task: ScenarioMiningTaskSpec
    ) -> tuple[Any, tuple[InteractionCandidate, ...]]:
        """Inspect the runtime map before policy selection without exposing labels."""
        task.validate()
        key = self._static_key(task)
        cached = self._static_scenes.get(key)
        if cached is not None:
            return cached.map_tokens, cached.candidates
        try:
            adapter, space = self.adapters[task.adapter_id], self.spaces[task.functional_scenario]
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
            layouts: dict[str, _CachedLayout] = {}
            for index in range(len(space.candidates)):
                config = space.decode(
                    NormalizedScenarioAction(index, (0.0,) * space.continuous_dim, space.options[0])
                )
                layout = adapter.resolve_layout(env, task, config, space.candidates)
                adversary_route = RoutePolyline.from_env(
                    env, {"route_id": "adversary", "lane_sequence": layout.adversary_route}
                )
                sut_route = RoutePolyline.from_env(
                    env, {"route_id": "sut", "lane_sequence": layout.sut_route}
                )
                layouts[layout.candidate] = _CachedLayout(layout, adversary_route, sut_route)
                candidates.append(InteractionCandidate.from_routes(layout, adversary_route, sut_route))
            static = _StaticScene(tokens, tuple(candidates), layouts)
            self._static_scenes[key] = static
            return static.map_tokens, static.candidates
        finally:
            env.close()

    def reset(
        self,
        task: ScenarioMiningTaskSpec,
        action: NormalizedScenarioAction,
        *,
        episode_seed: int | None = None,
        environment_overrides: Mapping[str, Any] | None = None,
    ) -> ExecutableEpisode:
        task.validate()
        run_seed = task.geometry_seed if episode_seed is None else int(episode_seed)
        try:
            adapter, space = self.adapters[task.adapter_id], self.spaces[task.functional_scenario]
        except KeyError as error:
            raise ValueError(f"no executable contract for task {task.task_id}") from error
        config = space.decode(action)
        static = self._static_scene(task)
        candidate = str(config["route_or_conflict_candidate"])
        try:
            cached = static.layouts[candidate]
        except KeyError as error:
            raise RuntimeError(f"static layout is missing candidate {candidate!r}") from error
        config = self._resolve_spawn_config(adapter, cached, config)
        if environment_overrides:
            env = adapter.build_env(task, config, cached.layout, environment_overrides)
        else:
            env = adapter.build_env(task, config, cached.layout)
        try:
            observation, _ = adapter.reset(env, task, config, task.geometry_seed)
            adapter.validate_runtime(env, task, config)
            runtime_tokens = tokenize_road_network(env.current_map.road_network)
            if runtime_tokens.map_hash != task.map_hash:
                raise RuntimeError(f"runtime map hash changed between layout and execution: expected {task.map_hash}, got {runtime_tokens.map_hash}")
            map_tokens = static.map_tokens
            adversary = self._adversary(env)
            sut_adapter, sut_profile = self.sut_registry.create(task.sut_ref)
            sut_adapter.reset(env, task, config, run_seed)
            layout = cached.layout
            sut = spawn_sut(
                env,
                lane_index=layout.sut_lane,
                longitudinal_m=float(config["sut_spawn_m"]),
                speed_mps=float(config["sut_initial_speed_mps"]),
                destination=layout.sut_destination,
                route=layout.sut_route,
                adapter=sut_adapter,
                profile=sut_profile,
                seed=run_seed,
            )
            self._assert_vehicle_applied(adversary, layout.adversary_lane, float(config["adversary_spawn_m"]), float(config["adversary_initial_speed_mps"]), layout.adversary_destination)
            self._assert_vehicle_applied(sut, layout.sut_lane, float(config["sut_spawn_m"]), float(config["sut_initial_speed_mps"]), layout.sut_destination)
            self._assert_vehicle_route(adversary, layout.adversary_route)
            self._assert_vehicle_route(sut, layout.sut_route)
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
                applied, map_tokens, layout, cached.adversary_route, cached.sut_route, run_seed,
            )
        except Exception:
            env.close()
            raise
