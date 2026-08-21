"""Headless MetaDrive family adapter base with reset/config provenance."""
from __future__ import annotations

from typing import Any, Mapping

from ..task_spec import MetaTestTaskSpec


class MetaDriveFamilyAdapter:
    family = ""

    def env_config(self, task: MetaTestTaskSpec, config: Mapping[str, float | str]) -> dict[str, Any]:
        raise NotImplementedError

    def build_env(self, task: MetaTestTaskSpec, config: Mapping[str, float | str]) -> Any:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        if task.scenario_family != self.family:
            raise ValueError(f"{self.family} adapter cannot execute {task.scenario_family}")
        return MetaDriveEnv(self.env_config(task, config))

    def reset(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str], seed: int) -> tuple[Any, Mapping[str, Any]]:
        # MetaDrive's procedural maps use ``start_seed``/``num_scenarios`` at
        # construction time; passing an arbitrary reset seed is invalid when
        # only one scenario is configured.
        del seed
        observation, info = env.reset()
        info = dict(info)
        info["meta_testing_task_id"] = task.task_id
        info["meta_testing_config"] = dict(config)
        setattr(env, "_meta_testing_initial_config", dict(config))
        setattr(env, "_meta_testing_observation", observation)
        return observation, info

    def validate_runtime(self, env: Any, task: MetaTestTaskSpec, config: Mapping[str, float | str]) -> None:
        if getattr(env, "_meta_testing_initial_config", None) != dict(config):
            raise RuntimeError("outer configuration was not recorded by simulator reset")
        if task.scenario_family != self.family:
            raise RuntimeError("runtime scenario family mismatch")
