"""Deferred adapter for process/service-backed controllers.

It intentionally requires a supplied callable and performs no implicit IPC.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from .base import ControllerProfile


class ExternalSUTAdapter:
    name = "external"

    def __init__(self, controller: Callable[[Mapping[str, Any]], Any] | None = None) -> None:
        self.controller = controller

    def reset(self, env: Any, task: Any, config: Mapping[str, Any], seed: int) -> None:
        if self.controller is None:
            raise RuntimeError("external SUT adapter requires an explicit controller callable")
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset(env=env, task=task, config=config, seed=seed)

    def attach(self, env: Any, vehicle: Any, profile: ControllerProfile, seed: int) -> Any:
        del env, profile, seed
        return vehicle

    def observe(self, env: Any, vehicle: Any) -> Mapping[str, Any]:
        return {"env": env, "vehicle": vehicle}

    def step(self, observation: Mapping[str, Any]) -> Any:
        if self.controller is None:
            raise RuntimeError("external SUT adapter requires an explicit controller callable")
        return self.controller(observation)

    def metadata(self, profile: ControllerProfile) -> dict[str, Any]:
        return {"adapter": self.name, "profile": profile.profile_id, "model_input_fields": (), "profile_is_model_input": False}
