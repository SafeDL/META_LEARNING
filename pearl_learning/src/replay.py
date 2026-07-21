"""Strictly task-indexed replay and independent context sampling."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray; action: np.ndarray; reward: float; next_obs: np.ndarray; terminated: bool; truncated: bool
    task_id: str


class TaskReplayBuffer:
    def __init__(self, capacity: int = 200_000): self.capacity, self._data = capacity, []
    def __len__(self) -> int: return len(self._data)
    def add_episode(self, transitions: Iterable[Transition], expected_task_id: str) -> None:
        rows = list(transitions)
        if any(row.task_id != expected_task_id for row in rows): raise ValueError("cross-task transition rejected")
        self._data.extend(rows)
        if len(self._data) > self.capacity: del self._data[:len(self._data) - self.capacity]
    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        if not self._data: raise RuntimeError("cannot sample an empty task buffer")
        indices = rng.integers(len(self._data), size=batch_size)
        return [self._data[int(i)] for i in indices]
    def sample_context(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        # Deliberately a distinct method: caller never receives query episodes.
        return self.sample(batch_size, rng)


class TaskReplayBuffers:
    def __init__(self, task_ids: Iterable[str], capacity: int = 200_000): self.buffers = {task_id: TaskReplayBuffer(capacity) for task_id in task_ids}
    def add_episode(self, task_id: str, transitions: Iterable[Transition]) -> None:
        if task_id not in self.buffers: raise KeyError(f"unknown task buffer {task_id}")
        self.buffers[task_id].add_episode(transitions, task_id)
    def sample_per_task(self, task_ids: list[str], batch_size: int, rng: np.random.Generator) -> list[list[Transition]]:
        return [self.buffers[task_id].sample(batch_size, rng) for task_id in task_ids]
    def context_per_task(self, task_ids: list[str], batch_size: int, rng: np.random.Generator) -> list[list[Transition]]:
        return [self.buffers[task_id].sample_context(batch_size, rng) for task_id in task_ids]
