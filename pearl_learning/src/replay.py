"""Task-isolated, episode-aware replay for PEARL prior/posterior collection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np


CollectionMode = Literal["prior_support", "posterior_rollout", "deterministic_query"]


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    termination_reason: str
    task_id: str
    episode_id: str
    case_id: str
    collection_mode: CollectionMode
    posterior_version: int


@dataclass(frozen=True)
class ReplayEpisode:
    task_id: str
    episode_id: str
    case_id: str
    collection_mode: CollectionMode
    posterior_version: int
    terminated: bool
    truncated: bool
    termination_reason: str
    transitions: tuple[Transition, ...]


class TaskReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self._episodes: list[ReplayEpisode] = []
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def episodes(self) -> tuple[ReplayEpisode, ...]:
        return tuple(self._episodes)

    def add_episode(self, transitions: Iterable[Transition], expected_task_id: str) -> None:
        rows = tuple(transitions)
        if not rows:
            raise ValueError("empty episodes cannot enter replay")
        head = rows[0]
        if any(row.task_id != expected_task_id for row in rows):
            raise ValueError("cross-task transition rejected")
        if any((row.episode_id, row.case_id, row.collection_mode, row.posterior_version) != (head.episode_id, head.case_id, head.collection_mode, head.posterior_version) for row in rows):
            raise ValueError("a replay episode must have consistent provenance")
        tail = rows[-1]
        if not (tail.terminated or tail.truncated):
            raise ValueError("replay accepts only completed episodes")
        if any(row.terminated or row.truncated for row in rows[:-1]):
            raise ValueError("only the final transition may end a replay episode")
        episode = ReplayEpisode(
            expected_task_id, head.episode_id, head.case_id, head.collection_mode, head.posterior_version,
            bool(tail.terminated), bool(tail.truncated), tail.termination_reason, rows,
        )
        self._episodes.append(episode); self._size += len(rows)
        while self._episodes and self._size > self.capacity:
            self._size -= len(self._episodes.pop(0).transitions)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        if not self._episodes:
            raise RuntimeError("cannot sample an empty task buffer")
        all_rows = [row for episode in self._episodes for row in episode.transitions]
        indices = rng.integers(len(all_rows), size=int(batch_size))
        return [all_rows[int(index)] for index in indices]

    def sample_episode_balanced(self, total_size: int, per_episode: int, rng: np.random.Generator) -> list[list[Transition]]:
        if not self._episodes:
            raise RuntimeError("cannot sample context from an empty task buffer")
        per_episode = max(1, int(per_episode))
        episode_count = min(len(self._episodes), max(1, int(total_size) // per_episode))
        indices = rng.choice(len(self._episodes), size=episode_count, replace=False)
        groups: list[list[Transition]] = []
        for index in np.asarray(indices).reshape(-1):
            rows = list(self._episodes[int(index)].transitions)
            chosen = rng.choice(len(rows), size=per_episode, replace=len(rows) < per_episode)
            groups.append([rows[int(item)] for item in np.asarray(chosen).reshape(-1)])
        return groups


class TaskReplayBuffers:
    def __init__(self, task_ids: Iterable[str], capacity: int = 200_000):
        self.buffers = {task_id: TaskReplayBuffer(capacity) for task_id in task_ids}

    def add_episode(self, task_id: str, transitions: Iterable[Transition]) -> None:
        if task_id not in self.buffers:
            raise KeyError(f"unknown task buffer {task_id}")
        self.buffers[task_id].add_episode(transitions, task_id)

    def sample_per_task(self, task_ids: list[str], batch_size: int, rng: np.random.Generator) -> list[list[Transition]]:
        return [self.buffers[task_id].sample(batch_size, rng) for task_id in task_ids]

    def context_per_task(self, task_ids: list[str], total_size: int, per_episode: int, rng: np.random.Generator) -> list[list[list[Transition]]]:
        return [self.buffers[task_id].sample_episode_balanced(total_size, per_episode, rng) for task_id in task_ids]
