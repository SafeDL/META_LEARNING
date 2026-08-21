"""Inner replay and strictly on-policy Outer rollout storage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator
import random

import torch


@dataclass(frozen=True)
class InnerTransition:
    episode_id: str
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


@dataclass
class InnerReplay:
    capacity: int = 100_000
    rows: list[InnerTransition] = field(default_factory=list)

    def add(self, row: InnerTransition) -> None:
        self.rows.append(row)
        del self.rows[:-self.capacity]

    def sample(self, count: int, *, excluded_episode_ids: set[str] | None = None, rng: random.Random | None = None) -> list[InnerTransition]:
        eligible = [row for row in self.rows if row.episode_id not in (excluded_episode_ids or set())]
        if len(eligible) < count:
            raise ValueError("not enough leakage-free inner transitions")
        return (rng or random).sample(eligible, count)


@dataclass(frozen=True)
class OuterRolloutStep:
    map_embedding: torch.Tensor
    latent: torch.Tensor
    history: torch.Tensor
    candidate: torch.Tensor
    continuous: torch.Tensor
    option: torch.Tensor
    old_log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


@dataclass
class OuterRolloutBuffer:
    rows: list[OuterRolloutStep] = field(default_factory=list)
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def add(self, row: OuterRolloutStep) -> None:
        if self.advantages is not None:
            raise RuntimeError("clear the finished PPO rollout before adding rows")
        self.rows.append(row)

    def finish(self, *, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = 0.0) -> None:
        if not self.rows:
            raise ValueError("cannot finish an empty PPO rollout")
        values = [float(row.value.detach().cpu()) for row in self.rows] + [float(last_value)]
        advantages = torch.zeros(len(self.rows), dtype=torch.float32)
        gae = 0.0
        for index in range(len(self.rows) - 1, -1, -1):
            nonterminal = 0.0 if self.rows[index].done else 1.0
            delta = self.rows[index].reward + gamma * values[index + 1] * nonterminal - values[index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[index] = gae
        self.returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
        self.advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

    def minibatches(self, batch_size: int, *, generator: torch.Generator | None = None) -> Iterator[dict[str, torch.Tensor]]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("finish PPO rollout before requesting minibatches")
        order = torch.randperm(len(self.rows), generator=generator)
        for indices in order.split(batch_size):
            rows = [self.rows[int(index)] for index in indices]
            yield {
                "map_embedding": torch.stack([row.map_embedding for row in rows]),
                "latent": torch.stack([row.latent for row in rows]),
                "history": torch.stack([row.history for row in rows]),
                "candidate": torch.stack([row.candidate for row in rows]).long().reshape(-1),
                "continuous": torch.stack([row.continuous for row in rows]),
                "option": torch.stack([row.option for row in rows]).long().reshape(-1),
                "old_logprob": torch.stack([row.old_log_prob for row in rows]).reshape(-1),
                "advantage": self.advantages[indices],
                "return": self.returns[indices],
            }

    def clear(self) -> None:
        self.rows.clear()
        self.advantages = self.returns = None
