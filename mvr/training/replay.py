"""Inner replay and strictly on-policy Outer rollout storage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping
import random

import torch


@dataclass(frozen=True)
class InnerTransition:
    episode_id: str
    task_id: str
    support_group_id: str | None
    geometry_hash: str
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    map_tokens: Any
    interactions: tuple[Any, ...]
    logical_domain_bounds: Mapping[str, tuple[float, float]]
    latent: torch.Tensor
    concrete: torch.Tensor
    schedule_state: Any


@dataclass
class InnerReplay:
    capacity: int = 100_000
    rows: list[InnerTransition] = field(default_factory=list)

    def add(self, row: InnerTransition) -> None:
        self.rows.append(row)
        del self.rows[:-self.capacity]

    def sample(
        self,
        count: int,
        *,
        excluded_episode_ids: set[str] | None = None,
        rng: random.Random | None = None,
        positive_fraction: float = 0.0,
    ) -> list[InnerTransition]:
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive_fraction must lie in [0, 1]")
        eligible = [row for row in self.rows if row.episode_id not in (excluded_episode_ids or set())]
        if len(eligible) < count:
            raise ValueError("not enough leakage-free inner transitions")
        sampler = rng or random
        positive_count = min(
            int(round(count * positive_fraction)),
            sum(float(row.reward) > 0.0 for row in eligible),
        )
        if positive_count == 0:
            return sampler.sample(eligible, count)
        positives = [row for row in eligible if float(row.reward) > 0.0]
        selected = sampler.sample(positives, positive_count)
        selected_ids = {id(row) for row in selected}
        remaining = [row for row in eligible if id(row) not in selected_ids]
        needed = count - positive_count
        non_positive = [row for row in remaining if float(row.reward) <= 0.0]
        if len(non_positive) >= needed:
            return selected + sampler.sample(non_positive, needed)
        return selected + non_positive + sampler.sample(
            [row for row in remaining if float(row.reward) > 0.0],
            needed - len(non_positive),
        )


@dataclass(frozen=True)
class SupportGroup:
    """One task-local support/query partition used to reconstruct a latent."""

    support_group_id: str
    task_id: str
    support_episodes: tuple[Any, ...]
    query_episodes: Mapping[str, Any]

    def validate(self) -> None:
        support_ids = {episode.episode_id for episode in self.support_episodes}
        query_ids = set(self.query_episodes)
        if not support_ids or not query_ids:
            raise ValueError("support groups require support and query episodes")
        if support_ids & query_ids:
            raise ValueError("support and query episodes must be disjoint")


@dataclass
class ContextReplay:
    """Group-level support storage; transitions refer to groups by id only."""

    groups: dict[str, SupportGroup] = field(default_factory=dict)

    def add(self, group: SupportGroup) -> None:
        group.validate()
        if group.support_group_id in self.groups:
            raise ValueError(f"duplicate support group {group.support_group_id!r}")
        self.groups[group.support_group_id] = group

    def get(self, group_id: str) -> SupportGroup:
        try:
            return self.groups[group_id]
        except KeyError as error:
            raise KeyError(f"missing support group {group_id!r}") from error


@dataclass(frozen=True)
class OuterRolloutStep:
    scene_embedding: torch.Tensor
    candidate_embeddings: torch.Tensor
    candidate_mask: torch.Tensor
    latent: torch.Tensor
    expert_index: torch.Tensor
    candidate: torch.Tensor
    continuous: torch.Tensor
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
                "scene_embedding": torch.stack([row.scene_embedding for row in rows]),
                "candidate_embeddings": torch.nn.utils.rnn.pad_sequence(
                    [row.candidate_embeddings for row in rows], batch_first=True
                ),
                "candidate_mask": torch.nn.utils.rnn.pad_sequence(
                    [row.candidate_mask for row in rows], batch_first=True, padding_value=False
                ),
                "latent": torch.stack([row.latent for row in rows]),
                "expert": torch.stack([row.expert_index for row in rows]).long().reshape(-1),
                "candidate": torch.stack([row.candidate for row in rows]).long().reshape(-1),
                "continuous": torch.stack([row.continuous for row in rows]),
                "old_logprob": torch.stack([row.old_log_prob for row in rows]).reshape(-1),
                "advantage": self.advantages[indices],
                "return": self.returns[indices],
            }

    def clear(self) -> None:
        self.rows.clear()
        self.advantages = self.returns = None
