"""The single online few-shot protocol shared by training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ..context.outcome_schema import encode_outcome
from ..failure.novelty import NoveltyTracker
from ..failure.reward import outer_reward
from ..model import HierarchicalMetaTester
from ..scenario.executor import ScenarioExecutor
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.task_spec import MetaTestTaskSpec
from ..state import PhysicalStateExtractor
from .replay import InnerTransition, OuterRolloutBuffer, OuterRolloutStep
from .runner import HierarchicalRunner, Rollout


@dataclass(frozen=True)
class OnlineEpisode:
    episode_id: str
    rollout: Rollout
    token: torch.Tensor
    latent_before: torch.Tensor
    latent_after: torch.Tensor
    map_tokens: Any
    config: torch.Tensor
    option_index: torch.Tensor
    outcome: Mapping[str, Any]


@dataclass
class OnlineMetaTestResult:
    episodes: list[OnlineEpisode]
    inner_transitions: list[InnerTransition]
    outer_rollout: OuterRolloutBuffer


class OnlineMetaTest:
    """Execute a fixed testing budget without exposing SUT identity to the model."""

    def __init__(self, model: HierarchicalMetaTester, executor: ScenarioExecutor, runner: HierarchicalRunner) -> None:
        self.model = model
        self.executor = executor
        self.runner = runner
        self._map_cache: dict[str, torch.Tensor] = {}
        self._tokens_cache: dict[str, Any] = {}

    def _map_embedding(self, tokens: Any) -> torch.Tensor:
        cached = self._map_cache.get(tokens.map_hash)
        if cached is not None:
            return cached
        with torch.no_grad():
            _, embedding = self.model.encode_map(tokens)
        embedding = embedding.detach()
        if not any(parameter.requires_grad for parameter in self.model.map_encoder.parameters()):
            self._map_cache[tokens.map_hash] = embedding
        return embedding

    def run(self, task: MetaTestTaskSpec, budget: int, *, deterministic: bool = False, posterior_support_limit: int | None = None) -> OnlineMetaTestResult:
        if budget < 1 or self.model.outer_history_dim != 0:
            raise ValueError("online meta-test requires a positive budget and zero outer history")
        if self.model.state_dim != PhysicalStateExtractor.dimension:
            raise ValueError("online Inner rollout requires the physical state schema")
        if posterior_support_limit is not None and not 0 <= posterior_support_limit <= budget:
            raise ValueError("posterior support limit must lie within the episode budget")
        space = self.model.parameter_spaces[task.parameter_space_id]
        tokens = self._tokens_cache.get(task.map_hash)
        if tokens is None:
            setup = self.executor.reset(task, NormalizedScenarioAction(0, (0.0,) * space.continuous_dim, space.options[0]))
            try:
                tokens = setup.map_tokens
            finally:
                setup.env.close()
            self._tokens_cache[task.map_hash] = tokens
        map_embedding = self._map_embedding(tokens)
        device = self.model.device
        latent, _ = self.model.posterior.prior(device=device)
        evidence_tokens: list[torch.Tensor] = []
        result = OnlineMetaTestResult([], [], OuterRolloutBuffer())
        novelty = NoveltyTracker()
        for index in range(budget):
            with torch.no_grad():
                scene = self.model.select_scene(task.parameter_space_id, map_embedding.unsqueeze(0), latent, deterministic=deterministic)
            action = NormalizedScenarioAction(int(scene.candidate_index.item()), tuple(float(value) for value in scene.continuous.squeeze(0).tolist()), space.options[int(scene.option_index.item())])
            episode = self.executor.reset(task, action)
            option_index = scene.option_index.detach()
            def inner_action(state_values: np.ndarray) -> np.ndarray:
                state = torch.as_tensor(state_values, dtype=torch.float32, device=device).reshape(1, -1)
                with torch.no_grad():
                    value = self.model.act_inner(state, map_embedding.unsqueeze(0), latent, option_index, scene.continuous, deterministic=deterministic)
                return value.squeeze(0).cpu().numpy()

            try:
                rollout = self.runner.rollout(episode, task.scenario_family, action.option.value, inner_action)
            finally:
                episode.env.close()
            signature = rollout.signature
            outcome = dict(rollout.outcome)
            outcome.update({"is_valid_episode": signature.is_valid_episode, "is_failure": signature.is_failure})
            outcome_tensor = encode_outcome(outcome).to(device).unsqueeze(0)
            trajectory = rollout.trajectory.to(device).unsqueeze(0)
            mask = torch.ones(trajectory.shape[:2], dtype=torch.bool, device=device)
            with torch.no_grad():
                token = self.model.episode_token_builder(map_embedding.unsqueeze(0), scene.continuous, option_index, trajectory, mask, outcome_tensor).squeeze(0)
            evidence_tokens.append(token)
            support = torch.stack(evidence_tokens).unsqueeze(0)
            support_mask = torch.ones((1, len(evidence_tokens)), dtype=torch.bool, device=device)
            latent_after = latent
            if posterior_support_limit is None or index < posterior_support_limit:
                with torch.no_grad():
                    latent_after, _ = self.model.infer_posterior(support, support_mask)
            reward = outer_reward(signature, novel=novelty.observe(signature))
            result.outer_rollout.add(OuterRolloutStep(map_embedding.cpu(), latent.squeeze(0).cpu(), torch.empty(0), scene.candidate_index.cpu(), scene.continuous.squeeze(0).cpu(), scene.option_index.cpu(), scene.log_prob.cpu(), scene.value.cpu(), reward, index + 1 == budget))
            episode_id = f"{task.task_id}:{index}"
            result.inner_transitions.extend(InnerTransition(episode_id, row["state"], row["action"], row["reward_inner"], row["next_state"], row["done"], tokens, latent.squeeze(0).detach().cpu(), option_index.squeeze(0).detach().cpu(), scene.continuous.squeeze(0).detach().cpu()) for row in rollout.transitions)
            result.episodes.append(OnlineEpisode(episode_id, rollout, token.detach().cpu(), latent.detach().cpu().clone(), latent_after.detach().cpu().clone(), tokens, scene.continuous.squeeze(0).detach().cpu(), option_index.squeeze(0).detach().cpu(), outcome))
            latent = latent_after
        result.outer_rollout.finish()
        return result
