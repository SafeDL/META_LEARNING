"""The single online few-shot protocol shared by training and evaluation."""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ..context.outcome_schema import encode_outcome
from ..failure.novelty import NoveltyTracker
from ..failure.reward import outer_reward
from ..model import TransferableScenarioMiner
from ..policy.universal_scene_policy import UniversalSceneAction
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.concrete import ConcreteScenario
from ..scenario.executor import ScenarioExecutor
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.task_spec import ScenarioMiningTaskSpec
from ..provenance import content_hash
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
    scene_embedding: torch.Tensor
    config: torch.Tensor
    option_index: torch.Tensor
    outcome: Mapping[str, Any]
    concrete_scenario: ConcreteScenario


@dataclass
class OnlineMetaTestResult:
    episodes: list[OnlineEpisode]
    inner_transitions: list[InnerTransition]
    outer_rollout: OuterRolloutBuffer


class OnlineMetaTest:
    """Execute a fixed testing budget without exposing SUT identity to the model."""

    garbage_collection_interval = 12

    def __init__(self, model: TransferableScenarioMiner, executor: ScenarioExecutor, runner: HierarchicalRunner) -> None:
        self.model = model
        self.executor = executor
        self.runner = runner
        self._scene_cache: dict[str, tuple[Any, Any, Any]] = {}
        self._closed_episode_count = 0

    def _inner_policy_hash(self) -> str:
        return content_hash({
            component: {
                name: value.detach().cpu().tolist()
                for name, value in self.model.training_components()[component].state_dict().items()
            }
            for component in ("shared_feature_encoder", "option_embedding", "inner_sac")
        })

    def _scene_encoding(self, task: ScenarioMiningTaskSpec) -> tuple[Any, Any, Any]:
        cached = self._scene_cache.get(task.geometry_hash)
        if cached is not None:
            return cached
        tokens, candidates = self.executor.enumerate_interactions(task)
        with torch.no_grad():
            encoding = self.model.encode_scene(tokens, candidates)
        map_frozen = not self.model.training or not any(
            parameter.requires_grad for parameter in self.model.map_encoder.parameters()
        )
        if map_frozen:
            self._scene_cache[task.geometry_hash] = (tokens, candidates, encoding)
        return tokens, candidates, encoding

    def run(
        self,
        task: ScenarioMiningTaskSpec,
        budget: int,
        *,
        deterministic: bool = False,
        posterior_support_limit: int | None = None,
        episode_index_offset: int = 0,
        scene_action_provider: Callable[[ScenarioMiningTaskSpec, int, tuple[Any, ...], Any], NormalizedScenarioAction] | None = None,
        inner_action_provider: Callable[[np.ndarray], np.ndarray] | None = None,
        rollout_step_callback: Callable[[Any, int, Mapping[str, Any]], None] | None = None,
        environment_overrides: Mapping[str, Any] | None = None,
    ) -> OnlineMetaTestResult:
        if budget < 1:
            raise ValueError("online meta-test requires a positive budget")
        if self.model.state_dim != PhysicalStateExtractor.dimension:
            raise ValueError("online Inner rollout requires the physical state schema")
        if posterior_support_limit is not None and not 0 <= posterior_support_limit <= budget:
            raise ValueError("posterior support limit must lie within the episode budget")
        if episode_index_offset < 0:
            raise ValueError("episode index offset must be non-negative")
        space = mvr_parameter_spaces()[task.functional_scenario]
        tokens, candidates, encoding = self._scene_encoding(task)
        scene_embedding = encoding.global_embedding
        device = self.model.device
        latent, _ = self.model.context_encoder.prior(device=device)
        inner_policy_hash = self._inner_policy_hash()
        evidence_tokens: list[torch.Tensor] = []
        result = OnlineMetaTestResult([], [], OuterRolloutBuffer())
        novelty = NoveltyTracker()
        for index in range(budget):
            episode_index = episode_index_offset + index
            if scene_action_provider is None:
                with torch.no_grad():
                    scene = self.model.select_scene(encoding, latent, deterministic=deterministic)
            else:
                provided = scene_action_provider(task, episode_index, candidates, space)
                provided.validate(space.continuous_dim)
                scene = UniversalSceneAction(
                    expert_index=torch.zeros(1, dtype=torch.long, device=device),
                    candidate_index=torch.tensor([provided.candidate_index], dtype=torch.long, device=device),
                    continuous=torch.as_tensor(provided.continuous, dtype=torch.float32, device=device).unsqueeze(0),
                    option_index=torch.tensor([space.options.index(provided.option)], dtype=torch.long, device=device),
                    log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                    value=torch.zeros(1, dtype=torch.float32, device=device),
                )
            action = NormalizedScenarioAction(int(scene.candidate_index.item()), tuple(float(value) for value in scene.continuous.squeeze(0).tolist()), space.options[int(scene.option_index.item())])
            episode_seed = task.geometry_seed + episode_index
            episode = self.executor.reset(
                task,
                action,
                episode_seed=episode_seed,
                environment_overrides=environment_overrides,
            )
            concrete = ConcreteScenario.from_applied(
                task,
                episode.applied_scenario,
                inner_policy_hash,
                latent=latent.squeeze(0).detach().cpu().tolist(),
                episode_seed=episode_seed,
            )
            option_index = scene.option_index.detach()
            def inner_action(state_values: np.ndarray) -> np.ndarray:
                if inner_action_provider is not None:
                    return np.asarray(inner_action_provider(state_values), dtype=np.float32)
                state = torch.as_tensor(state_values, dtype=torch.float32, device=device).reshape(1, -1)
                with torch.no_grad():
                    value = self.model.act_inner(state, scene_embedding.unsqueeze(0), latent, option_index, scene.continuous, deterministic=deterministic)
                return value.squeeze(0).cpu().numpy()

            try:
                rollout = self.runner.rollout(
                    episode,
                    task.functional_scenario,
                    action.option.value,
                    inner_action,
                    step_callback=rollout_step_callback,
                )
            finally:
                episode.env.close()
                # MetaDrive/Panda3D keeps cyclic scene references after the
                # engine singleton is closed.  Collect cyclic references at
                # a fixed cadence: per-episode collection dominates formal
                # Stage1 runtime without improving resource boundedness.
                self._closed_episode_count += 1
                if self._closed_episode_count % self.garbage_collection_interval == 0:
                    gc.collect()
            signature = rollout.signature
            outcome = dict(rollout.outcome)
            outcome.update({"is_valid_episode": signature.is_valid_episode, "is_failure": signature.is_failure})
            outcome_tensor = encode_outcome(outcome).to(device).unsqueeze(0)
            trajectory = rollout.trajectory.to(device).unsqueeze(0)
            mask = torch.ones(trajectory.shape[:2], dtype=torch.bool, device=device)
            with torch.no_grad():
                token = self.model.episode_token_builder(scene_embedding.unsqueeze(0), scene.continuous, option_index, trajectory, mask, outcome_tensor).squeeze(0)
            evidence_tokens.append(token)
            support = torch.stack(evidence_tokens).unsqueeze(0)
            support_mask = torch.ones((1, len(evidence_tokens)), dtype=torch.bool, device=device)
            latent_after = latent
            if posterior_support_limit is None or index < posterior_support_limit:
                with torch.no_grad():
                    latent_after, _ = self.model.infer_posterior(support, support_mask)
            reward = outer_reward(signature, novel=novelty.observe(signature))
            result.outer_rollout.add(OuterRolloutStep(
                scene_embedding.cpu(), encoding.candidate_embeddings.cpu(), encoding.candidate_mask.cpu(),
                latent.squeeze(0).cpu(), scene.expert_index.cpu(), scene.candidate_index.cpu(),
                scene.continuous.squeeze(0).cpu(), scene.option_index.cpu(), scene.log_prob.cpu(),
                scene.value.cpu(), reward, index + 1 == budget,
            ))
            episode_id = f"{task.task_id}:{episode_index}"
            result.inner_transitions.extend(
                InnerTransition(
                    episode_id, task.geometry_hash, row["state"], row["action"], row["reward_inner"], row["next_state"],
                    row["done"], tokens, candidates, latent.squeeze(0).detach().cpu(),
                    option_index.squeeze(0).detach().cpu(), scene.continuous.squeeze(0).detach().cpu(),
                    np.asarray(row["state"][-4:], dtype=np.float32).copy(),
                    bool(row["maneuver_update_mask"]),
                )
                for row in rollout.transitions
            )
            result.episodes.append(OnlineEpisode(
                episode_id, rollout, token.detach().cpu(), latent.detach().cpu().clone(),
                latent_after.detach().cpu().clone(), tokens, scene_embedding.detach().cpu(),
                scene.continuous.squeeze(0).detach().cpu(), option_index.squeeze(0).detach().cpu(), outcome,
                concrete,
            ))
            latent = latent_after
        result.outer_rollout.finish()
        return result
