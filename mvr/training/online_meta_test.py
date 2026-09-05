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
    interactions: tuple[Any, ...]
    logical_domain_bounds: Mapping[str, tuple[float, float]]
    logical_parameter_mask: tuple[bool, ...]
    candidate_index: int
    continuous: tuple[float, ...]
    outcome: Mapping[str, Any]
    concrete_scenario: ConcreteScenario


@dataclass
class OnlineMetaTestResult:
    episodes: list[OnlineEpisode]
    inner_transitions: list[InnerTransition]
    outer_rollout: OuterRolloutBuffer


def _inner_learning_blocks(
    transitions: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in transitions:
        if bool(row["info"].get("inner_policy_decision", False)) and current:
            blocks.append(current)
            current = []
        current.append(row)
    if current:
        blocks.append(current)
    return blocks


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
            for component in ("task_structure_encoder", "shared_feature_encoder", "inner_sac")
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
        episode_seed_provider: Callable[[ScenarioMiningTaskSpec, int], int] | None = None,
        initial_latent: torch.Tensor | None = None,
        use_scene_context: bool = True,
        use_latent_context: bool = True,
        inner_gamma: float = 0.99,
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
        if not 0.0 < inner_gamma <= 1.0:
            raise ValueError("inner_gamma must lie in (0, 1]")
        space = mvr_parameter_spaces()[task.functional_scenario]
        tokens, candidates, encoding = self._scene_encoding(task)
        scene_embedding = self.model.encode_task_structure(
            encoding.global_embedding, dict(task.logical_domain_bounds), task.logical_parameter_mask
        )
        continuous_bounds = torch.as_tensor(
            [task.logical_domain_bounds[name] for name in task.logical_domain_bounds],
            dtype=torch.float32,
            device=self.model.device,
        )
        device = self.model.device
        latent, _ = self.model.context_encoder.prior(device=device)
        if initial_latent is not None:
            latent = torch.as_tensor(initial_latent, dtype=torch.float32, device=device).reshape(1, -1)
            if latent.shape[1] != self.model.context_encoder.latent_dim:
                raise ValueError("initial latent dimension does not match the model")
        inner_policy_hash = self._inner_policy_hash()
        evidence_tokens: list[torch.Tensor] = []
        result = OnlineMetaTestResult([], [], OuterRolloutBuffer())
        novelty = NoveltyTracker()
        for index in range(budget):
            episode_index = episode_index_offset + index
            if scene_action_provider is None:
                with torch.no_grad():
                    scene = self.model.universal_scene_policy.sample(
                        scene_embedding, encoding.candidate_embeddings,
                        encoding.candidate_mask, latent, deterministic,
                        continuous_mask=torch.as_tensor(task.logical_parameter_mask, device=device),
                        continuous_bounds=continuous_bounds,
                    )
            else:
                provided = scene_action_provider(task, episode_index, candidates, space)
                provided.validate(space.continuous_dim)
                scene = UniversalSceneAction(
                    expert_index=torch.zeros(1, dtype=torch.long, device=device),
                    candidate_index=torch.tensor([provided.candidate_index], dtype=torch.long, device=device),
                    continuous=torch.as_tensor(provided.continuous, dtype=torch.float32, device=device).unsqueeze(0),
                    log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                    value=torch.zeros(1, dtype=torch.float32, device=device),
                )
            action = NormalizedScenarioAction(
                int(scene.candidate_index.item()),
                tuple(float(value) for value in scene.continuous.squeeze(0).tolist()),
            )
            episode_seed = (
                task.geometry_seed + episode_index
                if episode_seed_provider is None
                else int(episode_seed_provider(task, episode_index))
            )
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
            candidate_embedding = encoding.candidate_embeddings[
                int(scene.candidate_index.item())
            ].unsqueeze(0)
            concrete_input = self.model.concrete_features(
                candidate_embedding, scene.continuous, task.logical_parameter_mask
            )
            def inner_action(state_values: np.ndarray) -> np.ndarray:
                if inner_action_provider is not None:
                    return np.asarray(inner_action_provider(state_values), dtype=np.float32)
                state = torch.as_tensor(state_values, dtype=torch.float32, device=device).reshape(1, -1)
                with torch.no_grad():
                    policy_scene = scene_embedding.unsqueeze(0) if use_scene_context else torch.zeros_like(scene_embedding).unsqueeze(0)
                    policy_latent = latent if use_latent_context else torch.zeros_like(latent)
                    value = self.model.act_inner(
                        state, policy_scene, policy_latent, concrete_input,
                        deterministic=deterministic,
                    )
                return value.squeeze(0).cpu().numpy()

            try:
                rollout = self.runner.rollout(
                    episode,
                    task.functional_scenario,
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
                token = self.model.episode_token_builder(
                    scene_embedding.unsqueeze(0), concrete_input, trajectory, mask,
                    outcome_tensor,
                ).squeeze(0)
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
                torch.as_tensor(task.logical_parameter_mask, dtype=torch.bool),
                continuous_bounds.cpu(),
                latent.squeeze(0).cpu(), scene.expert_index.cpu(), scene.candidate_index.cpu(),
                scene.continuous.squeeze(0).cpu(), scene.log_prob.cpu(),
                scene.value.cpu(), reward, index + 1 == budget,
            ))
            episode_id = f"{task.task_id}:{episode_index}"
            for block in _inner_learning_blocks(rollout.transitions):
                first = block[0]
                last = block[-1]
                result.inner_transitions.append(InnerTransition(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    support_group_id=None,
                    geometry_hash=task.geometry_hash,
                    state=first["state"],
                    action=first["raw_policy_action"],
                    reward=float(sum(
                        inner_gamma ** offset * float(row["reward_inner"])
                        for offset, row in enumerate(block)
                    )),
                    next_state=last["next_state"],
                    done=last["done"],
                    duration_steps=len(block),
                    map_tokens=tokens,
                    interactions=candidates,
                    logical_domain_bounds=dict(task.logical_domain_bounds),
                    logical_parameter_mask=task.logical_parameter_mask,
                    latent=latent.squeeze(0).detach().cpu(),
                    candidate_index=int(scene.candidate_index.item()),
                    continuous=tuple(float(value) for value in scene.continuous.squeeze(0).tolist()),
                ))
            result.episodes.append(OnlineEpisode(
                episode_id, rollout, token.detach().cpu(), latent.detach().cpu().clone(),
                latent_after.detach().cpu().clone(), tokens, candidates,
                dict(task.logical_domain_bounds), task.logical_parameter_mask,
                int(scene.candidate_index.item()), tuple(float(value) for value in scene.continuous.squeeze(0).tolist()), outcome,
                concrete,
            ))
            latent = latent_after
        result.outer_rollout.finish()
        return result
