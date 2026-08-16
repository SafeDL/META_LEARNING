"""No-gradient, episode-balanced few-shot evaluation."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping
import numpy as np
import torch

from .collector import Rollout, collect_episode
from .io import content_hash
from .metrics import summarize
from .observation import OBS_FIELDS
from .task_env import LogicalMergeEnv, freeze_physical_task_descriptor
from .task_representation import (
    INTERACTION_OBSERVATION_INDEXES,
    representation_target,
)
from .support_selection import DYNAMIC_POLICIES, order_support_cases


_INTERVENTION_MASKS = {
    "geometry_descriptor": (
        tuple(range(OBS_FIELDS.index("num_incoming_branches"), len(OBS_FIELDS))),
        "geometry",
    ),
    "interaction_state": (
        tuple(OBS_FIELDS.index(name) for name in (
            "arrival_time_difference", "euclidean_distance", "relative_route_speed",
            "closing_speed", "ttc", "conflict_angle",
        )),
        "interaction",
    ),
    "visible_priority": (
        tuple(OBS_FIELDS.index(name) for name in ("adversary_priority", "sut_priority")),
        "rule",
    ),
}

QUERY_EXECUTION_MODES = {"posterior_mean_deterministic", "posterior_sampled"}
EVALUATION_REGIMES = {
    "meta_validation": "validation_known_logical_type",
    "meta_test_template": "id_known_logical_type",
    "meta_test_logical": "ood_unseen_logical_type",
}


def evaluation_regime(split: str) -> str:
    try:
        return EVALUATION_REGIMES[split]
    except KeyError as error:
        raise ValueError(f"unsupported evaluation split: {split!r}") from error


_CONTEXT_PROTOCOLS = {"fixed_nested"}


def _context_protocol(config: Mapping[str, Any]) -> str:
    protocol = str(config["evaluation"].get("context_protocol", "fixed_nested"))
    if protocol not in _CONTEXT_PROTOCOLS:
        raise ValueError(f"unsupported evaluation context protocol: {protocol}")
    return protocol


def _fixed_episode_context_block(
    rollout: Rollout,
    per_episode: int,
    *,
    base_seed: int,
    task_id: str,
) -> tuple[list[Any], dict[str, Any]]:
    if per_episode <= 0 or not rollout.transitions:
        raise ValueError("a fixed context block requires a positive size and a non-empty rollout")
    sample_seed = int(content_hash({
        "seed": int(base_seed),
        "task_id": str(task_id),
        "case_id": str(rollout.record["case_id"]),
        "episode_id": rollout.episode_id,
        "purpose": "fixed_episode_context_block",
    })[:16], 16)
    rng = np.random.default_rng(sample_seed)
    indexes = np.asarray(rng.choice(
        len(rollout.transitions),
        size=int(per_episode),
        replace=len(rollout.transitions) < int(per_episode),
    )).reshape(-1)
    audit = {
        "episode_id": rollout.episode_id,
        "case_id": str(rollout.record["case_id"]),
        "episode_length": len(rollout.transitions),
        "sample_seed": sample_seed,
        "transition_indexes": [int(index) for index in indexes],
    }
    audit["sample_hash"] = content_hash(audit)
    return [rollout.transitions[int(index)] for index in indexes], audit


def _posterior_context(
    rollouts: list[Rollout],
    fixed_blocks: list[list[Any]],
    fixed_audits: list[dict[str, Any]],
    *,
    total_size: int,
    per_episode: int,
) -> tuple[list[list[Any]], dict[str, Any]]:
    capacity = int(total_size) // int(per_episode)
    if capacity < 1:
        raise ValueError("context_sample_size_eval must hold at least one episode block")
    if len(rollouts) > capacity:
        raise ValueError(
            f"fixed nested context has {len(rollouts)} episodes but capacity is {capacity}; "
            "increase context_sample_size_eval or reduce K"
        )
    if len(fixed_blocks) != len(rollouts) or len(fixed_audits) != len(rollouts):
        raise RuntimeError("fixed context blocks are not aligned with support rollouts")
    context = [list(block) for block in fixed_blocks]
    episode_audits = [dict(row) for row in fixed_audits]
    hashes = [str(row["sample_hash"]) for row in episode_audits]
    return context, {
        "context_episode_count": len(context),
        "context_transition_count": int(sum(len(group) for group in context)),
        "context_episode_sample_hashes": hashes,
        "context_sample_hash": content_hash(hashes),
        "context_episode_samples": episode_audits,
    }


def _add_posterior_deltas(results: dict[str, Any]) -> None:
    ordered = sorted(results, key=int)
    if not ordered:
        return
    reference_mu = np.asarray(results[ordered[0]]["posterior_mean"], dtype=float)
    reference_log_var = np.asarray(results[ordered[0]]["posterior_log_variance"], dtype=float)
    previous_mu = reference_mu
    previous_log_var = reference_log_var
    for key in ordered:
        row = results[key]
        mu = np.asarray(row["posterior_mean"], dtype=float)
        log_var = np.asarray(row["posterior_log_variance"], dtype=float)
        row["posterior_change"] = {
            "mean_l2_from_k0": float(np.linalg.norm(mu - reference_mu)),
            "mean_l2_from_previous_k": float(np.linalg.norm(mu - previous_mu)),
            "log_variance_l2_from_k0": float(np.linalg.norm(log_var - reference_log_var)),
            "log_variance_l2_from_previous_k": float(np.linalg.norm(log_var - previous_log_var)),
        }
        previous_mu = mu
        previous_log_var = log_var


def _initial_observations(env: LogicalMergeEnv, cases: list[dict[str, Any]]) -> np.ndarray:
    """Read candidate initial observations without taking an environment step."""
    observations = []
    try:
        for case in cases:
            observation, _ = env.reset(options={"case": case})
            observations.append(np.asarray(observation, dtype=np.float32))
    finally:
        env.close()
    return np.stack(observations)


def _posterior_action_disagreement(agent: Any, observations: np.ndarray, mu: torch.Tensor,
                                   log_var: torch.Tensor, *, seed: int, samples: int = 16,
                                   descriptor: Any = None, posterior_version: int = 0) -> list[float]:
    """Score initial states by deterministic-action variance across posterior samples.

    This is a decision-uncertainty proxy, not an estimate of information gain:
    no candidate rollout, reward, termination signal, or query observation is
    consumed here.
    """
    if observations.ndim != 2 or len(observations) == 0:
        raise ValueError("candidate observations must be a non-empty matrix")
    if samples < 2:
        raise ValueError("posterior action disagreement needs at least two latent samples")
    noise = np.random.default_rng(int(seed)).standard_normal((samples, int(mu.shape[-1]))).astype(np.float32)
    count = len(observations) * samples
    latent = mu.detach().repeat(count, 1)
    latent = latent + torch.as_tensor(noise, device=agent.device).repeat(len(observations), 1) * torch.exp(0.5 * log_var.detach()).repeat(count, 1)
    observation = torch.as_tensor(observations, dtype=torch.float32, device=agent.device).repeat_interleave(samples, dim=0)
    route = agent.compute_route(descriptor, mu, log_var, posterior_version) if descriptor is not None else None
    with torch.no_grad():
        actions = agent.act(observation, latent, True, route).reshape(len(observations), samples, -1)
        scores = actions.var(dim=1, unbiased=False).sum(dim=-1)
    return [float(score) for score in scores.detach().cpu()]


def _mask_context_fields(context: list[list[Any]], indexes: tuple[int, ...]) -> list[list[Any]]:
    """Return a support-context copy with declared observable fields set to zero."""
    result: list[list[Any]] = []
    for episode in context:
        masked_episode = []
        for transition in episode:
            observation = np.asarray(transition.obs, dtype=np.float32).copy()
            next_observation = np.asarray(transition.next_obs, dtype=np.float32).copy()
            observation[list(indexes)] = 0.0
            next_observation[list(indexes)] = 0.0
            masked_episode.append(replace(transition, obs=observation, next_obs=next_observation))
        result.append(masked_episode)
    return result


def _posterior_block_l2_shift(agent: Any, reference: torch.Tensor, changed: torch.Tensor) -> dict[str, float]:
    """Measure posterior-mean sensitivity for the declared latent blocks."""
    blocks = (
        ("geometry", int(agent.geometry_dim)),
        ("interaction", int(agent.interaction_dim)),
        ("rule", int(agent.rule_dim)),
    )
    start = 0; result: dict[str, float] = {}
    for name, width in blocks:
        result[name] = float(torch.linalg.vector_norm(changed[:, start:start + width] - reference[:, start:start + width]).item())
        start += width
    return result


def _expert_action_audit(
    agent: Any,
    task: Any,
    config: Mapping[str, Any],
    cases: list[dict[str, Any]],
    latent: torch.Tensor,
) -> dict[str, Any]:
    """Compare anonymous expert actions on query initial states post hoc."""
    env = LogicalMergeEnv(task, config, cases)
    observations = _initial_observations(env, cases)
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    expanded_latent = latent.detach().expand(len(tensor), -1)
    with torch.no_grad():
        actions = agent.expert_action_means(tensor, expanded_latent)
    pairwise = {}
    for left in range(actions.shape[1]):
        for right in range(left + 1, actions.shape[1]):
            distance = torch.linalg.vector_norm(actions[:, left] - actions[:, right], dim=-1)
            pairwise[f"{left}:{right}"] = float(distance.mean())
    return {
        "posthoc_only": True,
        "uses_query_initial_observations": True,
        "uses_query_outcomes": False,
        "expert_action_means": actions.detach().cpu().tolist(),
        "pairwise_mean_action_l2": pairwise,
    }


def evaluate_fewshot(
    agent: Any,
    config: Mapping[str, Any],
    tasks: list[Any],
    casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    split: str,
    query_cases_per_task: int | None = None,
    provenance: Mapping[str, Any] | None = None,
    support_selection: str = "fixed",
    adaptation_mode: str = "posterior_sampled",
    query_latent_mode: str = "adaptive",
    query_route_mode: str = "adaptive",
    knockout_expert: int | None = None,
    mechanism_audit: bool = False,
    query_execution_mode: str = "posterior_mean_deterministic",
) -> dict[str, Any]:
    if adaptation_mode not in {"posterior_sampled", "posterior_deterministic", "no_context"}:
        raise ValueError(f"unsupported adaptation mode: {adaptation_mode}")
    if query_latent_mode not in {"adaptive", "frozen_prior"}:
        raise ValueError(f"unsupported query_latent_mode: {query_latent_mode}")
    if query_route_mode not in {"adaptive", "frozen_prior", "uniform"}:
        raise ValueError(f"unsupported query_route_mode: {query_route_mode}")
    if query_execution_mode not in QUERY_EXECUTION_MODES:
        raise ValueError(f"unsupported query_execution_mode: {query_execution_mode}")
    regime = evaluation_regime(split)
    if (query_route_mode != "adaptive" or knockout_expert is not None) and agent.actor_architecture != "posterior_routed_moe":
        raise ValueError("query route interventions require a MoE actor")
    device = agent.device
    shots = sorted(set(int(shot) for shot in config["evaluation"]["shots"]))
    if not shots or shots[0] != 0:
        raise ValueError("few-shot evaluation must include K=0")
    query_limit = query_cases_per_task or int(config["evaluation"]["query_cases_per_task"])
    before = agent.parameter_hash()
    module_hashes_before = agent.module_hashes()
    all_results: dict[str, Any] = {}
    selection_by_task: dict[str, Any] = {}
    support_key, query_key = (
        ("validation_support", "validation_query")
        if split == "meta_validation"
        else ("test_support", "test_query")
    )
    base_seed = int(config["evaluation"]["context_sampling_seed"])
    protocol = _context_protocol(config)
    total_size = int(config["pearl"]["context_sample_size_eval"])
    per_episode = int(config["pearl"]["context_transitions_per_episode"])
    capacity = total_size // per_episode
    if protocol == "fixed_nested" and max(shots) > capacity:
        raise ValueError(f"K={max(shots)} exceeds fixed nested context capacity {capacity}")
    output_provenance = dict(provenance or {})
    for task in tasks:
        book = casebooks[task.task_id]
        descriptor = None
        if agent.actor_architecture == "posterior_routed_moe":
            descriptor = freeze_physical_task_descriptor(task, config, book[support_key])
        selection_seed = int(content_hash({
            "seed": base_seed, "task": task.task_id, "policy": support_selection,
        })[:16], 16)
        if support_selection in DYNAMIC_POLICIES:
            ordered_support, static_selection = order_support_cases(
                book[support_key], "fixed", seed=selection_seed,
            )
            selection = {
                **static_selection,
                "policy": support_selection,
                "seed": selection_seed,
                "selected_case_ids": [],
                "selection_rounds": [],
                "score": "posterior deterministic-action disagreement on initial observations",
                "uses_initial_observations": True,
                "uses_past_support_rollout_outcomes": True,
                "uses_unexecuted_rollout_outcomes": False,
                "uses_query_cases": False,
                "uses_hidden_rules": False,
                "initial_observation_environment_steps": 0,
                "initial_observation_count": 0,
            }
        else:
            ordered_support, selection = order_support_cases(
                book[support_key], support_selection, seed=selection_seed,
            )
        if max(shots) > len(ordered_support):
            raise ValueError(f"task {task.task_id} lacks support cases for K={max(shots)}")
        env = LogicalMergeEnv(task, config, book[query_key])
        results: dict[str, Any] = {}
        support_rollouts: list[Rollout] = []
        support_episode_lengths: list[int] = []
        fixed_blocks: list[list[Any]] = []
        fixed_audits: list[dict[str, Any]] = []
        expert_audit_latents: dict[str, torch.Tensor] = {}
        context: list[list[Any]] = []
        context_audit = {
            "context_episode_count": 0,
            "context_transition_count": 0,
            "context_episode_sample_hashes": [],
            "context_sample_hash": content_hash([]),
            "context_episode_samples": [],
        }
        try:
            prior_mu, prior_log_var = agent.prior()
            mu, log_var = prior_mu, prior_log_var
            prior_route = agent.compute_route(
                descriptor, prior_mu, prior_log_var, 0
            ) if descriptor is not None else None
            for shot in range(max(shots) + 1):
                route_mu = prior_mu if adaptation_mode == "no_context" else mu
                route_log_var = prior_log_var if adaptation_mode == "no_context" else log_var
                route = agent.compute_route(
                    descriptor, route_mu, route_log_var, shot
                ) if descriptor is not None else None
                if shot in shots:
                    query_cases = book[query_key][:query_limit]
                    policy_mu = (
                        prior_mu
                        if adaptation_mode == "no_context" or query_latent_mode == "frozen_prior"
                        else mu
                    )
                    policy_log_var = (
                        prior_log_var
                        if adaptation_mode == "no_context" or query_latent_mode == "frozen_prior"
                        else log_var
                    )
                    query_route = route
                    if query_route_mode == "frozen_prior":
                        query_route = agent.intervene_route(
                            prior_route, posterior_version=shot, mode="frozen_prior"
                        )
                    elif query_route_mode == "uniform":
                        query_route = agent.intervene_route(
                            route, posterior_version=shot, mode="uniform"
                        )
                    if knockout_expert is not None:
                        query_route = agent.intervene_route(
                            query_route,
                            posterior_version=shot,
                            mode="expert_knockout",
                            expert_index=knockout_expert,
                        )
                    query_latents = []
                    queries = []
                    for index, case in enumerate(query_cases):
                        if query_execution_mode == "posterior_mean_deterministic":
                            query_z = policy_mu
                            collection_mode = "deterministic_query"
                        else:
                            sample_seed = int(content_hash({
                                "base_seed": base_seed,
                                "task_id": task.task_id,
                                "shot": shot,
                                "case_id": str(case["case_id"]),
                                "purpose": "posterior_sampled_query",
                            })[:16], 16)
                            query_z = agent.sample_latent_seeded(policy_mu, policy_log_var, sample_seed)
                            collection_mode = "posterior_sampled_query"
                        query_latents.append(content_hash(query_z.detach().cpu().tolist()))
                        queries.append(collect_episode(
                            env,
                            task,
                            case,
                            agent,
                            query_z,
                            collection_mode,
                            device,
                            episode_id=f"{task.task_id}:query:{shot}:{index}",
                            posterior_version=shot,
                            route_context=query_route,
                        ))
                    records = [rollout.record for rollout in queries]
                    results[str(shot)] = {
                        "summary": summarize(
                            records,
                            case_metadata={str(case["case_id"]): case for case in query_cases},
                        ),
                        "records": records,
                        "posterior_mean": mu.detach().cpu().tolist(),
                        "posterior_log_variance": log_var.detach().cpu().tolist(),
                        "posterior_variance": torch.exp(log_var).detach().cpu().tolist(),
                        "posterior_used_for_policy": (
                            adaptation_mode != "no_context" and query_latent_mode != "frozen_prior"
                        ),
                        "query_execution_mode": query_execution_mode,
                        "query_latent_hashes": query_latents,
                        "router_audit": None if query_route is None else {
                            **query_route.audit_dict(), "parameter_hash": before,
                        },
                        "query_route_hashes": [record.get("route_hash") for record in records],
                        "query_route_consistent": len({record.get("route_hash") for record in records}) == 1,
                        **context_audit,
                        "support_episode_lengths": list(support_episode_lengths),
                        "support_environment_steps": int(sum(support_episode_lengths)),
                        "support_case_ids": [str(case["case_id"]) for case in ordered_support[:shot]],
                        "expert_action_audit": None,
                    }
                    if mechanism_audit and agent.actor_architecture == "posterior_routed_moe":
                        expert_audit_latents[str(shot)] = policy_mu.detach().clone()
                if shot == max(shots):
                    break
                if support_selection in DYNAMIC_POLICIES:
                    remaining = ordered_support[shot:]
                    candidate_ids = [str(item["case_id"]) for item in remaining]
                    initial = _initial_observations(env, remaining)
                    scores = _posterior_action_disagreement(
                        agent,
                        initial,
                        mu,
                        log_var,
                        seed=int(content_hash({"seed": selection_seed, "round": shot})[:16], 16),
                        descriptor=descriptor,
                        posterior_version=shot,
                    )
                    score_by_id = {
                        str(item["case_id"]): float(score)
                        for item, score in zip(remaining, scores)
                    }
                    chosen_offset = max(
                        range(len(remaining)), key=lambda index: (scores[index], -index),
                    )
                    case = remaining.pop(chosen_offset)
                    ordered_support[shot:] = [case, *remaining]
                    selection["selected_case_ids"].append(str(case["case_id"]))
                    selection["initial_observation_count"] += len(candidate_ids)
                    selection["selection_rounds"].append({
                        "round": shot,
                        "candidate_case_ids": candidate_ids,
                        "scores": score_by_id,
                        "selected_case_id": str(case["case_id"]),
                        "initial_observation_count": len(candidate_ids),
                        "posterior_variance_mean": float(torch.exp(log_var).mean().detach()),
                    })
                else:
                    case = ordered_support[shot]
                if adaptation_mode == "no_context":
                    support_z = agent.sample_latent(prior_mu, prior_log_var)
                    mode = "prior_support"
                elif adaptation_mode == "posterior_deterministic":
                    support_z = mu
                    mode = "prior_support" if shot == 0 else "posterior_rollout"
                else:
                    support_z = agent.sample_latent(mu, log_var)
                    mode = "prior_support" if shot == 0 else "posterior_rollout"
                rollout = collect_episode(
                    env,
                    task,
                    case,
                    agent,
                    support_z,
                    mode,
                    device,
                    episode_id=f"{task.task_id}:support:{shot}",
                    posterior_version=shot,
                    route_context=route,
                )
                support_rollouts.append(rollout)
                support_episode_lengths.append(len(rollout.transitions))
                if protocol == "fixed_nested":
                    block, audit = _fixed_episode_context_block(
                        rollout,
                        per_episode,
                        base_seed=base_seed,
                        task_id=task.task_id,
                    )
                    fixed_blocks.append(block)
                    fixed_audits.append(audit)
                context, context_audit = _posterior_context(
                    support_rollouts,
                    fixed_blocks,
                    fixed_audits,
                    total_size=total_size,
                    per_episode=per_episode,
                )
                with torch.no_grad():
                    mu, log_var = agent.infer_posterior([context])
        finally:
            env.close()
        # MetaDrive owns a single global engine, so post-hoc audit environments
        # must only be created after the task's rollout environment is closed.
        for shot, latent in expert_audit_latents.items():
            results[shot]["expert_action_audit"] = _expert_action_audit(
                agent, task, config, book[query_key][:query_limit], latent,
            )
        _add_posterior_deltas(results)
        all_results[task.task_id] = results
        selection_by_task[task.task_id] = selection
    after = agent.parameter_hash()
    module_hashes_after = agent.module_hashes()
    if before != after or module_hashes_before != module_hashes_after:
        raise RuntimeError("few-shot evaluation changed PEARL parameters")
    return {
        "split": split,
        "evaluation_regime": regime,
        "query_execution_mode": query_execution_mode,
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "module_hashes_before": module_hashes_before,
        "module_hashes_after": module_hashes_after,
        "no_gradient_adaptation": True,
        "no_topology_ablation": bool(config.get("ablation", {}).get("no_topology", False)),
        "adaptation_mode": adaptation_mode,
        "mechanism_intervention": {
            "query_latent_mode": query_latent_mode,
            "query_route_mode": query_route_mode,
            "knockout_expert": knockout_expert,
            "support_collection": "adaptive_unintervened",
            "mechanism_audit": bool(mechanism_audit),
        },
        "support_selection": support_selection,
        "support_selection_by_task": selection_by_task,
        "context_protocol": {
            "name": protocol,
            "sample_size": total_size,
            "transitions_per_episode": per_episode,
            "episode_capacity": capacity,
            "seed": base_seed,
        },
        "provenance": output_provenance,
        "tasks": all_results,
    }


def infer_support_posteriors(agent: Any, config: Mapping[str, Any], tasks: list[Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
                             split: str, shots: list[int] | None = None,
                             provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Infer task posteriors from support only; never create query rollouts."""
    requested = sorted(set(int(shot) for shot in (shots or config["evaluation"]["shots"])))
    if not requested or requested[0] < 0:
        raise ValueError("posterior diagnostic shots must be non-negative")
    support_key = "validation_support" if split == "meta_validation" else "test_support"
    base_seed = int(config["evaluation"]["context_sampling_seed"])
    protocol = _context_protocol(config)
    total_size = int(config["pearl"]["context_sample_size_eval"])
    per_episode = int(config["pearl"]["context_transitions_per_episode"])
    capacity = total_size // per_episode
    if protocol == "fixed_nested" and max(requested) > capacity:
        raise ValueError(f"K={max(requested)} exceeds fixed nested context capacity {capacity}")
    output_provenance = dict(provenance or {})
    before = agent.parameter_hash()
    module_hashes_before = agent.module_hashes()
    all_results: dict[str, Any] = {}
    for task in tasks:
        book = casebooks[task.task_id]
        descriptor = None
        if agent.actor_architecture == "posterior_routed_moe":
            descriptor = freeze_physical_task_descriptor(task, config, book[support_key])
        if max(requested) > len(book[support_key]):
            raise ValueError(f"task {task.task_id} lacks support cases for K={max(requested)}")
        env = LogicalMergeEnv(task, config, book[support_key])
        rollouts: list[Rollout] = []
        episode_lengths: list[int] = []
        support_records: list[dict[str, object]] = []
        fixed_blocks: list[list[Any]] = []
        fixed_audits: list[dict[str, Any]] = []
        context_audit = {
            "context_episode_count": 0,
            "context_transition_count": 0,
            "context_episode_sample_hashes": [],
            "context_sample_hash": content_hash([]),
            "context_episode_samples": [],
        }
        task_results: dict[str, Any] = {}
        try:
            mu, log_var = agent.prior()
            for shot in range(max(requested) + 1):
                route = agent.compute_route(
                    descriptor, mu, log_var, shot
                ) if descriptor is not None else None
                if shot in requested:
                    task_results[str(shot)] = {
                        "posterior_mean": mu.detach().cpu().tolist(),
                        "posterior_log_variance": log_var.detach().cpu().tolist(),
                        "posterior_variance": torch.exp(log_var).detach().cpu().tolist(),
                        "router_audit": None if route is None else {
                            **route.audit_dict(), "parameter_hash": before,
                        },
                        **context_audit,
                        "support_case_ids": [str(case["case_id"]) for case in book[support_key][:shot]],
                        "support_episode_lengths": list(episode_lengths),
                        "support_episode_records": list(support_records),
                        "support_environment_steps": int(sum(episode_lengths)),
                    }
                if shot == max(requested):
                    break
                case = book[support_key][shot]
                rollout = collect_episode(
                    env, task, case, agent, agent.sample_latent(mu, log_var),
                    "prior_support" if shot == 0 else "posterior_rollout", agent.device,
                    episode_id=f"{task.task_id}:posterior_support:{shot}",
                    posterior_version=shot, route_context=route,
                )
                rollouts.append(rollout)
                episode_lengths.append(len(rollout.transitions))
                support_records.append(dict(rollout.record))
                if protocol == "fixed_nested":
                    block, audit = _fixed_episode_context_block(
                        rollout,
                        per_episode,
                        base_seed=base_seed,
                        task_id=task.task_id,
                    )
                    fixed_blocks.append(block)
                    fixed_audits.append(audit)
                context, context_audit = _posterior_context(
                    rollouts,
                    fixed_blocks,
                    fixed_audits,
                    total_size=total_size,
                    per_episode=per_episode,
                )
                with torch.no_grad():
                    mu, log_var = agent.infer_posterior([context])
        finally:
            env.close()
        _add_posterior_deltas(task_results)
        all_results[task.task_id] = task_results
    after = agent.parameter_hash()
    module_hashes_after = agent.module_hashes()
    if before != after or module_hashes_before != module_hashes_after:
        raise RuntimeError("support posterior diagnostic changed model parameters")
    return {
        "schema": "logical_merge_support_posterior_diagnostic_v1", "split": split,
        "taskbook_hash": output_provenance.get("taskbook_hash"),
        "uses_query_cases": False, "parameter_hash_before": before, "parameter_hash_after": after,
        "module_hashes_before": module_hashes_before, "module_hashes_after": module_hashes_after,
        "no_gradient_adaptation": True,
        "context_protocol": {
            "name": protocol, "sample_size": total_size,
            "transitions_per_episode": per_episode, "episode_capacity": capacity,
            "seed": base_seed,
        },
        "provenance": output_provenance, "tasks": all_results,
    }


def audit_task_representation(agent: Any, config: Mapping[str, Any], tasks: list[Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
                              split: str, shots: list[int] | None = None,
                              provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit optional factor heads using only support rollouts.

    Geometry and entry-order labels are read only after inference to score the
    decoder.  They are never passed to the agent here, so this is a post-hoc
    semantic audit rather than an adaptation input or a query-performance
    measurement.
    """
    if not bool(getattr(agent, "disentangled", False)):
        raise ValueError("task-representation audit requires a disentangled PEARL checkpoint")
    requested = sorted(set(int(shot) for shot in (shots or config["evaluation"]["shots"])))
    if not requested or requested[0] < 1:
        raise ValueError("task-representation audit requires positive support-episode counts")
    support_key = "validation_support" if split == "meta_validation" else "test_support"
    base_seed = int(config["evaluation"]["context_sampling_seed"])
    protocol = _context_protocol(config)
    total_size = int(config["pearl"]["context_sample_size_eval"])
    per_episode = int(config["pearl"]["context_transitions_per_episode"])
    capacity = total_size // per_episode
    if protocol == "fixed_nested" and max(requested) > capacity:
        raise ValueError(f"K={max(requested)} exceeds fixed nested context capacity {capacity}")
    before = agent.parameter_hash(); all_results: dict[str, Any] = {}
    aggregate: dict[str, list[float]] = {
        str(shot): [] for shot in requested
    }
    interaction_aggregate: dict[str, list[float]] = {
        str(shot): [] for shot in requested
    }
    rule_log_losses: dict[str, list[float]] = {str(shot): [] for shot in requested}
    rule_correct: dict[str, list[float]] = {str(shot): [] for shot in requested}
    intervention_aggregate: dict[str, dict[str, dict[str, list[float]]]] = {
        str(shot): {name: {block: [] for block in ("geometry", "interaction", "rule")} for name in _INTERVENTION_MASKS}
        for shot in requested
    }
    for task in tasks:
        book = casebooks[task.task_id]
        descriptor = None
        if agent.actor_architecture == "posterior_routed_moe":
            descriptor = freeze_physical_task_descriptor(task, config, book[support_key])
        if max(requested) > len(book[support_key]):
            raise ValueError(f"task {task.task_id} lacks support cases for K={max(requested)}")
        posthoc_target = representation_target(task)
        env = LogicalMergeEnv(task, config, book[support_key])
        rollouts: list[Rollout] = []
        episode_lengths: list[int] = []
        fixed_blocks: list[list[Any]] = []
        fixed_audits: list[dict[str, Any]] = []
        task_results: dict[str, Any] = {}
        try:
            mu, log_var = agent.prior()
            context: list[list[Any]] | None = None
            for shot in range(1, max(requested) + 1):
                case = book[support_key][shot - 1]
                route = agent.compute_route(
                    descriptor, mu, log_var, shot - 1
                ) if descriptor is not None else None
                rollout = collect_episode(
                    env, task, case, agent, agent.sample_latent(mu, log_var),
                    "prior_support" if shot == 1 else "posterior_rollout", agent.device,
                    episode_id=f"{task.task_id}:representation_support:{shot - 1}",
                    posterior_version=shot - 1,
                    route_context=route,
                )
                rollouts.append(rollout)
                episode_lengths.append(len(rollout.transitions))
                if protocol == "fixed_nested":
                    block, audit = _fixed_episode_context_block(
                        rollout,
                        per_episode,
                        base_seed=base_seed,
                        task_id=task.task_id,
                    )
                    fixed_blocks.append(block)
                    fixed_audits.append(audit)
                context, context_audit = _posterior_context(
                    rollouts,
                    fixed_blocks,
                    fixed_audits,
                    total_size=total_size,
                    per_episode=per_episode,
                )
                with torch.no_grad():
                    mu, log_var = agent.infer_posterior([context])
                if shot not in requested:
                    continue
                with torch.no_grad():
                    decoded = agent.decode_task_representation(mu)
                    context_tensor = agent.context_tensor([context])
                    interaction_target = context_tensor[..., list(INTERACTION_OBSERVATION_INDEXES)].mean(dim=(1, 2))
                    geometry_target = torch.as_tensor(posthoc_target["geometry"][None], dtype=torch.float32, device=agent.device)
                    rule_target = float(np.asarray(posthoc_target["entry_order"]))
                    geometry_mse = torch.mean((decoded["geometry"] - geometry_target) ** 2)
                    interaction_mse = torch.mean((decoded["interaction"] - interaction_target) ** 2)
                    rule_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        decoded["entry_order_logit"], torch.full_like(decoded["entry_order_logit"], rule_target),
                    )
                    rule_probability = float(decoded["entry_order_probability"].item())
                    interventions: dict[str, Any] = {}
                    for name, (indexes, target_block) in _INTERVENTION_MASKS.items():
                        masked_mu, _ = agent.infer_posterior([_mask_context_fields(context, indexes)])
                        shifts = _posterior_block_l2_shift(agent, mu, masked_mu)
                        for block, value in shifts.items():
                            intervention_aggregate[str(shot)][name][block].append(value)
                        interventions[name] = {
                            "masked_observation_fields": [OBS_FIELDS[index] for index in indexes],
                            "expected_latent_block": target_block,
                            "posterior_mean_block_l2_shift": shifts,
                        }
                key = str(shot)
                geometry_value, interaction_value, rule_value = (float(geometry_mse), float(interaction_mse), float(rule_bce))
                aggregate[key].append(geometry_value); interaction_aggregate[key].append(interaction_value); rule_log_losses[key].append(rule_value)
                rule_correct[key].append(float((rule_probability >= 0.5) == bool(rule_target)))
                task_results[key] = {
                    "posterior_mean": mu.detach().cpu().tolist(),
                    "posterior_log_variance": log_var.detach().cpu().tolist(),
                    "posterior_variance": torch.exp(log_var).detach().cpu().tolist(),
                    **context_audit,
                    "support_case_ids": [str(item["case_id"]) for item in book[support_key][:shot]],
                    "support_environment_steps": int(sum(episode_lengths)),
                    "geometry_prediction": decoded["geometry"].detach().cpu().tolist(),
                    "geometry_target_for_posthoc_audit": geometry_target.detach().cpu().tolist(),
                    "interaction_prediction": decoded["interaction"].detach().cpu().tolist(),
                    "interaction_target_from_support": interaction_target.detach().cpu().tolist(),
                    "entry_order_probability": rule_probability,
                    "entry_order_target_for_posthoc_audit": rule_target,
                    "geometry_mse": geometry_value,
                    "interaction_mse": interaction_value,
                    "entry_order_bce": rule_value,
                    "entry_order_correct": bool(rule_correct[key][-1]),
                    "router_audit": None if route is None else {
                        **route.audit_dict(), "parameter_hash": before,
                    },
                    "observation_mask_interventions": interventions,
                }
        finally:
            env.close()
        all_results[task.task_id] = task_results
    after = agent.parameter_hash()
    if before != after:
        raise RuntimeError("task-representation audit changed model parameters")
    summary = {}
    for key in (str(shot) for shot in requested):
        intervention_summary: dict[str, Any] = {}
        for name, (_, target_block) in _INTERVENTION_MASKS.items():
            means = {block: float(np.mean(values)) for block, values in intervention_aggregate[key][name].items()}
            total = float(sum(means.values()))
            intervention_summary[name] = {
                "expected_latent_block": target_block,
                "mean_posterior_mean_block_l2_shift": means,
                "expected_block_shift_share": None if total <= 0.0 else float(means[target_block] / total),
            }
        summary[key] = {
            "task_count": len(aggregate[key]),
            "mean_geometry_mse": float(np.mean(aggregate[key])),
            "mean_interaction_mse": float(np.mean(interaction_aggregate[key])),
            "mean_entry_order_bce": float(np.mean(rule_log_losses[key])),
            "entry_order_accuracy": float(np.mean(rule_correct[key])),
            "observation_mask_intervention_sensitivity": intervention_summary,
        }
    return {
        "schema": "logical_merge_task_representation_audit_v1",
        "split": split,
        "uses_query_cases": False,
        "uses_task_id_or_hash_as_model_input": False,
        "uses_hidden_rule_labels_for_posthoc_audit": True,
        "intervention_protocol": {
            "kind": "support-context observable-field masking",
            "uses_query_cases": False,
            "uses_counterfactual_environment_rollouts": False,
            "uses_hidden_rule_labels_as_intervention_input": False,
            "interpretation": "posterior sensitivity diagnostic, not causal factor-disentanglement proof",
        },
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "no_gradient_adaptation": True,
        "context_protocol": {
            "name": protocol,
            "sample_size": total_size,
            "transitions_per_episode": per_episode,
            "episode_capacity": capacity,
            "seed": base_seed,
        },
        "provenance": dict(provenance or {}),
        "summary": summary,
        "tasks": all_results,
    }


def compact_fewshot_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep reportable few-shot metrics while dropping per-episode records."""
    tasks: dict[str, Any] = {}
    for task_id, shots in result["tasks"].items():
        tasks[task_id] = {
            shot: {
                key: value[key]
                for key in (
                    "summary",
                    "posterior_mean",
                    "posterior_log_variance",
                    "posterior_variance",
                    "posterior_change",
                    "posterior_used_for_policy",
                    "context_episode_count",
                    "context_transition_count",
                    "context_episode_sample_hashes",
                    "context_sample_hash",
                    "context_episode_samples",
                    "support_episode_lengths",
                    "support_environment_steps",
                    "support_case_ids",
                    "router_audit",
                    "query_route_hashes",
                    "query_route_consistent",
                    "query_execution_mode",
                    "query_latent_hashes",
                    "expert_action_audit",
                )
                if key in value
            }
            for shot, value in shots.items()
        }
    compact = {
        key: result[key]
        for key in (
            "split",
            "parameter_hash_before",
            "parameter_hash_after",
            "module_hashes_before",
            "module_hashes_after",
            "no_gradient_adaptation",
            "no_topology_ablation",
            "adaptation_mode",
            "evaluation_regime",
            "query_execution_mode",
            "context_protocol",
            "mechanism_intervention",
            "provenance",
        )
        if key in result
    } | {"tasks": tasks}
    if "support_selection" in result:
        compact["support_selection"] = result["support_selection"]
        compact["support_selection_by_task"] = result["support_selection_by_task"]
    return compact


def validation_score(result: Mapping[str, Any], shot: int = 5) -> tuple[float, float, float, float, float]:
    """Few-shot-sensitive lexicographic checkpoint score."""
    tasks = list(result["tasks"].values())
    if not tasks:
        raise ValueError("validation result has no task summaries")
    strict = lambda task, key: float(task[str(key)]["summary"]["valid_critical_strict_rate"])
    target_shot = str(shot)
    strict_at = float(np.mean([float(task[target_shot]["summary"]["valid_critical_strict_rate"]) for task in tasks]))
    aucs = []
    for task in tasks:
        xs = np.asarray(sorted(int(key) for key in task), dtype=float)
        ys = np.asarray([strict(task, int(key)) for key in xs], dtype=float)
        aucs.append(float(np.trapz(ys, xs) / max(float(xs[-1] - xs[0]), 1.0)))
    gain = float(np.mean([strict(task, shot) - strict(task, 0) for task in tasks]))
    invalid = float(np.mean([float(task[target_shot]["summary"]["invalid_rate"]) for task in tasks]))
    ttc = float(np.mean([float(task[target_shot]["summary"]["median_min_ttc"]) for task in tasks]))
    return strict_at, float(np.mean(aucs)), gain, -invalid, -ttc
