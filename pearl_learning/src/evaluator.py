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
from .task_env import LogicalMergeEnv
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


def _sample_episode_context(rollouts: list[Rollout], total_size: int, per_episode: int, rng: np.random.Generator) -> list[list[Any]]:
    if not rollouts:
        raise ValueError("cannot infer a posterior without a support episode")
    count = min(len(rollouts), max(1, int(total_size) // int(per_episode)))
    indexes = rng.choice(len(rollouts), size=count, replace=False)
    groups = []
    for index in np.asarray(indexes).reshape(-1):
        rows = rollouts[int(index)].transitions
        chosen = rng.choice(len(rows), size=int(per_episode), replace=len(rows) < int(per_episode))
        groups.append([rows[int(item)] for item in np.asarray(chosen).reshape(-1)])
    return groups


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
                                   log_var: torch.Tensor, *, seed: int, samples: int = 16) -> list[float]:
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
    with torch.no_grad():
        mean, _ = agent.actor(observation, latent)
        actions = torch.tanh(mean).reshape(len(observations), samples, -1)
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


def evaluate_fewshot(agent: Any, config: Mapping[str, Any], tasks: list[Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
                     split: str, query_cases_per_task: int | None = None,
                     provenance: Mapping[str, Any] | None = None,
                     support_selection: str = "fixed") -> dict[str, Any]:
    device = agent.device
    shots = list(config["evaluation"]["shots"])
    query_limit = query_cases_per_task or int(config["evaluation"]["query_cases_per_task"])
    before = agent.parameter_hash(); all_results: dict[str, Any] = {}; selection_by_task: dict[str, Any] = {}
    support_key, query_key = ("validation_support", "validation_query") if split == "meta_validation" else ("test_support", "test_query")
    base_seed = int(config["evaluation"]["context_sampling_seed"])
    output_provenance = dict(provenance or {})
    for task in tasks:
        book = casebooks[task.task_id]
        selection_seed = int(content_hash({"seed": base_seed, "task": task.task_id, "policy": support_selection})[:16], 16)
        if support_selection in DYNAMIC_POLICIES:
            # Validate the frozen pool through the same metadata path while
            # leaving its order undecided until each support posterior exists.
            ordered_support, static_selection = order_support_cases(book[support_key], "fixed", seed=selection_seed)
            selection = {
                **static_selection, "policy": support_selection, "seed": selection_seed,
                "selected_case_ids": [], "selection_rounds": [],
                "score": "posterior deterministic-action disagreement on initial observations",
                "uses_initial_observations": True, "uses_past_support_rollout_outcomes": True,
                "uses_unexecuted_rollout_outcomes": False, "uses_query_cases": False,
                "uses_hidden_rules": False, "initial_observation_environment_steps": 0,
                "initial_observation_count": 0,
            }
        else:
            ordered_support, selection = order_support_cases(book[support_key], support_selection, seed=selection_seed)
        if max(shots) > len(ordered_support):
            raise ValueError(f"task {task.task_id} lacks support cases for K={max(shots)}")
        env = LogicalMergeEnv(task, config, book[query_key])
        results: dict[str, Any] = {}
        support_rollouts: list[Rollout] = []
        support_episode_lengths: list[int] = []
        try:
            mu, log_var = agent.prior()
            for shot in range(max(shots) + 1):
                if shot in shots:
                    query_cases = book[query_key][:query_limit]
                    queries = [collect_episode(env, task, case, agent, mu, "deterministic_query", device, episode_id=f"{task.task_id}:query:{shot}:{index}", posterior_version=shot) for index, case in enumerate(query_cases)]
                    records = [rollout.record for rollout in queries]
                    posterior_mean = mu.detach().cpu().tolist()
                    results[str(shot)] = {
                        "summary": summarize(records, case_metadata={str(case["case_id"]): case for case in query_cases}), "records": records, "posterior_mean": posterior_mean,
                        "posterior_variance": torch.exp(log_var).detach().cpu().tolist(), "context_episode_count": len(support_rollouts),
                        "support_episode_lengths": list(support_episode_lengths),
                        "support_environment_steps": int(sum(support_episode_lengths)),
                        "support_case_ids": [str(case["case_id"]) for case in ordered_support[:shot]],
                    }
                if shot == max(shots):
                    break
                if support_selection in DYNAMIC_POLICIES:
                    remaining = ordered_support[shot:]
                    candidate_ids = [str(item["case_id"]) for item in remaining]
                    initial = _initial_observations(env, remaining)
                    scores = _posterior_action_disagreement(
                        agent, initial, mu, log_var,
                        seed=int(content_hash({"seed": selection_seed, "round": shot})[:16], 16),
                    )
                    score_by_id = {str(item["case_id"]): float(score) for item, score in zip(remaining, scores)}
                    # Stable lower-index tie breaking keeps a rerun identical.
                    chosen_offset = max(range(len(remaining)), key=lambda index: (scores[index], -index))
                    case = remaining.pop(chosen_offset)
                    ordered_support[shot:] = [case, *remaining]
                    selection["selected_case_ids"].append(str(case["case_id"]))
                    selection["initial_observation_count"] += len(candidate_ids)
                    selection["selection_rounds"].append({
                        "round": shot, "candidate_case_ids": candidate_ids,
                        "scores": score_by_id,
                        "selected_case_id": str(case["case_id"]),
                        "initial_observation_count": len(candidate_ids),
                        "posterior_variance_mean": float(torch.exp(log_var).mean().detach()),
                    })
                else:
                    case = ordered_support[shot]
                rollout = collect_episode(env, task, case, agent, agent.sample_latent(mu, log_var), "prior_support" if shot == 0 else "posterior_rollout", device, episode_id=f"{task.task_id}:support:{shot}", posterior_version=shot)
                support_rollouts.append(rollout)
                support_episode_lengths.append(len(rollout.transitions))
                rng = np.random.default_rng(int(content_hash({"seed": base_seed, "task": task.task_id, "shot": shot})[:16], 16))
                context = _sample_episode_context(support_rollouts, int(config["pearl"]["context_sample_size_eval"]), int(config["pearl"]["context_transitions_per_episode"]), rng)
                mu, log_var = agent.infer_posterior([context])
        finally:
            env.close()
        all_results[task.task_id] = results
        selection_by_task[task.task_id] = selection
    after = agent.parameter_hash()
    if before != after:
        raise RuntimeError("meta-test changed model parameters, target critics, or alpha")
    return {"split": split, "parameter_hash_before": before, "parameter_hash_after": after, "no_gradient_adaptation": True, "no_topology_ablation": bool(config.get("ablation", {}).get("no_topology", False)), "support_selection": support_selection, "support_selection_by_task": selection_by_task, "context_protocol": {"sample_size": int(config["pearl"]["context_sample_size_eval"]), "transitions_per_episode": int(config["pearl"]["context_transitions_per_episode"]), "seed": base_seed}, "provenance": output_provenance, "tasks": all_results}


def infer_support_posteriors(agent: Any, config: Mapping[str, Any], tasks: list[Any], casebooks: Mapping[str, Mapping[str, list[dict[str, Any]]]],
                             split: str, shots: list[int] | None = None,
                             provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Infer task posteriors from support only; never create query rollouts."""
    requested = sorted(set(int(shot) for shot in (shots or config["evaluation"]["shots"])))
    if not requested or requested[0] < 0:
        raise ValueError("posterior diagnostic shots must be non-negative")
    support_key = "validation_support" if split == "meta_validation" else "test_support"
    base_seed = int(config["evaluation"]["context_sampling_seed"])
    output_provenance = dict(provenance or {})
    before = agent.parameter_hash(); all_results: dict[str, Any] = {}
    for task in tasks:
        book = casebooks[task.task_id]
        if max(requested) > len(book[support_key]):
            raise ValueError(f"task {task.task_id} lacks support cases for K={max(requested)}")
        env = LogicalMergeEnv(task, config, book[support_key])
        rollouts: list[Rollout] = []
        episode_lengths: list[int] = []
        task_results: dict[str, Any] = {}
        try:
            mu, log_var = agent.prior()
            for shot in range(max(requested) + 1):
                if shot in requested:
                    task_results[str(shot)] = {
                        "posterior_mean": mu.detach().cpu().tolist(),
                        "posterior_variance": torch.exp(log_var).detach().cpu().tolist(),
                        "context_episode_count": len(rollouts),
                        "support_case_ids": [str(case["case_id"]) for case in book[support_key][:shot]],
                        "support_environment_steps": int(sum(episode_lengths)),
                    }
                if shot == max(requested):
                    break
                case = book[support_key][shot]
                rollout = collect_episode(env, task, case, agent, agent.sample_latent(mu, log_var), "prior_support" if shot == 0 else "posterior_rollout", agent.device, episode_id=f"{task.task_id}:posterior_support:{shot}", posterior_version=shot)
                rollouts.append(rollout); episode_lengths.append(len(rollout.transitions))
                rng = np.random.default_rng(int(content_hash({"seed": base_seed, "task": task.task_id, "shot": shot})[:16], 16))
                context = _sample_episode_context(rollouts, int(config["pearl"]["context_sample_size_eval"]), int(config["pearl"]["context_transitions_per_episode"]), rng)
                mu, log_var = agent.infer_posterior([context])
        finally:
            env.close()
        all_results[task.task_id] = task_results
    after = agent.parameter_hash()
    if before != after:
        raise RuntimeError("support posterior diagnostic changed model parameters")
    return {
        "schema": "logical_merge_support_posterior_diagnostic_v1", "split": split,
        "taskbook_hash": output_provenance.get("taskbook_hash"),
        "uses_query_cases": False, "parameter_hash_before": before, "parameter_hash_after": after,
        "no_gradient_adaptation": True,
        "context_protocol": {"sample_size": int(config["pearl"]["context_sample_size_eval"]), "transitions_per_episode": int(config["pearl"]["context_transitions_per_episode"]), "seed": base_seed},
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
        if max(requested) > len(book[support_key]):
            raise ValueError(f"task {task.task_id} lacks support cases for K={max(requested)}")
        posthoc_target = representation_target(task)
        env = LogicalMergeEnv(task, config, book[support_key])
        rollouts: list[Rollout] = []
        episode_lengths: list[int] = []
        task_results: dict[str, Any] = {}
        try:
            mu, log_var = agent.prior()
            context: list[list[Any]] | None = None
            for shot in range(1, max(requested) + 1):
                case = book[support_key][shot - 1]
                rollout = collect_episode(
                    env, task, case, agent, agent.sample_latent(mu, log_var),
                    "prior_support" if shot == 1 else "posterior_rollout", agent.device,
                    episode_id=f"{task.task_id}:representation_support:{shot - 1}",
                    posterior_version=shot - 1,
                )
                rollouts.append(rollout); episode_lengths.append(len(rollout.transitions))
                rng = np.random.default_rng(int(content_hash({"seed": base_seed, "task": task.task_id, "shot": shot - 1})[:16], 16))
                context = _sample_episode_context(
                    rollouts, int(config["pearl"]["context_sample_size_eval"]),
                    int(config["pearl"]["context_transitions_per_episode"]), rng,
                )
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
                    "posterior_variance": torch.exp(log_var).detach().cpu().tolist(),
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
            "sample_size": int(config["pearl"]["context_sample_size_eval"]),
            "transitions_per_episode": int(config["pearl"]["context_transitions_per_episode"]),
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
                "summary": value["summary"],
                "support_environment_steps": value["support_environment_steps"],
                "support_case_ids": list(value.get("support_case_ids", [])),
            }
            for shot, value in shots.items()
        }
    compact = {
        key: result[key]
        for key in ("split", "parameter_hash_before", "parameter_hash_after", "no_gradient_adaptation", "no_topology_ablation", "context_protocol", "provenance")
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
