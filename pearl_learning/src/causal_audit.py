"""No-gradient latent/context interventions for the method-flow gate."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import torch

from .collector import Rollout, collect_episode
from .io import content_hash
from .metrics import summarize
from .task_env import LogicalMergeEnv


def _context_block(rollout: Rollout, size: int) -> list[Any]:
    if not rollout.transitions:
        raise ValueError("support rollout is empty")
    indexes = np.linspace(0, len(rollout.transitions) - 1, num=size, dtype=int)
    return [rollout.transitions[int(index)] for index in indexes]


def _support_context(agent: Any, task: Any, config: Mapping[str, Any], cases: list[dict[str, Any]], k: int) -> list[list[Any]]:
    if k > len(cases):
        raise ValueError(f"{task.task_id} lacks K={k} support cases")
    per_episode = int(config["pearl"]["context_transitions_per_episode"])
    env = LogicalMergeEnv(task, config, cases)
    context: list[list[Any]] = []
    try:
        mu, log_var = agent.prior(tasks=[task])
        for index, case in enumerate(cases[:k]):
            rollout = collect_episode(
                env, task, case, agent, mu, "prior_support", agent.device,
                episode_id=f"{task.task_id}:causal_support:{index}", posterior_version=index,
            )
            context.append(_context_block(rollout, per_episode))
            with torch.no_grad():
                mu, log_var = agent.infer_posterior([context], [task])
    finally:
        env.close()
    return context


def _initial_states(task: Any, config: Mapping[str, Any], cases: list[dict[str, Any]]) -> np.ndarray:
    env = LogicalMergeEnv(task, config, cases)
    states = []
    try:
        for case in cases:
            observation, _ = env.reset(options={"case": case})
            states.append(np.asarray(observation, dtype=np.float32))
    finally:
        env.close()
    return np.stack(states)


def _actor_means(agent: Any, states: np.ndarray, latent: torch.Tensor) -> np.ndarray:
    observation = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
    expanded = latent.detach().expand(len(states), -1)
    with torch.no_grad():
        actions = agent.act(observation, expanded, deterministic=True)
    return actions.detach().cpu().numpy()


def _bootstrap(values: np.ndarray, seed: int, samples: int = 2000) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "bootstrap_mean_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "values": values.tolist(),
    }


def _pair_action_stats(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    def _longitudinal_component(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim == 1:
            return np.abs(values)
        if values.shape[-1] >= 2:
            return np.abs(values[:, 1])
        if values.shape[-1] == 1:
            return np.abs(values[:, 0])
        raise ValueError("action tensor must be at least one-dimensional")
    return {
        "action_l2": _bootstrap(np.linalg.norm(left - right, axis=-1), seed),
        "longitudinal_absolute_change": _bootstrap(_longitudinal_component(left - right), seed + 1),
    }


def _trajectory_rollouts(agent: Any, task: Any, config: Mapping[str, Any], cases: list[dict[str, Any]], latent: torch.Tensor, label: str) -> list[Rollout]:
    env = LogicalMergeEnv(task, config, cases)
    rows = []
    try:
        for index, case in enumerate(cases):
            rows.append(collect_episode(
                env, task, case, agent, latent, "deterministic_query", agent.device,
                episode_id=f"{task.task_id}:{label}:{index}", posterior_version=0,
            ))
    finally:
        env.close()
    return rows


def _trajectory_delta(left: Rollout, right: Rollout) -> dict[str, float]:
    count = min(len(left.transitions), len(right.transitions))
    left_actions = np.stack([row.action for row in left.transitions[:count]])
    right_actions = np.stack([row.action for row in right.transitions[:count]])
    left_states = np.stack([row.obs for row in left.transitions[:count]])
    right_states = np.stack([row.obs for row in right.transitions[:count]])
    def _longitudinal_component(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return np.abs(values)
        if values.shape[-1] >= 2:
            return np.abs(values[:, 1])
        if values.shape[-1] == 1:
            return np.abs(values[:, 0])
        raise ValueError("action tensor must be at least one-dimensional")
    return {
        "aligned_steps": count,
        "action_trajectory_l2": float(np.linalg.norm(left_actions - right_actions)),
        "mean_longitudinal_action_difference": float(np.mean(_longitudinal_component(left_actions - right_actions))),
        "state_trajectory_l2": float(np.linalg.norm(left_states - right_states)),
    }


def audit_task_context_interventions(
    agent: Any,
    config: Mapping[str, Any],
    target_task: Any,
    wrong_evidence_task: Any,
    target_book: Mapping[str, list[dict[str, Any]]],
    wrong_book: Mapping[str, list[dict[str, Any]]],
    *,
    split: str,
    wrong_support_key: str | None = None,
    shots: list[int] = [1, 2, 4],
) -> dict[str, Any]:
    """Compare B prior, B evidence, A evidence under the unchanged B prior."""
    if agent.actor_architecture != "dense":
        raise ValueError("method-flow causal audit currently requires the dense actor")
    support_key = "validation_support" if split == "meta_validation" else "test_support"
    wrong_support_key = wrong_support_key or support_key
    query_key = "validation_query" if split == "meta_validation" else "test_query"
    query_cases = list(target_book[query_key])[: int(config["evaluation"]["query_cases_per_task"])]
    state_bank = _initial_states(target_task, config, query_cases)
    before = agent.parameter_hash()
    prior_mu, prior_log_var = agent.prior(tasks=[target_task])
    zero = torch.zeros_like(prior_mu)
    prior_actions = _actor_means(agent, state_bank, prior_mu)
    zero_actions = _actor_means(agent, state_bank, zero)
    results: dict[str, Any] = {}
    association_rows: list[dict[str, float | str | int]] = []
    for k in sorted(set(int(value) for value in shots)):
        correct_context = _support_context(agent, target_task, config, list(target_book[support_key]), k)
        wrong_context = _support_context(agent, wrong_evidence_task, config, list(wrong_book[wrong_support_key]), k)
        with torch.no_grad():
            correct_mu, correct_log_var = agent.infer_posterior([correct_context], [target_task])
            # Crucial intervention: evidence comes from A, but p(z|e) remains
            # the conditional prior for target Task B.
            wrong_mu, wrong_log_var = agent.infer_posterior([wrong_context], [target_task])
        correct_actions = _actor_means(agent, state_bank, correct_mu)
        wrong_actions = _actor_means(agent, state_bank, wrong_mu)
        seed = int(content_hash({"task": target_task.task_id, "k": k, "audit": "action_adaptation"})[:8], 16)
        trajectories = {
            "prior": _trajectory_rollouts(agent, target_task, config, query_cases, prior_mu, f"prior_k{k}"),
            "correct": _trajectory_rollouts(agent, target_task, config, query_cases, correct_mu, f"correct_k{k}"),
            "wrong": _trajectory_rollouts(agent, target_task, config, query_cases, wrong_mu, f"wrong_k{k}"),
        }
        summaries = {
            name: summarize(
                [rollout.record for rollout in rows],
                case_metadata={str(case["case_id"]): case for case in query_cases},
            )
            for name, rows in trajectories.items()
        }
        paired = []
        for index, case in enumerate(query_cases):
            prior_record = trajectories["prior"][index].record
            correct_record = trajectories["correct"][index].record
            wrong_record = trajectories["wrong"][index].record
            paired.append({
                "case_id": case["case_id"],
                "correct_minus_prior_return": float(correct_record["episode_return"]) - float(prior_record["episode_return"]),
                "correct_minus_wrong_return": float(correct_record["episode_return"]) - float(wrong_record["episode_return"]),
                "correct_minus_prior_valid_critical": int(bool(correct_record["valid_critical_strict"])) - int(bool(prior_record["valid_critical_strict"])),
                "correct_minus_wrong_valid_critical": int(bool(correct_record["valid_critical_strict"])) - int(bool(wrong_record["valid_critical_strict"])),
                "correct_vs_prior_trajectory": _trajectory_delta(trajectories["correct"][index], trajectories["prior"][index]),
                "correct_vs_wrong_trajectory": _trajectory_delta(trajectories["correct"][index], trajectories["wrong"][index]),
            })
        latent_shifts = {
            "correct_prior_l2": float(torch.linalg.vector_norm(correct_mu - prior_mu)),
            "wrong_prior_l2": float(torch.linalg.vector_norm(wrong_mu - prior_mu)),
            "correct_wrong_l2": float(torch.linalg.vector_norm(correct_mu - wrong_mu)),
        }
        action_means = {
            "correct_prior": float(np.linalg.norm(correct_actions - prior_actions, axis=-1).mean()),
            "wrong_prior": float(np.linalg.norm(wrong_actions - prior_actions, axis=-1).mean()),
            "correct_wrong": float(np.linalg.norm(correct_actions - wrong_actions, axis=-1).mean()),
        }
        variance_means = {
            "prior": float(torch.exp(prior_log_var).mean()),
            "correct": float(torch.exp(correct_log_var).mean()),
            "wrong": float(torch.exp(wrong_log_var).mean()),
        }
        for comparison, left, right in (
            ("correct_prior", "correct", "prior"),
            ("wrong_prior", "wrong", "prior"),
            ("correct_wrong", "correct", "wrong"),
        ):
            association_rows.append({
                "k": k,
                "comparison": comparison,
                "latent_l2": latent_shifts[f"{comparison}_l2"],
                "action_l2_mean": action_means[comparison],
                "posterior_variance_mean_abs_difference": abs(variance_means[left] - variance_means[right]),
            })
        results[str(k)] = {
            "target_prior_task_id": target_task.task_id,
            "correct_evidence_task_id": target_task.task_id,
            "wrong_evidence_task_id": wrong_evidence_task.task_id,
            "wrong_context_preserves_target_prior": True,
            "latents": {
                "prior": prior_mu.detach().cpu().tolist(), "correct": correct_mu.detach().cpu().tolist(),
                "wrong": wrong_mu.detach().cpu().tolist(), "zero": zero.detach().cpu().tolist(),
            },
            "posterior_variance": {
                "prior": torch.exp(prior_log_var).detach().cpu().tolist(),
                "correct": torch.exp(correct_log_var).detach().cpu().tolist(),
                "wrong": torch.exp(wrong_log_var).detach().cpu().tolist(),
            },
            "latent_l2": latent_shifts,
            "action_adaptation": {
                "correct_prior": _pair_action_stats(correct_actions, prior_actions, seed),
                "wrong_prior": _pair_action_stats(wrong_actions, prior_actions, seed + 2),
                "correct_wrong": _pair_action_stats(correct_actions, wrong_actions, seed + 4),
                "zero_prior": _pair_action_stats(zero_actions, prior_actions, seed + 6),
            },
            "trajectory_summaries": summaries,
            "paired_query_gains": paired,
            "paired_gain_means": {
                key: float(np.mean([float(row[key]) for row in paired]))
                for key in (
                    "correct_minus_prior_return", "correct_minus_wrong_return",
                    "correct_minus_prior_valid_critical", "correct_minus_wrong_valid_critical",
                )
            },
        }
    after = agent.parameter_hash()
    if before != after:
        raise RuntimeError("causal action audit changed model parameters")
    latent_values = np.asarray([float(row["latent_l2"]) for row in association_rows])
    action_values = np.asarray([float(row["action_l2_mean"]) for row in association_rows])
    variance_values = np.asarray([
        float(row["posterior_variance_mean_abs_difference"]) for row in association_rows
    ])
    correlation = lambda left, right: (
        None
        if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12
        else float(np.corrcoef(left, right)[0, 1])
    )
    return {
        "schema": "latent_context_causal_audit_v1",
        "task_id": target_task.task_id,
        "split": split,
        "fixed_dynamic_state_bank_size": len(state_bank),
        "state_bank_hash": content_hash(state_bank.tolist()),
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "no_gradient": True,
        "writes_replay_or_context": False,
        "latent_action_association": {
            "rows": association_rows,
            "pearson_latent_l2_vs_action_l2": correlation(latent_values, action_values),
            "pearson_posterior_variance_difference_vs_action_l2": correlation(variance_values, action_values),
        },
        "shots": results,
    }
