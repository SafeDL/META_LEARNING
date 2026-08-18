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


def _gaussian_symmetric_kl(
    mean_left: torch.Tensor,
    log_std_left: torch.Tensor,
    mean_right: torch.Tensor,
    log_std_right: torch.Tensor,
) -> float:
    """Mean symmetric KL between two diagonal pre-squash Gaussians."""
    var_left = torch.exp(2.0 * log_std_left).clamp_min(1e-7)
    var_right = torch.exp(2.0 * log_std_right).clamp_min(1e-7)
    kl_left_right = 0.5 * (
        var_left / var_right
        + (mean_right - mean_left).square() / var_right
        - 1.0
        + torch.log(var_right / var_left)
    )
    kl_right_left = 0.5 * (
        var_right / var_left
        + (mean_left - mean_right).square() / var_left
        - 1.0
        + torch.log(var_left / var_right)
    )
    return float((kl_left_right + kl_right_left).sum(dim=-1).mean().detach())


def stage_b_actor_critic_diagnostics(
    agent: Any,
    state_bank: np.ndarray,
    z_correct: torch.Tensor,
    z_wrong: torch.Tensor,
    *,
    action_grid_points: int = 41,
) -> dict[str, Any]:
    """No-gradient Actor/Critic diagnostics for one correct/wrong latent pair.

    Separates the Stage-B failure into one of three mechanisms on the frozen
    dynamic-state bank:

    * latent geometry -- are the two posteriors nearly collinear?
    * pre-tanh actor   -- does the task signal die inside the tanh squash?
    * critic Q-grid    -- does the Critic itself demand different actions
      under the two latents?

    No environment step, replay write, or parameter change is involved.
    """
    if agent.actor_architecture != "dense":
        raise ValueError("Stage-B diagnostics currently require the dense actor")
    observations = torch.as_tensor(state_bank, dtype=torch.float32, device=agent.device)
    state_count = len(state_bank)

    def _latent_geometry() -> dict[str, float]:
        left = z_correct.detach().float().reshape(-1)
        right = z_wrong.detach().float().reshape(-1)
        norm_left = float(torch.linalg.vector_norm(left))
        norm_right = float(torch.linalg.vector_norm(right))
        difference = float(torch.linalg.vector_norm(left - right))
        cosine = float(
            torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1)[0]
        )
        separation_ratio = difference / (0.5 * (norm_left + norm_right) + 1e-8)
        return {
            "correct_norm_l2": norm_left,
            "wrong_norm_l2": norm_right,
            "correct_wrong_l2": difference,
            "correct_wrong_cosine": cosine,
            "separation_ratio": separation_ratio,
        }

    def _actor_pre_tanh() -> dict[str, Any]:
        with torch.no_grad():
            raw_correct, log_std_correct = agent.actor(observations, z_correct.detach().expand(state_count, -1))
            raw_wrong, log_std_wrong = agent.actor(observations, z_wrong.detach().expand(state_count, -1))
        tanh_correct = torch.tanh(raw_correct)
        tanh_wrong = torch.tanh(raw_wrong)
        saturation_correct = 1.0 - tanh_correct.square()
        saturation_wrong = 1.0 - tanh_wrong.square()
        return {
            "raw_mean_correct": raw_correct.detach().cpu().tolist(),
            "raw_mean_wrong": raw_wrong.detach().cpu().tolist(),
            "log_std_correct": log_std_correct.detach().cpu().tolist(),
            "log_std_wrong": log_std_wrong.detach().cpu().tolist(),
            "tanh_action_correct": tanh_correct.detach().cpu().tolist(),
            "tanh_action_wrong": tanh_wrong.detach().cpu().tolist(),
            "raw_mean_l2": float((raw_correct - raw_wrong).norm(dim=-1).mean().detach()),
            "tanh_action_l2": float((tanh_correct - tanh_wrong).norm(dim=-1).mean().detach()),
            "saturation_complement_mean": {
                "correct": float(saturation_correct.mean().detach()),
                "wrong": float(saturation_wrong.mean().detach()),
            },
            "symmetric_kl_pre_squash": _gaussian_symmetric_kl(
                raw_correct, log_std_correct, raw_wrong, log_std_wrong,
            ),
        }

    def _latent_interpolation() -> dict[str, Any]:
        alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
        direction = (z_correct.detach() - z_wrong.detach()).expand(state_count, -1)
        base = z_wrong.detach().expand(state_count, -1)
        actions = {}
        with torch.no_grad():
            for alpha in alphas:
                action = agent.act(observations, base + alpha * direction, deterministic=True)
                actions[f"alpha_{alpha:.2f}"] = action.detach().cpu().tolist()
        means = np.asarray([np.mean(actions[f"alpha_{alpha:.2f}"], axis=0) for alpha in alphas], dtype=float)
        return {
            "alphas": list(alphas),
            "per_state_actions": actions,
            "mean_action_by_alpha": {
                f"alpha_{alpha:.2f}": means[index].tolist() for index, alpha in enumerate(alphas)
            },
            "max_pairwise_action_l2": float(
                np.max([np.linalg.norm(means[index] - means[index + 1]) for index in range(len(alphas) - 1)])
            ),
        }

    def _critic_q_grid() -> dict[str, Any]:
        grid = np.linspace(-1.0, 1.0, int(action_grid_points), dtype=np.float32)
        # The grid sweeps the longitudinal component; the remaining action
        # components (steering) are held at zero.  Mechanism runs use a 1-D
        # action space, so the sweep covers the full action there.
        grid_actions = np.zeros((len(grid), agent.action_dim), dtype=np.float32)
        grid_actions[:, 0] = grid
        action_tensor = torch.as_tensor(grid_actions, dtype=torch.float32, device=agent.device)
        result: dict[str, Any] = {}
        for name, z in (("correct", z_correct), ("wrong", z_wrong)):
            latent = z.detach().expand(state_count, -1)
            with torch.no_grad():
                q_values = []
                for observation in observations:
                    expanded = observation.unsqueeze(0).expand(len(grid), -1)
                    expanded_latent = latent[:1].expand(len(grid), -1)
                    q1 = agent.q1(expanded, action_tensor, expanded_latent).squeeze(-1)
                    q2 = agent.q2(expanded, action_tensor, expanded_latent).squeeze(-1)
                    q_values.append(torch.minimum(q1, q2).detach().cpu().numpy())
                actor_action = agent.act(observations, latent, deterministic=True)
                actor_q = []
                for index, observation in enumerate(observations):
                    q1_actor = agent.q1(observation.unsqueeze(0), actor_action[index].unsqueeze(0), z.detach())
                    q2_actor = agent.q2(observation.unsqueeze(0), actor_action[index].unsqueeze(0), z.detach())
                    actor_q.append(float(torch.minimum(q1_actor, q2_actor).squeeze().detach()))
            q_min = np.stack(q_values)  # [states, grid]
            best_indexes = np.argmax(q_min, axis=1)
            argmax_q = np.asarray([q_min[index, best_indexes[index]] for index in range(state_count)])
            actor_q = np.asarray(actor_q)
            result[name] = {
                "argmax_action": [float(grid[int(best_indexes[index])]) for index in range(state_count)],
                "actor_deterministic_action": actor_action.detach().cpu().tolist(),
                "argmax_q_value": argmax_q.tolist(),
                "actor_q_value": actor_q.tolist(),
                "actor_regret": [float(argmax_q[index] - actor_q[index]) for index in range(state_count)],
            }
        argmax_correct = np.asarray(result["correct"]["argmax_action"])
        argmax_wrong = np.asarray(result["wrong"]["argmax_action"])
        return {
            "action_grid_points": int(action_grid_points),
            "action_grid": grid.tolist(),
            "argmax_action_distance_mean": float(np.mean(np.abs(argmax_correct - argmax_wrong))),
            "argmax_action_distance_max": float(np.max(np.abs(argmax_correct - argmax_wrong))),
            "actor_regret_mean": {
                "correct": float(np.mean(result["correct"]["actor_regret"])),
                "wrong": float(np.mean(result["wrong"]["actor_regret"])),
            },
            **result,
        }

    before = agent.parameter_hash()
    payload = {
        "state_bank_size": int(state_count),
        "latent_geometry": _latent_geometry(),
        "actor_pre_tanh": _actor_pre_tanh(),
        "latent_interpolation": _latent_interpolation(),
        "critic_q_grid": _critic_q_grid(),
    }
    after = agent.parameter_hash()
    if before != after:
        raise RuntimeError("Stage-B diagnostics changed model parameters")
    return payload


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
        stage_b_diagnostics = stage_b_actor_critic_diagnostics(
            agent, state_bank, correct_mu, wrong_mu,
        )
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
            "stage_b_diagnostics": stage_b_diagnostics,
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
