"""No-gradient latent/context interventions for the method-flow gate."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import torch

from .collector import Rollout, collect_episode
from .io import content_hash
from .metrics import summarize
from .replay import Transition, select_context_rows
from .task_env import LogicalMergeEnv


def _context_block(rollout: Rollout, size: int, scheme: str, rng: np.random.Generator) -> list[Any]:
    if not rollout.transitions:
        raise ValueError("support rollout is empty")
    return select_context_rows(list(rollout.transitions), int(size), scheme, rng)


def _support_context(agent: Any, task: Any, config: Mapping[str, Any], cases: list[dict[str, Any]], k: int) -> list[list[Any]]:
    if k > len(cases):
        raise ValueError(f"{task.task_id} lacks K={k} support cases")
    per_episode = int(config["pearl"]["context_transitions_per_episode"])
    scheme = str(config["pearl"].get("context_transition_sampling", "random"))
    if scheme not in CONTEXT_SAMPLING_SCHEMES:
        raise ValueError(f"unsupported pearl.context_transition_sampling: {scheme!r}")
    env = LogicalMergeEnv(task, config, cases)
    context: list[list[Any]] = []
    try:
        mu, log_var = agent.prior(tasks=[task])
        for index, case in enumerate(cases[:k]):
            rollout = collect_episode(
                env, task, case, agent, mu, "prior_support", agent.device,
                episode_id=f"{task.task_id}:causal_support:{index}", posterior_version=index,
            )
            # Deterministic per-episode selector seed so the causal audit is
            # reproducible without sharing the training RNG stream.  The
            # selection itself is the same canonical function training uses.
            rng = np.random.default_rng(int(content_hash({
                "task_id": task.task_id, "case_id": str(case["case_id"]),
                "purpose": "causal_support_context_block",
            })[:16], 16))
            context.append(_context_block(rollout, per_episode, scheme, rng))
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


# ---------------------------------------------------------------------------
# Stage-A localization audit.  These functions never train: they probe the
# exact PEARL context preprocessing and the frozen Context Encoder on
# collected support trajectories, answering two questions:
#
#   1. Does the context PEARL actually receives still contain task information?
#   2. If it does, does the trained Context Encoder preserve a
#      task-discriminative posterior component?
#
# The trigger was Round 2: D_cw 3.6-7.7 (>= 0.5) but R_sep 0.068-0.141 and
# cos ~ 0.9997, i.e. the posterior moved along a shared "has-context"
# direction.  Gate 2, the training sampler, and the causal audit all look at
# different data (full trajectory summary + raw reward vs. random 8
# transitions/episode with r/200 vs. linspace terminal-inclusive blocks), so
# the input-side confounder must be excluded before blaming the Encoder.
# ---------------------------------------------------------------------------


def _transition_context_row(transition: Transition, reward_scale: float) -> np.ndarray:
    """Replicate PEARL's exact per-transition context preprocessing.

    Must stay byte-compatible with ``PEARLAgent.context_tensor``: obs, action,
    reward / context_reward_scale, next_obs, terminated, truncated.
    """
    return np.asarray(
        np.concatenate(
            [
                np.asarray(transition.obs),
                np.asarray(transition.action),
                [float(transition.reward) / float(reward_scale)],
                np.asarray(transition.next_obs),
                [float(transition.terminated), float(transition.truncated)],
            ]
        ),
        dtype=np.float32,
    )


def context_rows(episodes: list[list[Transition]], reward_scale: float) -> np.ndarray:
    return np.stack([
        _transition_context_row(transition, reward_scale)
        for episode in episodes
        for transition in episode
    ])


CONTEXT_SAMPLING_SCHEMES = ("random", "linspace", "terminal_stratified_v1", "conflict_window")


def _conflict_near_indexes(rows: list[Transition], count: int) -> list[int]:
    """Rank non-terminal transitions by public-dynamics conflict proximity.

    Only label-free observation fields are used: |arrival-time difference|,
    distance-to-conflict (both vehicles), and TTC.  Hidden entry order, task
    ids, and query outcomes never influence the ranking.
    """
    scored = []
    for index, transition in enumerate(rows):
        obs = np.asarray(transition.obs, dtype=float)
        score = abs(obs[16]) + 0.5 * (abs(obs[0]) + abs(obs[8])) + 0.25 * abs(obs[20])
        scored.append((float(score), index))
    scored.sort(key=lambda pair: pair[0])
    return [index for _, index in scored[: int(count)]]


def sample_context_scheme(
    episodes: list[list[Transition]],
    per_episode: int,
    scheme: str,
    rng: np.random.Generator,
) -> list[list[Transition]]:
    """Select ``per_episode`` transitions per episode under one frozen scheme.

    ``random`` and ``terminal_stratified_v1`` delegate to the canonical
    training selector in :mod:`replay`; ``linspace`` mirrors the historical
    causal-audit block and ``conflict_window`` is a Stage-A sampling ablation
    (1 terminal + 3 conflict-near + rest random).  The terminal transition is
    included exactly once by the inclusive schemes.
    """
    if scheme not in CONTEXT_SAMPLING_SCHEMES:
        raise ValueError(f"unsupported context sampling scheme: {scheme!r}")
    size = max(1, int(per_episode))
    groups: list[list[Transition]] = []
    for episode in episodes:
        rows = list(episode)
        count = len(rows)
        if not rows:
            continue
        if scheme in ("random", "terminal_stratified_v1"):
            groups.append(select_context_rows(rows, size, scheme, rng))
        elif scheme == "linspace":
            indexes = [int(item) for item in np.linspace(0, count - 1, num=size, dtype=int)]
            groups.append([rows[index] for index in indexes])
        else:  # conflict_window: 1 terminal + 3 conflict-near + rest random
            near_count = min(3, count - 1, size - 1)
            if count <= 1:
                groups.append([rows[0]] * size)
            else:
                near = _conflict_near_indexes(rows[:-1], near_count)
                remaining = [index for index in range(count - 1) if index not in near]
                extra = size - 1 - len(near)
                if extra > 0:
                    chosen = rng.choice(len(remaining), size=extra, replace=len(remaining) < extra)
                    tail = [remaining[int(item)] for item in np.asarray(chosen).reshape(-1)]
                else:
                    tail = []
                indexes = [count - 1] + near + tail
                groups.append([rows[index] for index in indexes])
    return groups


def posterior_separation(
    agent: Any,
    correct_context: list[list[Transition]],
    wrong_context: list[list[Transition]],
    task: Any,
) -> dict[str, Any]:
    """D_cw, prior-relative R_sep and cosine under the frozen Context Encoder.

    R_sep = D_cw / (0.5 * (||mu_c - mu_p||_2 + ||mu_w - mu_p||_2) + eps) with
    mu_p the prior mean (zero for the unit-normal prior).  Defining it against
    the prior keeps the metric valid once a Structure-Aware prior has a
    non-zero mean, and penalizes a correct/wrong shift that is tiny relative
    to the common shift away from the prior.
    """
    prior_mu, _ = agent.prior(tasks=[task])
    with torch.no_grad():
        mu_c, _ = agent.infer_posterior([correct_context], [task])
        mu_w, _ = agent.infer_posterior([wrong_context], [task])
    d_cw = float(torch.linalg.vector_norm(mu_c - mu_w))
    c_prior = float(torch.linalg.vector_norm(mu_c - prior_mu))
    w_prior = float(torch.linalg.vector_norm(mu_w - prior_mu))
    r_sep = d_cw / (0.5 * (c_prior + w_prior) + 1e-8)
    cosine = float(
        torch.nn.functional.cosine_similarity(mu_c.reshape(-1).float(), mu_w.reshape(-1).float(), dim=-1)
    )
    return {
        "correct_wrong_l2": d_cw,
        "correct_prior_l2": c_prior,
        "wrong_prior_l2": w_prior,
        "prior_relative_separation_ratio": r_sep,
        "correct_wrong_cosine": cosine,
        "correct_mean": mu_c.detach().cpu().tolist(),
        "wrong_mean": mu_w.detach().cpu().tolist(),
    }


def _fit_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Tiny linear logistic regression (LBFGS) returning test predictions."""
    x = torch.as_tensor(train_features, dtype=torch.float32)
    y = torch.as_tensor(train_labels, dtype=torch.float32)
    x_test = torch.as_tensor(test_features, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    weights = torch.randn((x.shape[1], 1), generator=generator) * 0.1
    bias = torch.zeros(1)
    weights.requires_grad_(True)
    bias.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [weights, bias], lr=0.5, max_iter=80, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.sigmoid(x_test @ weights + bias).squeeze(-1).numpy()


def logistic_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    folds: int = 4,
) -> dict[str, Any]:
    """Grouped (per-episode) k-fold linear logistic probe on Task.

    Held-out accuracy answers whether the features retain task information.
    ``groups`` prevents episode leakage between train and test folds.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    groups = np.asarray(groups, dtype=int).reshape(-1)
    if features.ndim != 2 or features.shape[0] != len(labels) or len(groups) != len(labels):
        raise ValueError("probe features, labels and groups must align row-wise")
    unique = np.unique(groups)
    if len(unique) < int(folds):
        raise ValueError("fewer episode groups than probe folds")
    order = np.random.default_rng(int(seed)).permutation(len(unique))
    fold_ids = np.array_split(order, int(folds))
    predictions = np.full(len(labels), np.nan)
    per_fold = []
    for fold_index, fold in enumerate(fold_ids):
        test_mask = np.isin(groups, unique[fold])
        train_mask = ~test_mask
        train_labels = labels[train_mask]
        if len(np.unique(train_labels)) < 2:
            # Degenerate fold: report chance instead of fabricating a fit.
            predictions[test_mask] = 0.5
        else:
            mean = features[train_mask].mean(axis=0)
            std = features[train_mask].std(axis=0) + 1e-8
            x_train = (features[train_mask] - mean) / std
            x_test = (features[test_mask] - mean) / std
            probs = _fit_logistic(
                x_train, train_labels, x_test, int(seed) + int(fold[0]) + fold_index
            )
            predictions[test_mask] = probs
        fold_labels = labels[test_mask]
        fold_correct = (np.asarray(predictions[test_mask]) > 0.5).astype(int) == fold_labels
        per_fold.append(float(fold_correct.mean()))
    predicted = (predictions > 0.5).astype(int)
    episode_hits = []
    for group in unique:
        mask = groups == group
        episode_hits.append(int((predicted[mask] == labels[mask]).mean() > 0.5))
    return {
        "folds": int(folds),
        "chance_accuracy": 0.5,
        "rows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "episodes": int(len(unique)),
        "transition_accuracy": float((predicted == labels).mean()),
        "episode_majority_accuracy": float(np.mean(episode_hits)),
        "per_fold_transition_accuracy": per_fold,
    }


def stage_a_context_posterior_diagnostics(
    agent: Any,
    config: Mapping[str, Any],
    target_task: Any,
    wrong_task: Any,
    target_book: Mapping[str, list[dict[str, Any]]],
    wrong_book: Mapping[str, list[dict[str, Any]]],
    *,
    split: str = "meta_validation",
    support_cases: int = 4,
    sampling_draws: int = 3,
    probe_folds: int = 4,
) -> dict[str, Any]:
    """Zero-training-step Stage-A localization audit.

    Support trajectories are collected once with the frozen checkpoint's prior
    policy, then the same episodes are re-cut under three sampling schemes and
    probed for task identifiability at three levels: exact PEARL input
    (preprocessing + sampler), frozen-Encoder posterior separation, and input
    channels (reward vs. dynamics).  No parameter changes, no replay writes,
    no gradient steps.
    """
    support_key = "validation_support" if split == "meta_validation" else "test_support"
    per_episode = int(config["pearl"]["context_transitions_per_episode"])
    reward_scale = float(agent.reward_scale)
    before = agent.parameter_hash()

    episodes: dict[str, list[list[Transition]]] = {}
    collected_steps = 0
    for task, book in ((target_task, target_book), (wrong_task, wrong_book)):
        cases = list(book[support_key])[: int(support_cases)]
        env = LogicalMergeEnv(task, config, cases)
        try:
            mu, log_var = agent.prior(tasks=[task])
            rows: list[list[Transition]] = []
            for index, case in enumerate(cases):
                # Training collection samples the latent from the prior; the
                # diagnostics mirror that exactly (frozen model, no gradient).
                z = agent.sample_latent(mu, log_var)
                rollout = collect_episode(
                    env, task, case, agent, z, "prior_support", agent.device,
                    episode_id=f"{task.task_id}:stage_a_diag:{index}", posterior_version=0,
                )
                rows.append(rollout.transitions)
                collected_steps += len(rollout.transitions)
            episodes[task.task_id] = rows
        finally:
            env.close()

    task_ids = (target_task.task_id, wrong_task.task_id)

    def sampled_groups(scheme: str, task_id: str, rng: np.random.Generator) -> list[list[Transition]]:
        return sample_context_scheme(episodes[task_id], per_episode, scheme, rng)

    def labelled_rows(scheme: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        blocks, block_labels, block_groups = [], [], []
        for label, task_id in enumerate(task_ids):
            for group_index, group in enumerate(sampled_groups(scheme, task_id, rng)):
                rows = context_rows([group], reward_scale)
                blocks.append(rows)
                block_labels.append(np.full(len(rows), label, dtype=int))
                block_groups.append(np.full(len(rows), f"{task_id}:{group_index}"))
        features = np.concatenate(blocks)
        labels = np.concatenate(block_labels)
        _, group_ids = np.unique(np.concatenate(block_groups), return_inverse=True)
        return features, labels, group_ids

    # --- Experiment A: exact-PEARL-input probe --------------------------
    exact_input_probe: dict[str, Any] = {}
    for scheme in ("random", "linspace"):
        seed = int(content_hash({"stage_a_probe": scheme})[:8], 16)
        features, labels, groups = labelled_rows(scheme, seed)
        matches = None
        if scheme == "random":
            # Byte-level cross-check against the agent's own preprocessing.
            rng = np.random.default_rng(seed)
            context_by_task = [sampled_groups(scheme, task_id, rng) for task_id in task_ids]
            tensor_rows = agent.context_tensor(context_by_task).detach().cpu().numpy()
            manual = context_rows([group for groups in context_by_task for group in groups], reward_scale)
            matches = bool(np.array_equal(tensor_rows.reshape(-1, tensor_rows.shape[-1]), manual))
            if not matches:
                raise RuntimeError("diagnostic context rows differ from PEARLAgent.context_tensor")
        exact_input_probe[scheme] = {
            "matches_agent_context_tensor": matches,
            "probe": logistic_probe_accuracy(
                features, labels, groups,
                seed=int(content_hash({"probe": scheme, "seed": seed})[:8], 16),
                folds=probe_folds,
            ),
        }

    # --- Experiment B: sampling ablation on the frozen encoder ----------
    sampling_ablation: dict[str, Any] = {}
    for scheme in ("random", "terminal_stratified_v1", "conflict_window"):
        draws = []
        for draw in range(int(sampling_draws)):
            rng = np.random.default_rng(
                int(content_hash({"stage_a_ablation": scheme, "draw": draw})[:8], 16)
            )
            correct = sampled_groups(scheme, target_task.task_id, rng)
            wrong = sampled_groups(scheme, wrong_task.task_id, rng)
            draws.append(posterior_separation(agent, correct, wrong, target_task))
        sampling_ablation[scheme] = {
            "draws": draws,
            "mean_prior_relative_separation_ratio": float(np.mean([
                row["prior_relative_separation_ratio"] for row in draws
            ])),
            "mean_correct_wrong_l2": float(np.mean([row["correct_wrong_l2"] for row in draws])),
            "mean_correct_wrong_cosine": float(np.mean([row["correct_wrong_cosine"] for row in draws])),
        }

    # --- Experiment C: channel ablation ---------------------------------
    seed = int(content_hash({"stage_a_channel": 1})[:8], 16)
    features, labels, groups = labelled_rows("random", seed)
    obs_dim = int(agent.observation_dim)
    act_dim = int(agent.action_dim)
    reward_index = obs_dim + act_dim
    done_start = reward_index + 1 + obs_dim
    masks: dict[str, np.ndarray] = {
        "full": np.ones(features.shape[1], dtype=bool),
        "without_reward": np.ones(features.shape[1], dtype=bool),
        "without_dynamics": np.zeros(features.shape[1], dtype=bool),
    }
    masks["without_reward"][reward_index] = False
    masks["without_dynamics"][[reward_index, done_start, done_start + 1]] = True
    channel_ablation = {
        name: logistic_probe_accuracy(
            features[:, mask], labels, groups,
            seed=int(content_hash({"probe_channel": name})[:8], 16),
            folds=probe_folds,
        )
        for name, mask in masks.items()
    }

    after = agent.parameter_hash()
    if before != after:
        raise RuntimeError("Stage-A diagnostics changed model parameters")
    return {
        "schema": "gate3_stage_a_diagnostics_v1",
        "target_task_id": target_task.task_id,
        "wrong_evidence_task_id": wrong_task.task_id,
        "split": split,
        "support_cases_per_task": int(support_cases),
        "context_transitions_per_episode": per_episode,
        "context_reward_scale": reward_scale,
        "collection_policy": "checkpoint prior policy, frozen, no gradients",
        "collection_environment_steps": int(collected_steps),
        "training_updates": 0,
        "no_gradient": True,
        "writes_replay_or_context": False,
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "exact_pearl_input_probe": exact_input_probe,
        "context_sampling_ablation": sampling_ablation,
        "context_channel_ablation": channel_ablation,
    }
