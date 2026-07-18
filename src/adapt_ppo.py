"""
PPO adaptation with optional g_phi warm-start.

Init modes:
  "maml"         — load encoder weights from FOMAML checkpoint
  "joint"        — load encoder weights from joint-pretrain checkpoint
  "random"       — freshly Xavier-initialised (no transfer)

All critic / value-head parameters are freshly initialised regardless of mode.
"""

import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env_wrapper import make_env
from src.networks import PPOActorCritic, SharedEncoder, load_encoder_weights, xavier_init


def run_ppo_adapt(
    topology: str,
    init_mode: str,               # "maml" | "joint" | "random"
    encoder_ckpt: Optional[str],  # path to .pt if init_mode != "random"
    total_steps: int,
    seed: int,
    cfg: dict,
    out_dir: str,
    device: str = "cuda",
) -> dict:
    """
    Run PPO adaptation on *topology*.  Returns a dict of results.

    Results saved to: <out_dir>/results.json
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg = cfg["env"]
    ppo_cfg = cfg["ppo"]

    env = make_env(topology, seed=seed,
                   num_scenarios=env_cfg["num_scenarios"],
                   horizon=env_cfg["horizon"])

    # ── Build policy ──────────────────────────────────────────────────────────
    policy = PPOActorCritic(
        obs_dim=env_cfg["obs_dim"],
        act_dim=env_cfg["act_dim"],
        hidden=cfg["encoder"]["hidden"],
    ).to(device)

    if init_mode in ("maml", "joint") and encoder_ckpt is not None:
        load_encoder_weights(policy.encoder, encoder_ckpt, device=device)
        # Re-init critic and actor head (not the encoder trunk)
        xavier_init(policy.actor.head)
        xavier_init(policy.critic.net)
        policy.actor.log_std.data.zero_()
        print(f"[PPO] Loaded encoder from {encoder_ckpt}  (init_mode={init_mode})")
    else:
        print("[PPO] Random initialisation")

    optimizer = optim.Adam(policy.parameters(), lr=ppo_cfg["lr"])

    # Adaptation loop parameters
    n_steps   = ppo_cfg["n_steps"]
    n_epochs  = ppo_cfg["n_epochs"]
    batch_sz  = ppo_cfg["batch_size"]
    gamma     = ppo_cfg["gamma"]
    gae_lam   = ppo_cfg["gae_lambda"]
    clip_eps  = ppo_cfg["clip_eps"]
    vf_coef   = ppo_cfg["vf_coef"]
    ent_coef  = ppo_cfg["entropy_coef"]
    max_grad  = ppo_cfg["max_grad_norm"]
    log_int   = cfg["adapt"]["log_interval"]

    # ── Training loop ─────────────────────────────────────────────────────────
    obs_buf  = torch.zeros((n_steps, env_cfg["obs_dim"]), dtype=torch.float32).to(device)
    act_buf  = torch.zeros((n_steps, env_cfg["act_dim"]), dtype=torch.float32).to(device)
    logp_buf = torch.zeros(n_steps, dtype=torch.float32).to(device)
    rew_buf  = torch.zeros(n_steps, dtype=torch.float32).to(device)
    done_buf = torch.zeros(n_steps, dtype=torch.float32).to(device)
    val_buf  = torch.zeros(n_steps, dtype=torch.float32).to(device)

    global_step = 0
    episode_returns: list = []
    cur_ep_return = 0.0
    log_entries: list = []
    collision_count = 0
    total_episodes = 0

    obs, _ = env.reset(seed=seed)
    t0 = time.time()

    while global_step < total_steps:
        # ── Collect rollout ────────────────────────────────────────────────────
        for s in range(n_steps):
            obs_t = torch.FloatTensor(obs).to(device)
            obs_buf[s] = obs_t

            with torch.no_grad():
                action, logp, _, value = policy.get_action_and_value(obs_t.unsqueeze(0))
            action = action.squeeze(0)

            obs_buf[s]  = obs_t
            act_buf[s]  = action
            logp_buf[s] = logp.squeeze()
            val_buf[s]  = value.squeeze()

            a_np = action.cpu().numpy()
            next_obs, reward, done, _, info = env.step(a_np)
            rew_buf[s]  = reward
            done_buf[s] = float(done)
            cur_ep_return += reward
            if info.get("collision", False):
                collision_count += 1

            obs = next_obs
            if done:
                episode_returns.append(cur_ep_return)
                cur_ep_return = 0.0
                total_episodes += 1
                obs, _ = env.reset()

        global_step += n_steps

        # ── GAE advantages (with correct bootstrap when episode still running) ──
        with torch.no_grad():
            # obs_t holds the current observation after the rollout ended
            bootstrap_val = policy.get_value(obs_t.unsqueeze(0)).item()
        advantages = torch.zeros_like(rew_buf)
        last_gae   = 0.0
        for t in reversed(range(n_steps)):
            # If last step and episode not done, bootstrap from next-state value
            if t == n_steps - 1:
                next_val = 0.0 if done_buf[t] else bootstrap_val
            else:
                next_val = val_buf[t + 1].item() if not done_buf[t] else 0.0
            delta    = rew_buf[t] + gamma * next_val - val_buf[t]
            last_gae = delta + gamma * gae_lam * (1 - done_buf[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO update ────────────────────────────────────────────────────────
        idx_all = np.arange(n_steps)
        for _ in range(n_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n_steps, batch_sz):
                idx = idx_all[start: start + batch_sz]
                _, new_logp, entropy, new_val = policy.get_action_and_value(
                    obs_buf[idx], act_buf[idx])
                ratio   = (new_logp - logp_buf[idx]).exp()
                adv_b   = advantages[idx]
                pg1     = ratio * adv_b
                pg2     = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_b
                pg_loss = -torch.min(pg1, pg2).mean()
                vf_loss = ((new_val - returns[idx]) ** 2).mean()
                loss    = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
                optimizer.step()

        # ── Logging ────────────────────────────────────────────────────────────
        # Use boundary-crossing check: fires whenever we pass a log_int multiple,
        # even when global_step increments by n_steps (which may not be divisible by log_int).
        if global_step // log_int > (global_step - n_steps) // log_int:
            mean_ret = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            elapsed  = time.time() - t0
            entry = {
                "step": global_step,
                "mean_return": round(float(mean_ret), 4),
                "collision_rate": collision_count / max(total_episodes, 1),
                "elapsed_s": round(elapsed, 1),
            }
            log_entries.append(entry)
            print(f"[PPO-{init_mode}|{topology}|seed{seed}] "
                  f"step={global_step:6d}  ret={mean_ret:.3f}  "
                  f"collisions={collision_count}/{total_episodes}")

    env.close()

    # ── AUC: trapezoidal integration of mean_return over steps, normalised ────
    if len(log_entries) >= 2:
        steps_arr = np.array([e["step"] for e in log_entries], dtype=float)
        rets_arr  = np.array([e["mean_return"] for e in log_entries], dtype=float)
        return_auc = float(np.trapz(rets_arr, steps_arr) / max(total_steps, 1))
    else:
        return_auc = float(log_entries[0]["mean_return"]) if log_entries else 0.0

    results = {
        "algorithm":        "ppo",
        "topology":         topology,
        "init_mode":        init_mode,
        "seed":             seed,
        "total_steps":      total_steps,
        "return_auc":       return_auc,
        "final_mean_return": log_entries[-1]["mean_return"] if log_entries else None,
        "total_collisions": collision_count,
        "total_episodes":   total_episodes,
        "episode_returns":  episode_returns,
        "log":              log_entries,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save policy checkpoint
    torch.save(policy.state_dict(), os.path.join(out_dir, "policy_final.pt"))

    return results
