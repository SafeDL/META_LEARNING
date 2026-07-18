"""
SAC adaptation with optional g_phi warm-start.

Architecture:
  Actor  : SharedEncoder(obs) → head → (mean, log_std)  [tanh squash]
  Q1, Q2 : raw (obs, action)  → independent MLPs        [freshly init'd]

Init modes: "maml" | "joint" | "random"
"""

import json
import os
import random
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.env_wrapper import make_env
from src.networks import (
    SACActorNetwork, SACQNetwork, SharedEncoder,
    load_encoder_weights, xavier_init,
)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str):
        self.capacity = capacity
        self.device   = device
        self.ptr = 0
        self.size = 0
        self.obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew  = np.zeros(capacity,            dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity,            dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.act[self.ptr]      = action
        self.rew[self.ptr]      = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(self.device),
            torch.FloatTensor(self.act[idx]).to(self.device),
            torch.FloatTensor(self.rew[idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.done[idx]).to(self.device),
        )


def run_sac_adapt(
    topology: str,
    init_mode: str,
    encoder_ckpt: Optional[str],
    total_steps: int,
    seed: int,
    cfg: dict,
    out_dir: str,
    device: str = "cuda",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env_cfg = cfg["env"]
    sac_cfg = cfg["sac"]
    obs_dim = env_cfg["obs_dim"]
    act_dim = env_cfg["act_dim"]
    hidden  = cfg["encoder"]["hidden"]

    env = make_env(topology, seed=seed,
                   num_scenarios=env_cfg["num_scenarios"],
                   horizon=env_cfg["horizon"])

    # ── Build networks ────────────────────────────────────────────────────────
    encoder = SharedEncoder(obs_dim, hidden).to(device)
    actor   = SACActorNetwork(encoder, act_dim, hidden).to(device)
    qf1     = SACQNetwork(obs_dim, act_dim, hidden).to(device)
    qf2     = SACQNetwork(obs_dim, act_dim, hidden).to(device)
    qf1_tgt = SACQNetwork(obs_dim, act_dim, hidden).to(device)
    qf2_tgt = SACQNetwork(obs_dim, act_dim, hidden).to(device)
    qf1_tgt.load_state_dict(qf1.state_dict())
    qf2_tgt.load_state_dict(qf2.state_dict())

    if init_mode in ("maml", "joint") and encoder_ckpt is not None:
        load_encoder_weights(encoder, encoder_ckpt, device=device)
        xavier_init(actor.mean_head)
        xavier_init(actor.log_std_head)
        print(f"[SAC] Loaded encoder from {encoder_ckpt}  (init_mode={init_mode})")
    else:
        print("[SAC] Random initialisation")

    actor_opt = optim.Adam(actor.parameters(), lr=sac_cfg["lr"])
    qf_opt    = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()),
                           lr=sac_cfg["lr"])

    # Automatic entropy tuning
    target_entropy = -act_dim
    log_alpha       = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt       = optim.Adam([log_alpha], lr=sac_cfg["lr"])
    alpha           = log_alpha.exp().item()

    replay = ReplayBuffer(sac_cfg["buffer_size"], obs_dim, act_dim, device)

    gamma        = sac_cfg["gamma"]
    tau          = sac_cfg["tau"]
    batch_sz     = sac_cfg["batch_size"]
    learn_start  = sac_cfg["learning_starts"]
    train_freq   = sac_cfg["train_freq"]
    grad_steps   = sac_cfg["gradient_steps"]
    log_int      = cfg["adapt"]["log_interval"]

    episode_returns = []
    cur_return      = 0.0
    collision_count = 0
    total_episodes  = 0
    log_entries     = []
    t0 = time.time()

    obs, _ = env.reset(seed=seed)

    for global_step in range(1, total_steps + 1):
        # ── Select action ─────────────────────────────────────────────────────
        if global_step < learn_start:
            action = env.action_space.sample()
        else:
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_t)
            action = action.squeeze(0).cpu().numpy()

        next_obs, reward, done, _, info = env.step(action)
        replay.add(obs, action, reward, next_obs, float(done))
        cur_return += reward
        if info.get("collision", False):
            collision_count += 1
        obs = next_obs

        if done:
            episode_returns.append(cur_return)
            cur_return = 0.0
            total_episodes += 1
            obs, _ = env.reset()

        # ── Training ──────────────────────────────────────────────────────────
        if global_step >= learn_start and global_step % train_freq == 0:
            for _ in range(grad_steps):
                o, a, r, no, d = replay.sample(batch_sz)

                # ── Critic update ─────────────────────────────────────────────
                with torch.no_grad():
                    na, na_logp, _ = actor.get_action(no)
                    q1_next = qf1_tgt(no, na)
                    q2_next = qf2_tgt(no, na)
                    q_next  = torch.min(q1_next, q2_next) - alpha * na_logp
                    q_target = r + gamma * (1 - d) * q_next

                q1_pred = qf1(o, a)
                q2_pred = qf2(o, a)
                qf_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

                qf_opt.zero_grad()
                qf_loss.backward()
                qf_opt.step()

                # ── Actor update ──────────────────────────────────────────────
                new_a, new_logp, _ = actor.get_action(o)
                q1_new = qf1(o, new_a)
                q2_new = qf2(o, new_a)
                actor_loss = (alpha * new_logp - torch.min(q1_new, q2_new)).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # ── Alpha update ──────────────────────────────────────────────
                with torch.no_grad():
                    _, log_pi, _ = actor.get_action(o)
                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                alpha = log_alpha.exp().item()

                # Soft target update
                for p, tp in zip(qf1.parameters(), qf1_tgt.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
                for p, tp in zip(qf2.parameters(), qf2_tgt.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        # ── Logging ────────────────────────────────────────────────────────────
        if global_step % log_int == 0:
            mean_ret = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            elapsed  = time.time() - t0
            entry = {
                "step": global_step,
                "mean_return": round(float(mean_ret), 4),
                "collision_rate": collision_count / max(total_episodes, 1),
                "elapsed_s": round(elapsed, 1),
            }
            log_entries.append(entry)
            print(f"[SAC-{init_mode}|{topology}|seed{seed}] "
                  f"step={global_step:6d}  ret={mean_ret:.3f}  "
                  f"collisions={collision_count}/{total_episodes}")

    env.close()

    if len(log_entries) >= 2:
        steps_arr = np.array([e["step"] for e in log_entries], dtype=float)
        rets_arr  = np.array([e["mean_return"] for e in log_entries], dtype=float)
        return_auc = float(np.trapz(rets_arr, steps_arr) / max(total_steps, 1))
    else:
        return_auc = float(log_entries[0]["mean_return"]) if log_entries else 0.0

    results = {
        "algorithm":        "sac",
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
    torch.save({"actor": actor.state_dict(), "encoder": encoder.state_dict()},
               os.path.join(out_dir, "policy_final.pt"))
    return results
