#!/usr/bin/env python
"""
R011 — Joint pre-training baseline.

Trains a PPO agent on a mixture of all training topologies simultaneously
(no MAML structure).  Saves the encoder checkpoint for use as the
joint-pretrain warm-start baseline.

Produces:  results/joint_pretrain/encoder_final.pt

Usage:
  python scripts/run_joint_pretrain.py --config configs/default.yaml
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env_wrapper import make_env
from src.networks import PPOActorCritic, save_checkpoint

TRAIN_TOPOLOGIES = ["highway", "merge", "t_junction", "intersection"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="results/joint_pretrain")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device   = args.device
    ppo_cfg  = cfg["ppo"]
    env_cfg  = cfg["env"]
    obs_dim  = env_cfg["obs_dim"]
    act_dim  = env_cfg["act_dim"]
    hidden   = cfg["encoder"]["hidden"]

    total_steps   = cfg["fomaml"]["total_env_steps"]   # same budget as FOMAML
    rollout_steps = ppo_cfg["n_steps"]
    n_epochs      = ppo_cfg["n_epochs"]
    batch_sz      = ppo_cfg["batch_size"]
    gamma         = ppo_cfg["gamma"]
    gae_lam       = ppo_cfg["gae_lambda"]
    clip_eps      = ppo_cfg["clip_eps"]
    vf_coef       = ppo_cfg["vf_coef"]
    ent_coef      = ppo_cfg["entropy_coef"]
    max_grad      = ppo_cfg["max_grad_norm"]
    lr            = ppo_cfg["lr"]

    # MetaDrive uses a global engine — only ONE env can be open at a time.
    # We create a fresh env per rollout batch and close it immediately after.
    policy    = PPOActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    obs_buf  = torch.zeros((rollout_steps, obs_dim)).to(device)
    act_buf  = torch.zeros((rollout_steps, act_dim)).to(device)
    logp_buf = torch.zeros(rollout_steps).to(device)
    rew_buf  = torch.zeros(rollout_steps).to(device)
    done_buf = torch.zeros(rollout_steps).to(device)
    val_buf  = torch.zeros(rollout_steps).to(device)

    log_entries   = []
    episode_returns = []
    cur_return    = 0.0
    global_step   = 0
    rng           = np.random.default_rng(args.seed)
    t0            = time.time()

    topo_cycle = list(TRAIN_TOPOLOGIES)
    topo_idx   = 0

    print(f"[R011] Joint pre-training: {total_steps} env steps, device={device}")

    while global_step < total_steps:
        # Pick topology, open env, collect one rollout, then close env
        topo = topo_cycle[topo_idx % len(topo_cycle)]
        topo_idx += 1
        env = make_env(topo, seed=args.seed + topo_idx,
                       num_scenarios=env_cfg["num_scenarios"])
        obs, _ = env.reset()

        for s in range(rollout_steps):
            obs_t = torch.FloatTensor(obs).to(device)
            obs_buf[s] = obs_t
            with torch.no_grad():
                action, logp, _, value = policy.get_action_and_value(obs_t.unsqueeze(0))
            action = action.squeeze(0)
            act_buf[s]  = action
            logp_buf[s] = logp.squeeze()
            val_buf[s]  = value.squeeze()

            next_obs, reward, done, _, _ = env.step(action.cpu().numpy())
            rew_buf[s]  = reward
            done_buf[s] = float(done)
            cur_return += reward
            obs = next_obs
            if done:
                episode_returns.append(cur_return)
                cur_return = 0.0
                obs, _ = env.reset()

        env.close()   # close immediately — do not hold multiple MetaDrive envs open
        global_step   += rollout_steps

        # GAE — bootstrap from current obs value when episode still running
        obs_t_now = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            bootstrap_val = policy.get_value(obs_t_now.unsqueeze(0)).item()

        advantages = torch.zeros_like(rew_buf)
        last_gae   = 0.0
        for t_idx in reversed(range(rollout_steps)):
            if t_idx == rollout_steps - 1:
                nv = 0.0 if done_buf[t_idx] else bootstrap_val
            else:
                nv = val_buf[t_idx + 1].item() if not done_buf[t_idx] else 0.0
            delta    = rew_buf[t_idx] + gamma * nv - val_buf[t_idx]
            last_gae = delta + gamma * gae_lam * (1 - done_buf[t_idx]) * last_gae
            advantages[t_idx] = last_gae
        returns     = advantages + val_buf
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx_all = np.arange(rollout_steps)
        for _ in range(n_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, rollout_steps, batch_sz):
                idx = idx_all[start: start + batch_sz]
                _, new_logp, entropy, new_val = policy.get_action_and_value(
                    obs_buf[idx], act_buf[idx])
                ratio   = (new_logp - logp_buf[idx]).exp()
                adv_b   = advantages[idx]
                pg_loss = -torch.min(ratio * adv_b,
                                     ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_b).mean()
                vf_loss = ((new_val - returns[idx]) ** 2).mean()
                loss    = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
                optimizer.step()

        if global_step % cfg["adapt"]["log_interval"] == 0:
            mean_ret = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            entry = {"step": global_step, "mean_return": round(float(mean_ret), 4),
                     "elapsed_s": round(time.time() - t0, 1)}
            log_entries.append(entry)
            print(f"[JointPretrain] step={global_step:7d}  ret={mean_ret:.3f}")

    ckpt_path = os.path.join(args.out_dir, "encoder_final.pt")
    save_checkpoint(policy.encoder, ckpt_path)
    print(f"\n[R011] Done. Encoder saved → {ckpt_path}")

    with open(os.path.join(args.out_dir, "log.json"), "w") as f:
        json.dump(log_entries, f, indent=2)


if __name__ == "__main__":
    main()
