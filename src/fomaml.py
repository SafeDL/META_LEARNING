"""
First-Order MAML (FOMAML) meta-training for the adversarial NPC agent.

Algorithm overview
------------------
For each outer iteration:
  1. Sample meta_batch topology tasks from T_train.
  2. For each task:
     a. Collect rollout_batch steps with current policy theta.
     b. Compute PPO loss, take k inner gradient steps with inner_lr
        (using first-order approximation — no second derivatives).
     c. Collect a second rollout with adapted weights theta_i'.
     d. Compute outer loss on this second rollout.
  3. Average outer losses, update theta with outer_lr (Adam).
  4. Save encoder g_phi checkpoint periodically.

Hyperparameters (from configs/default.yaml):
  inner_lr:     0.01
  outer_lr:     0.001
  inner_steps:  3
  meta_batch:   8
  total_env_steps: 200000
  rollout_batch: 64
"""

import copy
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.env_wrapper import make_env, set_normalisation_stats
from src.networks import PPOActorCritic, save_checkpoint

TRAIN_TOPOLOGIES = ["highway", "merge", "t_junction", "intersection"]


# ─── Rollout collector ────────────────────────────────────────────────────────

def collect_rollout(env, policy: PPOActorCritic, n_steps: int, device: str):
    """Collect up to n_steps transitions; return batched tensors."""
    obs_list, act_list, logp_list, rew_list, done_list, val_list = [], [], [], [], [], []
    obs, _ = env.reset()
    done = False
    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, logp, _, value = policy.get_action_and_value(obs_t)
        a_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, done, _, _ = env.step(a_np)
        obs_list.append(obs)
        act_list.append(a_np)
        logp_list.append(logp.item())
        rew_list.append(reward)
        done_list.append(float(done))
        val_list.append(value.item())
        obs = next_obs
        if done:
            obs, _ = env.reset()
            done = False

    obs_t   = torch.FloatTensor(np.array(obs_list)).to(device)
    act_t   = torch.FloatTensor(np.array(act_list)).to(device)
    logp_t  = torch.FloatTensor(logp_list).to(device)
    rew_t   = torch.FloatTensor(rew_list).to(device)
    done_t  = torch.FloatTensor(done_list).to(device)
    val_t   = torch.FloatTensor(val_list).to(device)
    return obs_t, act_t, logp_t, rew_t, done_t, val_t


def compute_returns(rewards, dones, values, gamma=0.99, gae_lambda=0.95):
    """GAE-Lambda advantage + return estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae   = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta    = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_loss(policy, obs, actions, old_logp, advantages, returns,
             clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """Single PPO loss for inner-loop updates."""
    _, new_logp, entropy, value = policy.get_action_and_value(obs, actions)
    ratio  = (new_logp - old_logp).exp()
    pg1    = ratio * advantages
    pg2    = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    pg_loss = -torch.min(pg1, pg2).mean()
    vf_loss = ((value - returns) ** 2).mean()
    ent_loss = -entropy.mean()
    return pg_loss + vf_coef * vf_loss + ent_coef * ent_loss


# ─── FOMAML trainer ───────────────────────────────────────────────────────────

class FOMAMLTrainer:

    def __init__(self, cfg: dict, device: str = "cpu", out_dir: str = "results/meta_train"):
        self.cfg       = cfg
        self.device    = device
        self.out_dir   = out_dir
        os.makedirs(out_dir, exist_ok=True)

        fomaml_cfg   = cfg["fomaml"]
        self.inner_lr    = fomaml_cfg["inner_lr"]
        self.outer_lr    = fomaml_cfg["outer_lr"]
        self.inner_steps = fomaml_cfg["inner_steps"]
        self.meta_batch  = fomaml_cfg["meta_batch"]
        self.total_steps = fomaml_cfg["total_env_steps"]
        self.rollout_batch = fomaml_cfg["rollout_batch"]

        ppo_cfg      = cfg["ppo"]
        self.gamma       = ppo_cfg["gamma"]
        self.gae_lambda  = ppo_cfg["gae_lambda"]
        self.clip_eps    = ppo_cfg["clip_eps"]
        self.vf_coef     = ppo_cfg["vf_coef"]
        self.ent_coef    = ppo_cfg["entropy_coef"]

        # MetaDrive has a single global engine; we create/destroy envs one at a time.
        self.num_scenarios = cfg["env"]["num_scenarios"]
        self._env_seed = 0   # bumped per outer iteration

        # Meta-policy (theta)
        self.meta_policy = PPOActorCritic(
            obs_dim=cfg["env"]["obs_dim"],
            act_dim=cfg["env"]["act_dim"],
            hidden=cfg["encoder"]["hidden"],
        ).to(device)

        # Outer optimizer updates encoder g_phi only (proposal: "record gradient w.r.t. g_phi only")
        self.outer_opt = optim.Adam(self.meta_policy.encoder.parameters(), lr=self.outer_lr)

        self.log: List[dict] = []

    def _inner_adapt(self, env, base_params: dict) -> PPOActorCritic:
        """
        Clone meta-policy, collect a rollout, take k SGD steps on PPO loss.
        Returns adapted policy (with first-order gradients w.r.t. base_params).
        """
        adapted = copy.deepcopy(self.meta_policy)
        adapted.load_state_dict(base_params)

        for _ in range(self.inner_steps):
            obs, acts, logp, rews, dones, vals = collect_rollout(
                env, adapted, self.rollout_batch, self.device)
            adv, ret = compute_returns(rews, dones, vals, self.gamma, self.gae_lambda)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            loss = ppo_loss(adapted, obs, acts, logp, adv, ret,
                            self.clip_eps, self.vf_coef, self.ent_coef)

            grads = torch.autograd.grad(loss, adapted.parameters(),
                                        create_graph=False, allow_unused=True)
            with torch.no_grad():
                for p, g in zip(adapted.parameters(), grads):
                    if g is not None:
                        p.data -= self.inner_lr * g

        return adapted

    def train(self) -> None:
        global_steps  = 0
        outer_iter    = 0
        rng           = np.random.default_rng(42)
        t0            = time.time()

        while global_steps < self.total_steps:
            # ── Sample meta-batch of topology tasks ──────────────────────────
            tasks = rng.choice(TRAIN_TOPOLOGIES, size=self.meta_batch, replace=True)
            base_params = {k: v.clone() for k, v in self.meta_policy.state_dict().items()}

            # Accumulate encoder gradients across tasks (FOMAML first-order approximation).
            # We only update g_phi (encoder), so accumulate grads only for encoder params.
            enc_params     = list(self.meta_policy.encoder.parameters())
            enc_grad_acc   = [torch.zeros_like(p) for p in enc_params]
            mean_outer_loss = 0.0

            for task_i, topo in enumerate(tasks):
                # Create a fresh env for this task (MetaDrive has a global engine)
                env = make_env(topo, seed=self._env_seed + task_i,
                               num_scenarios=self.num_scenarios)

                # Inner adaptation (returns deep-copied policy with adapted params)
                adapted = self._inner_adapt(env, {k: v.clone() for k, v in base_params.items()})

                # Query rollout with adapted policy
                obs, acts, logp, rews, dones, vals = collect_rollout(
                    env, adapted, self.rollout_batch, self.device)
                adv, ret = compute_returns(rews, dones, vals, self.gamma, self.gae_lambda)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Query loss is computed directly on `adapted` — gradients flow through
                # adapted.encoder.parameters() (FOMAML: treat as grad w.r.t. meta params)
                query_loss = ppo_loss(adapted, obs, acts, logp, adv, ret,
                                      self.clip_eps, self.vf_coef, self.ent_coef)

                # Gradient only w.r.t. encoder params of adapted policy
                adapted_enc = list(adapted.encoder.parameters())
                task_grads  = torch.autograd.grad(query_loss, adapted_enc,
                                                  allow_unused=True, create_graph=False)
                for i, tg in enumerate(task_grads):
                    if tg is not None:
                        enc_grad_acc[i] = enc_grad_acc[i] + tg / self.meta_batch

                mean_outer_loss += query_loss.item() / self.meta_batch
                global_steps += self.rollout_batch * (self.inner_steps + 1)
                env.close()   # close immediately — MetaDrive singleton engine

            self._env_seed += self.meta_batch

            # ── Outer gradient step (encoder only) ───────────────────────────
            self.meta_policy.load_state_dict(base_params)   # restore base weights
            self.outer_opt.zero_grad()
            for p, g in zip(enc_params, enc_grad_acc):
                p.grad = g.detach()
            nn.utils.clip_grad_norm_(enc_params, 0.5)
            self.outer_opt.step()

            outer_iter += 1

            # ── Logging ────────────────────────────────────────────────────────
            if outer_iter % 20 == 0:
                elapsed = time.time() - t0
                entry = {
                    "outer_iter": outer_iter,
                    "global_steps": global_steps,
                    "outer_loss": round(mean_outer_loss, 6),
                    "elapsed_s": round(elapsed, 1),
                }
                self.log.append(entry)
                print(f"[FOMAML] iter={outer_iter:5d}  steps={global_steps:7d}  "
                      f"loss={mean_outer_loss:.4f}  t={elapsed:.0f}s")

            # ── Checkpoint ────────────────────────────────────────────────────
            if outer_iter % 200 == 0 or global_steps >= self.total_steps:
                ckpt_path = os.path.join(self.out_dir, f"encoder_step{global_steps}.pt")
                save_checkpoint(self.meta_policy.encoder, ckpt_path,
                                extra={"outer_iter": outer_iter,
                                       "global_steps": global_steps})
                print(f"  Saved checkpoint → {ckpt_path}")

        # Final checkpoint
        final_ckpt = os.path.join(self.out_dir, "encoder_final.pt")
        save_checkpoint(self.meta_policy.encoder, final_ckpt)
        print(f"\nMeta-training complete.  Encoder saved → {final_ckpt}")

        # Save log
        log_path = os.path.join(self.out_dir, "fomaml_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"Training log → {log_path}")

    def close(self):
        pass  # envs are closed per-task during training
