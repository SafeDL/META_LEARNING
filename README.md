# META_LEARNING

Experiment code for meta-learning an adversarial non-player-character (NPC) policy in MetaDrive road scenarios. The controlled NPC is trained to create challenging interactions with a traffic vehicle selected as the target ("ego") vehicle.

The project compares a FOMAML-pretrained shared encoder with a joint-pretrained encoder and random initialization, then adapts policies with PPO, SAC, or TD3 on held-out road topologies.

## What is implemented

- A Gymnasium-compatible adversarial NPC environment with a 38-dimensional normalized observation and a two-dimensional continuous action `[acceleration, steering]`.
- An adversarial reward composed of negative time-to-collision, a collision bonus, and an action penalty:

  `-min(TTC, ttc_scale) / ttc_scale + collision_bonus * collision - action_reg * ||action||²`

- A shared tanh encoder with the default shape `38 -> 256 -> 256`.
- FOMAML meta-training on `highway`, `merge`, `t_junction`, and `intersection`.
- A joint PPO pretraining baseline on the same four topologies.
- PPO, SAC, and TD3 adaptation on the held-out `roundabout` and `y_junction` topologies.
- Aggregation of return-AUC tables and bootstrap ATE statistics for MAML versus joint pretraining.

Only the encoder is transferred to adaptation. Policy heads, value functions, and Q-functions are initialized afresh for every adaptation run.

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/default.yaml` | Default environment, reward, network, and training hyperparameters. |
| `src/env_wrapper.py` | MetaDrive wrapper, observation construction, and adversarial reward. |
| `src/fomaml.py` | FOMAML trainer. |
| `src/adapt_ppo.py`, `src/adapt_sac.py`, `src/adapt_td3.py` | Per-algorithm adaptation implementations. |
| `src/networks.py` | Shared encoder, policy/critic networks, initialization, and checkpoint helpers. |
| `src/analysis.py` | Result loading, ATE estimation, and optional diversity utilities. |
| `scripts/run_sanity.py` | Environment and short PPO smoke test. |
| `scripts/run_meta_train.py` | FOMAML encoder training. |
| `scripts/run_joint_pretrain.py` | Joint-pretraining encoder baseline. |
| `scripts/run_adapt.py` | One adaptation run. |
| `scripts/deploy_b1_grid.sh` | Sequential launch of the 54-run main grid (Bash/CUDA). |
| `scripts/compute_results.py` | CSV, JSON, and optional LaTeX result tables. |
| `ref_code/metadrive-scenario/` | MetaDrive scenario reference source included as ordinary source code. |

## Requirements and installation

Use Python 3.9 or newer. A CUDA-enabled PyTorch installation is recommended for full-scale experiments; the training and adaptation entry points select `cuda` when it is available and otherwise select `cpu`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`metadrive-simulator` is listed in `requirements.txt`. If it is unavailable at runtime, `src/env_wrapper.py` automatically uses its lightweight toy environment instead. This makes the scripts runnable for code checks, but toy-environment results are not MetaDrive simulation results.

## Reproducing the main experiment

All commands below are run from the repository root. Pass `--device cpu` to force CPU execution, or `--device cuda` to require CUDA.

### 1. Run the sanity checks

```bash
python scripts/run_sanity.py --config configs/default.yaml --out_dir results/sanity
```

This performs an environment smoke test over every configured topology and a 1,024-step random-initialized PPO run on `roundabout`. It writes `results/sanity/sanity_summary.json` and the smoke-run result.

### 2. Train the two encoder initializations

```bash
python scripts/run_meta_train.py \
  --config configs/default.yaml \
  --out_dir results/meta_train \
  --seed 42 \
  --device cuda

python scripts/run_joint_pretrain.py \
  --config configs/default.yaml \
  --out_dir results/joint_pretrain \
  --seed 42 \
  --device cuda
```

Both use the configured `fomaml.total_env_steps` budget (200,000 by default) and save an encoder checkpoint named `encoder_final.pt`.

### 3. Run an adaptation cell

Each main-grid cell is defined by algorithm, held-out topology, initialization mode, and seed. `maml` and `joint` require the corresponding encoder checkpoint; `random` does not.

```bash
# FOMAML initialization
python scripts/run_adapt.py \
  --algo ppo \
  --topology roundabout \
  --init_mode maml \
  --encoder_ckpt results/meta_train/encoder_final.pt \
  --seed 0 \
  --config configs/default.yaml \
  --device cuda

# Joint-pretrained initialization
python scripts/run_adapt.py \
  --algo sac \
  --topology y_junction \
  --init_mode joint \
  --encoder_ckpt results/joint_pretrain/encoder_final.pt \
  --seed 0 \
  --config configs/default.yaml \
  --device cuda

# Random-initialization baseline
python scripts/run_adapt.py \
  --algo td3 \
  --topology roundabout \
  --init_mode random \
  --seed 0 \
  --config configs/default.yaml \
  --device cuda
```

Use `--total_steps N` to override the configured budget for one run. The override is applied to PPO, SAC, and TD3 settings for that invocation.

### 4. Launch the full grid

The main grid contains `3 algorithms × 2 held-out topologies × 3 initialization modes × 3 seeds = 54` runs. The Bash script runs them sequentially, skips cells with an existing `results.json`, and writes one log per run under `results/logs/`.

```bash
# Inspect commands without launching runs
bash scripts/deploy_b1_grid.sh --dry-run

# Run the grid after both encoder checkpoints have been generated
bash scripts/deploy_b1_grid.sh
```

The deployment script is intended for a Bash environment and invokes adaptation with `--device cuda`. On Windows without Bash, run the individual `run_adapt.py` commands instead (or use WSL/Git Bash).

### 5. Aggregate results

```bash
python scripts/compute_results.py --results_root results

# Also print LaTeX table source
python scripts/compute_results.py --results_root results --latex
```

The ATE calculation uses 10,000 bootstrap resamples with a 95% percentile confidence interval. A MAML-versus-joint cell passes the configured criterion when the ATE is above `0.10` and the lower confidence bound is positive.

## Outputs

| Output | Description |
| --- | --- |
| `results/meta_train/encoder_final.pt` | FOMAML encoder checkpoint. |
| `results/joint_pretrain/encoder_final.pt` | Joint-pretrained encoder checkpoint. |
| `results/<algo>/<topology>_<init>_seed<seed>/results.json` | Metrics, episode returns, and periodic logs for one adaptation cell. |
| `results/<algo>/<topology>_<init>_seed<seed>/policy_final.pt` | Final algorithm policy and encoder state for one cell. |
| `results/logs/*.log` | Per-run output from the batch deployment script. |
| `results/tables/main_table.csv` | Mean ± standard deviation of return AUC. |
| `results/tables/ate_table.csv` | MAML-versus-joint ATE table; created when each comparison has at least two seeds. |
| `results/tables/summary.json` | Structured summary of detected runs and ATE coverage. |

When an `results/ablation/` directory with supported result files is present, the aggregation script also writes `results/tables/ablation_table.csv`.

## Configuration

Edit `configs/default.yaml` to change experiment settings. The most relevant sections are:

- `env`: scenario count, episode horizon, and observation/action dimensions.
- `topology`: train and held-out topology lists.
- `reward`: TTC scale, collision bonus, and action regularization.
- `encoder`: shared-encoder width and activation.
- `fomaml`, `ppo`, `sac`, and `td3`: training budgets and optimizer/algorithm parameters.
- `adapt`: adaptation logging interval and canonical step budget.

Generated `results/` and `logs/` directories are excluded by `.gitignore`.
