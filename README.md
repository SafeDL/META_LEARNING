# Main Experiment Only

This directory contains only the code needed to reproduce the main experiment from this project.

Included scope:

- FOMAML meta-training on four training topologies:
  - `highway`
  - `merge`
  - `t_junction`
  - `intersection`
- Joint-pretrain baseline on the same four training topologies
- Adaptation on two held-out topologies:
  - `roundabout`
  - `y_junction`
- Three adaptation algorithms:
  - `PPO`
  - `SAC`
  - `TD3`
- Three initialization modes:
  - `maml`
  - `joint`
  - `random`

Excluded on purpose:

- Ablations
- Rescue runs
- Stability runs
- Loop-3 follow-up scripts
- Extra paper-specific analysis

## Directory layout

- `configs/default.yaml`: hyperparameters
- `src/`: environment, networks, meta-training, adaptation, and analysis
- `scripts/run_meta_train.py`: train the meta-learned encoder
- `scripts/run_joint_pretrain.py`: train the joint-pretrain baseline encoder
- `scripts/run_adapt.py`: run one adaptation cell
- `scripts/deploy_b1_grid.sh`: run the full 54-cell main grid
- `scripts/compute_results.py`: aggregate paper-style result tables
- `scripts/run_sanity.py`: smoke test before long runs

## Main grid

- `3 algorithms x 2 held-out topologies x 3 init modes x 3 seeds = 54 runs`

Held-out topologies:

- `roundabout`
- `y_junction`

Main metric:

- `return_auc`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need a working MetaDrive installation and a PyTorch-compatible machine.

## Reproduce

1. Sanity check

```bash
python scripts/run_sanity.py --config configs/default.yaml
```

2. Train the FOMAML encoder

```bash
python scripts/run_meta_train.py --config configs/default.yaml
```

3. Train the joint-pretrain encoder

```bash
python scripts/run_joint_pretrain.py --config configs/default.yaml
```

4. Run the full main experiment grid

```bash
bash scripts/deploy_b1_grid.sh
```

5. Aggregate the main tables

```bash
python scripts/compute_results.py --results_root results
```

## Expected outputs

- `results/meta_train/encoder_final.pt`
- `results/joint_pretrain/encoder_final.pt`
- `results/<algo>/<topology>_<init>_seed<seed>/results.json`
- `results/tables/main_table.csv`
- `results/tables/ate_table.csv`
- `results/tables/summary.json`

## Notes

- `joint` means a jointly pre-trained encoder learned on the four training topologies.
- `random` means no pretrained encoder is loaded before adaptation.
- Only the encoder is transferred; adaptation heads and critics are re-initialized.
