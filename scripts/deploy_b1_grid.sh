#!/bin/bash
# Deploy the B1 main adaptation grid:
#   3 init modes x 3 algos x 2 held-out topologies x 3 seeds = 54 runs
#
# Usage:
#   bash scripts/deploy_b1_grid.sh
#   bash scripts/deploy_b1_grid.sh --dry-run

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="$BASE/configs/default.yaml"
MAML_CKPT="$BASE/results/meta_train/encoder_final.pt"
JOINT_CKPT="$BASE/results/joint_pretrain/encoder_final.pt"
DRY="${1:-}"

ALGOS=(ppo sac td3)
TOPOS=(roundabout y_junction)
INIT_MODES=(maml joint random)
SEEDS=(0 1 2)

mkdir -p "$BASE/results/logs"

run_id=0
done_count=0
skip_count=0

for algo in "${ALGOS[@]}"; do
  for topo in "${TOPOS[@]}"; do
    for init in "${INIT_MODES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_id=$((run_id + 1))
        tag="R$(printf '%03d' "$run_id")_${algo}_${topo}_${init}_seed${seed}"
        logfile="$BASE/results/logs/${tag}.log"
        out_dir="$BASE/results/${algo}/${topo}_${init}_seed${seed}"
        results_json="$out_dir/results.json"

        if [ "$init" = "maml" ]; then
          ckpt_arg="--encoder_ckpt $MAML_CKPT"
        elif [ "$init" = "joint" ]; then
          ckpt_arg="--encoder_ckpt $JOINT_CKPT"
        else
          ckpt_arg=""
        fi

        cmd="PYTHONPATH=$BASE $PYTHON_BIN $BASE/scripts/run_adapt.py \
          --algo $algo --topology $topo --init_mode $init \
          --seed $seed --config $CFG $ckpt_arg \
          --out_root $BASE/results --device cuda"

        if [ -n "$DRY" ]; then
          if [ -f "$results_json" ]; then
            echo "[SKIP] $tag (results.json exists)"
          else
            echo "[DRY]  $tag: $cmd"
          fi
        else
          if [ -f "$results_json" ]; then
            echo "[SKIP] $tag (already done)"
            skip_count=$((skip_count + 1))
          else
            echo "[RUN]  $tag ..."
            (cd "$BASE" && eval "$cmd") > "$logfile" 2>&1
            if [ $? -eq 0 ]; then
              echo "[OK]   $tag"
              done_count=$((done_count + 1))
            else
              echo "[FAIL] $tag -- see $logfile"
            fi
          fi
        fi
      done
    done
  done
done

echo ""
echo "Done: $done_count new runs completed, $skip_count skipped (already done). Total: $run_id runs."
