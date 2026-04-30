#!/usr/bin/env bash
# DM0 phase-1 (freeze ViT). Run from anywhere:
#     bash lerobot/bt/dm0/run_dm0_freeze_vit.sh
#     bash run_dm0_freeze_vit.sh
#
# Auto-generates norm_stats.json on first run (DM0 needs dexbotic-style
# q01/q99 over delta-action chunks; the dataset's `meta/stats.json` is *not*
# used by DM0, and the one shipped with DM0-base is a [-1,1] placeholder).

set -euo pipefail

export HF_HOME="/mnt/r/share/zwy/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
# mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

export WANDB_API_KEY="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"

# export WANDB_DISABLED=true

# Anchor everything to the lerobot repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_PATH="/mnt/r/share/zwy/datasets/r1_pro_data_convert_chassis_v3_newnorm"
DM0_BASE="./checkpoints/DM0-base"
NORM_STATS="./norm_stats/r1_pro_chassis_v3.json"

# 1) compute norm_stats.json if missing (one-shot, ~tens of seconds).
if [[ ! -f "${NORM_STATS}" ]]; then
  echo "[run_dm0_freeze_vit] ${NORM_STATS} not found, computing it now..."
  mkdir -p "$(dirname "${NORM_STATS}")"
  python3 examples/dm0/compute_norm_stats.py \
    --repo-id "${DATASET_PATH}" \
    --output "${NORM_STATS}" \
    --chunk-size 50 \
    --non-delta-mask 14 15 \
    --max-action-dim 32 --max-state-dim 32
fi

# 2) launch training.
NUM_GPUS="${NUM_GPUS:-8}"
LAUNCHER=(accelerate launch --num_processes="${NUM_GPUS}" --mixed_precision=bf16)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  LAUNCHER+=(--multi_gpu)
fi

# Generate one shared RUN_ID for *all* ranks. Without this each rank calls
# datetime.now() independently and they typically share the same second, leading
# to the same output_dir; rank 0's wandb.init() then mkdirs that dir and the
# remaining ranks trip cfg.validate()'s "output_dir already exists" check on a
# fast retry. Including $$ (launcher PID) makes the id unique even when two
# launches happen within the same second.
export DM0_RUN_ID="${DM0_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

# Image augmentation is ON by default and matches dexbotic policy_dm0 / policy_color_dm0.
# Pass --no-aug to disable, or tune via --aug-prob / --aug-size.
"${LAUNCHER[@]}" bt/dm0/train_dm0_r1_pro.py \
  --task=train \
  --dataset-repo="local/r1_pro_chassis_v3" \
  --dataset-root="${DATASET_PATH}" \
  --norm-stats-path="${NORM_STATS}" \
  --dm0-base="${DM0_BASE}" \
  --ema-decay=0.99 \
  --wandb \
  --wandb-project="dm0_r1_pro_chassis_v3_lerobot"

# "${LAUNCHER[@]}" bt/dm0/train_dm0_r1_pro.py \
#   --task=lora_train \
#   --dataset-repo="local/r1_pro_chassis_v3" \
#   --dataset-root="${DATASET_PATH}" \
#   --norm-stats-path="${NORM_STATS}" \
#   --phase1-ckpt="./outputs/bt/dm0/train-20260427_053604/checkpoints/last/pretrained_model" \
#   --wandb \
#   --wandb-project="dm0_r1_pro_chassis_v3_lerobot_lora_vit"