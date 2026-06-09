#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/mnt/r/share/zwy/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

export WANDB_API_KEY="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"

# Anchor everything to the lerobot repo root（与 run_dm0_full_train.sh 一致）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"      # -> .../shallowMerge/lerobot
cd "${REPO_ROOT}"

# 路径都相对 ${REPO_ROOT}（也就是 lerobot/）。
DATASET_PATH="/mnt/r/share/zwy/datasets/r1_pro_data_convert_chassis_v3_newnorm"
NORM_STATS="./norm_stats/r1_pro_chassis_v3.json"
# dexbotic/ 与 lerobot/ 是同级目录，所以从 lerobot/ 出发要写 ../dexbotic/...
ALOHA_CKPT="../dexbotic/checkpoints/DM0-table30_plug_in_network_cable"

# 1) compute norm_stats.json if missing (与 run_dm0_full_train.sh 同样的兜底)。
if [[ ! -f "${NORM_STATS}" ]]; then
  echo "[run_dm0_aloha_r1_train] ${NORM_STATS} not found, computing it now..."
  mkdir -p "$(dirname "${NORM_STATS}")"
  python3 examples/dm0/compute_norm_stats.py \
    --repo-id "${DATASET_PATH}" \
    --output "${NORM_STATS}" \
    --chunk-size 50 \
    --non-delta-mask 14 15 \
    --max-action-dim 32 --max-state-dim 32
fi

# 2) launch training。
NUM_GPUS="${NUM_GPUS:-8}"
LAUNCHER=(accelerate launch --num_processes="${NUM_GPUS}" --mixed_precision=bf16)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  LAUNCHER+=(--multi_gpu)
fi

# 一份 run id 给所有 rank 共用，避免每个 rank 各自 datetime.now() 撞目录。
export DM0_RUN_ID="${DM0_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}${RANDOM}}"

"${LAUNCHER[@]}" bt/dm0/train_dm0_aloha_r1_pro.py \
  --task=full_train \
  --aloha-ckpt="${ALOHA_CKPT}" \
  --dataset-repo="local/r1_pro_chassis_v3" \
  --dataset-root="${DATASET_PATH}" \
  --norm-stats-path="${NORM_STATS}" \
  --ema-decay=0.99 \
  --steps=30000 \
  --save-freq=2000 \
  --lr=5e-6 \
  --decay-lr=5e-7 \
  --warmup-steps=2000 \
  --new-weight-std=0.02 \
  --wandb \
  --wandb-project="dm0_aloha_table30_to_r1pro"
