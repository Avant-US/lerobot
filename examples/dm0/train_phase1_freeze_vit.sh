#!/usr/bin/env bash
# DM0 phase-1 training: freeze ViT, fine-tune LLM (+ projector / action head).
# Mirrors `dexbotic/playground/benchmarks/real/r1_pro_dm0_freeze_lora.py --task=train`.
#
# Edit the variables below or override them via `VAR=... ./train_phase1_freeze_vit.sh`.

set -euo pipefail

# ----- required -------------------------------------------------------------
DATASET_REPO_ID="${DATASET_REPO_ID:?Set DATASET_REPO_ID to your LeRobotDataset repo id}"
DM0_BASE="${DM0_BASE:-./checkpoints/DM0-base}"
NORM_STATS_PATH="${NORM_STATS_PATH:?Set NORM_STATS_PATH to a norm_stats.json (run examples/dm0/compute_norm_stats.py first)}"

# ----- run-level ------------------------------------------------------------
DATE_TAG="${DATE_TAG:-$(date +%m%d)}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/dm0_r1_pro/phase1_freeze_vit-${DATE_TAG}}"
JOB_NAME="${JOB_NAME:-dm0_phase1_freeze_vit}"

# ----- policy hyper-params (match r1_pro_dm0_freeze_lora.py) ----------------
NUM_IMAGES="${NUM_IMAGES:-3}"
ORIGINAL_ACTION_DIM="${ORIGINAL_ACTION_DIM:-23}"
NON_DELTA_MASK="${NON_DELTA_MASK:-[14,15]}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
LR="${LR:-2.5e-5}"
DECAY_LR="${DECAY_LR:-2.5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
STEPS="${STEPS:-6000}"

# ----- training-loop params -------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-1}"

# ----- wandb (set WANDB_ENABLE=true to opt in) ------------------------------
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-dm0_r1_pro_chassis_v3_freeze_vit}"

# ----- accelerate (multi-gpu / grad_accum live here, not on lerobot-train) --
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
ACCELERATE_ARGS=()
if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
  ACCELERATE_ARGS+=("--config_file=${ACCELERATE_CONFIG_FILE}")
fi

accelerate launch "${ACCELERATE_ARGS[@]}" -m lerobot.scripts.lerobot_train \
  --policy.type=dm0 \
  --policy.model_name_or_path="${DM0_BASE}" \
  --policy.norm_stats_path="${NORM_STATS_PATH}" \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.num_images="${NUM_IMAGES}" \
  --policy.original_action_dim="${ORIGINAL_ACTION_DIM}" \
  --policy.non_delta_mask="${NON_DELTA_MASK}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${CHUNK_SIZE}" \
  --policy.optimizer_lr="${LR}" \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${STEPS}" \
  --policy.scheduler_decay_lr="${DECAY_LR}" \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}"
