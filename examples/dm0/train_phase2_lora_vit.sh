#!/usr/bin/env bash
# DM0 phase-2 training: freeze base policy, attach LoRA on the ViT (`out_proj`).
# Mirrors `dexbotic/playground/benchmarks/real/r1_pro_dm0_freeze_lora.py --task=lora_train`.
#
# Requires a phase-1 checkpoint produced by `train_phase1_freeze_vit.sh`.
# `wrap_with_peft` will freeze every base parameter for you, so we don't pass
# `freeze_vision_encoder` / `train_expert_only` here.
#
# Edit the variables below or override them via `VAR=... ./train_phase2_lora_vit.sh`.

set -euo pipefail

# ----- required -------------------------------------------------------------
PHASE1_CKPT="${PHASE1_CKPT:?Set PHASE1_CKPT to phase-1 .../checkpoints/last/pretrained_model}"
DATASET_REPO_ID="${DATASET_REPO_ID:?Set DATASET_REPO_ID to your LeRobotDataset repo id}"

# ----- run-level ------------------------------------------------------------
DATE_TAG="${DATE_TAG:-$(date +%m%d)}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/dm0_r1_pro/phase2_lora_vit-${DATE_TAG}}"
JOB_NAME="${JOB_NAME:-dm0_phase2_lora_vit}"

# ----- LoRA hyper-params ----------------------------------------------------
# `r` is the only LoRA knob exposed by `PeftConfig`. `lora_alpha=32` and `lora_dropout=0.05`
# (matching r1_pro_dm0_freeze_lora.py) are baked into `DM0Policy._get_default_peft_targets`;
# tweak that method if you need different values.
LORA_R="${LORA_R:-16}"

# ----- optimizer / schedule -------------------------------------------------
LR="${LR:-2.5e-5}"
DECAY_LR="${DECAY_LR:-2.5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
STEPS="${STEPS:-6000}"

# ----- training-loop params -------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-1}"

# ----- wandb ----------------------------------------------------------------
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-dm0_r1_pro_chassis_v3_lora_vit}"

# ----- accelerate -----------------------------------------------------------
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
ACCELERATE_ARGS=()
if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
  ACCELERATE_ARGS+=("--config_file=${ACCELERATE_CONFIG_FILE}")
fi

accelerate launch "${ACCELERATE_ARGS[@]}" -m lerobot.scripts.lerobot_train \
  --policy.path="${PHASE1_CKPT}" \
  --policy.gradient_checkpointing=true \
  --policy.optimizer_lr="${LR}" \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${STEPS}" \
  --policy.scheduler_decay_lr="${DECAY_LR}" \
  --policy.push_to_hub=false \
  --peft.method_type=lora \
  --peft.r="${LORA_R}" \
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
