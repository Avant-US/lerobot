#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# train_r1pro_chassis_multi.sh — 多 GPU 分布式训练
#
# 严格对齐 OpenPI JAX 命令:
#   uv run python scripts/train.py pi05_r1pro_chassis \
#       --exp_name $EXPNAME --batch_size 256 \
#       --num_train_steps 100000 --save_interval 500 --keep_period 2500
#
# 使用 accelerate 进行多 GPU 训练。per-GPU batch_size 按 GPU 数量等分，
# 保持 effective batch_size = 256 (与 OpenPI CLI 覆盖对齐)。
#
# 用法:
#   # 4 GPU (每 GPU batch_size=64, effective=256)
#   bash bt/pi05/alig/trainr1/train_r1pro_chassis_multi.sh --num-gpus 4
#
#   # 8 GPU (每 GPU batch_size=32, effective=256)
#   bash bt/pi05/alig/trainr1/train_r1pro_chassis_multi.sh --num-gpus 8
#
#   # 指定 GPU
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash bt/pi05/alig/trainr1/train_r1pro_chassis_multi.sh --num-gpus 4
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEROBOT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VENV_PATH="/mnt/r/Venv/lerobot-venv"

DATA_DIR="${SCRIPT_DIR}/data/r1_pro_chassis_v30"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/r1pro_chassis_aligned_multi"

# ── 默认参数 (严格对齐 CLI 覆盖) ──────────────────────────────────────
NUM_GPUS=4
EFFECTIVE_BATCH_SIZE=256    # CLI: --batch_size 256
STEPS=100000                # CLI: --num_train_steps 100000
SEED=42
KEEP_PERIOD=2500            # CLI: --keep_period 2500

# ── 解析参数 ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            NUM_GPUS=$2
            shift 2
            ;;
        --output)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --steps)
            STEPS=$2
            shift 2
            ;;
        --keep-period)
            KEEP_PERIOD=$2
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 计算 per-GPU batch size
PER_GPU_BATCH=$((EFFECTIVE_BATCH_SIZE / NUM_GPUS))
if [ $((PER_GPU_BATCH * NUM_GPUS)) -ne ${EFFECTIVE_BATCH_SIZE} ]; then
    echo "ERROR: effective_batch_size (${EFFECTIVE_BATCH_SIZE}) 必须能被 num_gpus (${NUM_GPUS}) 整除"
    exit 1
fi

# ── 环境检查 ──────────────────────────────────────────────────────────
echo "=========================================="
echo " Pi0.5 多 GPU 对齐训练"
echo "=========================================="
echo ""
echo "  GPU 数量:        ${NUM_GPUS}"
echo "  Per-GPU Batch:   ${PER_GPU_BATCH}"
echo "  Effective Batch: ${EFFECTIVE_BATCH_SIZE}"
echo "  Steps:           ${STEPS}"
echo ""

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "ERROR: 虚拟环境不存在: ${VENV_PATH}"
    exit 1
fi

if [ ! -d "${DATA_DIR}" ] || [ ! -f "${DATA_DIR}/meta/info.json" ]; then
    echo "ERROR: 数据集不存在。请先运行 prepare_data.sh"
    exit 1
fi

if [ -d "${OUTPUT_DIR}" ]; then
    echo "WARNING: 输出目录已存在: ${OUTPUT_DIR}"
    exit 1
fi

# ── 激活虚拟环境 ──────────────────────────────────────────────────────
source "${VENV_PATH}/bin/activate"
cd "${LEROBOT_ROOT}"

# ── 执行训练 ──────────────────────────────────────────────────────────
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --multi_gpu \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=local/r1_pro_chassis_v30 \
    --dataset.root="${DATA_DIR}" \
    --policy.path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.push_to_hub=false \
    --policy.optimizer_weight_decay=1e-10 \
    --policy.loss_include_padding=true \
    --policy.ema_decay=0.99 \
    --policy.augmentation_enabled=true \
    --policy.gradient_checkpointing=true \
    --batch_size=${PER_GPU_BATCH} \
    --steps=${STEPS} \
    --seed=${SEED} \
    --log_freq=100 \
    --save_freq=500 \
    --eval_freq=-1 \
    --num_workers=2 \
    --output_dir="${OUTPUT_DIR}" \
    --wandb.enable=true \
    --wandb.project=pi05_r1pro_chassis_alignment \
    --rename_map='{"observation.images.head_rgb":"observation.images.base_0_rgb","observation.images.left_wrist_rgb":"observation.images.left_wrist_0_rgb","observation.images.right_wrist_rgb":"observation.images.right_wrist_0_rgb"}'
TRAIN_EXIT_CODE=$?

# ── Checkpoint 清理 (keep_period) ────────────────────────────────────
if [ ${TRAIN_EXIT_CODE} -eq 0 ] && [ "${KEEP_PERIOD}" -gt 0 ]; then
    echo ""
    echo "──────────────────────────────────────────"
    echo " Checkpoint 清理 (keep_period=${KEEP_PERIOD})"
    echo "──────────────────────────────────────────"
    CKPT_DIR="${OUTPUT_DIR}/checkpoints"
    if [ -d "${CKPT_DIR}" ]; then
        python "${SCRIPT_DIR}/cleanup_checkpoints.py" \
            --checkpoint-dir "${CKPT_DIR}" \
            --keep-period "${KEEP_PERIOD}"
    fi
fi

exit ${TRAIN_EXIT_CODE}
