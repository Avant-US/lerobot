#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# train_r1pro_chassis.sh — OpenPI 对齐的 Pi0.5 训练 (单 GPU)
#
# 严格对齐 OpenPI JAX 命令:
#   uv run python scripts/train.py pi05_r1pro_chassis \
#       --exp_name $EXPNAME --batch_size 256 \
#       --num_train_steps 100000 --save_interval 500 --keep_period 2500
#
# 对齐的关键差异 (@#2 标记项):
#   P0-1: Weight Decay      1e-10 (OpenPI) vs 0.01 (LeRobot默认)
#   P0-2: Loss 截断         32维  (OpenPI) vs 23维 (LeRobot默认)
#   P0-3: LR Schedule       post_warmup (PI05Config preset 已设置)
#   P1-1: EMA               0.99  (OpenPI) vs None (LeRobot默认)
#   P1-2: 数据增强           启用  (OpenPI) vs 禁用 (LeRobot默认)
#   P2:   Batch/Seed/dtype   256/42/bf16 vs 8/1000/fp32
#
# 注意: LR schedule 的 decay_steps=30000 未被 CLI 覆盖,
#       cosine 在 step 30000 完成后 LR 钳位在 2.5e-6 直到 step 100000
#
# 前置条件:
#   1. 运行 prepare_data.sh 准备数据集
#   2. 确保 lerobot pi0.5_base 预训练权重可访问
#
# 用法:
#   # 标准训练 (100000 步, batch_size=256, 需多 GPU 或梯度累积)
#   bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh
#
#   # 快速冒烟测试 (200 步, batch_size=4)
#   bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --smoke-test
#
#   # 梯度累积 (单 GPU 无法容纳 batch_size=256 时)
#   bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --grad-accum 8
#
#   # 指定 GPU
#   CUDA_VISIBLE_DEVICES=0 bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEROBOT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VENV_PATH="/mnt/r/Venv/lerobot-venv"

DATA_DIR="${SCRIPT_DIR}/data/r1_pro_chassis_v30"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/r1pro_chassis_aligned"

# ── 默认参数 (严格对齐 CLI: --batch_size 256 --num_train_steps 100000
#    --save_interval 500 --keep_period 2500) ─────────────────────
BATCH_SIZE=256
STEPS=100000
SEED=42
LOG_FREQ=100
SAVE_FREQ=500
KEEP_PERIOD=2500
EVAL_FREQ=-1
NUM_WORKERS=2
PRETRAINED="lerobot/pi05_base"
GRAD_ACCUM=1

# OpenPI 对齐的 policy 参数
DTYPE="bfloat16"
WEIGHT_DECAY="1e-10"
LOSS_INCLUDE_PADDING="true"
EMA_DECAY="0.99"
AUGMENTATION_ENABLED="true"
GRADIENT_CHECKPOINTING="true"

# WandB
WANDB_ENABLE="true"
WANDB_PROJECT="pi05_r1pro_chassis_alignment"

# ── 解析命令行参数 ────────────────────────────────────────────────────
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)
            echo ">>> 冒烟测试模式: 200 步, batch_size=4"
            BATCH_SIZE=4
            STEPS=200
            SAVE_FREQ=100
            KEEP_PERIOD=100
            LOG_FREQ=20
            GRAD_ACCUM=1
            WANDB_ENABLE="false"
            OUTPUT_DIR="${SCRIPT_DIR}/outputs/smoke_test"
            shift
            ;;
        --steps)
            STEPS=$2
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --output)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --no-wandb)
            WANDB_ENABLE="false"
            shift
            ;;
        --no-augmentation)
            AUGMENTATION_ENABLED="false"
            shift
            ;;
        --no-ema)
            EMA_DECAY=""
            shift
            ;;
        --pretrained)
            PRETRAINED=$2
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM=$2
            shift 2
            ;;
        --keep-period)
            KEEP_PERIOD=$2
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# 梯度累积: 实际 per-step batch = BATCH_SIZE / GRAD_ACCUM
if [ "${GRAD_ACCUM}" -gt 1 ]; then
    MICRO_BATCH=$((BATCH_SIZE / GRAD_ACCUM))
    if [ $((MICRO_BATCH * GRAD_ACCUM)) -ne ${BATCH_SIZE} ]; then
        echo "ERROR: batch_size (${BATCH_SIZE}) 必须能被 grad_accum (${GRAD_ACCUM}) 整除"
        exit 1
    fi
    echo ">>> 梯度累积模式: micro_batch=${MICRO_BATCH}, accum_steps=${GRAD_ACCUM}, effective_batch=${BATCH_SIZE}"
    BATCH_SIZE=${MICRO_BATCH}
fi

# ── 环境检查 ──────────────────────────────────────────────────────────
echo "=========================================="
echo " Pi0.5 对齐训练 — OpenPI pi05_r1pro_chassis"
echo "=========================================="
echo ""

# 检查虚拟环境
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "ERROR: 虚拟环境不存在: ${VENV_PATH}"
    exit 1
fi

# 检查数据集
if [ ! -d "${DATA_DIR}" ] || [ ! -f "${DATA_DIR}/meta/info.json" ]; then
    echo "ERROR: 数据集不存在或不完整: ${DATA_DIR}"
    echo "请先运行: bash bt/pi05/alig/trainr1/prepare_data.sh"
    exit 1
fi

# 检查输出目录冲突
if [ -d "${OUTPUT_DIR}" ]; then
    echo "WARNING: 输出目录已存在: ${OUTPUT_DIR}"
    echo "  请删除或选择不同的输出目录"
    echo "  使用 --output <path> 指定"
    exit 1
fi

# ── 激活虚拟环境 ──────────────────────────────────────────────────────
source "${VENV_PATH}/bin/activate"

# ── 打印配置摘要 ──────────────────────────────────────────────────────
echo "  对齐目标 CLI:"
echo "    uv run python scripts/train.py pi05_r1pro_chassis \\"
echo "        --batch_size 256 --num_train_steps 100000 \\"
echo "        --save_interval 500 --keep_period 2500"
echo ""
echo "  训练参数:"
echo "    Pretrained:      ${PRETRAINED}"
echo "    Data:            ${DATA_DIR}"
echo "    Output:          ${OUTPUT_DIR}"
echo "    Steps:           ${STEPS}"
echo "    Batch Size:      ${BATCH_SIZE}  (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "    Grad Accum:      ${GRAD_ACCUM}"
echo "    Save Freq:       ${SAVE_FREQ}"
echo "    Keep Period:     ${KEEP_PERIOD}"
echo "    Seed:            ${SEED}"
echo "    dtype:           ${DTYPE}"
echo "    num_workers:     ${NUM_WORKERS}"
echo ""
echo "  对齐参数 (@#2):"
echo "    Weight Decay:    ${WEIGHT_DECAY}  (OpenPI=1e-10)"
echo "    Loss Padding:    ${LOSS_INCLUDE_PADDING}  (OpenPI=True, 32维)"
echo "    EMA Decay:       ${EMA_DECAY:-disabled}  (OpenPI=0.99)"
echo "    Augmentation:    ${AUGMENTATION_ENABLED}  (OpenPI=True)"
echo "    LR Phase Mode:   post_warmup  (PI05Config preset)"
echo "    LR decay_steps:  30000 (cosine 完成后 LR 钳位在 2.5e-6)"
echo "    Grad Checkpoint: ${GRADIENT_CHECKPOINTING}"
echo ""
echo "  WandB:            ${WANDB_ENABLE} (project: ${WANDB_PROJECT})"
echo ""

# ── 构建训练命令 ──────────────────────────────────────────────────────
cd "${LEROBOT_ROOT}"

CMD="python -m lerobot.scripts.lerobot_train"
CMD="${CMD} --dataset.repo_id=local/r1_pro_chassis_v30"
CMD="${CMD} --dataset.root=${DATA_DIR}"
CMD="${CMD} --policy.path=${PRETRAINED}"

# 模型参数
CMD="${CMD} --policy.dtype=${DTYPE}"

# 优化器 (@#2 Weight Decay)
CMD="${CMD} --policy.optimizer_weight_decay=${WEIGHT_DECAY}"

# Loss 对齐 (@#2)
CMD="${CMD} --policy.loss_include_padding=${LOSS_INCLUDE_PADDING}"

# EMA 对齐 (@#2)
if [ -n "${EMA_DECAY}" ]; then
    CMD="${CMD} --policy.ema_decay=${EMA_DECAY}"
fi

# 数据增强 (@#2)
CMD="${CMD} --policy.augmentation_enabled=${AUGMENTATION_ENABLED}"

# 不推送到 Hub
CMD="${CMD} --policy.push_to_hub=false"

# 特征名映射 (数据集相机名 → pi05_base 模型期望的名称)
RENAME_MAP='{"observation.images.head_rgb":"observation.images.base_0_rgb","observation.images.left_wrist_rgb":"observation.images.left_wrist_0_rgb","observation.images.right_wrist_rgb":"observation.images.right_wrist_0_rgb"}'

# 内存优化
CMD="${CMD} --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING}"

# 训练参数
CMD="${CMD} --batch_size=${BATCH_SIZE}"
CMD="${CMD} --steps=${STEPS}"
CMD="${CMD} --seed=${SEED}"
CMD="${CMD} --log_freq=${LOG_FREQ}"
CMD="${CMD} --save_freq=${SAVE_FREQ}"
CMD="${CMD} --eval_freq=${EVAL_FREQ}"
CMD="${CMD} --num_workers=${NUM_WORKERS}"

# 输出
CMD="${CMD} --output_dir=${OUTPUT_DIR}"

# WandB
CMD="${CMD} --wandb.enable=${WANDB_ENABLE}"
CMD="${CMD} --wandb.project=${WANDB_PROJECT}"

# 额外参数
if [ -n "${EXTRA_ARGS}" ]; then
    CMD="${CMD} ${EXTRA_ARGS}"
fi

echo "──────────────────────────────────────────"
echo " 启动训练"
echo "──────────────────────────────────────────"
echo ""
echo "命令:"
echo "  ${CMD}"
echo ""

# ── 执行训练 ──────────────────────────────────────────────────────────
${CMD} --rename_map="${RENAME_MAP}"
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
