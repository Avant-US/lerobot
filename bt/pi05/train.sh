#!/bin/bash
# Pi0.5 微调脚本 - 在 LIBERO 37-episode 子集上训练 100 步
#
# 用法:
#   ./bt/pi05/train.sh              # 完整流程: 准备数据 + 训练
#   ./bt/pi05/train.sh prepare      # 仅准备数据
#   ./bt/pi05/train.sh train        # 仅训练 (需先准备数据)
#   ./bt/pi05/train.sh help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="/home/luogang/VENV/lerobot-venv"
EPISODE_FILE="$SCRIPT_DIR/selected_episodes.json"
export HF_HOME="$HOME/hfhome"
mkdir -p "$HF_HOME"

# ─── 配置 ──────────────────────────────────────────────────────────────
DATASET_REPO_ID="HuggingFaceVLA/libero"
PRETRAINED_PATH="lerobot/pi05_base"
TOKENIZER_NAME="${PI05_TOKENIZER_NAME:-google/paligemma-3b-pt-224}"
NORMALIZATION_MODE="${PI05_NORMALIZATION_MODE:-QUANTILES}"
NUM_EPISODES=37
SEED=42
STEPS="${PI05_STEPS:-100}"
BATCH_SIZE="${PI05_BATCH_SIZE:-4}"
LOG_FREQ="${PI05_LOG_FREQ:-10}"
SAVE_FREQ="${PI05_SAVE_FREQ:-100}"
NUM_WORKERS="${PI05_NUM_WORKERS:-4}"
OUTPUT_DIR="$SCRIPT_DIR/outputs/pi05_libero_37ep"

# ─── 激活虚拟环境 ──────────────────────────────────────────────────────
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[INFO] 已激活虚拟环境: $VENV_DIR"
else
    echo "[WARN] 虚拟环境不存在: $VENV_DIR，使用当前 Python 环境"
fi

echo "[INFO] Hugging Face 缓存目录: $HF_HOME"

# 训练前依赖检查：PI0.5 tokenizer 需要 sentencepiece
if ! python -c "import sentencepiece" >/dev/null 2>&1; then
    echo "[INFO] 检测到缺少 sentencepiece，正在安装..."
    export PATH="$HOME/.local/bin:$PATH"
    uv pip install sentencepiece
fi

cd "$REPO_ROOT"
MODE=${1:-all}

# ─── 准备数据 ──────────────────────────────────────────────────────────
do_prepare() {
    echo "============================================"
    echo " 步骤 1: 准备数据 (抽取 $NUM_EPISODES episodes)"
    echo "============================================"
    python -m bt.pi05.prepare_data \
        --num-episodes "$NUM_EPISODES" \
        --seed "$SEED"
}

# ─── 训练 ──────────────────────────────────────────────────────────────
do_train() {
    echo ""
    echo "============================================"
    echo " 步骤 2: Pi0.5 训练 ($STEPS steps)"
    echo "============================================"

    if [ ! -f "$EPISODE_FILE" ]; then
        echo "[ERROR] Episode 文件不存在: $EPISODE_FILE"
        echo "        请先运行: $0 prepare"
        exit 1
    fi

    if [ -d "$OUTPUT_DIR" ]; then
        echo "[WARN] 输出目录已存在，将添加时间戳后缀"
        OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
    fi

    LOCAL_ROOT="$HF_HOME/lerobot/$DATASET_REPO_ID"

    echo "[INFO] Dataset:    $DATASET_REPO_ID"
    echo "[INFO] Episodes:   $EPISODE_FILE"
    echo "[INFO] Local root: $LOCAL_ROOT"
    echo "[INFO] Steps:      $STEPS"
    echo "[INFO] Batch size: $BATCH_SIZE"
    echo "[INFO] Tokenizer:  $TOKENIZER_NAME"
    echo "[INFO] Norm mode:  $NORMALIZATION_MODE"
    echo "[INFO] Output:     $OUTPUT_DIR"
    echo ""

    python -m bt.pi05.train_pi05_local \
        --repo-id="$DATASET_REPO_ID" \
        --local-root="$LOCAL_ROOT" \
        --episode-file="$EPISODE_FILE" \
        --pretrained-path="$PRETRAINED_PATH" \
        --tokenizer-name="$TOKENIZER_NAME" \
        --output-dir="$OUTPUT_DIR" \
        --steps="$STEPS" \
        --batch-size="$BATCH_SIZE" \
        --num-workers="$NUM_WORKERS" \
        --log-freq="$LOG_FREQ" \
        --save-freq="$SAVE_FREQ" \
        --seed="$SEED" \
        --dtype=bfloat16 \
        --normalization-mode="$NORMALIZATION_MODE" \
        --gradient-checkpointing \
        --train-expert-only

    echo ""
    echo "============================================"
    echo " 训练完成! 输出保存在: $OUTPUT_DIR"
    echo "============================================"
}

# ─── 主逻辑 ────────────────────────────────────────────────────────────
case "$MODE" in
    prepare)
        do_prepare
        ;;
    train)
        do_train
        ;;
    all)
        do_prepare
        echo ""
        do_train
        ;;
    help|*)
        echo "Pi0.5 LIBERO 微调脚本"
        echo ""
        echo "用法: $0 {all|prepare|train|help}"
        echo ""
        echo "  all       - 完整流程: 准备数据 + 训练 (默认)"
        echo "  prepare   - 仅从 LIBERO 数据集抽取 $NUM_EPISODES episodes"
        echo "  train     - 仅训练 $STEPS 步 (需先 prepare)"
        echo "  help      - 显示此帮助"
        echo ""
        echo "配置:"
        echo "  数据集:       $DATASET_REPO_ID"
        echo "  预训练模型:   $PRETRAINED_PATH"
        echo "  Episode 数:   $NUM_EPISODES"
        echo "  训练步数:     $STEPS"
        echo "  Batch size:   $BATCH_SIZE"
        echo "  输出目录:     $OUTPUT_DIR"
        ;;
esac
