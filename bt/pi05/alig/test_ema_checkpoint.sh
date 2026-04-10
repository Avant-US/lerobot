#!/bin/bash
# EMA Checkpoint 端到端验证
#
# 验证:
#   1. ema_state.safetensors / raw_trainable_params.safetensors 文件存在
#   2. model.safetensors 保存的是 EMA 参数
#   3. 保存的 EMA 数值与内存一致
#   4. EMA 数值满足递推公式 ema = 0.99 * ema_old + 0.01 * param_new
#
# 用法:
#   ./bt/pi05/alig/test_ema_checkpoint.sh
#   EMA_STEPS=10 EMA_DECAY=0.995 ./bt/pi05/alig/test_ema_checkpoint.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_DIR="/mnt/r/Venv/lerobot-venv"

STEPS="${EMA_STEPS:-5}"
DECAY="${EMA_DECAY:-0.99}"

# ─── 激活虚拟环境 ──────────────────────────────────────────────────────
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[INFO] 已激活虚拟环境: $VENV_DIR"
else
    echo "[WARN] 虚拟环境不存在: $VENV_DIR，使用当前 Python 环境"
fi

cd "$REPO_ROOT"

echo "============================================"
echo " EMA Checkpoint 验证"
echo "  Steps:     $STEPS"
echo "  EMA decay: $DECAY"
echo "============================================"
echo ""

python -m bt.pi05.alig.test_ema_checkpoint \
    --steps="$STEPS" \
    --ema-decay="$DECAY"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "============================================"
    echo " EMA Checkpoint 验证: 全部通过 ✓"
    echo "============================================"
else
    echo "============================================"
    echo " EMA Checkpoint 验证: 存在失败 ✗"
    echo "============================================"
fi

exit $exit_code
