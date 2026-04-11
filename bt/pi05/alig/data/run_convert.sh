#!/bin/bash
# R1 Pro 数据集转换为 LeRobot v3.0 格式 + Pi0.5 兼容性验证
#
# 用法:
#   bash bt/pi05/alig/data/run_convert.sh [--forward-pass] [--norm-stats-path]
#
# 选项:
#   --forward-pass  运行 Level 3 前向传播验证 (需要 GPU)
#   --norm-stats-path    chassis 数据集使用 OpenPI 的 norm_stats.json (导入模式)
#
# 步骤:
#   1. 转换 r1_pro_test_data (4 episodes, ~2.8GB) — 精确计算模式
#   2. 转换 r1_pro_data_convert_chassis (采样 10 episodes, ~7GB)
#   3. 运行 Pi0.5 Level 1+2 验证
#   4. 如指定 --forward-pass，运行 Level 3 验证

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$LEROBOT_ROOT"

# 数据路径
INPUT_TEST="/mnt/r/share/lkx/pi/data/r1_pro_test_data"
INPUT_CHASSIS="/mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis"
OUTPUT_TEST="$SCRIPT_DIR/r1_pro_test_data_v30"
OUTPUT_CHASSIS="$SCRIPT_DIR/r1_pro_chassis_v30"

# OpenPI norm stats 路径
OPENPI_NORM_STATS="/mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json"

FORWARD_PASS=""
NORM_STATS_ARG=""

for arg in "$@"; do
    case "$arg" in
        --forward-pass) FORWARD_PASS="--run-forward-pass" ;;
        --norm-stats-path)   NORM_STATS_ARG="--norm-stats-path $OPENPI_NORM_STATS" ;;
    esac
done

echo "============================================================"
echo "Step 1: 转换 r1_pro_test_data (4 episodes, 精确计算模式)"
echo "============================================================"
python "$SCRIPT_DIR/convert_r1pro_to_lerobot.py" \
    --input "$INPUT_TEST" \
    --output "$OUTPUT_TEST"

echo ""
echo "============================================================"
if [[ -n "$NORM_STATS_ARG" ]]; then
    echo "Step 2: 转换 r1_pro_data_convert_chassis (采样 10 episodes, 导入模式)"
else
    echo "Step 2: 转换 r1_pro_data_convert_chassis (采样 10 episodes, 精确计算模式)"
fi
echo "============================================================"
python "$SCRIPT_DIR/convert_r1pro_to_lerobot.py" \
    --input "$INPUT_CHASSIS" \
    --output "$OUTPUT_CHASSIS" \
    --sample-episodes 10 \
    $NORM_STATS_ARG

echo ""
echo "============================================================"
echo "Step 3: Pi0.5 验证 — r1_pro_test_data"
echo "============================================================"
python "$SCRIPT_DIR/verify_pi05.py" \
    --dataset-dir "$OUTPUT_TEST" \
    $FORWARD_PASS

echo ""
echo "============================================================"
echo "Step 4: Pi0.5 验证 — r1_pro_data_convert_chassis"
echo "============================================================"
python "$SCRIPT_DIR/verify_pi05.py" \
    --dataset-dir "$OUTPUT_CHASSIS" \
    $FORWARD_PASS

echo ""
echo "============================================================"
echo "全部完成!"
echo "============================================================"
