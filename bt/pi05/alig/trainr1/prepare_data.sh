#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# prepare_data.sh — 数据集准备 (OpenPI R1 Pro Chassis → LeRobot v3.0)
#
# 功能:
#   1. 使用 convert_r1pro_to_lerobot.py 将 OpenPI v2.1 数据集转换为 LeRobot v3.0
#   2. 注入 OpenPI 的 norm_stats.json (确保 quantile 归一化精确对齐)
#   3. 运行验证确认数据集完整性
#
# 用法:
#   bash bt/pi05/alig/trainr1/prepare_data.sh [--sample N]
#
# 选项:
#   --sample N    仅采样 N 个 episodes (用于快速测试)
#   无参数        使用全部 64 episodes (正式训练)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEROBOT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VENV_PATH="/mnt/r/Venv/lerobot-venv"

RAW_DATA="/mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis"
# 输出到 /mnt/r (大容量)，然后创建本地 symlink
# (v2.1→v3.0 转换需要两份副本同时存在，本地磁盘空间不够)
OUTPUT_DATA_ACTUAL="/mnt/r/share/lkx/pi/data/r1_pro_chassis_lerobot_v30"
OUTPUT_DATA="${SCRIPT_DIR}/data/r1_pro_chassis_v30"
NORM_STATS="/mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json"
CONVERT_SCRIPT="${LEROBOT_ROOT}/bt/pi05/alig/dataprocess/convert_r1pro_to_lerobot.py"

# ── 解析参数 ──────────────────────────────────────────────────────────
SAMPLE_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            SAMPLE_ARGS="--sample-episodes $2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ── 环境检查 ──────────────────────────────────────────────────────────
echo "=========================================="
echo " 数据集准备: OpenPI R1 Pro Chassis → LeRobot v3.0"
echo "=========================================="
echo ""
echo "  虚拟环境:    ${VENV_PATH}"
echo "  源数据:      ${RAW_DATA}"
echo "  输出目录:    ${OUTPUT_DATA}"
echo "  Norm Stats:  ${NORM_STATS}"
echo "  采样参数:    ${SAMPLE_ARGS:-全部 episodes}"
echo ""

# 检查虚拟环境
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "ERROR: 虚拟环境不存在: ${VENV_PATH}"
    exit 1
fi

# 检查源数据
if [ ! -d "${RAW_DATA}" ]; then
    echo "ERROR: 源数据目录不存在: ${RAW_DATA}"
    exit 1
fi

# 检查 norm_stats
if [ ! -f "${NORM_STATS}" ]; then
    echo "ERROR: norm_stats.json 不存在: ${NORM_STATS}"
    exit 1
fi

# 检查转换脚本
if [ ! -f "${CONVERT_SCRIPT}" ]; then
    echo "ERROR: 转换脚本不存在: ${CONVERT_SCRIPT}"
    exit 1
fi

# ── 激活虚拟环境 ──────────────────────────────────────────────────────
source "${VENV_PATH}/bin/activate"
echo "已激活虚拟环境: $(which python)"
echo ""

# ── 清理旧输出 (如果存在) ─────────────────────────────────────────────
if [ -L "${OUTPUT_DATA}" ]; then
    rm -f "${OUTPUT_DATA}"
fi
if [ -d "${OUTPUT_DATA}" ]; then
    echo "WARNING: 输出目录已存在，将清除: ${OUTPUT_DATA}"
    rm -rf "${OUTPUT_DATA}"
fi
if [ -d "${OUTPUT_DATA_ACTUAL}" ]; then
    echo "WARNING: 实际输出目录已存在，将清除: ${OUTPUT_DATA_ACTUAL}"
    rm -rf "${OUTPUT_DATA_ACTUAL}"
fi

# ── 运行转换 ──────────────────────────────────────────────────────────
echo "──────────────────────────────────────────"
echo " Phase 1: 数据转换 (含 norm_stats 注入)"
echo "──────────────────────────────────────────"
cd "${LEROBOT_ROOT}"

python "${CONVERT_SCRIPT}" \
    --input "${RAW_DATA}" \
    --output "${OUTPUT_DATA_ACTUAL}" \
    --norm-stats-path "${NORM_STATS}" \
    ${SAMPLE_ARGS}

# 创建 symlink 到本地路径
mkdir -p "$(dirname "${OUTPUT_DATA}")"
ln -sfn "${OUTPUT_DATA_ACTUAL}" "${OUTPUT_DATA}"
echo "已创建 symlink: ${OUTPUT_DATA} -> ${OUTPUT_DATA_ACTUAL}"

# ── 验证结果 ──────────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo " Phase 2: 验证数据集"
echo "──────────────────────────────────────────"

# 检查关键文件存在
echo "检查关键文件..."
for f in "meta/info.json" "meta/stats.json"; do
    if [ -f "${OUTPUT_DATA}/${f}" ]; then
        echo "  OK: ${f}"
    else
        echo "  FAIL: ${f} 缺失!"
        exit 1
    fi
done

# 检查 stats.json 包含分位数
python -c "
import json
with open('${OUTPUT_DATA}/meta/stats.json') as f:
    stats = json.load(f)
for key in ['observation.state', 'action']:
    assert key in stats, f'stats.json 缺少 {key}'
    assert 'q01' in stats[key], f'{key} 缺少 q01'
    assert 'q99' in stats[key], f'{key} 缺少 q99'
    dim = len(stats[key]['q01'])
    print(f'  {key}: q01/q99 维度 = {dim}')
print('  分位数验证通过!')
"

# 检查 info.json
python -c "
import json
with open('${OUTPUT_DATA}/meta/info.json') as f:
    info = json.load(f)
print(f'  版本: {info[\"codebase_version\"]}')
print(f'  Episodes: {info[\"total_episodes\"]}')
print(f'  Frames: {info[\"total_frames\"]}')
print(f'  FPS: {info[\"fps\"]}')
assert info['codebase_version'] == 'v3.0', '版本必须是 v3.0'
"

echo ""
echo "=========================================="
echo " 数据集准备完成!"
echo " 输出: ${OUTPUT_DATA}"
echo "=========================================="
