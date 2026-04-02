#!/bin/bash
# str_groot 环境初始化脚本
# 安装 starVLA 及其依赖, 在安装前检测版本冲突, 冲突时停下来让用户解决
#
# 用法:
#   bash bt/str_groot_1/setup_env.sh
#
#   # 指定已有的 starVLA 代码库路径 (跳过克隆):
#   STARVLA_REPO_PATH=/path/to/starVLA bash bt/str_groot_1/setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 如果已有 starVLA 代码库, 设置此变量跳过克隆:
STARVLA_REPO_PATH="${STARVLA_REPO_PATH:-}"

# ----------------------------------------------------------------
# 辅助函数: 冲突检测
# 使用 pip install --dry-run 预检, 捕获冲突信息
# ----------------------------------------------------------------
check_conflicts() {
  local req_file="$1"
  echo "  检测依赖冲突..."
  local conflict_output
  conflict_output=$(pip install --dry-run -r "$req_file" 2>&1) || true

  # 检查是否存在不兼容提示
  local conflicts
  conflicts=$(echo "$conflict_output" | grep -iE "(incompatible|conflict|ERROR)" || true)

  if [ -n "$conflicts" ]; then
    echo ""
    echo "================================================================"
    echo "  WARNING: 检测到依赖冲突, 请先手动解决再重新运行此脚本"
    echo "================================================================"
    echo ""
    echo "冲突详情:"
    echo "$conflicts"
    echo ""
    echo "完整的 dry-run 输出:"
    echo "$conflict_output" | tail -30
    echo ""
    echo "建议:"
    echo "  1. 检查当前环境中冲突包的版本: pip show <package_name>"
    echo "  2. 决定保留哪个版本, 手动 pip install <package>==<version>"
    echo "  3. 或创建新的虚拟环境: python -m venv .venv && source .venv/bin/activate"
    echo "  4. 解决冲突后重新运行: bash $0"
    exit 1
  fi
  echo "  未检测到冲突。"
}

check_single_pkg_conflict() {
  local pkg="$1"
  local install_args="${2:-}"
  echo "  检测 $pkg 冲突..."
  local conflict_output
  # shellcheck disable=SC2086
  conflict_output=$(pip install --dry-run $install_args "$pkg" 2>&1) || true

  local conflicts
  conflicts=$(echo "$conflict_output" | grep -iE "(incompatible|conflict|ERROR)" || true)

  if [ -n "$conflicts" ]; then
    echo ""
    echo "================================================================"
    echo "  WARNING: 安装 $pkg 与当前环境冲突"
    echo "================================================================"
    echo ""
    echo "冲突详情:"
    echo "$conflicts"
    echo ""
    echo "建议: 手动解决冲突后重新运行此脚本。"
    exit 1
  fi
}

# ================================================================
# Step 1/4: 获取 starVLA 代码库 (https://github.com/Avant-US/starVLA/tree/bt260402 改了些bug)
# ================================================================
echo "==> 1/4 获取 starVLA 代码库..."
if [ -z "$STARVLA_REPO_PATH" ]; then
  STARVLA_REPO_PATH="${REPO_ROOT}/.cache/starVLA_repo"
  if [ ! -d "$STARVLA_REPO_PATH" ]; then
    git clone --depth 1 https://github.com/Avant-US/starVLA/tree/bt260402 "$STARVLA_REPO_PATH"
  else
    echo "  已存在: $STARVLA_REPO_PATH, 跳过克隆。"
  fi
else
  echo "  使用已有代码库: $STARVLA_REPO_PATH"
fi

if [ ! -f "$STARVLA_REPO_PATH/requirements.txt" ]; then
  echo "ERROR: $STARVLA_REPO_PATH/requirements.txt 不存在, 请检查路径是否正确。"
  exit 1
fi

# ================================================================
# Step 2/4: 安装 starVLA 的依赖 (requirements.txt) -- 先检测冲突
# ================================================================
echo "==> 2/4 安装 starVLA 的依赖 (requirements.txt)..."
check_conflicts "$STARVLA_REPO_PATH/requirements.txt"
pip install -r "$STARVLA_REPO_PATH/requirements.txt"

# ================================================================
# Step 3/4: 安装 flash-attn -- 先检测冲突
# ================================================================
echo "==> 3/4 安装 flash-attn..."
check_single_pkg_conflict "flash-attn" "--no-build-isolation"
pip install flash-attn --no-build-isolation || {
  echo "WARNING: flash-attn 安装失败。"
  echo "  可能原因: 缺少 CUDA toolkit 或编译工具链。"
  echo "  可手动安装: pip install flash-attn --no-build-isolation"
  echo "  或跳过 (starVLA 可在无 flash-attn 下运行, 但性能较低)。"
}

# ================================================================
# Step 4/4: 安装 starVLA 包 -- 先检测冲突
# ================================================================
echo "==> 4/4 安装 starVLA 包..."
check_single_pkg_conflict "$STARVLA_REPO_PATH"
pip install -e "$STARVLA_REPO_PATH"

# ================================================================
# 验证安装
# ================================================================
echo "==> 验证安装..."
python -c "from starVLA.model.framework.QwenGR00T import Qwen_GR00T; print('starVLA import OK')"
python -c "from deployment.model_server.tools.image_tools import to_pil_preserve; print('deployment import OK')"
echo ""
echo "==> 安装完成! starVLA 代码库路径: $STARVLA_REPO_PATH"
