#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_CONFIG="${ROOT_DIR}/bt/hil/interactive_keyboard_safe_record.json"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "当前会话没有图形输入环境，不能直接运行 Keyboard-v0 交互版 safe demo。"
  echo "请在 XRDP / 本地图形桌面会话中重试。"
  exit 1
fi

mkdir -p "${ROOT_DIR}/bt/hil/.cache/hf_home" "${ROOT_DIR}/bt/hil/.cache/hf_datasets"

# XRDP 下优先走 Mesa 软件渲染，避免 MuJoCo passive viewer 在 GLX/DRI 路径上崩溃。
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export LIBGL_DRI3_DISABLE="${LIBGL_DRI3_DISABLE:-1}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/bt/hil/.cache/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/bt/hil/.cache/hf_datasets}"

source "${ROOT_DIR}/lerobot-venv/bin/activate"

if [[ "${1:-}" == "--config_path" ]]; then
  python "${ROOT_DIR}/bt/hil/run_interactive_keyboard_safe.py" "$@"
else
  python "${ROOT_DIR}/bt/hil/run_interactive_keyboard_safe.py" --config_path "${DEFAULT_CONFIG}" "$@"
fi
