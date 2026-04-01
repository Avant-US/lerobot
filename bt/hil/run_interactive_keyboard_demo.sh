#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "当前会话没有图形输入环境，不能直接运行 Keyboard-v0 交互版 demo。"
  echo "请在带桌面的会话中重试，或者先运行 bt/hil/scripted_takeover_demo.py 头less 版本。"
  exit 1
fi

source "${ROOT_DIR}/lerobot-venv/bin/activate"

python -m lerobot.rl.gym_manipulator \
  --config_path "${ROOT_DIR}/bt/hil/interactive_keyboard_record.json"
