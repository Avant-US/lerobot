#!/usr/bin/env bash
set -euo pipefail

echo "==> 当前用户: $(whoami)"
echo "==> DISPLAY: ${DISPLAY:-<empty>}"
echo

echo "==> xrdp 服务"
systemctl is-enabled xrdp 2>/dev/null || true
systemctl is-active xrdp 2>/dev/null || true
echo

echo "==> 3389 监听"
ss -ltn '( sport = :3389 )' || true
echo

echo "==> 会话文件"
ls -l "${HOME}/.xsession" "${HOME}/.xsessionrc" 2>/dev/null || true
echo

echo "==> 提示"
echo "如果 3389 已监听，但 Windows 仍连不上，请优先检查云防火墙 / 安全组。"
