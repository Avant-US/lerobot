#!/usr/bin/env bash
set -euo pipefail

TARGET_USER="Luogang"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      TARGET_USER="${2:?missing value for --user}"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: sudo bash bt/hil/setup_xrdp.sh [--user Luogang]"
      exit 1
      ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "请用 root 权限运行：sudo bash bt/hil/setup_xrdp.sh --user ${TARGET_USER}"
  exit 1
fi

if ! getent passwd "${TARGET_USER}" >/dev/null; then
  echo "找不到用户: ${TARGET_USER}"
  exit 1
fi

TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"

echo "==> 安装 xrdp / xorgxrdp / xfce4 / dbus-x11"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  xrdp \
  xorgxrdp \
  xfce4 \
  dbus-x11

echo "==> 配置 xrdp 读取 SSL 证书"
adduser xrdp ssl-cert || true

echo "==> 为 ${TARGET_USER} 写入 XFCE 会话文件"
cat > "${TARGET_HOME}/.xsession" <<'EOF'
#!/bin/sh
exec startxfce4
EOF

cat > "${TARGET_HOME}/.xsessionrc" <<'EOF'
export XDG_SESSION_DESKTOP=xfce
export XDG_CURRENT_DESKTOP=XFCE
export DESKTOP_SESSION=xfce
EOF

chown "${TARGET_USER}:${TARGET_USER}" "${TARGET_HOME}/.xsession" "${TARGET_HOME}/.xsessionrc"
chmod 644 "${TARGET_HOME}/.xsession" "${TARGET_HOME}/.xsessionrc"

echo "==> 启动并设置 xrdp 开机自启"
systemctl enable xrdp
systemctl restart xrdp

if command -v ufw >/dev/null 2>&1; then
  UFW_STATUS="$(ufw status 2>/dev/null | sed -n '1p' || true)"
  if [[ "${UFW_STATUS}" == "Status: active" ]]; then
    echo "==> 检测到 ufw 已启用，放行 3389/tcp"
    ufw allow 3389/tcp
  fi
fi

echo
echo "==> XRDP 服务状态"
systemctl --no-pager --full status xrdp || true
echo
echo "==> 3389 监听检查"
ss -ltn '( sport = :3389 )' || true
echo
echo "==> 完成"
echo "Windows 11 侧请使用“远程桌面连接(mstsc)”连接到服务器 IP。"
echo "登录用户名: ${TARGET_USER}"
echo "登录密码: 该 Linux 用户自己的系统密码"
echo
echo "如果你在云平台上，还需要确认云防火墙已经放行 TCP 3389。"
