# Windows 11 + XRDP

这份说明用于替代 `MobaXterm + X11 forwarding` 方案。

`MobaXterm` 那条路在当前 `gym_hil + MuJoCo passive viewer` 上会卡在 `GLXBadDrawable / X_GLXSwapBuffers`，根因是 OpenGL/GLX 的交互 viewer 通过 X11 forwarding 不稳定。

`xrdp` 方案的思路是：

1. 在服务器上启动完整的远程桌面会话。
2. 在 Windows 11 上直接用系统自带的“远程桌面连接”客户端。
3. 在远程桌面里打开终端，再运行 `bt/hil/run_interactive_keyboard_demo.sh`。

## 服务器安装

需要 root 权限，在服务器上执行：

```bash
cd ~/SRC/Robot/lerobot
sudo bash bt/hil/setup_xrdp.sh --user Luogang
```

如果你想做安装后检查：

```bash
bash bt/hil/check_xrdp.sh
```

## Windows 11 连接

1. 打开“远程桌面连接”
   - 快捷键：`Win + R`
   - 输入：`mstsc`

2. 在“计算机”里填服务器 IP

3. 点“连接”

4. 登录：
   - 用户名：`Luogang`
   - 密码：该 Linux 用户自己的系统密码

## 连接后运行 demo

在远程桌面会话里的终端执行：

```bash
cd ~/SRC/Robot/lerobot
source ./lerobot-venv/bin/activate
bash bt/hil/run_interactive_keyboard_safe.sh
```

这样 `gym_hil` 的窗口就在远程桌面里显示，你在 Windows 11 上敲真实键盘即可操作它。

> 如果普通版 `run_interactive_keyboard_demo.sh` 在 XRDP 中崩溃，优先使用这个 safe 版本。

safe 版交互约定：

- `Space`：开始/停止 intervention
- `Enter`：成功结束当前 episode
- `Backspace`：失败结束当前 episode
- `ESC`：中止当前 run，并丢弃未完成的 partial episode
- `O`：打开夹爪
- `C`：关闭夹爪

如果你的键盘没有 `右 Ctrl`，直接使用 `O/C` 即可。
如果按 `ESC` 时还没有保存过任何完整 episode，safe 脚本会删除不完整输出目录，并只写一份 summary。

## 如果连不上

优先检查：

1. 服务器上 `xrdp` 是否已启动
2. 服务器上 3389 是否监听
3. 云平台防火墙 / 安全组是否放行 TCP 3389
4. 本机网络是否能访问服务器 3389 端口
