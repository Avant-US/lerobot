# bt/hil Demo

这个目录放的是“方案一：最少改动接入”的本地 demo，不改 `src/` 下的 LeRobot 代码，只在 demo 侧复用已有的 `gym_hil + processor + dataset` 流程。

## 文件说明

- `scripted_takeover_demo.py`
  - 头less smoke test。
  - 用 `gym_hil` 的 `PandaPickCubeBase-v0` 包装出与 HIL 路径一致的 4 维动作空间。
  - 通过 `info["teleop_action"]` 和 `info["is_intervention"]` 注入脚本化“人工接管”。
  - 复用 `InterventionActionProcessorStep` 和 `LeRobotDataset` 完成录制。
- `scripted_takeover_schedule.json`
  - 脚本化接管动作序列，可直接修改。
- `interactive_keyboard_record.json`
  - 交互版配置，面向带图形桌面的 `PandaPickCubeKeyboard-v0`。
- `run_interactive_keyboard_demo.sh`
  - 图形桌面下启动交互版 demo 的快捷脚本。
- `interactive_keyboard_safe_record.json`
  - safe 版默认配置，输出到单独目录。
- `run_interactive_keyboard_safe.py`
  - safe 版交互录制包装器。
  - 复用 `gym_manipulator` 的 env / processor，但把记录方式改成更保守的路径：
    1. 相机记录为 PNG 图片，不编码 MP4。
    2. 显式 `dataset.finalize()`。
    3. 成功写盘后直接硬退出 Python 进程，绕开 MuJoCo viewer 的析构崩溃。
- `run_interactive_keyboard_safe.sh`
  - safe 版启动脚本。
  - 默认开启 Mesa 软件渲染，更适合 XRDP 会话。
- `setup_xrdp.sh`
  - 给服务器安装并配置 `xrdp + xfce4` 的脚本。
- `check_xrdp.sh`
  - 安装后的快速检查脚本。
- `XRDP_WINDOWS11.md`
  - Windows 11 侧使用 `xrdp` 的连接说明。

## 为什么有两个 demo

当前机器会话没有可用的图形输入连接，`Keyboard-v0` / `Gamepad-v0` 依赖 `pynput` 和 X/Wayland 输入，无法在当前头less 会话中直接自动化调试跑通。

所以这里保留了：

1. 交互版：给开发者在桌面会话里实际接管机械臂用。
2. 头less 版：给当前机器做自动 smoke test，用来验证最少改动接入链路已经跑通。

## 头less 版运行

在仓库根目录执行：

```bash
source ./lerobot-venv/bin/activate
python bt/hil/scripted_takeover_demo.py --clean
```

运行后会在 `bt/hil/output/` 下生成：

- `scripted_takeover_dataset/`
- `scripted_takeover_summary.json`

## 交互版运行

要求：

- 有图形桌面会话
- `DISPLAY` 或 `WAYLAND_DISPLAY` 可用

执行：

```bash
bash bt/hil/run_interactive_keyboard_demo.sh
```

## 交互版 safe 运行

如果普通交互版在 `XRDP` / 远程桌面里崩溃，优先改用 safe 版。

执行：

```bash
bash bt/hil/run_interactive_keyboard_safe.sh
```

safe 版的主要差别：

1. 图像按 `image` 类型记录为 PNG，不走 MP4 视频编码。
2. 不使用默认的并行视频编码子进程。
3. 只有按 `Enter` 或 `Backspace` 才会结束当前 episode，不再因为 `100` 步上限自动结束。
4. 写盘完成后直接退出 Python 进程，避免 MuJoCo passive viewer 在解释器退出阶段崩溃。
5. 夹爪支持备用按键：`O` 打开、`C` 关闭，不依赖 `右 Ctrl`。

safe 版的代价：

1. 数据目录体积会比 MP4 版本大。
2. 脚本结束时不会优雅关闭 viewer，而是直接退出进程。
3. `ESC` 会中止当前 run 并丢弃未完成的 partial episode。
   如果此时还没有保存过任何完整 episode，脚本会删除这个不完整输出目录，并只保留 summary json。
4. 更适合作为“远程交互稳定版 demo”，不是最漂亮的生产形态。

## XRDP 远程桌面方案

如果 `MobaXterm + X11 forwarding` 在 MuJoCo viewer 上出现 `GLXBadDrawable`，建议改用 `xrdp`。

服务器安装：

```bash
sudo bash bt/hil/setup_xrdp.sh --user Luogang
```

Windows 11 连接说明见：

- `bt/hil/XRDP_WINDOWS11.md`

## 接入验证点

头less demo 已覆盖以下关键链路：

1. `gym_hil` 环境创建
2. `GymHILAdapterProcessorStep`
3. `InterventionActionProcessorStep`
4. `LeRobotDataset.add_frame()`
5. `LeRobotDataset.save_episode()`
6. `LeRobotDataset.finalize()`

也就是说，它验证的是“最少改动接入”的核心 HIL 记录通路，而不是单独验证某个 GUI 输入设备。
