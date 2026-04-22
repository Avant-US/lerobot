# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a PyTorch-based robotics library by HuggingFace for real-world robot learning. It provides a unified framework for recording robot data, training policies, and deploying them on real robots or in simulation. Source code lives in `src/lerobot/`.

## Common Commands

### Installation
```bash
pip install -e ".[dev,test]"
# Or with uv (preferred):
uv sync --extra dev --extra test
```

### Testing
```bash
# Run all tests
pytest tests -vv

# Run a single test file
pytest tests/test_available.py -vv

# Run a single test function
pytest tests/test_available.py::test_function_name -vv

# With coverage
pytest tests --cov=lerobot
```

### Linting & Formatting
```bash
# Ruff (linter + formatter)
ruff check src/lerobot/          # lint
ruff check --fix src/lerobot/    # lint with auto-fix
ruff format src/lerobot/         # format

# Type checking (gradually being adopted)
mypy src/lerobot/

# Pre-commit (runs all checks)
pre-commit run --all-files
```

### End-to-End Training Tests
```bash
make test-act-ete-train DEVICE=cpu
make test-diffusion-ete-train DEVICE=cpu
make test-end-to-end DEVICE=cpu
```

### CLI Entry Points
All CLI commands are `lerobot-*` scripts: `lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-replay`, `lerobot-teleoperate`, `lerobot-calibrate`, `lerobot-find-cameras`, `lerobot-find-port`, `lerobot-setup-motors`, `lerobot-dataset-viz`, `lerobot-info`, `lerobot-edit-dataset`.

Training example:
```bash
lerobot-train \
  --policy.type=act \
  --env.type=aloha \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --batch_size=8 \
  --steps=100000
```

## Architecture

### Configuration System
Uses **draccus** (pinned v0.10.0) for dataclass-based configs with CLI override support. Configs are defined as typed dataclasses in `src/lerobot/configs/`. CLI arguments map to nested config fields via dot notation (e.g., `--policy.type=act`).

### Core Modules

- **`policies/`** — ML policies (ACT, Diffusion, TDMPC, VQ-BeT, SmolVLA, Gr00t, PI0, XVLA, WallX). Each policy has its own subdirectory with a config, model, and policy wrapper.
- **`datasets/`** — `LeRobotDataset` format: Parquet files for tabular data + MP4 videos for image observations. Integrates with HuggingFace Hub for storage/streaming.
- **`robots/`** — Hardware abstraction for physical robots (SO100, Koch, etc.). Base `Robot` class with a config-driven design.
- **`motors/`** — Motor bus drivers (Dynamixel, Feetech, CAN-based).
- **`cameras/`** — Camera backends (OpenCV, Intel RealSense).
- **`teleoperators/`** — Teleoperation input devices (gamepads, keyboards, phones, leader arms).
- **`envs/`** — Simulation environment wrappers (ALOHA, PushT, LIBERO, MetaWorld) built on gymnasium.
- **`processor/`** — Data pipeline: normalization, image transforms, tokenization.
- **`scripts/`** — CLI entry point implementations. Each `lerobot-*` command maps to a `lerobot_*.py` file here.
- **`optim/`** — Optimizers and LR schedulers.
- **`rl/`** — Reinforcement learning utilities.
- **`async_inference/`** — gRPC-based async policy serving for real-time robot control.
- **`transport/`** — Protocol buffer definitions for distributed communication.

### Data Flow
1. **Record** trajectories via robots/teleoperators → stored as LeRobotDataset (Parquet + MP4)
2. **Upload** to HuggingFace Hub
3. **Train** a policy with `lerobot-train` (uses accelerate for mixed-precision/multi-GPU)
4. **Evaluate** in simulation with `lerobot-eval` or deploy on real robots

### Key Patterns
- Policies follow a common interface: `select_action()` for inference, `forward()` for training loss computation.
- Robot hardware is configured through dataclass configs, instantiated via a registry pattern.
- Optional dependencies are organized as pip extras (e.g., `pip install lerobot[aloha,pusht]` for simulation envs).

## Code Style

- **Line length**: 110 characters
- **Formatter/Linter**: Ruff (target Python 3.12)
- **Import style**: isort via Ruff with `combine-as-imports`, `lerobot` as known first-party
- **Quote style**: double quotes
- **Type checking**: MyPy enabled for `envs`, `configs`, `optim`, `model`, `cameras`, `motors`, `transport` modules; other modules have `ignore_errors = true`

## Design Overview
请参考 LeRobot 的论文 `@bt/docs/bt/ov/LeRobot: An Open-Source Library for End-to-End Robot Learning.pdf`, 参考 LeRobot 的官网 https://huggingface.co/docs/lerobot/main/en/index, 参考 @bt/docs/bt/ov/ 中的各个md文档, 参考 @bt/docs/ 中的各个md文档, 参考 @bt/ 中的各个md文档, 也可参考网上与 LeRobot 的设计相关的文章与讨论. 重要的是要以深入分析本地的 LeRobot 代码库 @lerobot (/home/Luogang/SRC/Robot/lerobot)为基础 , 对 LeRobot 的设计与架构进行分析, 生成一份描述 LeRobot 的设计与架构的技术文档, 文档要紧密结合机器人行业, 要考虑到软件工程的方方面面, 要写得比现有的@bt/docs/bt/ov/lrb_arch_cc.md要好(比如考虑它所没考虑到的,想得更周全,方案比它的好), 要图文并茂(比如要有架构图,各种UML图,mermaid等等). 文档写到 @bt/docs/bt/ov/lrb_arch_cc2.md 中.