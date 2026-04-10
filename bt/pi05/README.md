# Pi0.5 LIBERO 微调实验

在 [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) 数据集上对 [Pi0.5](https://huggingface.co/docs/lerobot/main/en/pi05) 进行微调训练的完整流程。

## 概览

| 项目 | 说明 |
|------|------|
| 策略模型 | [Pi0.5 (π₀.₅)](https://huggingface.co/lerobot/pi05_base) — Vision-Language-Action 模型 |
| 数据集 | [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) — LIBERO 机器人操作基准 |
| 采样 | 从 1693 个 episode 中随机抽取 37 个 (seed=42) |
| 训练 | 100 步微调, batch_size=4, bfloat16 精度 |

## 前置条件

### 1. 虚拟环境

使用 `uv` 创建的 Python 虚拟环境，包含 `lerobot[pi]` 依赖:

```bash
# 创建虚拟环境 (如已创建可跳过)
uv venv --python 3.12 /mnt/r/Venv/lerobot-venv

# 激活虚拟环境
source /mnt/r/Venv/lerobot-venv/bin/activate

# 安装 lerobot 及 pi 依赖
uv pip install -e ".[pi]"
```

### 2. Hugging Face 缓存目录

本实验固定使用：

```bash
export HF_HOME=~/hfhome/
mkdir -p ~/hfhome/
```

`bt/pi05/train.sh` 和 `bt/pi05/prepare_data.py` 都会自动设置该环境变量，因此通过这两个入口运行时无需手动再设置。所有 Hugging Face 相关缓存（模型、数据集、metadata）都会写入 `~/hfhome/`。

### 3. 关键依赖版本

| 包 | 版本 |
|---|---|
| lerobot | 0.5.1 |
| torch | 2.10.0+cu128 |
| transformers | 5.5.0 |
| scipy | 1.17.1 |

### 4. 硬件要求

- GPU: 至少 1 张支持 bfloat16 的 NVIDIA GPU (建议 ≥24GB 显存)
- 如果显存不足, 可在 `train.sh` 中减小 `BATCH_SIZE` 或保持 `train_expert_only=true`

## 文件结构

```
bt/pi05/
├── README.md                  # 本文档
├── __init__.py                # Python 包标识
├── prepare_data.py            # 数据采样与下载脚本
├── train.sh                   # 训练入口 shell 脚本
├── train_pi05_local.py        # 训练 Python 入口 (绕过全量策略导入)
├── selected_episodes.json     # (运行后生成) 抽取的 episode 列表
├── outputs/                   # (运行后生成) 训练输出
│   └── pi05_libero_37ep/
│       └── checkpoints/
└── alig/                      # OpenPI 对齐: EMA + Loss 截断
    ├── __init__.py
    ├── aligdesign.md           # 对齐设计方案 (EMA + Loss 截断)
    ├── test_ema_checkpoint.sh  # EMA checkpoint 端到端验证 shell 入口
    ├── test_ema_checkpoint.py  # EMA checkpoint 端到端验证 Python 脚本
    ├── pi05_alig*.md           # 对齐分析文档 (多版本)
    └── mdldiff.md / pi05_diffanalyz.md  # 模型差异分析

tests/policies/pi0_pi05/
├── test_pi05.py               # PI0.5 基础功能测试
├── test_pi05_alignment.py     # EMA + Loss 截断对齐单元测试
└── ...
```

## 使用方法

### 一键执行 (推荐)

```bash
cd /home/luogang/SRC/Robot/lerobot
./bt/pi05/train.sh
```

此命令会自动完成: 数据准备 → 训练。

### 分步执行

#### 步骤 1: 数据准备

从 LIBERO 数据集随机抽取 37 个 episode 并下载到本地:

```bash
./bt/pi05/train.sh prepare
```

或直接运行 Python 脚本:

```bash
python -m bt.pi05.prepare_data --num-episodes 37 --seed 42
```

参数说明:
- `--num-episodes`: 抽取的 episode 数量 (默认 37)
- `--seed`: 随机种子 (默认 42)
- `--skip-download`: 只生成 episode 列表，不下载数据

此步骤会:
1. 用 seed=42 从 1693 个 episode 中随机抽取 37 个
2. 将 episode 索引列表保存到 `selected_episodes.json`
3. 使用 `HF_HOME=~/hfhome/` 作为 Hugging Face 缓存根目录
4. 通过 `LeRobotDataset` 下载选中的 episode 数据到本地缓存
5. 输出数据集摘要信息 (帧数、特征、FPS 等)

#### 步骤 2: 训练

在抽取的 37 个 episode 上对 Pi0.5 微调 100 步:

```bash
./bt/pi05/train.sh train
```

### 其他命令

```bash
./bt/pi05/train.sh help     # 查看帮助和配置
```

## 技术细节

### 数据集说明

LIBERO 是一个用于研究**终身机器人学习 (lifelong robot learning)** 的基准数据集:

- **总 episode 数**: 1,693
- **总帧数**: 273,465
- **任务数**: 40 个 (涵盖 spatial, object, goal, long-horizon 等类型)
- **FPS**: 10 Hz
- **机器人**: Franka Panda

每个 episode 包含:

| 特征 | 维度 | 说明 |
|------|------|------|
| `observation.images.image` | 256×256 RGB | 主相机视角 (agentview) |
| `observation.images.image2` | 256×256 RGB | 腕部相机视角 (eye-in-hand) |
| `observation.state` | 8 维 float32 | 末端执行器位姿 + 夹爪状态 |
| `action` | 7 维 float32 | 6D 末端执行器增量 + 夹爪 |

数据集已采用 LeRobot v3.0 格式 (Parquet + MP4)，无需额外转换。

### 随机抽样

使用 `random.sample(range(1693), 37)` 在 seed=42 下生成的 episode 索引:

```
[13, 51, 54, 61, 65, 178, 191, 209, 228, 285, 407, 447, 451, 457, 476,
 501, 563, 569, 859, 864, 919, 1034, 1116, 1149, 1206, 1209, 1232,
 1309, 1330, 1385, 1436, 1466, 1508, 1516, 1518, 1554, 1657]
```

### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `policy.type` | `pi05` | π₀.₅ 策略 |
| `policy.pretrained_path` | `lerobot/pi05_base` | HuggingFace 上的预训练基座模型 |
| `policy.dtype` | `bfloat16` | 混合精度训练 |
| `policy.gradient_checkpointing` | `true` | 梯度检查点，降低显存占用 |
| `policy.train_expert_only` | `true` | 只训练 action expert 和投影层，冻结 VLM |
| `normalization_mapping` | `QUANTILES` | STATE/ACTION 优先使用 quantile normalization |
| `steps` | `100` | 训练步数 |
| `batch_size` | `4` | 批大小 |
| `log_freq` | `10` | 每 10 步记录日志 |
| `save_freq` | `100` | 每 100 步保存 checkpoint |
| `eval_freq` | `10000` | 评估频率 (大于 steps，即本次不评估) |
| `wandb.enable` | `false` | 关闭 W&B 日志 |

**关于归一化**: Pi0.5 默认优先使用 `QUANTILES` 归一化。如果外部仍传入 `MEAN_STD`，训练脚本在检测到数据集已包含 quantile 统计时会自动切回 `QUANTILES`。如果数据集尚未计算 quantile 统计量，再使用 `MEAN_STD`，或先运行:

```bash
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=HuggingFaceVLA/libero
```

**关于 `train_expert_only`**: 设为 `true` 会冻结 VLM 主干，只训练 action expert 和投影层。这大幅降低显存需求，适合快速实验。如需全参数微调，设为 `false`。

## 脚本使用说明

### 训练脚本

#### `train.sh` — 训练入口

一键完成数据准备 + Pi0.5 微调训练。

```bash
# 完整流程 (数据准备 + 训练)
./bt/pi05/train.sh

# 仅准备数据
./bt/pi05/train.sh prepare

# 仅训练 (需先 prepare)
./bt/pi05/train.sh train

# 查看帮助
./bt/pi05/train.sh help
```

可通过环境变量覆盖默认配置:

```bash
PI05_STEPS=200 PI05_BATCH_SIZE=2 PI05_LOG_FREQ=20 ./bt/pi05/train.sh train
```

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `PI05_STEPS` | `100` | 训练步数 |
| `PI05_BATCH_SIZE` | `4` | 批大小 |
| `PI05_LOG_FREQ` | `10` | 日志频率 |
| `PI05_SAVE_FREQ` | `100` | Checkpoint 保存频率 |
| `PI05_NUM_WORKERS` | `4` | DataLoader worker 数 |
| `PI05_TOKENIZER_NAME` | `google/paligemma-3b-pt-224` | Tokenizer |
| `PI05_NORMALIZATION_MODE` | `QUANTILES` | 归一化模式 |

#### `train_pi05_local.py` — 训练 Python 入口

`train.sh` 内部调用的 Python 脚本。绕过 `lerobot.policies.__init__` 的全量导入链（避免导入 groot 等不必要的依赖），直接构建 PI05Policy 并运行训练循环。

```bash
python -m bt.pi05.train_pi05_local \
    --repo-id="HuggingFaceVLA/libero" \
    --local-root="$HF_HOME/lerobot/HuggingFaceVLA/libero" \
    --episode-file="bt/pi05/selected_episodes.json" \
    --pretrained-path="lerobot/pi05_base" \
    --output-dir="bt/pi05/outputs/my_run" \
    --steps=100 --batch-size=4 --dtype=bfloat16 \
    --gradient-checkpointing --train-expert-only
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo-id` | (必填) | HuggingFace 数据集 ID |
| `--local-root` | (必填) | 本地数据根目录 |
| `--episode-file` | (必填) | episode 列表 JSON 文件 |
| `--pretrained-path` | `lerobot/pi05_base` | 预训练模型 |
| `--output-dir` | (必填) | 输出目录 |
| `--steps` | `100` | 训练步数 |
| `--batch-size` | `4` | 批大小 |
| `--lr` | `2.5e-5` | 学习率 |
| `--dtype` | `bfloat16` | 精度 (`bfloat16` / `float32`) |
| `--gradient-checkpointing` | `false` | 启用梯度检查点 |
| `--train-expert-only` | `false` | 仅训练 action expert |

#### `prepare_data.py` — 数据采样与下载

从 LIBERO 数据集随机抽取 episode 并下载到本地:

```bash
python -m bt.pi05.prepare_data --num-episodes 37 --seed 42
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-episodes` | `37` | 抽取 episode 数 |
| `--seed` | `42` | 随机种子 |
| `--skip-download` | `false` | 只生成列表不下载 |

---

### OpenPI 对齐测试脚本

对齐功能通过两个配置项控制（在 `PI05Config` 中添加）:

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ema_decay` | `None` | EMA 衰减系数。设为 `0.99` 以对齐 OpenPI |
| `loss_include_padding` | `False` | 设为 `True` 使 loss 包含 padding 维度 (OpenPI 行为) |

#### `alig/test_ema_checkpoint.sh` — EMA Checkpoint 端到端验证

验证 EMA 模式下 checkpoint 的保存是否正确。

```bash
# 默认参数 (5 步, decay=0.99)
./bt/pi05/alig/test_ema_checkpoint.sh

# 自定义参数
EMA_STEPS=10 EMA_DECAY=0.995 ./bt/pi05/alig/test_ema_checkpoint.sh
```

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EMA_STEPS` | `5` | 训练步数 |
| `EMA_DECAY` | `0.99` | EMA 衰减系数 |

验证内容 (共 13 项):

| # | 验证项 | 说明 |
|---|--------|------|
| 1-4 | EMA 递推公式 | `ema_new = decay * ema_old + (1-decay) * param_new` |
| 5 | EMA 与训练参数分叉 | 训练后 EMA 参数 ≠ 原始参数 |
| 6-8 | 文件结构 | `model.safetensors`、`ema_state.safetensors`、`raw_trainable_params.safetensors` 存在 |
| 9 | model.safetensors = EMA | 推理权重文件中保存的是 EMA 参数 |
| 10 | model.safetensors ≠ raw | 推理权重 ≠ 原始训练参数 |
| 11 | ema_state 数值一致 | 文件中的 EMA 值与内存一致 |
| 12 | raw_trainable 数值一致 | 文件中的训练参数与内存一致 |
| 13 | save 后参数恢复 | 保存后模型参数恢复为训练参数 |

也可直接运行 Python 脚本:

```bash
python -m bt.pi05.alig.test_ema_checkpoint --steps=5 --ema-decay=0.99
```

#### `tests/policies/pi0_pi05/test_pi05_alignment.py` — 对齐单元测试

使用 pytest 运行的单元测试，覆盖 Loss 截断对齐和 EMA 对齐两个方面。需要 CUDA + HuggingFace token。

```bash
# 运行全部对齐测试
python -m pytest tests/policies/pi0_pi05/test_pi05_alignment.py -v

# 运行单个测试
python -m pytest tests/policies/pi0_pi05/test_pi05_alignment.py::test_ema_update_formula -v
```

测试列表:

| 测试名 | 类别 | 说明 |
|--------|------|------|
| `test_loss_truncation_gradient_coverage` | Loss 截断 | `loss_include_padding=False` 时 padding 列梯度为 0；`=True` 时全部列有梯度 |
| `test_ema_disabled_by_default` | EMA | `ema_decay=None` 时 `update()` 无操作 |
| `test_ema_init_on_first_update` | EMA | 首次 `update()` 延迟初始化，EMA = 当前参数 |
| `test_ema_update_formula` | EMA | 验证公式 `ema = 0.99 * ema_old + 0.01 * param_new` |
| `test_ema_inference_uses_ema_params` | EMA | `select_action` 使用 EMA 参数，输出与裸参数不同 |
| `test_ema_swap_restore_roundtrip` | EMA | `_swap_to_ema` + `_restore_from_backup` 完美往返 |
| `test_ema_nested_swap_protection` | EMA | 嵌套调用 `_swap_to_ema` 返回 None，防止重复交换 |

---

## 参考文档

- [LeRobot Pi0.5 文档](https://huggingface.co/docs/lerobot/main/en/pi05)
- [LeRobot LIBERO 文档](https://huggingface.co/docs/lerobot/main/en/libero)
- [LeRobot 安装指南](https://huggingface.co/docs/lerobot/main/en/installation)
- [Pi0.5 论文 (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi05)
- [LIBERO 论文](https://arxiv.org/abs/2306.03310)
