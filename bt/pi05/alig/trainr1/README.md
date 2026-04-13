# trainr1/ — OpenPI 对齐的 Pi0.5 训练代码

用 LeRobot 框架复现 OpenPI JAX `pi05_r1pro_chassis` 的训练结果。

## 对齐目标 CLI 命令

```bash
uv run python scripts/train.py pi05_r1pro_chassis \
    --exp_name $EXPNAME --batch_size 256 \
    --num_train_steps 100000 --save_interval 500 --keep_period 2500
```

## 快速开始

```bash
cd /home/Luogang/SRC/Robot/lerobot

# 1. 数据准备 (全部 64 episodes + OpenPI norm_stats 注入)
bash bt/pi05/alig/trainr1/prepare_data.sh

# 2. 静态验证
python bt/pi05/alig/trainr1/compare_openpi_lerobot.py --plot

# 3. 冒烟测试 (200 步, batch_size=4)
bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --smoke-test

# 4. 正式训练 (100000 步, batch_size=256)
# 多 GPU (推荐, 4x A100)
bash bt/pi05/alig/trainr1/train_r1pro_chassis_multi.sh --num-gpus 4

# 或单 GPU + 梯度累积
bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --grad-accum 8
```

## 对齐参数一览

| 对齐项 | OpenPI 值 | CLI 覆盖 |
|--------|----------|---------|
| Weight Decay | 1e-10 | `--policy.optimizer_weight_decay=1e-10` |
| Loss 截断 | 32维(含padding) | `--policy.loss_include_padding=true` |
| LR Schedule | post_warmup, decay_steps=30000 | PI05Config preset 自动 |
| EMA | 0.99 | `--policy.ema_decay=0.99` |
| 数据增强 | 启用 | `--policy.augmentation_enabled=true` |
| Batch Size | **256** (CLI覆盖) | `--batch_size=256` |
| Steps | **100000** (CLI覆盖) | `--steps=100000` |
| Save Freq | **500** (CLI覆盖) | `--save_freq=500` |
| Keep Period | **2500** (CLI覆盖) | 训练后 `cleanup_checkpoints.py` |
| num_workers | 2 | `--num_workers=2` |
| Seed | 42 | `--seed=42` |
| Norm Stats | OpenPI json | `prepare_data.sh --norm-stats-path` |

**注意**: LR schedule 的 `decay_steps=30000` 未被 CLI 覆盖。cosine 在 step 30000 完成后，LR 钳位在 `2.5e-6` 直到训练结束 (step 100000)。

**无需修改 `lerobot/src/` 代码** — 所有对齐通过 CLI 参数实现。

## 文件说明

| 文件 | 用途 |
|------|------|
| `config.py` | 集中式参数定义 |
| `prepare_data.sh` | 数据集准备 (v2.1→v3.0 + norm_stats) |
| `train_r1pro_chassis.sh` | 单 GPU 训练 (支持梯度累积) |
| `train_r1pro_chassis_multi.sh` | 多 GPU 训练 |
| `cleanup_checkpoints.py` | Checkpoint 清理 (keep_period 逻辑) |
| `verify_lr_schedule.py` | LR 调度对齐验证 (含钳位期) |
| `verify_norm_stats.py` | 归一化统计验证 |
| `verify_training.py` | 训练曲线对比 |
| `compare_openpi_lerobot.py` | 端到端验证 |

## 训练脚本选项

```bash
# 冒烟测试 (200 步)
bash train_r1pro_chassis.sh --smoke-test

# 自定义步数
bash train_r1pro_chassis.sh --steps 5000

# 梯度累积 (单 GPU 无法容纳 batch_size=256 时)
bash train_r1pro_chassis.sh --grad-accum 8  # micro_batch=32, effective=256

# 禁用 EMA (消融实验)
bash train_r1pro_chassis.sh --no-ema

# 禁用增强 (消融实验)
bash train_r1pro_chassis.sh --no-augmentation

# 多 GPU
bash train_r1pro_chassis_multi.sh --num-gpus 4  # per-GPU batch=64, effective=256

# Checkpoint 清理
python cleanup_checkpoints.py --checkpoint-dir outputs/checkpoints --keep-period 2500 --dry-run
```

## 内存需求

Pi0.5 (~2.3B 参数), bfloat16 + gradient_checkpointing:

| batch_size | 估计 VRAM | 推荐配置 |
|-----------|----------|---------|
| 32 | ~36 GB | 单 A100-80GB |
| 64 | ~42 GB | 单 A100-80GB |
| 256 | ~50-70 GB | 4x A100-80GB 或单GPU+梯度累积 |

## 相关文档

- `aligdesign_2v2.md` — 综合设计文档 v2 (mermaid 图, 严格对齐 CLI)
- `aligdesign_2.md` — 综合设计文档 v1 (已过时, 参数不匹配)
- `pi05_alig_3.md` — 差异分析 v3
