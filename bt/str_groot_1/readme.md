# str_groot_1 — StarVLA Qwen-GR00T 在 LIBERO 上的训练

## 概述

将 StarVLA 的 Qwen3-VL + GR00T FlowMatching action head（`str_groot`）适配到 LeRobot 框架，
在 `HuggingFaceVLA/libero` 数据集上进行微调训练。

模型结构：Qwen3-VL-4B (VLM backbone) → DiT-B FlowMatching (action head)，总参数 ~5B，可训练参数 ~161M（冻结 VLM 时）。

## 文件说明

| 文件 | 说明 |
|---|---|
| `train_str_groot_libero.py` | 训练入口脚本 |
| `test_str_groot_libero37.py` | 分阶段验证脚本（config → dataset → model → forward → predict → e2e） |
| `test_episodes_37.json` | 上次测试使用的 37 条 episode 索引 |

## 快速端到端验证

从 LIBERO（共 1693 条 episode）中随机抽 37 条，跑 2 步训练，验证完整流水线：

```bash
python bt/str_groot_1/train_str_groot_libero.py \
  --episodes 0 1 3 4 11 12 13 15 16 18 24 25 27 28 29 30 32 35 36 37 \
             39 40 41 42 46 48 49 50 53 54 56 57 58 60 61 62 63 \
  --steps 2 \
  --batch-size 1 \
  --freeze-vlm \
  --no-save-checkpoint \
  --starvla-checkpoint "" \
  --log-freq 1 \
  --num-workers 0
```

> `--starvla-checkpoint ""` 表示不加载 StarVLA 预训练权重（action head 随机初始化），仅用于快速验证流水线。
> 正式训练时去掉此参数，默认使用 `StarVLA/Qwen3VL-GR00T-Bridge-RT-1`。

### 验证通过的输出（2026-03-20）

```
dataset.num_frames  = 5338
dataset.num_episodes = 37
num_learnable_params = 161473799 (161M)
num_total_params     = 4599289607 (5B)

step:1 smpl:1 ep:0 epch:0.00 loss:1.108 grdn:4.057  lr:5.5e-05 updt_s:1.388
step:2 smpl:2 ep:0 epch:0.00 loss:2.147 grdn:12.221 lr:1.0e-05 updt_s:0.072
End of training
```

## 正式训练

```bash
# 全量训练（全部 1693 episodes, 50k steps）
python bt/str_groot_1/train_str_groot_libero.py \
  --steps 50000 \
  --batch-size 8 \
  --freeze-vlm \
  --wandb

# 指定 episode 子集训练
python bt/str_groot_1/train_str_groot_libero.py \
  --episodes 0 1 3 4 11 12 13 15 16 18 24 25 27 28 29 30 32 35 36 37 \
  --steps 100 \
  --batch-size 8
```

## 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset-repo` | `HuggingFaceVLA/libero` | HuggingFace 数据集 repo |
| `--episodes` | 全部 | 指定 episode 索引列表 |
| `--starvla-checkpoint` | `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` | 预训练权重（HF repo 或本地路径，空字符串跳过） |
| `--base-vlm` | `Qwen/Qwen3-VL-4B-Instruct` | VLM backbone |
| `--action-dim` | 7 | 动作维度 |
| `--state-dim` | 7 | 状态维度（会自动从数据集推断） |
| `--freeze-vlm` | False | 是否冻结 VLM 只训练 action head |
| `--steps` | 50000 | 训练步数 |
| `--batch-size` | 8 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--save-freq` | 5000 | checkpoint 保存间隔 |
| `--wandb` | False | 启用 WandB 日志 |

## 已修复的问题

### state_dim 维度不匹配 (2026-03-20)

**现象**：训练时 `state_encoder` 报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x8 and 7x1024)`。

**原因**：`StrGrootConfig.state_dim` 默认为 7，但 LIBERO 数据集的 `observation.state` 实际维度为 8。
`factory.py` 在创建 policy 前会从数据集自动填充 `input_features`，但 `validate_features()` 没有将
`state_dim` 同步为数据集的真实维度，导致 `state_encoder` 的 MLP 用错误的输入维度构建。

**修复**（`configuration_str_groot.py`）：在 `validate_features()` 中，当 `observation.state` 已存在于
`input_features` 时，自动将 `state_dim` 更新为数据集提供的实际维度。
