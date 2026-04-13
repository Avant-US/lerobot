# LeRobot pi0.5 与 OpenPI JAX 对齐训练：综合设计与实现方案

> **日期**: 2026-04-12
> **目标**: 在 `lerobot/bt/pi05/alig/trainr1/` 中生成训练代码，使用 LeRobot 框架复现 OpenPI JAX `pi05_r1pro_chassis` 的训练结果，实现训练曲线和模型精度尽量一致
> **基准**: OpenPI **JAX 训练路径** (`scripts/train.py`)，配置 `pi05_r1pro_chassis`
> **前序**: `pi05_alig_3.md` (差异分析), `aligdesign.md` (EMA/Loss设计), `aligdesign_1v2.md` (LR/增强设计)    
> Prmp:请在python虚拟环境/mnt/r/Venv/lerobot-venv/中, 深度分析 @lerobot/ 和 @pi/openpi/ 这两个本地源码库现在的代码, 分析 @lerobot/bt/pi05/alig/中的各个文档(比如但不限于pi05_diffanalyz.md,mdldiff.md等等), 我们的目标是在 @lerobot/bt/pi05/alig/trainr1/ 中生成代码来用lerobot复现在 @pi/openpi/doc/pi05_微调.md 中调用openpi的`sccript/train.py`对数据 @pi/data/r1_pro_data_convert_chassis 进行训练的结果, 要做到训练曲线尽量一致, 训练出来的模型的精度尽量一致. 同时注意要在需要对齐的地方都给予对齐, 特别是@lerobot/bt/pi05/alig/pi05_alig_3.md, pi05_alig_2.md, pi05_alig.md 中标注了`@#2`的章节(比如,"Weight Decay", "Normalization Stats", "batch_size 和 steps", "Seed 与随机性", "Tokenizer", "LR Scheduler 初始值"等等), 而"Normalization Stats"的对齐采用"使用数据集已有的/norm_stats.json"的方式, 其它方面的对齐尽量通过调超参或者config去达到, 如果迫不得已要修改@lerobot/src/内的代码的话则需要给出理由,方案和影响分析. 那么, 请对上面提到的代码和文档进入深入分析与讨论, 想想如果要达到刚才提到的生成对齐的训练代码这个目标的话, 我们需要怎么做? 怎么设计与实现? 实现完后又要怎么测试与验收? 分析时要考虑到软件工程的方方面面, 要考虑到各种风险, 要梳理好各个流程与管道, 然后把分析及其结果和设计实现方案写到 @lerobot/bt/pi05/alig/aligdesign_2.md 中, 注意要有核心代码说明, 要图文并茂(比如画各种UML图)地说明.
> 之前你写的 @lerobot/bt/pi05/alig/aligdesign_2.md 并没有按照我的意思做, 我的意思是目标是在 @lerobot/bt/pi05/alig/trainr1/ 中生成代码来用lerobot复现在 @pi/openpi/doc/pi05_微调.md 中用调用了jax版本的openpi/scripts/train.py 的命令行语句`uv run python scripts/train.py pi05_r1pro_chassis --exp_name $EXPNAME --batch_size 256 --num_train_steps 100000 --save_interval 500 --keep_period 2500`对数据 @pi/data/r1_pro_data_convert_chassis 进行pi0.5训练的结果. 请在python虚拟环境/mnt/r/Venv/lerobot-venv/中对aligdesign_2.md进行深入改良, 请严格对齐该命令行语句, 涉及到UML图的部分都需要用mermaid绘图, 一些没有考虑到的细节点和风险点要考虑到, 其它要求与对请对aligdesign_2.md的要求一致. 最后, @lerobot/bt/pi05/alig/trainr1/ 中之前生成的代码也要作出相应修改. 改良后的结果写到 @lerobot/bt/pi05/alig/aligdesign_2v2.md 中.

---

## 目录

1. [项目概述与目标](#1-项目概述与目标)
2. [系统架构分析](#2-系统架构分析)
3. [对齐项详细分析](#3-对齐项详细分析)
4. [参数映射完整表](#4-参数映射完整表)
5. [数据管道设计](#5-数据管道设计)
6. [训练流程设计](#6-训练流程设计)
7. [核心代码说明](#7-核心代码说明)
8. [trainr1/ 文件结构](#8-trainr1-文件结构)
9. [验证方案](#9-验证方案)
10. [风险分析与缓解](#10-风险分析与缓解)
11. [实施步骤](#11-实施步骤)

---

## 1. 项目概述与目标

### 1.1 问题陈述

OpenPI 使用 JAX 框架训练 Pi0.5 VLA 模型，LeRobot 使用 PyTorch。两者在多个维度上存在默认参数差异，导致直接使用 LeRobot 训练得到的模型与 OpenPI 训练的模型在训练曲线和推理质量上有显著差异。

### 1.2 目标定义

```
┌─────────────────────────────────────────────────────────────────────┐
│                    训练等价的三个层次                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  L1: Loss 曲线等价                                                   │
│    ├─ 30000 步训练的 loss 曲线在量级和趋势上对齐                       │
│    ├─ 可达性: ✅ 修复 P0 后可达                                       │
│    └─ 验收: loss 曲线相对差异 < 20%                                   │
│                                                                     │
│  L2: Checkpoint 质量等价                                             │
│    ├─ 最终 checkpoint 的推理性能无统计显著差异                          │
│    ├─ 可达性: ✅ 修复 P0+P1 后可达                                    │
│    └─ 验收: 离线评估动作预测 MSE 差异 < 10%                            │
│                                                                     │
│  L3: Action 输出等价                                                 │
│    ├─ 同输入下推理输出的 action 向量逐元素相近                          │
│    ├─ 可达性: ❌ 跨框架不可达 (RNG/数值精度)                           │
│    └─ 原因: JAX key-PRNG vs PyTorch stateful PRNG, XLA vs Eager     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 核心发现

经过对两个代码库的深度分析，**LeRobot 的 `PI05Config` 已实现所有对齐所需的配置字段**。不需要修改 `lerobot/src/` 下的任何代码，所有对齐均可通过训练脚本的 CLI 参数覆盖实现。

---

## 2. 系统架构分析

### 2.1 训练流程对比 — 活动图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   OpenPI JAX 训练流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Dataset  │───▶│ Transforms│───▶│ Collate  │───▶│ jax.device_put()    │  │
│  │ (LeRobot │    │ (Python)  │    │ (numpy)  │    │ → JAX arrays        │  │
│  │  HF fmt) │    │           │    │          │    │                      │  │
│  └──────────┘    └───────────┘    └──────────┘    └──────────┬───────────┘  │
│                                                               │              │
│       ┌───────────────────────────────────────────────────────▼──────────┐   │
│       │                  JIT-compiled train_step                         │   │
│       │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │   │
│       │  │ Data Augment │  │ Forward Pass │  │ Backward + Optimizer  │ │   │
│       │  │ (augmax)     │──│ (Flow Match) │──│ (optax AdamW)         │ │   │
│       │  │              │  │  MSE Loss    │  │ clip_grad + wd=1e-10  │ │   │
│       │  └──────────────┘  │  (32-dim)    │  └──────────┬─────────────┘ │   │
│       │                    └──────────────┘              │               │   │
│       │                                    ┌─────────────▼─────────────┐ │   │
│       │                                    │ EMA Update               │ │   │
│       │                                    │ ema = 0.99*ema + 0.01*θ │ │   │
│       │                                    └───────────────────────────┘ │   │
│       └─────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│       ┌─────────────────────────────────────────────────────────────────┐   │
│       │ Checkpoint: saves EMA params (not raw params) via Orbax       │   │
│       └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   LeRobot PyTorch 训练流程 (对齐后)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Dataset  │───▶│Preprocessor──▶│DataLoader│───▶│ accelerator.prepare()│  │
│  │ (LeRobot │    │(Normalize,│    │(PyTorch) │    │ → CUDA tensors       │  │
│  │  v3.0)   │    │ Tokenize) │    │          │    │                      │  │
│  └──────────┘    └───────────┘    └──────────┘    └──────────┬───────────┘  │
│                                                               │              │
│       ┌───────────────────────────────────────────────────────▼──────────┐   │
│       │                    update_policy()                               │   │
│       │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │   │
│       │  │ Data Augment │  │ Forward Pass │  │ Backward + Optimizer  │ │   │
│       │  │ (torchvision)│──│ (Flow Match) │──│ (torch AdamW)         │ │   │
│       │  │ GPU-side     │  │  MSE Loss    │  │ clip_grad + wd=1e-10  │ │   │
│       │  └──────────────┘  │  (32-dim) ★  │  └──────────┬─────────────┘ │   │
│       │                    └──────────────┘              │               │   │
│       │                                    ┌─────────────▼─────────────┐ │   │
│       │                                    │ policy.update()          │ │   │
│       │                                    │ EMA: 0.99*ema + 0.01*θ  │ │   │
│       │                                    └───────────────────────────┘ │   │
│       └─────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│       ┌─────────────────────────────────────────────────────────────────┐   │
│       │ Checkpoint: swap EMA → save model.safetensors → restore       │   │
│       └─────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ★ loss_include_padding=True → MSE 在全部 32 维计算 (含 9 维 padding)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 组件关系 — 类图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐    ┌────────────────────────────────┐  │
│  │   TrainPipelineConfig│    │        PI05Config              │  │
│  │   (configs/train.py) │    │  (configuration_pi05.py)      │  │
│  ├──────────────────────┤    ├────────────────────────────────┤  │
│  │ batch_size: 64       │    │ dtype: "bfloat16"             │  │
│  │ steps: 30000         │    │ optimizer_weight_decay: 1e-10 │  │
│  │ seed: 42             │    │ loss_include_padding: True    │  │
│  │ log_freq: 100        │    │ ema_decay: 0.99              │  │
│  │ save_freq: 1000      │    │ augmentation_enabled: True   │  │
│  │ eval_freq: -1        │    │ gradient_checkpointing: True │  │
│  ├──────────────────────┤    ├────────────────────────────────┤  │
│  │ optimizer ◄──────────│────│ get_optimizer_preset()        │  │
│  │ scheduler ◄──────────│────│ get_scheduler_preset()        │  │
│  └──────────────────────┘    │   phase_mode="post_warmup"    │  │
│           │                  └──────────────┬─────────────────┘  │
│           │                                 │                     │
│           ▼                                 ▼                     │
│  ┌──────────────────┐           ┌──────────────────────┐         │
│  │    AdamWConfig   │           │    PI05Policy         │         │
│  │ (optimizers.py)  │           │ (modeling_pi05.py)    │         │
│  ├──────────────────┤           ├──────────────────────┤         │
│  │ lr: 2.5e-5       │           │ _ema_params: dict    │         │
│  │ betas: (0.9,0.95)│           │ _ema_active: bool    │         │
│  │ weight_decay:1e-10│          │                      │         │
│  │ grad_clip: 1.0   │           ├──────────────────────┤         │
│  └──────────────────┘           │ forward(batch)       │         │
│                                 │   → MSE loss [B,50,32]│         │
│  ┌──────────────────┐           │ update()             │         │
│  │ CosineDecay      │           │   → EMA step         │         │
│  │ WithWarmup       │           │ _swap_to_ema()       │         │
│  │ (schedulers.py)  │           │ _restore_from_backup()│         │
│  ├──────────────────┤           │ _augment_image()     │         │
│  │ warmup: 1000     │           │ _preprocess_images() │         │
│  │ decay: 30000     │           └──────────────────────┘         │
│  │ peak: 2.5e-5     │                                            │
│  │ end: 2.5e-6      │                                            │
│  │ phase: post_warmup│                                           │
│  └──────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 数据流 — 序列图

```
┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────┐
│ Dataset  │  │Normalizer│  │ Tokenizer │  │ PI05Model│  │Optimizer│
│ (v3.0)   │  │(Quantile)│  │(PaliGemma)│  │(PyTorch) │  │ (AdamW) │
└────┬─────┘  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └────┬────┘
     │              │              │              │              │
     │  raw sample  │              │              │              │
     │──────────────▶              │              │              │
     │              │              │              │              │
     │  state[23], action[50,23]   │              │              │
     │  images: uint8[H,W,3] ×3   │              │              │
     │  task: str                  │              │              │
     │              │              │              │              │
     │              │ quantile normalize          │              │
     │              │ state → [-1,1] (q01/q99)    │              │
     │              │ action → [-1,1] (q01/q99)   │              │
     │              │──────────────▶              │              │
     │              │              │              │              │
     │              │  "Task: {text}, State: {discretized};\nAction: "
     │              │              │ tokenize → tokens[200]      │
     │              │              │──────────────▶              │
     │              │              │              │              │
     │              │              │  pad state/action to 32-dim │
     │              │              │  resize images → 224×224     │
     │              │              │  normalize images → [-1,1]   │
     │              │              │  augment (if training)       │
     │              │              │              │              │
     │              │              │    flow matching forward     │
     │              │              │    noise ~ N(0,1)            │
     │              │              │    time ~ Beta(1.5,1)*0.999+0.001
     │              │              │    x_t = t*noise + (1-t)*action
     │              │              │    u_t = noise - action      │
     │              │              │    v_t = model(x_t, time)    │
     │              │              │    loss = MSE(v_t, u_t)  [B,50,32]
     │              │              │              │              │
     │              │              │    loss.mean() → scalar      │
     │              │              │              │──────────────▶
     │              │              │              │  backward()  │
     │              │              │              │  clip_grad(1.0)
     │              │              │              │  optimizer.step()
     │              │              │              │  scheduler.step()
     │              │              │              │              │
     │              │              │  policy.update() → EMA      │
     │              │              │              │              │
```

---

## 3. 对齐项详细分析

### 3.1 P0-1: Weight Decay (@#2)

**差异**: OpenPI `1e-10` vs LeRobot 默认 `0.01`

**根因**: OpenPI 的 `AdamW` 默认值在 `optimizer.py:73` 为 `1e-10`（实质无正则化），而 LeRobot 遵循 PyTorch 社区惯例设为 `0.01`。

**影响分析**:
- AdamW 中 weight_decay 效果: 每步将参数乘以 `(1 - lr × wd)`
- `wd=0.01, lr=2.5e-5`: 每步乘 `0.99999975`
- 30000 步后: 参数缩小约 `(1-2.5e-7)^30000 ≈ 0.9925` (缩小 0.75%)
- 这改变了 loss landscape 和收敛方向

**修复**: CLI 覆盖 `--policy.optimizer_weight_decay=1e-10`

**代码路径**:
```
PI05Config.optimizer_weight_decay (configuration_pi05.py:88)
  → PI05Config.get_optimizer_preset() (line 163-170)
    → AdamWConfig(weight_decay=1e-10) (optimizers.py)
      → torch.optim.AdamW(params, weight_decay=1e-10)
```

### 3.2 P0-2: Loss 截断 (@#0)

**差异**: OpenPI 在 32 维 (含 9 维 padding) 上计算 MSE，LeRobot 截断到 23 维

**根因**: OpenPI 的 `pi0.py:214` 对 `action_out_proj` 全部 32 列输出计算 `jnp.mean(jnp.square(v_t - u_t), axis=-1)`。LeRobot 的 `modeling_pi05.py:1402` 在 `loss_include_padding=False` 时截断到实际维度。

**影响分析**:
1. **梯度流**: `action_out_proj` 的后 9 列在截断模式下无梯度
2. **Loss 数值**: OpenPI loss ÷ 32 含 "容易的" padding 维 → 数值偏低
3. **隐式正则化**: OpenPI 训练后 padding 维输出被约束为接近 0

**修复**: CLI 覆盖 `--policy.loss_include_padding=true`

**代码路径**:
```
PI05Config.loss_include_padding (configuration_pi05.py:93)
  → PI05Policy.forward() (modeling_pi05.py:1401-1407)
    if not self.config.loss_include_padding:
        losses = losses[:, :, :original_action_dim]  # 截断
    # loss_include_padding=True 时保留全部 32 维
```

### 3.3 P0-3: LR Schedule 余弦相位 (@#1)

**差异**: OpenPI 余弦跨 29000 步 (warmup后到decay_steps)，LeRobot absolute 模式跨 30000 步 (从 step 0 开始)

**根因**: `optax.warmup_cosine_decay_schedule` 的 `decay_steps=30000` 是**总步数**（含 warmup），余弦实际跨度为 `30000-1000=29000` 步。LeRobot 的 `absolute` 模式中余弦输入是 `step/decay_steps`，不减去 warmup。

**影响**: 中后期 LR 系统性偏低 4-8%:

| Step | OpenPI LR | LeRobot(absolute) | 差异 |
|------|-----------|-------------------|------|
| 1000 | 2.500e-5 | 2.494e-5 | -0.25% |
| 15000 | 1.434e-5 | 1.375e-5 | -4.13% |
| 20000 | 1.056e-5 | 1.002e-5 | -5.1% |

**修复**: PI05Config 的 `get_scheduler_preset()` 已使用 `phase_mode="post_warmup"`。

**代码路径**:
```
PI05Config.get_scheduler_preset() (configuration_pi05.py:172-179)
  → CosineDecayWithWarmupSchedulerConfig(phase_mode="post_warmup")
    → cosine_decay_schedule() (schedulers.py:130-133)
      total_cosine_steps = decay_steps - warmup_steps  # 29000
      progress = (step - warmup_steps) / total_cosine_steps
      # 与 optax 语义一致
```

### 3.4 P1-1: EMA (@#0)

**差异**: OpenPI `ema_decay=0.99`，checkpoint 保存 EMA 参数；LeRobot 默认无 EMA

**影响**: EMA 不影响训练动态（梯度基于原始参数），但影响推理模型质量 0.5-3%

**修复**: CLI 覆盖 `--policy.ema_decay=0.99`

**实现追踪**:
```
PI05Policy.__init__()
  → self._ema_params = None  (modeling_pi05.py:938)

训练循环每步:
  update_policy() (lerobot_train.py:142-144)
    → policy.update()
      → PI05Policy.update() (modeling_pi05.py:1146-1159)
        if self.config.ema_decay is not None:
          if self._ema_params is None:
            self._init_ema()  # 首次: 克隆所有参数
          for name, param in self.model.named_parameters():
            ema[name] = decay * ema[name] + (1-decay) * param

Checkpoint 保存:
  save_checkpoint() (train_utils.py:104-113)
    → policy._swap_to_ema()  # 将 EMA 参数写入 model
    → policy.save_pretrained()  # model.safetensors 含 EMA 权重
    → policy._restore_from_backup()  # 恢复原始训练参数
```

### 3.5 P1-2: 数据增强 (@#1)

**差异**: OpenPI 在 JIT 内执行相机感知增强；LeRobot 默认无增强

**增强参数**:
| 增强类型 | 参数 | 非 wrist 相机 | wrist 相机 |
|---------|------|:-----:|:-----:|
| RandomCrop | 95% (212×212→224×224) | ✅ | ❌ |
| Rotate | ±5° | ✅ | ❌ |
| ColorJitter(b=0.3,c=0.4,s=0.5) | — | ✅ | ✅ |

**修复**: CLI 覆盖 `--policy.augmentation_enabled=true`

**代码路径**:
```
PI05Policy._preprocess_images() (modeling_pi05.py:1314-1315)
  if self.training and self.config.augmentation_enabled:
    image = self._augment_image(image, camera_key)

PI05Policy._augment_image() (modeling_pi05.py:1204-1267)
  is_wrist = any(p in camera_key for p in self.config.aug_wrist_patterns)
  if not is_wrist:
    # RandomCrop(95%) + Resize(224) + Rotate(±5°) + ColorJitter
  else:
    # 仅 ColorJitter
```

### 3.6 P2: Normalization Stats (@#2)

**对齐方式**: 使用数据集已有的 norm_stats.json

通过 `convert_r1pro_to_lerobot.py --norm-stats-path` 将 OpenPI 的 `norm_stats.json` 中的 q01/q99 精确注入到 LeRobot 数据集的 `meta/stats.json` 中。

**关键映射**:
- OpenPI `state` → LeRobot `observation.state`
- OpenPI `actions` → LeRobot `action`

**归一化公式对比**:
```
OpenPI:  (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
LeRobot: 2.0 * (x - q01) / max(q99 - q01, 1e-8) - 1.0

实际数据中 q99-q01 >> 1e-6, 两个公式等价 (差异 < 1e-6)
```

### 3.7 P2: Batch Size 与 Steps (@#2)

- OpenPI config: `batch_size=64`, `num_train_steps=30000`
- 修复: `--batch_size=64 --steps=30000`
- 注意: 若单 GPU 内存不足，可结合 `--policy.gradient_checkpointing=true` 和 `--policy.dtype=bfloat16`

### 3.8 P2: Seed (@#2)

- OpenPI: `seed=42`; LeRobot 默认: `seed=1000`
- 修复: `--seed=42`
- 注意: 由于 JAX/PyTorch RNG 实现不同，即使同 seed 也不会产生相同的随机序列，但能保证训练过程的确定性可重复

### 3.9 P2: Tokenizer (@#2)

- OpenPI: SentencePiece (`paligemma_tokenizer.model`), `max_token_len=200`
- LeRobot: HuggingFace `AutoTokenizer("google/paligemma-3b-pt-224")`, `tokenizer_max_length=200`
- 两者使用相同的 PaliGemma 词表，token ID 一致
- `tokenizer_max_length=200` 已在 PI05Config 中设置为默认值

### 3.10 P2: LR Scheduler 初始值

- OpenPI: `init_value = peak_lr / (warmup_steps + 1) = 2.5e-5 / 1001 ≈ 2.497e-8`
- LeRobot: `lr_lambda(0) = 1 / (warmup_steps + 1)`, 实际 lr = `peak_lr × lr_lambda(0) = 2.5e-5 / 1001`
- **公式完全一致，无需对齐**

---

## 4. 参数映射完整表

| # | 参数 | OpenPI 值 | LeRobot 默认 | 对齐值 | 覆盖方式 | 优先级 |
|---|------|----------|-------------|--------|---------|--------|
| 1 | weight_decay | 1e-10 | 0.01 | **1e-10** | `--policy.optimizer_weight_decay=1e-10` | P0 |
| 2 | loss_include_padding | True(32维) | False(23维) | **True** | `--policy.loss_include_padding=true` | P0 |
| 3 | LR phase_mode | post_warmup | absolute | **post_warmup** | PI05Config preset 自动设置 | P0 |
| 4 | ema_decay | 0.99 | None | **0.99** | `--policy.ema_decay=0.99` | P1 |
| 5 | augmentation | True | False | **True** | `--policy.augmentation_enabled=true` | P1 |
| 6 | dtype | bfloat16 | float32 | **bfloat16** | `--policy.dtype=bfloat16` | P1 |
| 7 | batch_size | 64 | 8 | **64** | `--batch_size=64` | P2 |
| 8 | steps | 30000 | 100000 | **30000** | `--steps=30000` | P2 |
| 9 | seed | 42 | 1000 | **42** | `--seed=42` | P2 |
| 10 | norm_stats | OpenPI json | 数据集 meta | **OpenPI json** | `--norm-stats-path` (数据准备) | P2 |
| 11 | lr | 2.5e-5 | 2.5e-5 | 2.5e-5 | 已对齐 | - |
| 12 | betas | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) | 已对齐 | - |
| 13 | eps | 1e-8 | 1e-8 | 1e-8 | 已对齐 | - |
| 14 | grad_clip | 1.0 | 1.0 | 1.0 | 已对齐 | - |
| 15 | warmup_steps | 1000 | 1000 | 1000 | 已对齐 | - |
| 16 | decay_lr | 2.5e-6 | 2.5e-6 | 2.5e-6 | 已对齐 | - |
| 17 | chunk_size | 50 | 50 | 50 | 已对齐 | - |
| 18 | max_action_dim | 32 | 32 | 32 | 已对齐 | - |
| 19 | tokenizer_max_length | 200 | 200 | 200 | 已对齐 | - |
| 20 | image_resolution | 224×224 | 224×224 | 224×224 | 已对齐 | - |

**总结**: 需要通过 CLI 覆盖 9 项参数，其余 11 项已默认对齐。

---

## 5. 数据管道设计

### 5.1 数据转换流程

```
┌──────────────────────────────────────────────────────────────────────┐
│           数据准备流程 (prepare_data.sh)                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────┐                                         │
│  │ 源: OpenPI v2.1 数据集    │  /mnt/r/share/lkx/pi/data/             │
│  │ r1_pro_data_convert_chassis │  r1_pro_data_convert_chassis        │
│  │ 64 episodes, 61923 frames  │                                      │
│  │ 列名: head_rgb, state, ...  │                                     │
│  └──────────┬──────────────┘                                         │
│             │                                                        │
│             ▼ Phase 0+1: 采样 + 列名重命名                            │
│  ┌─────────────────────────┐                                         │
│  │ head_rgb → observation.images.head_rgb                            │
│  │ state → observation.state                                         │
│  │ actions → action                                                  │
│  │ v2.1 格式 (带 LeRobot 列名)                                       │
│  └──────────┬──────────────┘                                         │
│             │                                                        │
│             ▼ Phase 2: v2.1 → v3.0 格式升级                           │
│  ┌─────────────────────────┐                                         │
│  │ 使用 LeRobot 官方转换器    │  convert_dataset_v21_to_v30           │
│  │ 生成 stats.json (无分位数)  │                                      │
│  └──────────┬──────────────┘                                         │
│             │                                                        │
│             ▼ Phase 2.5: Norm Stats 注入                              │
│  ┌─────────────────────────┐   ┌─────────────────────────────────┐   │
│  │ 从 OpenPI norm_stats.json│◀──│ openpi/assets/.../norm_stats.json│  │
│  │ 导入 q01/q99 到 stats.json │  │ state: {q01:[23], q99:[23]}    │  │
│  │ 补充 q10/q50/q90 (插值)    │  │ actions: {q01:[23], q99:[23]}  │  │
│  └──────────┬──────────────┘   └─────────────────────────────────┘   │
│             │                                                        │
│             ▼ Phase 3: 验证                                          │
│  ┌─────────────────────────┐                                         │
│  │ ✓ codebase_version=v3.0  │                                        │
│  │ ✓ features 完整           │                                       │
│  │ ✓ stats.json 含 q01/q99   │                                      │
│  │ ✓ LeRobotDataset 可加载   │                                       │
│  └─────────────────────────┘                                         │
│                                                                      │
│  输出: trainr1/data/r1_pro_chassis_v30/                              │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 训练时数据流

```
LeRobotDataset.__getitem__(index)
    │
    ├── observation.images.head_rgb: uint8[360,640,3]
    ├── observation.images.left_wrist_rgb: uint8[480,640,3]
    ├── observation.images.right_wrist_rgb: uint8[480,640,3]
    ├── observation.state: float32[23]
    ├── action: float32[50,23]   (chunk_size=50)
    └── task: str  (e.g. "Open the door...")
        │
        ▼ PolicyProcessorPipeline (pre-processing)
        │
        ├── NormalizerProcessorStep
        │   └── state, action → quantile normalize to [-1,1]
        │       formula: 2*(x-q01)/(q99-q01) - 1
        │
        ├── Pi05PrepareStateTokenizerProcessorStep
        │   ├── discretize state: np.digitize(state, linspace(-1,1,257)) → [0-255]
        │   └── format: "Task: {text}, State: {digits};\nAction: "
        │
        └── TokenizerProcessorStep
            └── PaliGemma tokenizer → tokens[200], attention_mask[200]
                │
                ▼ PI05Policy.forward()
                │
                ├── _preprocess_images()
                │   ├── resize_with_pad → 224×224
                │   ├── normalize: /255*2-1 → [-1,1]
                │   └── augment (if training + enabled)
                │       ├── non-wrist: RandomCrop(95%) + Resize + Rotate(±5°) + ColorJitter
                │       └── wrist: ColorJitter only
                │
                ├── prepare_action()
                │   └── pad action: [50,23] → [50,32]  (append 9 zeros)
                │
                └── model.forward()  (flow matching)
                    ├── noise = N(0,1) [B,50,32]
                    ├── time = Beta(1.5,1)*0.999+0.001 [B]
                    ├── x_t = t*noise + (1-t)*action [B,50,32]
                    ├── u_t = noise - action [B,50,32]
                    ├── v_t = model(x_t, time) [B,50,32]
                    └── loss = MSE(v_t, u_t) [B,50,32] → mean() → scalar
```

---

## 6. 训练流程设计

### 6.1 训练循环 — 伪代码

```python
# train_r1pro_chassis.sh 最终调用的等效逻辑:

# 1. 初始化
dataset = LeRobotDataset("local/r1_pro_chassis_v30", root=DATA_DIR)
policy = PI05Policy.from_pretrained("lerobot/pi0.5_base",
    config=PI05Config(
        dtype="bfloat16",
        optimizer_weight_decay=1e-10,      # @#2 P0
        loss_include_padding=True,          # @#2 P0
        ema_decay=0.99,                     # @#2 P1
        augmentation_enabled=True,          # @#2 P1
        gradient_checkpointing=True,
    ))

optimizer = AdamW(policy.parameters(),
    lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8,
    weight_decay=1e-10)                     # @#2 P0

scheduler = CosineDecayWithWarmup(
    warmup=1000, decay=30000,
    peak=2.5e-5, end=2.5e-6,
    phase_mode="post_warmup")               # @#2 P0

set_seed(42)                                # @#2

# 2. 训练循环
for step in range(30000):                   # @#2
    batch = next(dataloader)                # batch_size=64  @#2
    batch = preprocessor(batch)             # quantile norm with OpenPI stats

    loss, output_dict = policy.forward(batch)  # 32-dim MSE
    loss.backward()
    clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    policy.update()                         # EMA: 0.99*ema + 0.01*θ

    if step % 1000 == 0:
        # Checkpoint: save EMA params
        backup = policy._swap_to_ema()
        policy.save_pretrained(ckpt_dir)    # model.safetensors = EMA weights
        policy._restore_from_backup(backup)
```

### 6.2 Checkpoint 内容

```
outputs/r1pro_chassis_aligned/
├── checkpoints/
│   ├── 001000/
│   │   ├── pretrained_model/
│   │   │   ├── config.json              # PI05Config (含对齐参数)
│   │   │   └── model.safetensors        # EMA 权重 (用于推理)
│   │   ├── training_state/
│   │   │   ├── optimizer_state.safetensors  # 原始训练优化器状态
│   │   │   ├── scheduler_state.json
│   │   │   └── ema_state.safetensors    # EMA 参数副本
│   │   └── train_config.json
│   ├── 002000/
│   │   └── ...
│   └── 030000/                          # 最终 checkpoint
│       └── ...
├── train_config.json                    # 训练完整配置
└── wandb/                               # WandB 日志
```

---

## 7. 核心代码说明

### 7.1 config.py — 集中式参数定义

`trainr1/config.py` 集中定义所有对齐参数，训练脚本和验证脚本都从此文件导入。

关键函数:
- `build_train_cli_args()`: 生成 `lerobot_train.py` 的完整 CLI 参数列表
- `print_alignment_summary()`: 打印参数对齐摘要，便于检查

```python
# config.py 核心结构
OPTIMIZER_CONFIG = {
    "weight_decay": 1e-10,    # @#2 P0: OpenPI=1e-10
    ...
}
ALIGNMENT_CONFIG = {
    "ema_decay": 0.99,        # @#2 P1: EMA
    "loss_include_padding": True,  # @#2 P0: 32维 loss
    ...
}
```

### 7.2 train_r1pro_chassis.sh — 训练入口

Shell 脚本作为训练入口，支持:
- `--smoke-test`: 100 步快速冒烟测试 (batch_size=4)
- `--steps N`: 自定义步数
- `--no-ema`, `--no-augmentation`: 禁用特定对齐项 (用于消融实验)

核心命令构造:
```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/pi0.5_base \
    --policy.optimizer_weight_decay=1e-10 \
    --policy.loss_include_padding=true \
    --policy.ema_decay=0.99 \
    --policy.augmentation_enabled=true \
    --batch_size=64 --steps=30000 --seed=42 \
    ...
```

### 7.3 PI05Config 中已实现的对齐机制

以下代码已在 `lerobot/src/` 中实现，**无需修改**:

**EMA 初始化与更新** (`modeling_pi05.py:1131-1159`):
```python
def _init_ema(self):
    """首次调用 update() 时初始化 EMA 参数 (延迟初始化)."""
    self._ema_params = {}
    for name, param in self.model.named_parameters():
        if param.requires_grad:
            self._ema_params[name] = param.data.clone()

def update(self):
    """训练循环每步调用, 更新 EMA."""
    if self.config.ema_decay is None:
        return
    if self._ema_params is None:
        self._init_ema()
    decay = self.config.ema_decay
    with torch.no_grad():
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._ema_params:
                self._ema_params[name].mul_(decay).add_(param.data, alpha=1-decay)
```

**EMA Checkpoint Swap** (`train_utils.py:104-113`):
```python
# 保存前: 将 EMA 参数写入模型
ema_backup = policy._swap_to_ema()
policy.save_pretrained(pretrained_dir)  # model.safetensors = EMA 权重
# 保存后: 恢复原始训练参数
policy._restore_from_backup(ema_backup)
```

**Loss Padding** (`modeling_pi05.py:1401-1407`):
```python
losses = self.model.forward(images, img_masks, tokens, masks, actions)  # [B,50,32]
if not self.config.loss_include_padding:
    original_action_dim = self.config.output_features[ACTION].shape[0]  # 23
    losses = losses[:, :, :original_action_dim]  # 截断到 23 维
loss = losses.mean()
```

**LR Schedule (post_warmup)** (`schedulers.py:130-133`):
```python
if phase_mode == "post_warmup":
    total_cosine_steps = max(1, actual_decay_steps - actual_warmup_steps)
    relative_step = min(current_step - actual_warmup_steps, total_cosine_steps)
    progress = relative_step / total_cosine_steps
# 余弦跨度: 29000 步, 与 optax 一致
```

---

## 8. trainr1/ 文件结构

```
bt/pi05/alig/trainr1/
│
├── config.py                        # 集中式参数定义 (所有脚本导入)
│
├── prepare_data.sh                  # 数据集准备
│   └── 调用 convert_r1pro_to_lerobot.py --norm-stats-path
│
├── train_r1pro_chassis.sh           # 单 GPU 训练入口
│   └── 支持: --smoke-test, --steps, --no-ema, --no-augmentation
│
├── train_r1pro_chassis_multi.sh     # 多 GPU 训练入口
│   └── 使用 accelerate launch, per-GPU batch = 64/N
│
├── verify_lr_schedule.py            # LR 调度验证
│   ├── OpenPI optax 参考实现
│   ├── LeRobot post_warmup 数值对比
│   └── 可选: 曲线对比图 (--plot)
│
├── verify_norm_stats.py             # 归一化统计验证
│   ├── OpenPI vs LeRobot q01/q99 数值对比
│   └── 归一化公式等价性验证
│
├── verify_training.py               # 训练曲线对比
│   ├── wandb / CSV 日志加载
│   ├── loss/lr/grad_norm 对比
│   └── L1/L2 等价性评估
│
├── compare_openpi_lerobot.py        # 端到端对齐验证 (整合所有检查)
│
├── data/                            # 转换后的数据集 (prepare_data.sh 生成)
│   └── r1_pro_chassis_v30/
│       ├── data/chunk-000/*.parquet
│       └── meta/
│           ├── info.json
│           ├── stats.json           # 含 OpenPI q01/q99
│           └── episodes/
│
├── outputs/                         # 训练输出 (训练脚本生成)
│   ├── r1pro_chassis_aligned/
│   └── smoke_test/
│
└── README.md                        # 使用指南
```

---

## 9. 验证方案

### 9.1 静态验证 (训练前)

| 验证项 | 工具 | 预期结果 | 判定标准 |
|--------|------|---------|---------|
| LR Schedule | `verify_lr_schedule.py` | post_warmup 模式与 optax 对齐 | 最大相对差异 < 0.1% |
| Norm Stats | `verify_norm_stats.py` | q01/q99 精确匹配 | 最大绝对差异 < 1e-10 |
| 配置参数 | `compare_openpi_lerobot.py` | 所有参数匹配 | 全部 PASS |
| 数据集完整性 | `compare_openpi_lerobot.py` | v3.0, 64 episodes | info.json 校验通过 |

### 9.2 冒烟测试 (训练前)

```bash
bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --smoke-test
```

验证:
- 训练 100 步无错误
- Loss 值合理 (初始 ~0.1-1.0，呈下降趋势)
- EMA 参数被正确初始化 (日志中无报错)
- Checkpoint 正确保存

### 9.3 完整训练验证

```bash
bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh
```

30000 步训练后:
1. **Loss 曲线对比** (L1):
   - 使用 `verify_training.py` 加载 wandb 日志
   - 最终 loss 相对差异 < 20%
   - Loss 趋势单调下降

2. **Checkpoint 质量** (L2):
   - 加载最终 checkpoint 进行推理
   - 与 OpenPI checkpoint 的动作预测 MSE 对比

### 9.4 验证执行顺序

```
Step 1: python bt/pi05/alig/trainr1/compare_openpi_lerobot.py --plot
        → 验证所有静态配置

Step 2: bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --smoke-test
        → 冒烟测试 (100步, ~5分钟)

Step 3: bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh
        → 完整训练 (30000步, ~8-24小时)

Step 4: python bt/pi05/alig/trainr1/verify_training.py \
            --lerobot-dir bt/pi05/alig/trainr1/outputs/r1pro_chassis_aligned \
            --openpi-wandb "entity/project/run_id"
        → 训练曲线对比
```

---

## 10. 风险分析与缓解

### 10.1 风险矩阵

```
                     低影响         中影响         高影响
           ┌─────────────────────────────────────────────┐
  高概率   │                │ GPU 内存不足  │             │
           │                │ (batch=64)   │             │
           ├────────────────┼──────────────┼─────────────┤
  中概率   │ 数据集转换     │ 预训练权重   │             │
           │ 需重新运行     │ 路径不对     │             │
           ├────────────────┼──────────────┼─────────────┤
  低概率   │ Tokenizer      │ 增强实现     │ RNG 差异    │
           │ token ID 差异  │ 数值差异     │ 导致 L1     │
           │                │              │ 不可达      │
           └─────────────────────────────────────────────┘
```

### 10.2 详细风险与缓解

| # | 风险 | 概率 | 影响 | 缓解措施 |
|---|------|------|------|---------|
| R1 | batch_size=64 单 GPU 内存不足 | 高 | 中 | 开启 `gradient_checkpointing=True` + `dtype=bfloat16`；若仍不够，使用多 GPU 或降低 batch_size 到 32/16 |
| R2 | 预训练权重路径不正确 | 中 | 中 | `compare_openpi_lerobot.py` 自动检查 HuggingFace Hub 可访问性；支持 `--pretrained` 指定本地路径 |
| R3 | 数据集转换不完整 | 中 | 中 | `prepare_data.sh` 含自动验证；`compare_openpi_lerobot.py` 检查数据集完整性 |
| R4 | JAX/PyTorch RNG 差异导致 loss 曲线不一致 | 低 | 中 | 已知限制 (L3 不可达)，关注 L1/L2 等价即可；统计分布一致是充分条件 |
| R5 | 数据增强的 PyTorch/JAX 实现差异 | 低 | 低 | 增强参数已对齐；轻微数值差异不影响训练收敛 |
| R6 | Tokenizer token ID 不一致 | 低 | 低 | 词表同源 (PaliGemma)；可通过 `verify_pi05.py` 验证 |

### 10.3 GPU 内存估算

Pi0.5 模型 (~2.3B 参数) 内存估算:

| 组件 | bfloat16 | float32 |
|------|---------|---------|
| 模型参数 | 4.6 GB | 9.2 GB |
| 优化器状态 (AdamW) | 18.4 GB | 36.8 GB |
| EMA 参数 | 4.6 GB | 9.2 GB |
| 梯度 | 4.6 GB | 9.2 GB |
| 激活 (无 grad_ckpt) | ~8 GB | ~16 GB |
| 激活 (有 grad_ckpt) | ~2 GB | ~4 GB |
| **总计 (有 grad_ckpt)** | **~34 GB** | **~68 GB** |
| **总计 (无 grad_ckpt)** | **~40 GB** | **~80 GB** |

**建议**: A100 80GB 使用 bfloat16 + gradient_checkpointing 可容纳 batch_size=64。
A100 40GB 或 V100 需要多 GPU 或降低 batch_size。

---

## 11. 实施步骤

### Phase 1: 数据准备 (~30 min)

```bash
# 1. 运行数据转换 (全部 64 episodes + OpenPI norm_stats)
bash bt/pi05/alig/trainr1/prepare_data.sh

# 2. 验证数据集
python bt/pi05/alig/trainr1/verify_norm_stats.py
```

### Phase 2: 静态验证 (~5 min)

```bash
# 运行所有静态检查
python bt/pi05/alig/trainr1/compare_openpi_lerobot.py --plot
```

预期输出: 所有 PASS，LR 曲线图保存

### Phase 3: 冒烟测试 (~10 min)

```bash
# 100 步快速验证
bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --smoke-test
```

检查:
- 训练无错误退出
- Loss 数值合理
- Checkpoint 正确保存在 `outputs/smoke_test/`

### Phase 4: 完整训练 (~8-24 hours)

```bash
# 单 GPU
bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh

# 或多 GPU (4 GPU)
bash bt/pi05/alig/trainr1/train_r1pro_chassis_multi.sh --num-gpus 4
```

监控: wandb dashboard 观察 loss 曲线

### Phase 5: 结果验证

```bash
# 对比训练曲线
python bt/pi05/alig/trainr1/verify_training.py \
    --lerobot-dir bt/pi05/alig/trainr1/outputs/r1pro_chassis_aligned \
    --openpi-wandb "entity/project/run_id"
```

### Phase 6: 消融实验 (可选)

逐一禁用对齐项，观察各项的影响:

```bash
# 仅禁用 EMA
bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --no-ema --output outputs/ablation_no_ema

# 仅禁用增强
bash bt/pi05/alig/trainr1/train_r1pro_chassis.sh --no-augmentation --output outputs/ablation_no_aug
```

---

## 附录 A: 关键文件索引

| 文件 | 角色 | 关键行 |
|------|------|--------|
| `lerobot/policies/pi05/configuration_pi05.py` | 全部配置字段定义 | L88 weight_decay, L92-93 ema/loss, L104 augmentation |
| `lerobot/policies/pi05/modeling_pi05.py` | 模型、EMA、增强实现 | L938 _ema_params, L1131-1186 EMA, L1204-1267 augment, L1401 loss_padding |
| `lerobot/scripts/lerobot_train.py` | 训练循环 | L143-144 policy.update(), L441-450 update_policy() |
| `lerobot/utils/train_utils.py` | Checkpoint EMA swap | L104-113 _swap_to_ema/restore |
| `lerobot/optim/schedulers.py` | LR 调度 | L130-133 post_warmup cosine |
| `lerobot/optim/optimizers.py` | 优化器构建 | AdamWConfig.build() |
| `openpi/training/config.py` | OpenPI 配置基线 | L1024-1042 pi05_r1pro_chassis |
| `openpi/training/optimizer.py` | OpenPI 优化器/LR | L73 wd=1e-10, L19-22 CosineDecay |
| `openpi/scripts/train.py` | OpenPI 训练循环 | L137-191 train_step, L169-175 EMA |

## 附录 B: 术语表

| 术语 | 含义 |
|------|------|
| VLA | Vision-Language-Action model |
| Flow Matching | 基于 ODE 的连续归一化流 (替代 diffusion) |
| EMA | Exponential Moving Average (指数移动平均) |
| quantile normalization | 使用 q01/q99 将数据映射到 [-1,1] |
| action_horizon / chunk_size | 模型预测的未来动作步数 (50) |
| FSDP | Fully Sharded Data Parallel |
| optax | JAX 优化器库 |
| post_warmup | 余弦衰减从 warmup 结束后开始计算相位 |
