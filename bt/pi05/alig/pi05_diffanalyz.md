# pi0.5 实现对比分析：OpenPI vs LeRobot

> **日期**: 2026-04-08
> **范围**: Physical Intelligence 官方 OpenPI 框架 vs HuggingFace LeRobot 框架中的 pi0.5 实现
> **分析方法**: 源代码逐文件对比、架构分析、训练/推理流程对比、生态系统集成分析、GitHub Issues/PRs 分析
> **代码库版本**: OpenPI (`/mnt/r/share/lkx/pi/openpi`), LeRobot (`/home/Luogang/SRC/Robot/lerobot`)

---

## 1. 执行摘要

Physical Intelligence 的 OpenPI 和 HuggingFace 的 LeRobot 都实现了 pi0.5 Vision-Language-Action (VLA) 模型。经过深入的代码级分析，**两者的核心算法完全一致**——相同的 flow matching loss、相同的 Beta 时间采样、相同的 ODE denoising、相同的 AdaRMSNorm 时间条件注入、相同的 discrete state 256-bin 量化。差异主要体现在**工程架构和生态系统集成**上：OpenPI 是以 JAX 为主的完整研究平台（含 PyTorch 辅助实现），面向研究复现和真实机器人部署；LeRobot 是纯 PyTorch 实现，深度集成 HuggingFace 生态（Hub、transformers、safetensors），面向社区使用和快速迭代。

### 总览对比表

| 维度 | OpenPI (Physical Intelligence) | LeRobot (HuggingFace) |
|------|-------------------------------|----------------------|
| **框架** | JAX/Flax (主) + PyTorch (辅) | PyTorch only |
| **pi0/pi0.5 代码共享** | 统一类，`pi05=True` 标志切换 | 独立 PI05Policy 类 |
| **transformers 集成** | 侵入式补丁（替换 site-packages 文件） | 干净扩展（pi_gemma.py 子类化） |
| **HuggingFace Hub** | 不支持（GCS + orbax） | 完整支持（from_pretrained, safetensors） |
| **RTC 支持** | 无 | 有（RTCProcessor） |
| **推理服务** | gRPC/WebSocket policy server | 直接 Python API 调用 |
| **机器人适配** | 内置多种适配器（ALOHA, DROID, LIBERO 等） | 通用接口，由数据集/环境配置驱动 |
| **License (代码)** | Apache 2.0 | Apache 2.0 |
| **License (权重)** | Gemma License (更严格) | Gemma License (更严格) |

### LIBERO Benchmark 性能对比

根据 LeRobot 官方文档，两者在 LIBERO 上的性能**非常接近**：

| 任务 | LeRobot | OpenPI |
|------|---------|--------|
| Libero Spatial | 97.0% | 98.8% |
| Libero Object | 99.0% | 98.2% |
| Libero Goal | 98.0% | 98.0% |
| Libero 10 | 96.0% | 92.4% |
| **平均** | **97.5%** | **96.85%** |

LeRobot 整体性能略优，尤其在多任务 Libero-10 上表现更好。

---

## 2. 背景

### 2.1 pi0.5 模型架构简介

pi0.5 是 Physical Intelligence 发布的第二代 VLA 模型（论文: [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)），在 pi0 基础上做了两项关键改进：

1. **Knowledge Insulation（知识隔离）**：将 robot state 从 suffix 中的连续向量改为 prefix 中的离散 token。state 被量化为 256 个 bin 后编码为文本，与 task description 一起作为 language token 输入 PaliGemma。这样 Action Expert 的梯度不会回传到 PaliGemma 的 vision-language 知识中，避免了 fine-tuning 过程中的 catastrophic forgetting。

2. **AdaRMSNorm 时间条件注入**：受 DiT (Scalable Diffusion Models with Transformers) 的 adaLN-Zero 启发，将 denoising timestep 通过 Adaptive RMS Normalization 注入到 Action Expert 的每一层，替代了 pi0 中的 concat+MLP 方式。这使得时间信息能更精细地调制每层的特征表示。

模型总体架构：
- **Vision Encoder**: SigLIP So400m/14（~400M 参数）
- **Language Model (Prefix Expert)**: PaliGemma / Gemma 2B（~2.5B 参数），标准 RMSNorm
- **Action Expert (Suffix Expert)**: Gemma 300M（~300M 参数），AdaRMSNorm
- **总参数量**: ~3.2B

### 2.2 开源版本与论文的差距

根据 GitHub Issues 分析，开源的 pi0.5 与论文描述的完整系统有**显著差距**：

| 论文描述 | 开源状态 | GitHub Issue |
|---------|---------|-------------|
| Cross-entropy + flow matching 组合 loss (Eq.1) | 仅实现 flow matching loss | [#816](https://github.com/Physical-Intelligence/openpi/issues/816) |
| High-level subtask language 输出 | 不支持 | [#813](https://github.com/Physical-Intelligence/openpi/issues/813) |
| VLM backbone 独立使用 | 不可用 | [#887](https://github.com/Physical-Intelligence/openpi/issues/887) |
| FAST autoregressive tokenizer head | pi0.5 不可用 | — |
| 两阶段训练（预训练+后训练） | 仅开源后训练阶段 | [#863](https://github.com/Physical-Intelligence/openpi/issues/863) |

实际训练流程是两阶段的：
1. **预训练阶段**：使用 FAST action tokens 的 autoregressive 训练
2. **后训练阶段**：flow matching 目标函数，产生连续 actions

开源版本仅包含后训练阶段的 flow matching 实现。

### 2.3 OpenPI 简介

Physical Intelligence 的官方开源实现。是一个完整的训练+推理平台，包含 JAX 和 PyTorch 两套模型实现、多种机器人的数据转换适配器、训练脚本、推理服务器等。主要面向研究者和机器人开发者进行结果复现和新机器人平台的 fine-tuning。

### 2.4 LeRobot 简介

HuggingFace 的开源机器人学习框架。pi0.5 作为其中的一个 policy 模块，深度集成 HuggingFace 生态。设计理念是让用户通过 `from_pretrained("lerobot/pi05_base")` 一行代码加载预训练模型，降低使用门槛。

---

## 3. 代码库总体对比

### 3.1 文件结构

| 功能 | OpenPI 文件 | 行数 | LeRobot 文件 | 行数 |
|------|-----------|------|-------------|------|
| **核心模型 (JAX)** | `models/pi0.py` | 279 | — | — |
| **核心模型 (PyTorch)** | `models_pytorch/pi0_pytorch.py` | 462 | `pi05/modeling_pi05.py` | 1294 |
| **Gemma 实现 (JAX)** | `models/gemma.py` | 459 | — | — |
| **Gemma 实现 (PyTorch)** | `models_pytorch/gemma_pytorch.py` | 281 | `pi_gemma.py` | 363 |
| **模型配置** | `models/pi0_config.py` | 117 | `pi05/configuration_pi05.py` | 169 |
| **训练配置** | `training/config.py` | 1083 | （集成在 configuration_pi05.py 中） | — |
| **Tokenizer** | `models/tokenizer.py` | 371 | （集成在 processor_pi05.py 中） | — |
| **数据处理** | `policies/policy.py` + `policy_config.py` | 229 | `pi05/processor_pi05.py` | 167 |
| **归一化** | `shared/normalize.py` | 146 | （使用框架通用 NormalizerProcessorStep） | — |
| **transformers 补丁** | `models_pytorch/transformers_replace/` | ~500+ | — | — |

### 3.2 代码组织差异

**OpenPI** 是一个 monorepo 式的完整平台：
- `models/` — JAX 模型实现（主要）
- `models_pytorch/` — PyTorch 模型实现（辅助）
- `policies/` — 多种机器人的 policy 适配器
- `training/` — 训练流水线、配置、checkpoint 管理
- `scripts/` — 训练、推理、数据处理脚本
- pi0 和 pi0.5 共享同一套代码，通过 `pi05=True` 标志切换行为

**LeRobot** 的 pi0.5 是一个自包含的 policy 模块：
- 仅 4 个 Python 文件（`__init__.py`, `configuration_pi05.py`, `modeling_pi05.py`, `processor_pi05.py`）
- 额外依赖 `pi_gemma.py`（共享于 pi0/pi0.5/pi0-FAST）
- PI05Policy 是独立类，不与 pi0 共享代码
- 遵循 HuggingFace `PreTrainedModel` 范式

### 3.3 依赖差异

| 依赖类别 | OpenPI | LeRobot |
|---------|--------|---------|
| **深度学习框架** | JAX, Flax (NNX), PyTorch | PyTorch |
| **Vision Encoder** | big_vision (JAX) / patched transformers (PyTorch) | transformers (标准) |
| **Language Model** | Flax Gemma (JAX) / patched transformers (PyTorch) | transformers (标准 + pi_gemma 扩展) |
| **Tokenizer** | SentencePiece (自定义封装) | HuggingFace tokenizers (AutoTokenizer) |
| **Checkpoint** | orbax (JAX) / safetensors (PyTorch) | safetensors |
| **分布式训练** | JAX sharding / PyTorch DDP | PyTorch (标准) |
| **监控** | Weights & Biases (wandb) | 框架通用 |

---

## 4. 模型架构深度对比

### 4.1 核心架构一致性

两个实现使用**完全相同的双专家 Transformer 架构**：

| 配置项 | PaliGemma (Prefix Expert) | Action Expert (Suffix Expert) |
|-------|--------------------------|------------------------------|
| **variant** | gemma_2b | gemma_300m |
| **width** | 2048 | 1024 |
| **depth** | 18 | 18 |
| **mlp_dim** | 16,384 | 4,096 |
| **num_heads** | 8 | 8 |
| **num_kv_heads** | 1 | 1 |
| **head_dim** | 256 | 256 |
| **AdaRMS** | False | True (pi0.5) / False (pi0) |

以上配置在两个实现中**完全一致**。

### 4.2 pi0/pi0.5 代码共享策略

**OpenPI**：统一实现，运行时切换
```python
# openpi/models/pi0_config.py
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    pi05: bool = False              # 切换 pi0 / pi0.5
    discrete_state_input: bool = None  # 自动跟随 pi05
    max_token_len: int = None         # 自动：pi0=48, pi0.5=200
```

```python
# openpi/models_pytorch/pi0_pytorch.py:103-109
if self.pi05:
    self.time_mlp_in = nn.Linear(...)    # pi0.5: AdaRMS 路径
    self.time_mlp_out = nn.Linear(...)
else:
    self.state_proj = nn.Linear(...)     # pi0: 连续 state 嵌入
    self.action_time_mlp_in = nn.Linear(...)  # pi0: concat+MLP
    self.action_time_mlp_out = nn.Linear(...)
```

**LeRobot**：独立实现
```python
# lerobot/policies/pi05/modeling_pi05.py:550
class PI05Pytorch(nn.Module):  # 仅 pi0.5，无 pi0 逻辑
    def __init__(self, config: PI05Config, ...):
        self.time_mlp_in = nn.Linear(...)   # 始终使用 AdaRMS
        self.time_mlp_out = nn.Linear(...)
        # 无 state_proj，无 action_time_mlp_*
```

**影响**：
- OpenPI 的统一方式便于维护和保证 pi0/pi0.5 的一致性，但增加了条件分支复杂度
- LeRobot 的独立方式代码更清晰，但 pi0 和 pi0.5 之间可能出现代码重复

### 4.3 Vision Encoder 实现差异

| 方面 | OpenPI (JAX) | OpenPI (PyTorch) | LeRobot |
|------|-------------|-----------------|---------|
| **实现来源** | big_vision SigLIP | patched transformers SigLIP | 标准 transformers PaliGemma vision tower |
| **安装方式** | JAX 原生 | 需手动 `cp` 到 site-packages | `pip install transformers` 即可 |
| **版本锁定** | 无（独立实现） | transformers==4.53.2 | 跟随 transformers 版本 |
| **图像格式** | channels-last `[B,H,W,C]` | channels-first `[B,C,H,W]` | channels-first `[B,C,H,W]` |
| **分辨率** | 224x224 | 224x224 | 224x224 |

**关键工程差异**：OpenPI 的 PyTorch 实现需要将修改过的 transformers 源文件复制到 `site-packages/transformers/` 目录：

```python
# openpi/models_pytorch/pi0_pytorch.py:118
msg = "transformers_replace is not installed correctly. Please install it with
       `uv pip install transformers==4.53.2` and
       `cp -r ./src/openpi/models_pytorch/transformers_replace/*
        .venv/lib/python3.11/site-packages/transformers/`."
```

LeRobot 则通过 `pi_gemma.py` 模块干净地扩展了 transformers，无需修改库文件：

```python
# lerobot/policies/pi_gemma.py
class PiGemmaForCausalLM(GemmaForCausalLM):
    """扩展标准 GemmaForCausalLM，添加 AdaRMS 支持"""
    ...
class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    """扩展 PaliGemma，使用 PiGemma 作为 language model"""
    ...
```

这是两者在工程实践上的**最显著差异**。OpenPI 的补丁方式脆弱、版本锁定严格；LeRobot 的扩展方式更易维护、版本兼容性更好。

### 4.4 AdaRMSNorm 时间条件注入

两者的 time MLP 处理**完全一致**：

```python
# OpenPI (pi0_pytorch.py:289-294) 和 LeRobot (modeling_pi05.py:705-709) 相同
def time_mlp_func(time_emb):
    x = self.time_mlp_in(time_emb)
    x = F.silu(x)                    # swish == silu，第一次激活
    x = self.time_mlp_out(x)
    return F.silu(x)                 # 第二次激活
```

OpenPI JAX 实现同样使用两次 swish（`pi0.py:164-167`），三个实现在此处**数值等价**。

AdaRMSNorm 的内部实现：
- 输入 `cond`（time embedding）通过一个 Dense 层生成 `3 * width` 维的 modulation 向量
- 拆分为 `scale, shift, gate` 三部分
- 归一化后的输入乘以 `(1 + scale)` 再加 `shift`
- `gate` 用于 gated residual connection: `output = x + y * gate`

### 4.5 Attention Masking

两者使用**相同的 attention masking 策略**：
- **Prefix tokens**（image + language）：双向注意力（bidirectional）
- **Suffix tokens**（action）：因果注意力（causal），不向 prefix 暴露
- Prefix tokens **不 attend** suffix tokens（隔离梯度）
- Suffix tokens **可 attend** prefix tokens（获取上下文）

实现方式：
```python
# 两者都使用 cumsum-based mask 构建
# OpenPI: make_attn_mask()  
# LeRobot: make_att_2d_masks()
att_masks = [1] + ([0] * (action_horizon - 1))  # 第一个 action token 打断因果链
attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # cumsum 构建因果矩阵
```

### 4.6 Sinusoidal Positional Encoding

两者使用**相同的正弦余弦位置编码**：
- `min_period=4e-3`, `max_period=4.0`
- 编码维度 = Action Expert width (1024)
- LeRobot 代码注释标注此函数为 "exact copy from openpi"

---

## 5. 训练流程对比

### 5.1 Loss 函数

两者使用**完全相同的 flow matching MSE loss**：

```python
# 两者共同的 loss 计算逻辑
time = Beta(1.5, 1.0) * 0.999 + 0.001   # 时间采样
noise = Normal(0, 1)                      # 噪声采样
x_t = time * noise + (1 - time) * actions  # 插值
u_t = noise - actions                      # 目标速度场
v_t = model(x_t, time)                     # 预测速度场
loss = MSE(v_t, u_t)                       # 均方误差
```

细微差异：
- OpenPI 返回 per-dimension loss 用于监控
- LeRobot 支持 `reduction` 参数（`"mean"` 或 `"none"`，后者用于 RA-BC 加权）

### 5.2 Optimizer 与 Scheduler

| 参数 | OpenPI (默认) | LeRobot (默认) |
|------|-------------|---------------|
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | 5e-5 (按具体配置) | 2.5e-5 |
| **Betas** | (0.9, 0.95) | (0.9, 0.95) |
| **Weight Decay** | 0.01 | 0.01 |
| **Gradient Clipping** | 1.0 | 1.0 |
| **Warmup Steps** | 10,000 (按具体配置) | 1,000 |
| **Decay Steps** | 1,000,000 (按具体配置) | 30,000 |
| **EMA Decay** | 0.999 | 不支持 |
| **Scheduler** | CosineDecaySchedule | CosineDecayWithWarmup |

### 5.3 Training Infrastructure

| 方面 | OpenPI | LeRobot |
|------|--------|---------|
| **JAX 训练** | `scripts/train.py`，FSDP sharding，TPU 支持 | 不支持 |
| **PyTorch 训练** | `scripts/train_pytorch.py`，DDP 分布式 | 标准 PyTorch 训练循环 |
| **Gradient Checkpointing** | 支持 | 支持 |
| **torch.compile** | 默认 `mode="max-autotune"` | 可选，通过 `compile_model` 配置 |
| **EMA** | 支持（decay=0.999） | 不支持 |
| **Mixed Precision** | bfloat16 | bfloat16 |

### 5.4 Fine-tuning 支持

**OpenPI**：
- LoRA 变体定义在 Gemma config 中（rank=16, alpha=16.0）
- `freeze_filter` 机制精细控制哪些参数可训练
- 丰富的命名配置（`pi05_aloha`, `pi05_droid`, `pi05_libero` 等）
- 通过 `CheckpointWeightLoader` 从 GCS 加载 base model

**LeRobot**：
- `freeze_vision_encoder`: 冻结视觉编码器
- `train_expert_only`: 仅训练 Action Expert
- PEFT 支持，提供 `_get_default_peft_targets()` 返回默认 LoRA 目标模块
- 通过 HuggingFace Hub `from_pretrained()` 加载 base model

### 5.5 命名训练配置

**OpenPI** 在 `training/config.py` 中维护了 ~20+ 个命名配置，覆盖多种机器人和任务：

| 配置名 | 机器人/任务 | action_horizon | batch_size |
|--------|-----------|---------------|------------|
| `pi05_base` | Base checkpoint | 50 | — |
| `pi05_aloha` | ALOHA | 50 | — |
| `pi05_libero` | LIBERO benchmark | 10 | 256 |
| `pi05_droid` | DROID | 15 | — |
| `pi05_full_droid_finetune` | DROID (全量) | 16 | 256 |
| `pi05_aloha_pen_uncap` | ALOHA 拧笔盖 | 50 | 64 |

**LeRobot** 使用通用的 `PI05Config` dataclass，用户自行设置参数。无预定义的命名配置。

---

## 6. 推理流程对比

### 6.1 ODE Denoising

两者使用**相同的 Euler 积分 ODE 求解器**：

```python
# 共同的 denoising 逻辑
x_t = noise                     # 从纯噪声开始 (t=1.0)
dt = -1.0 / num_steps           # 默认 num_steps=10
for step in range(num_steps):
    time = 1.0 - step / num_steps
    v_t = model.denoise_step(x_t, time)
    x_t = x_t + dt * v_t        # Euler 步进
# 最终 x_0 即为预测的 actions
```

### 6.2 KV Cache 优化

| 实现 | KV Cache |
|------|----------|
| **OpenPI JAX** | 有。prefix 只计算一次，后续 denoising 步骤复用 KV cache |
| **OpenPI PyTorch** | 无。每个 denoising 步骤重新计算完整 forward |
| **LeRobot** | 无。与 OpenPI PyTorch 相同，每步完整 forward |

这意味着 OpenPI 的 JAX 推理速度理论上**显著优于** PyTorch 实现。

### 6.3 Action Selection 机制

**OpenPI**：
- `sample_actions()` 返回原始 action tensor `(batch, action_horizon, action_dim)`
- 由 policy 层的 `infer()` 方法处理后续的 action 队列管理

**LeRobot**：
- `select_action()`: 内置 deque-based action queue，每次返回单个 action
- `predict_action_chunk()`: 返回完整 action chunk
- 自动管理 action queue 的填充和消耗

```python
# LeRobot action queue 机制
def select_action(self, batch):
    if len(self._action_queue) == 0:
        actions = self.predict_action_chunk(batch)
        self._action_queue.extend(actions)
    return self._action_queue.popleft()
```

### 6.4 推理服务架构

**OpenPI**：
- `scripts/serve_policy.py`: WebSocket/gRPC policy server
- 支持远程推理：机器人客户端通过网络调用 policy server
- 适合分离式部署（GPU 服务器 + 机器人控制器）

**LeRobot**：
- 直接 Python API 调用：`policy = PI05Policy.from_pretrained(...); action = policy.select_action(obs)`
- 无内置服务基础设施
- 适合嵌入式部署或自定义服务封装

---

## 7. 数据处理流程对比

### 7.1 State 离散化

两者使用**完全相同的 state 离散化方案**：

```python
# 共同逻辑：
# 1. State 归一化到 [-1, 1]
# 2. 量化为 256 个 bin
discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
# 3. 转换为字符串
state_str = " ".join(map(str, discretized))
# 4. 构建 prompt
prompt = f"Task: {task}, State: {state_str};\nAction: "
```

**OpenPI** 在 `PaligemmaTokenizer.tokenize()` 中实现（`tokenizer.py:22-48`）。
**LeRobot** 在 `Pi05PrepareStateTokenizerProcessorStep.__call__()` 中实现（`processor_pi05.py:57-85`）。

### 7.2 归一化策略

| 方面 | OpenPI | LeRobot |
|------|--------|---------|
| **pi0.5 默认** | Quantile normalization | Quantile normalization |
| **实现** | `NormStats` with `q01`, `q99` | `NormalizationMode.QUANTILES` |
| **公式** | `(x - q01) / (q99 - q01) * 2 - 1` | 相同 |
| **预计算** | `scripts/compute_norm_stats.py` | 集成在 `NormalizerProcessorStep` |
| **pi0 默认** | Z-score (`(x - mean) / std`) | Z-score |

### 7.3 图像预处理

两者使用**相同的 resize-with-pad 策略**：
- 保持宽高比，缩放到 224x224
- 不足部分 padding
- 归一化到 [-1, 1]（SigLIP 输入要求）
- LeRobot 代码注释标注 `resize_with_pad` 为 "exact copy" from OpenPI

### 7.4 Relative Actions（关键对齐问题）

**这是两者之间发现的最重要的实际差异之一。**

OpenPI 的 action 表示是**相对运动**（relative trajectory），而 LeRobot 最初使用**绝对位置**（absolute）。这导致了使用 OpenPI 预训练权重在 LeRobot 中 fine-tuning 时出现**分布不匹配**（distribution mismatch）。

LeRobot 在 [PR #2970](https://github.com/huggingface/lerobot/pull/2970)（2026年4月1日 merged，41条评论）中添加了 `RelativeActionsProcessorStep` 来修复此问题。

此外，[PR #2891](https://github.com/huggingface/lerobot/pull/2891) 发现了三个具体的对齐问题：

| 问题 | OpenPI 行为 | LeRobot (修复前) |
|------|-----------|-----------------|
| **Loss 截断** | 允许在 padded dims 上学习 | 截断到原始 action dims |
| **State padding** | 不在 tokenization 前 padding | 在 tokenization 前 padding |
| **Image padding 值** | 有效使用 0.0 | 使用 -1.0（导致归一化到 [-3,1]） |

这些差异曾导致 LeRobot v0.5.0 中精度从 ~98% 下降到 ~88%（[#3122](https://github.com/huggingface/lerobot/issues/3122)），后通过新 checkpoint 修复。

### 7.5 数据流水线架构

**OpenPI**：面向特定机器人的适配器模式
```
Raw obs → DataTransformFn (robot-specific) → PaligemmaTokenizer → Observation namedtuple → Model
```
每种机器人有独立的 policy adapter（如 `aloha_policy.py`, `droid_policy.py`），实现 camera 映射、action 维度适配等。

**LeRobot**：模块化 ProcessorStep 流水线
```
Raw obs → RenameObservations → AddBatchDim → Normalize → PrepareStateTokenizer → Tokenize → Device → Model
```
通用化设计，通过配置而非代码适配不同机器人。每个 step 可独立测试和复用。

---

## 8. 生态系统与集成对比

### 8.1 HuggingFace 生态集成

| 功能 | OpenPI | LeRobot |
|------|--------|---------|
| **from_pretrained()** | 不支持 | 支持 |
| **push_to_hub()** | 不支持 | 支持 |
| **safetensors** | PyTorch 版本支持 | 支持 |
| **HF Hub 托管** | 不使用（GCS） | `lerobot/pi05_base` 等 |
| **AutoConfig** | 不支持 | PI05Config 注册为 "pi05" |
| **PreTrainedModel 范式** | 不遵循 | 完全遵循 |

### 8.2 Transformers 库交互方式

这是两者**最重要的工程差异之一**。

**OpenPI 的侵入式补丁方式**：
```
openpi/models_pytorch/transformers_replace/
├── models/
│   ├── gemma/       # 修改后的 Gemma 实现（添加 adaRMS 支持）
│   ├── paligemma/   # 修改后的 PaliGemma 实现
│   └── siglip/      # 修改后的 SigLIP 实现 + 安装检查
```
- 需要 `cp -r` 将文件覆盖到 `site-packages/transformers/`
- 锁定 `transformers==4.53.2`
- 升级 transformers 可能导致不兼容
- 运行时通过 `check_whether_transformers_replace_is_installed_correctly()` 验证

**LeRobot 的干净扩展方式**：
```python
# pi_gemma.py: 通过继承扩展标准 transformers 类
class PiGemmaRMSNorm(nn.Module):
    """带 AdaRMS 条件的 RMSNorm"""
    def forward(self, x, adarms_cond=None): ...

class PiGemmaForCausalLM(GemmaForCausalLM):
    """扩展 Gemma，替换 RMSNorm 为 PiGemmaRMSNorm"""
    ...

class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    """使用 PiGemma 的 PaliGemma"""
    ...
```
- 无需修改 transformers 源码
- 版本兼容性更好
- 可独立升级 transformers

### 8.3 Checkpoint 格式与互操作性

**OpenPI → LeRobot 加载**：LeRobot 提供了 `_fix_pytorch_state_dict_keys()` 方法处理 key 映射：

```python
# LeRobot 的 key 修复逻辑 (modeling_pi05.py:1054-1114)
# 1. action_time_mlp_in.* → time_mlp_in.*  (pi0 → pi0.5 命名变更)
# 2. action_time_mlp_out.* → time_mlp_out.*
# 3. 跳过 state_proj.*  (pi0.5 无此层)
# 4. lm_head.weight → embed_tokens.weight  (权重共享处理)
```

**LeRobot → OpenPI 加载**：无直接支持，需手动进行反向 key 映射。

### 8.4 机器人平台支持

**OpenPI**：内置多种机器人适配器
| 机器人 | 适配文件 | action_dim |
|--------|---------|------------|
| ALOHA | `aloha_policy.py` | 14 |
| DROID | `droid_policy.py` | 32 |
| LIBERO | `libero_policy.py` | 7 |
| R1 Pro | `r1pro_chassis_policy.py` | 23 |

**LeRobot**：通用接口
- 通过 `max_action_dim`（默认 32）和 `max_state_dim`（默认 32）适配
- 机器人特定逻辑在数据集配置和环境层面处理
- 更灵活但需要用户自行配置

---

## 9. 各自独有特性

### 9.1 OpenPI 独有

| 特性 | 说明 |
|------|------|
| **JAX/TPU 支持** | 主要实现基于 JAX，原生支持 Google TPU 训练 |
| **FSDP Sharding** | 大模型分片训练支持 |
| **EMA (Exponential Moving Average)** | 训练时维护参数的指数移动平均，提升稳定性 |
| **命名训练配置** | ~20+ 预定义配置，一行命令启动训练 |
| **Policy Server** | gRPC/WebSocket 推理服务，支持远程部署 |
| **pi0-FAST 模型** | 支持 autoregressive action tokenization 的 pi0-FAST 变体 |
| **KV Cache (JAX)** | JAX 推理使用 KV cache 优化，仅计算一次 prefix |
| **机器人适配器** | 内置 ALOHA, DROID, LIBERO, R1Pro 等适配器 |
| **RLDS 数据格式** | 支持 TensorFlow RLDS 格式数据加载 |

### 9.2 LeRobot 独有

| 特性 | 说明 |
|------|------|
| **RTC (Real-Time Chunking)** | 通过 `RTCProcessor` 实现实时分块推理，降低延迟 |
| **HuggingFace Hub 集成** | `from_pretrained()`, `push_to_hub()`, 社区模型共享 |
| **干净的 transformers 扩展** | `pi_gemma.py` 子类化方式，无需补丁库文件 |
| **模块化 Processor 流水线** | `ProcessorStep` 链式设计，每个步骤可独立测试 |
| **Action Queue** | 内置 deque-based action 队列管理 |
| **Loss Reduction 模式** | 支持 `"mean"` 和 `"none"`（用于 RA-BC 加权） |
| **PEFT 默认目标** | `_get_default_peft_targets()` 提供默认 LoRA 目标模块 |
| **PreTrainedPolicy 范式** | 标准化的 policy 接口，与 HF 生态统一 |

---

## 10. 兼容性与互操作性

### 10.1 API 命名映射

| 概念 | OpenPI | LeRobot |
|------|--------|---------|
| action 预测步数 | `action_horizon` | `chunk_size` |
| action 维度 | `action_dim` | `max_action_dim` |
| token 最大长度 | `max_token_len` | `tokenizer_max_length` |
| 推理步数 | `num_steps` (sample_actions 参数) | `num_inference_steps` (config 属性) |
| pi0.5 标志 | `pi05=True` | 不需要（独立类） |
| 离散 state | `discrete_state_input=True` | 始终启用 |
| 时间采样 alpha | 硬编码 1.5 | `time_sampling_beta_alpha=1.5` |
| 时间采样 beta | 硬编码 1.0 | `time_sampling_beta_beta=1.0` |

### 10.2 Checkpoint 兼容性

| 方向 | 支持情况 | 说明 |
|------|---------|------|
| OpenPI JAX → LeRobot | 需转换 | 先 JAX→PyTorch 转换，再 key 映射 |
| OpenPI PyTorch → LeRobot | 支持 | `_fix_pytorch_state_dict_keys()` 自动处理 |
| LeRobot → OpenPI PyTorch | 需手动 | 反向 key 映射 |
| LeRobot → OpenPI JAX | 需转换 | PyTorch→JAX 转换 + key 映射 |

### 10.3 数值等价性

基于代码分析，在使用相同的权重和输入的情况下，两者的**模型前向传播应产生数值等价的输出**：
- Flow matching loss 公式完全一致
- Time MLP 激活函数完全一致（两次 silu/swish）
- Sinusoidal positional encoding 完全一致
- AdaRMSNorm 实现逻辑一致
- Attention masking 策略一致

PyTorch vs JAX 输出不一致问题已确认为**随机种子差异**所致，非权重转换错误（[OpenPI #810](https://github.com/Physical-Intelligence/openpi/issues/810)）。

可能的数值差异来源：
- 浮点精度差异（bfloat16 vs float32 的选择策略可能不同）
- PyTorch 与 JAX 底层算子的数值实现差异
- 随机数生成器差异（影响 noise 采样）

**重要注意**：即使模型前向传播数值等价，**数据预处理流水线的差异**（如 relative vs absolute actions、image padding 值、state padding 顺序）可能导致实际推理结果不同。使用 OpenPI 预训练权重在 LeRobot 中推理时，必须确保 processor pipeline 与 OpenPI 对齐（参见 7.4 节）。

---

## 11. 已知问题与局限性

### 11.1 论文与代码不一致（两者共同）

以下问题在 OpenPI GitHub Issues 中被提出，尚未得到官方回复：

1. **Tokenizer 不一致**：pi0.5 论文提及使用 FAST tokenizer，但代码实际使用 PaligemmaTokenizer（SentencePiece）。OpenPI 中 `pi05=True` 时仍走 PaligemmaTokenizer 路径。
   - 开源的 pi0.5 可能是论文描述的简化版本

2. **Loss 函数不一致**：论文 Equation 1 描述了 cross-entropy loss + flow matching loss 的组合，但代码中只实现了 flow matching MSE loss。
   - cross-entropy loss 可能仅用于论文中某些变体或实验

### 11.2 OpenPI 特有问题

| 问题 | 影响 | 严重度 | GitHub |
|------|------|--------|--------|
| transformers 补丁脆弱 | 升级 transformers 可能破坏 PyTorch 实现 | 高 | — |
| 版本锁定 transformers==4.53.2 | 无法使用新版 transformers 的功能和修复 | 中 | — |
| PyTorch 无 KV cache | PyTorch 推理性能低于 JAX | 中 | — |
| PyTorch 为辅助实现 | 可能滞后于 JAX 版本的功能更新 | 低 | — |
| **Fine-tuning 后真实部署失败** | 多个用户报告 loss 很低但实际抓取失败 | 高 | [#912](https://github.com/Physical-Intelligence/openpi/issues/912), [#906](https://github.com/Physical-Intelligence/openpi/issues/906), [#817](https://github.com/Physical-Intelligence/openpi/issues/817) |
| PyTorch 版 LIBERO 失败 | pi0.5 PyTorch 在 LIBERO 上运行失败 | 中 | [#878](https://github.com/Physical-Intelligence/openpi/issues/878) |

关于 PyTorch 与 JAX 输出不一致的问题已被确认为**随机种子差异**所致，非权重转换错误（[#810](https://github.com/Physical-Intelligence/openpi/issues/810)）。

### 11.3 LeRobot 特有问题

| 问题 | 影响 | 严重度 | GitHub |
|------|------|--------|--------|
| 无 JAX/TPU 支持 | 无法在 TPU 上训练 | 中 | — |
| 无 EMA | 训练稳定性可能不如 OpenPI | 中 | — |
| 无内置 policy server | 需自行实现推理服务 | 低 | — |
| 无 KV cache | 推理性能未优化 | 中 | — |
| **内存回归 (2-3x)** | PI0/PI05/PI0FAST 继承链构造模型 3 次丢弃 2 次 | 高 | [#3251](https://github.com/huggingface/lerobot/issues/3251), [PR #3254](https://github.com/huggingface/lerobot/pull/3254) |
| **transformers 版本敏感** | transformers 5.4.0 导致 LIBERO 0% 成功率 | 高 | [#3247](https://github.com/huggingface/lerobot/issues/3247) |
| **Checkpoint 版本不兼容** | v0.4.4 checkpoint 与 v0.5.0 不兼容 | 中 | [#3122](https://github.com/huggingface/lerobot/issues/3122) |
| **Relative actions 对齐** | 与 OpenPI 的 action 表示不一致（已修复） | 高（已修复） | [PR #2970](https://github.com/huggingface/lerobot/pull/2970) |

### 11.4 两者共同的实际部署挑战

值得注意的是，**两个实现都面临类似的真实世界部署困难**：fine-tuning 后 loss 很低（如 0.031），但实际机器人操作中出现"接近物体正确但抓取失败"的模式。这可能是 flow matching 方法本身在精细操控上的局限性，而非特定实现的问题。

---

## 12. 使用建议

### 12.1 选择决策矩阵

| 场景 | 推荐 | 原因 |
|------|------|------|
| **复现论文结果** | OpenPI | 官方实现，配置与论文一致 |
| **TPU/JAX 训练** | OpenPI | LeRobot 不支持 JAX |
| **已有 ALOHA/DROID 硬件** | OpenPI | 内置机器人适配器和 policy server |
| **HuggingFace 生态整合** | LeRobot | 原生 Hub 集成 |
| **快速实验 & 原型** | LeRobot | `from_pretrained()` 一行加载 |
| **社区模型共享** | LeRobot | 支持 push_to_hub() |
| **自定义机器人适配** | 视情况而定 | OpenPI 有模板；LeRobot 更灵活 |
| **需要 RTC** | LeRobot | OpenPI 不支持 |
| **PyTorch-only 环境** | LeRobot | 无需 JAX 依赖和 transformers 补丁 |
| **大规模分布式训练** | OpenPI | FSDP + EMA + 丰富的训练配置 |
| **长期维护性** | LeRobot | transformers 扩展方式更健壮 |

### 12.2 总结建议

- **研究者/复现**：选择 OpenPI。它是官方实现，有完整的训练配置和多机器人支持。
- **应用开发者**：选择 LeRobot。HF Hub 集成使模型分发和部署更便捷，代码更易维护。
- **新机器人适配**：两者都可，但 LeRobot 的通用接口和 OpenPI 的适配器模板各有优劣。如果你的机器人接近已有适配器（如 ALOHA），OpenPI 更省事；如果是全新平台，LeRobot 的通用化设计更灵活。

---

## 13. 结论

Physical Intelligence 的 OpenPI 和 HuggingFace 的 LeRobot 对 pi0.5 的实现在**核心模型算法层面完全一致**：相同的 flow matching loss、相同的 dual-expert Transformer 架构、相同的 AdaRMSNorm 时间条件注入、相同的 discrete state 256-bin 量化、相同的 ODE Euler denoising。LeRobot 在 LIBERO benchmark 上的性能（97.5%）与 OpenPI（96.85%）非常接近，甚至略优。

差异集中在**三个层面**：

**工程架构层面**：
1. OpenPI 是以 JAX 为主的**完整研究平台**，适合复现论文结果和大规模训练
2. LeRobot 是纯 PyTorch 的**生态系统组件**，适合快速实验和社区协作
3. 最显著的工程差异在于 transformers 库的交互方式：OpenPI 的侵入式补丁 vs LeRobot 的干净扩展

**数据处理层面**：
4. Relative vs absolute actions 表示差异是最关键的实际对齐问题（LeRobot 已修复）
5. Image padding 值、state padding 顺序、loss 截断维度等细节差异曾导致性能下降

**开源完整度层面**：
6. 两者都仅实现了论文描述系统的一个子集——flow matching post-training 阶段
7. 预训练阶段（FAST autoregressive）、组合 loss、HL subtask 输出等均未开源
8. 两者都面临 fine-tuning 后真实部署失败的共同挑战

两者互补而非竞争：OpenPI 定义了"正确的实现"，LeRobot 则让这个实现更容易被社区使用。选择时应根据具体场景（见第 12 节决策矩阵），而非简单地认为官方实现一定更好——LeRobot 在某些方面（如 transformers 维护性、RTC 支持、社区生态）已超越 OpenPI。

---

## 附录

### A. 文件对应关系表

| 功能 | OpenPI (JAX) | OpenPI (PyTorch) | LeRobot |
|------|-------------|-----------------|---------|
| 主模型 | `models/pi0.py` | `models_pytorch/pi0_pytorch.py` | `pi05/modeling_pi05.py` |
| Gemma/AdaRMS | `models/gemma.py` | `models_pytorch/gemma_pytorch.py` | `pi_gemma.py` |
| SigLIP | `models/siglip.py` | `transformers_replace/models/siglip/` | transformers 标准 |
| 配置 | `models/pi0_config.py` | 同 JAX | `pi05/configuration_pi05.py` |
| Tokenizer | `models/tokenizer.py` | 同 JAX | `pi05/processor_pi05.py` |
| Policy 接口 | `policies/policy.py` | 同 JAX | `pi05/modeling_pi05.py` (PI05Policy) |
| 训练配置 | `training/config.py` | 同 JAX | `pi05/configuration_pi05.py` |
| 归一化 | `shared/normalize.py` | 同 JAX | 框架通用 ProcessorStep |

### B. 关键代码对比片段

#### B.1 embed_suffix 中的 Time MLP（两者一致）

**OpenPI PyTorch** (`pi0_pytorch.py:289-298`):
```python
def time_mlp_func(time_emb):
    x = self.time_mlp_in(time_emb)
    x = F.silu(x)
    x = self.time_mlp_out(x)
    return F.silu(x)
time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
action_time_emb = action_emb
adarms_cond = time_emb
```

**LeRobot** (`modeling_pi05.py:705-713`):
```python
def time_mlp_func(time_emb):
    x = self.time_mlp_in(time_emb)
    x = F.silu(x)
    x = self.time_mlp_out(x)
    return F.silu(x)
time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
action_time_emb = action_emb
adarms_cond = time_emb
```

#### B.2 Flow Matching Loss（两者一致）

**OpenPI PyTorch** (`pi0_pytorch.py:317-350`):
```python
def forward(self, observation, actions, noise=None, time=None):
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    ...
    v_t = self.action_out_proj(suffix_out[:, -self.config.action_horizon:])
    return F.mse_loss(v_t, u_t, reduction="none").mean(dim=-1)
```

**LeRobot** (`modeling_pi05.py:730-780`):
```python
def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None):
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    ...
    v_t = self.action_out_proj(suffix_out[:, -self.config.chunk_size:])
    return F.mse_loss(v_t, u_t, reduction="none").mean(dim=-1)
```

### C. 参考资料

| 资料 | 链接 |
|------|------|
| pi0.5 论文 | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) |
| pi0 论文 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| OpenPI GitHub | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| LeRobot GitHub | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| LeRobot pi0.5 文档 | [huggingface.co/docs/lerobot/pi05](https://huggingface.co/docs/lerobot/pi05) |
| LeRobot pi0.5 Base 模型 | [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) |
| Knowledge Insulation | [pi.website/research/knowledge_insulation](https://www.pi.website/research/knowledge_insulation) |
| Flow Matching 论文 | Lipman et al., ICLR 2023 |
| DiT 论文 | Peebles & Xie, ICCV 2023 |

### D. 关键 GitHub Issues & PRs

**OpenPI**:
| Issue/PR | 标题 | 状态 |
|----------|------|------|
| [#634](https://github.com/Physical-Intelligence/openpi/pull/634) | Pi05 + PyTorch support | Merged |
| [#810](https://github.com/Physical-Intelligence/openpi/issues/810) | Pi05 output mismatch PyTorch vs JAX | Resolved (随机种子) |
| [#813](https://github.com/Physical-Intelligence/openpi/issues/813) | Can we get HL output? | Open |
| [#816](https://github.com/Physical-Intelligence/openpi/issues/816) | Tokenizer and Loss Function | Open |
| [#863](https://github.com/Physical-Intelligence/openpi/issues/863) | Two-stage training clarification | Open |
| [#887](https://github.com/Physical-Intelligence/openpi/issues/887) | Open-sourcing VLM backbone | Open |
| [#912](https://github.com/Physical-Intelligence/openpi/issues/912) | Robot reaches but fails to grasp | Open |

**LeRobot**:
| Issue/PR | 标题 | 状态 |
|----------|------|------|
| [PR #2727](https://github.com/huggingface/lerobot/pull/2727) | train_expert_only, freeze_vision_encoder | Merged |
| [PR #2891](https://github.com/huggingface/lerobot/pull/2891) | Align with OpenPI & image padding fix | Open |
| [PR #2970](https://github.com/huggingface/lerobot/pull/2970) | Relative action support | Merged (2026-04-01) |
| [#3122](https://github.com/huggingface/lerobot/issues/3122) | Accuracy drop in v0.5.0 | Fixed |
| [#3247](https://github.com/huggingface/lerobot/issues/3247) | 0% success with transformers 5.4.0 | Fixed |
| [#3251](https://github.com/huggingface/lerobot/issues/3251) | Memory regression 2-3x | Open |
| [#3226](https://github.com/huggingface/lerobot/issues/3226) | Parameter naming differences | Open |

---

## 附录 E. `action_sequence_keys` 配置项分析

### E.1 OpenPI 中的定义与作用

`action_sequence_keys` 是 OpenPI 的 `DataConfig` 中的一个配置项，定义在 `src/openpi/training/config.py:87-90`：

```python
# Names of keys that will be used by the data loader to generate the action sequence. The length of the
# sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
# LeRobot dataset is using different keys to represent the action.
action_sequence_keys: Sequence[str] = ("actions",)
```

它在 `data_loader.py:141-146` 中被使用，用于构造传给 LeRobot 数据集的 `delta_timestamps` 字典：

```python
dataset = lerobot_dataset.LeRobotDataset(
    data_config.repo_id,
    delta_timestamps={
        key: [t / dataset_meta.fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    },
)
```

#### 核心作用

1. **指定数据集中动作数据的字段名**：不同 LeRobot 数据集可能用不同的 key 名存储动作数据。`action_sequence_keys` 告诉数据加载器"去数据集的哪个字段里读动作序列"。

2. **构造时间窗口**：对于每个指定的 key，数据加载器以 `1/fps` 为步长，生成长度为 `action_horizon` 的相对时间戳列表 `[0, 1/fps, 2/fps, ..., (action_horizon-1)/fps]`，构成模型需要预测的 action chunk。

#### 各配置中的设定

| 配置 | action_sequence_keys | 原因 |
|------|---------------------|------|
| `DataConfig` 默认 | `("actions",)` | 通用默认（注意是复数） |
| `LeRobotAlohaDataConfig` | `("action",)` | ALOHA 数据集使用单数 key |
| R1 Pro 系列 | `("actions",)` | 转换脚本生成的 key 名为复数 |

### E.2 LeRobot 中的等价机制

LeRobot **没有** `action_sequence_keys` 这个配置项。它通过不同的机制实现了类似功能。

#### LeRobot 的实现路径

1. `PI05Config` 定义了 `action_delta_indices` 属性（`configuration_pi05.py:164-165`）：
   ```python
   @property
   def action_delta_indices(self) -> list:
       return list(range(self.chunk_size))
   ```

2. `resolve_delta_timestamps()` 函数（`datasets/factory.py:38-68`）遍历数据集的 `features`，对匹配 `ACTION` 常量的 key 生成时间戳：
   ```python
   for key in ds_meta.features:
       if key == ACTION and cfg.action_delta_indices is not None:
           delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
   ```

3. 其中 `ACTION` 是一个**硬编码常量**（`utils/constants.py:33`）：
   ```python
   ACTION = "action"
   ```

#### 关键差异

| 方面 | OpenPI | LeRobot |
|------|--------|---------|
| **action key 名** | 可配置，通过 `action_sequence_keys` | 硬编码为 `"action"` |
| **默认值** | `("actions",)` — 复数 | `"action"` — 单数 |
| **多 key 支持** | 支持（`Sequence[str]`，可指定多个 key） | 不支持（只认一个固定 key） |
| **时间窗口长度来源** | `action_horizon`（模型配置） | `chunk_size`（PI05Config） |
| **时间戳计算** | `[t / fps for t in range(action_horizon)]` | `[i / fps for i in action_delta_indices]` |
| **时间戳计算结果** | 等价 | 等价（`action_delta_indices = list(range(chunk_size))`） |

### E.3 实际影响分析

#### 为什么 OpenPI 默认是 `"actions"`（复数）而 LeRobot 是 `"action"`（单数）

这是两个生态系统的命名约定差异：

- **LeRobot 数据集规范**：统一使用 `"action"` 作为动作数据的 key 名（定义在 `utils/constants.py`），这是 LeRobot 框架层面的硬性约定
- **OpenPI 默认值**：使用 `"actions"`（复数），但通过 `action_sequence_keys` 允许适配不同数据集

因此，当 OpenPI 加载标准 LeRobot 格式的数据集时，需要将 `action_sequence_keys` 设为 `("action",)` 才能匹配。OpenPI 的 ALOHA 配置就是这么做的。而 R1 Pro 等自定义数据集可能在转换时就使用了 `"actions"` 作为 key 名，所以保持默认值即可。

#### 对 fine-tuning 的影响

1. **使用 LeRobot 框架 fine-tuning**：无需关心此配置项。LeRobot 的 `resolve_delta_timestamps()` 自动使用 `"action"` key，只要数据集遵循 LeRobot 规范即可。

2. **使用 OpenPI 框架 fine-tuning**：需要确保 `action_sequence_keys` 与你的数据集中实际的 key 名一致。如果数据集是标准 LeRobot 格式（`"action"`），应设为 `("action",)`。

3. **自定义数据集**：如果你的数据集使用了非标准 key 名，OpenPI 可以通过修改 `action_sequence_keys` 适配；LeRobot 则需要在数据转换阶段就将 key 名统一为 `"action"`。

#### 灵活性差异总结

OpenPI 的 `action_sequence_keys` 提供了一层**数据集 key 名到模型输入的可配置桥梁**，使得同一套训练代码可以适配不同命名约定的数据集。LeRobot 则通过**框架层面的命名约定**（硬编码 `ACTION = "action"`）消除了这种灵活性的需求——所有 LeRobot 生态内的数据集都必须遵循统一的 key 命名规范，因此不需要额外的映射配置。

两种设计各有取舍：
- OpenPI 更灵活，适合需要对接多种数据源的场景
- LeRobot 更简洁，依赖约定优于配置（convention over configuration），减少了用户出错的可能

---

## 附录 F. LeRobot 独有配置项与 OpenPI 对应关系

### F.1 概述

附录 E 分析了一个 OpenPI 独有的配置项（`action_sequence_keys`）。本附录反向分析：LeRobot 的 `PI05Config`（`src/lerobot/policies/pi05/configuration_pi05.py`）中有 9 类配置项在 OpenPI 的 `Pi0Config` / `DataConfig` / `TrainConfig` 中没有直接对应，或以截然不同的机制实现。

差异程度分为三级：
- **无对应**：OpenPI 中完全不存在该概念
- **不同机制**：两者解决同一问题，但实现方式和配置入口不同
- **结构差异**：语义相同，仅配置的组织方式不同（如 flat 字段 vs 独立 dataclass）

### F.2 总览对比表

| 配置项 | LeRobot 类型/默认值 | OpenPI 对应 | 差异程度 |
|--------|---------------------|------------|---------|
| `n_obs_steps` | `int = 1` | 无 | 无对应 |
| `n_action_steps` | `int = 50` | `ActionChunkBroker.action_horizon`（客户端侧） | 不同机制 |
| `rtc_config` | `RTCConfig \| None = None` | 无 | 无对应 |
| `empty_cameras` | `int = 0` | `image_masks` dict（per-adapter） | 不同机制 |
| `normalization_mapping` | `dict`（per-feature-type） | `use_quantile_norm: bool`（DataConfig） | 不同机制 |
| `freeze_vision_encoder` | `bool = False` | LoRA variant + `freeze_filter` | 不同机制 |
| `train_expert_only` | `bool = False` | 无直接对应 | 不同机制 |
| `optimizer_*` | 5 个 flat 字段 | `AdamW` dataclass（optimizer.py） | 结构差异 |
| `scheduler_*` | 3 个 flat 字段 | `CosineDecaySchedule` dataclass（optimizer.py） | 结构差异 |

### F.3 观测与动作步数控制

#### F.3.1 `n_obs_steps`

**LeRobot 定义**：`configuration_pi05.py:36`，`int = 1`，继承自基类 `PreTrainedConfig`。

**使用情况**：在 PI05 的模型代码（`modeling_pi05.py`）中**未被实际使用**。PI05 始终只处理当前时刻的单步观测，不支持观测历史堆叠（observation history stacking）。该字段仅因基类接口要求而存在。

**OpenPI 对应**：无。OpenPI 的 pi0.5 同样只使用单步观测，这是隐式行为而非显式配置。模型的 `Observation` 结构（`models/model.py:88-96`）只包含当前时刻的 images、state 和 tokenized_prompt，没有时间维度。

**实际影响**：无。两者行为一致，该字段在当前 pi0.5 实现中不产生任何效果。

#### F.3.2 `n_action_steps`

**LeRobot 定义**：`configuration_pi05.py:38`，`int = 50`。在 `__post_init__` 中验证 `n_action_steps <= chunk_size`。

**使用位置**：
- `modeling_pi05.py:1121-1123`：在 `reset()` 中设置 action deque 的最大长度（`maxlen=n_action_steps`）
- `modeling_pi05.py:1224-1225`：在 `select_action()` 中截取预测的 action chunk 的前 `n_action_steps` 步放入队列

**工作原理**：模型一次预测 `chunk_size` 步动作（action chunk），但只将前 `n_action_steps` 步放入执行队列。每次调用 `select_action()` 时从队列中弹出一个动作执行；队列耗尽后才重新调用模型预测。这意味着 `n_action_steps` 控制了"多久重新查询一次模型"。

```python
# LeRobot: modeling_pi05.py (简化)
def select_action(self, batch):
    if len(self._action_queue) == 0:
        actions = self.predict_action_chunk(batch)          # 预测 chunk_size 步
        self._action_queue.extend(actions[:n_action_steps]) # 只取前 n_action_steps 步
    return self._action_queue.popleft()                     # 每次执行一步
```

**OpenPI 对应**：OpenPI 的模型层（`policy.py:infer()`）直接返回完整的 action chunk（`action_horizon` 步），不内置队列管理。但客户端侧提供了 `ActionChunkBroker`（`packages/openpi-client/src/openpi_client/action_chunk_broker.py`）实现类似功能：

```python
# OpenPI: ActionChunkBroker (简化)
class ActionChunkBroker:
    def __init__(self, policy, action_horizon):  # action_horizon ≈ n_action_steps
        self._action_horizon = action_horizon
        self._cur_step = 0

    def infer(self, obs):
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)  # 查询模型
            self._cur_step = 0
        results = self._last_results[self._cur_step]       # 取当前步
        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None                       # 耗尽后重新查询
        return results
```

**关键差异**：
- **集成位置**：LeRobot 将 action queue 集成在模型类内部；OpenPI 将其分离到客户端 wrapper
- **配置方式**：LeRobot 在 PI05Config 中声明；OpenPI 在 ActionChunkBroker 构造函数中传入
- **默认行为**：LeRobot 默认 `n_action_steps=50`（等于 `chunk_size`，即用完整个 chunk）；OpenPI 的 ActionChunkBroker 是可选组件，不使用时直接返回完整 chunk

### F.4 实时分块推理

#### F.4.1 `rtc_config`

**LeRobot 定义**：`configuration_pi05.py:54`，`RTCConfig | None = None`。

**使用位置**：
- `modeling_pi05.py:609`：`_rtc_enabled()` 方法检查是否启用
- `modeling_pi05.py:1132-1133`：若配置非 None，初始化 `RTCProcessor`
- `modeling_pi05.py:842-854`：推理时在 denoising 步骤中可选地应用 Real-Time Chunking

**工作原理**：Real-Time Chunking (RTC) 是一种流式推理模式。启用后，模型不等待完整 action chunk 预测完毕，而是在 denoising 过程中增量地生成动作，降低首次动作输出的延迟。`RTCConfig`（定义在 `policies/rtc/configuration_rtc.py`）配置 RTC 的具体行为参数。

**OpenPI 对应**：**完全不存在**。OpenPI 不支持 RTC。模型始终执行完整的 ODE denoising（默认 10 步），生成完整 action chunk 后一次性返回。

**实际影响**：RTC 是 LeRobot 相对于 OpenPI 的一个功能性超集。如果应用场景对推理延迟敏感（如高频控制），LeRobot 可通过 RTC 获得优势；如果不需要流式推理，保持 `rtc_config=None` 即可，两者行为一致。

### F.5 图像输入处理

#### F.5.1 `empty_cameras`

**LeRobot 定义**：`configuration_pi05.py:62`，`int = 0`。

**使用位置**：
- `configuration_pi05.py:120-126`：在 `validate_features()` 中，为每个空相机添加一个 `obs_images.empty_camera_{i}` feature（shape `(3, 224, 224)`）
- `modeling_pi05.py:1200-1204`：推理时对缺失的图像 key 生成填充图像（像素值 -1）和零 mask

**工作原理**：pi0.5 预训练时使用了多个相机视角。如果你的机器人只有 1 个相机但模型期望 3 个，可以设 `empty_cameras=2` 来补齐。空相机会被填充为全黑图像并标记为无效（mask=0），模型在 attention 中忽略它们。

**OpenPI 对应**：OpenPI 使用 **per-adapter 的 `image_masks` 机制**。每个机器人 policy adapter（如 `aloha_policy.py:52-70`）内部硬编码了相机列表和缺失处理逻辑：

```python
# OpenPI: aloha_policy.py (简化)
images = {
    "base_0_rgb": base_image,
    "left_wrist_0_rgb": in_images.get("cam_left_wrist") or np.zeros_like(base_image),
    "right_wrist_0_rgb": in_images.get("cam_right_wrist") or np.zeros_like(base_image),
}
image_masks = {
    "base_0_rgb": np.True_,
    "left_wrist_0_rgb": np.True_ if "cam_left_wrist" in in_images else np.False_,
    "right_wrist_0_rgb": np.True_ if "cam_right_wrist" in in_images else np.False_,
}
```

**关键差异**：
- **LeRobot**：通过一个整数计数统一添加空相机，与具体机器人解耦
- **OpenPI**：在每个机器人 adapter 中逐个处理缺失相机，与具体机器人耦合
- LeRobot 更通用但缺乏对具体相机名称的语义控制；OpenPI 更精确但需要为每种机器人写适配代码

### F.6 归一化策略

#### F.6.1 `normalization_mapping`

**LeRobot 定义**：`configuration_pi05.py:66-72`，`dict[str, NormalizationMode]`，默认值：

```python
{
    "VISUAL": NormalizationMode.IDENTITY,    # 图像不做归一化
    "STATE": NormalizationMode.QUANTILES,    # state 用分位数归一化
    "ACTION": NormalizationMode.QUANTILES,   # action 用分位数归一化
}
```

**使用位置**：
- `processor_pi05.py:136`：传给 `NormalizerProcessorStep`（预处理）
- `processor_pi05.py:151`：传给 `UnnormalizerProcessorStep`（后处理）

**工作原理**：允许为不同类型的特征（VISUAL / STATE / ACTION）分别指定归一化模式。可选模式包括 `IDENTITY`（不归一化）、`QUANTILES`（分位数映射到 [-1, 1]）、`MEAN_STD`（z-score）等。

**OpenPI 对应**：OpenPI 使用一个**单一布尔开关** `use_quantile_norm: bool = False`（`training/config.py:85`）：

- `False`（pi0 默认）：所有特征使用 z-score 归一化
- `True`（pi0.5 默认）：所有特征使用 quantile 归一化

归一化在 `data_loader.py:188` 中统一应用：
```python
_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
```

**关键差异**：
- **粒度**：LeRobot 支持 per-feature-type 配置；OpenPI 只有全局开关
- **图像归一化**：LeRobot 显式将 VISUAL 设为 IDENTITY；OpenPI 的图像归一化在其他地方处理（SigLIP 预处理），不经过此开关
- **实际效果**：对于 pi0.5，两者的 STATE 和 ACTION 归一化行为一致（都用 quantile），差异主要在配置粒度

### F.7 冻结与微调控制

#### F.7.1 `freeze_vision_encoder`

**LeRobot 定义**：`configuration_pi05.py:81`，`bool = False`。

**使用位置**：
- `modeling_pi05.py:421-424`：在 `_set_requires_grad()` 中，若为 True，对 vision tower 的所有参数调用 `requires_grad_(False)` 并设为 eval 模式
- `modeling_pi05.py:432-433`：在 `train()` 方法中，即使进入训练模式，仍强制 vision tower 保持 eval

**工作原理**：一个布尔开关，冻结 SigLIP vision encoder 的所有参数。训练时只更新 language model 和 action expert 的参数。适用于下游任务视觉特征已足够好、不需要微调视觉编码器的场景。

**OpenPI 对应**：OpenPI 没有直接的 `freeze_vision_encoder` 开关。它通过 **LoRA variant + freeze_filter 机制**实现更灵活的参数冻结：

1. 在 `Pi0Config`（`pi0_config.py:21`）中设置 `paligemma_variant="gemma_2b_lora"`
2. 调用 `Pi0Config.get_freeze_filter()`（`pi0_config.py:88-117`）获取 NNX filter
3. 将 filter 传给 `TrainConfig.freeze_filter`（`config.py:495`）

```python
# OpenPI: pi0_config.py (简化)
def get_freeze_filter(self):
    filters = []
    if "lora" in self.paligemma_variant:
        filters.append(nnx_utils.PathRegex(".*llm.*"))       # 冻结 LLM 主干
    if has_lora:
        filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))  # 但保留 LoRA 可训练
    return nnx.All(*filters)
```

**关键差异**：
- **LeRobot**：简单的布尔开关，粒度为"vision encoder 全冻结或全不冻结"
- **OpenPI**：通过正则表达式 filter 精细控制任意参数子集的冻结，支持 LoRA 等混合策略
- OpenPI 的机制更强大但配置更复杂；LeRobot 更简单直观

#### F.7.2 `train_expert_only`

**LeRobot 定义**：`configuration_pi05.py:82`，`bool = False`。

**使用位置**：
- `modeling_pi05.py:425-428`：在 `_set_requires_grad()` 中，若为 True，对整个 PaliGemma VLM（vision + language model）的所有参数调用 `requires_grad_(False)` 并设为 eval 模式
- `modeling_pi05.py:434-435`：在 `train()` 方法中，强制 PaliGemma 保持 eval

**工作原理**：比 `freeze_vision_encoder` 更激进的冻结策略——冻结整个 PaliGemma（SigLIP + Gemma 2B），只训练 Action Expert（Gemma 300M）及投影层。适用于参数高效微调场景。

**OpenPI 对应**：OpenPI **没有直接对应的单一开关**。最接近的方式是组合使用 LoRA：
- `paligemma_variant="gemma_2b_lora"` + `action_expert_variant="gemma_300m"`
- 配合 `get_freeze_filter()` 冻结 PaliGemma 的非 LoRA 参数

但这并非"完全冻结 VLM + 完全训练 expert"的精确等价——LoRA 模式下 PaliGemma 仍有少量可训练参数（LoRA 适配器）。要完全冻结 VLM，OpenPI 需要手动构造更精确的 `freeze_filter`。

**关键差异**：
- **LeRobot**：一个布尔开关实现"只训练 expert"
- **OpenPI**：需要组合多个配置项，且默认通过 LoRA 方案（VLM 仍有少量可训练参数）
- LeRobot 的方式更适合快速实验；OpenPI 的方式更灵活但上手成本更高

### F.8 优化器与调度器

#### F.8.1 `optimizer_*` 系列字段

**LeRobot 定义**（`configuration_pi05.py:84-89`）：

| 字段 | 类型 | 默认值 |
|------|------|--------|
| `optimizer_lr` | `float` | `2.5e-5` |
| `optimizer_betas` | `tuple[float, float]` | `(0.9, 0.95)` |
| `optimizer_eps` | `float` | `1e-8` |
| `optimizer_weight_decay` | `float` | `0.01` |
| `optimizer_grad_clip_norm` | `float` | `1.0` |

通过 `get_optimizer_preset()`（`:142-149`）收集为 `AdamWConfig` 对象。

**OpenPI 对应**：独立的 `AdamW` dataclass（`training/optimizer.py:66-85`）：

| 字段 | OpenPI 名 | 默认值 |
|------|----------|--------|
| lr | （在 scheduler 上） | `2.5e-5`（peak_lr） |
| betas | `b1`, `b2` | `0.9`, `0.95` |
| eps | `eps` | `1e-8` |
| weight_decay | `weight_decay` | **`1e-10`** |
| grad_clip | `clip_gradient_norm` | `1.0` |

**关键差异**：

1. **`weight_decay` 值差异显著**：LeRobot 默认 `0.01`（标准 L2 正则化），OpenPI 默认 `1e-10`（实质上为零）。OpenPI 代码注释（`optimizer.py:72-73`）解释："Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value." 这意味着 OpenPI 的 pi0.5 训练**几乎不使用 weight decay**。

2. **learning rate 归属不同**：LeRobot 将 lr 放在 optimizer 字段上；OpenPI 将 `peak_lr` 放在 scheduler（`CosineDecaySchedule`）上，optimizer 不持有 lr。

3. **组织方式**：LeRobot 是 PI05Config 上的 flat 字段；OpenPI 是独立的 `AdamW` dataclass，通过 `TrainConfig.optimizer` 字段引用。

#### F.8.2 `scheduler_*` 系列字段

**LeRobot 定义**（`configuration_pi05.py:94-96`）：

| 字段 | 类型 | 默认值 |
|------|------|--------|
| `scheduler_warmup_steps` | `int` | `1_000` |
| `scheduler_decay_steps` | `int` | `30_000` |
| `scheduler_decay_lr` | `float` | `2.5e-6` |

通过 `get_scheduler_preset()`（`:151-157`）收集为 `CosineDecayWithWarmupSchedulerConfig` 对象。

**OpenPI 对应**：独立的 `CosineDecaySchedule` dataclass（`training/optimizer.py:16-31`）：

| 字段 | OpenPI 名 | 默认值 |
|------|----------|--------|
| warmup_steps | `warmup_steps` | `1_000` |
| peak_lr | `peak_lr` | `2.5e-5` |
| decay_steps | `decay_steps` | `30_000` |
| decay_lr | `decay_lr` | `2.5e-6` |

**差异分析**：

- **默认值完全一致**：warmup_steps=1000, peak_lr=2.5e-5, decay_steps=30000, decay_lr=2.5e-6
- **底层实现等价**：LeRobot 使用 PyTorch 自定义 scheduler；OpenPI 使用 `optax.warmup_cosine_decay_schedule`。两者都实现"线性 warmup → cosine decay"的学习率曲线
- **组织方式**：与 optimizer 一样，LeRobot 用 flat 字段，OpenPI 用独立 dataclass

注意：OpenPI 的具体训练配置（如 `pi05_aloha`）通常会覆盖这些默认值。例如 LoRA fine-tuning 配置中常见 `warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000`。LeRobot 代码注释提到 scheduler 会在 `total_steps < scheduler_decay_steps` 时自动缩放。

### F.9 设计哲学总结

这 9 类配置项的差异揭示了两个框架在配置设计上的根本分歧：

**LeRobot：All-in-One Config**

LeRobot 将几乎所有训练相关配置（模型结构、优化器、调度器、冻结策略、推理行为）打包在单一的 `PI05Config` dataclass 上。优势是：
- 一个 `config.json` 完整描述整个实验
- 与 HuggingFace Hub 的 `from_pretrained()` / `push_to_hub()` 无缝集成
- 用户只需理解一个配置类

代价是配置类职责过多（170 行），且无法灵活组合不同的 optimizer/scheduler/freeze 策略。

**OpenPI：Separation of Concerns**

OpenPI 将配置分散到多个独立 dataclass 中：
- `Pi0Config`：模型结构（`pi0_config.py`）
- `DataConfig` / `DataConfigFactory`：数据加载与预处理（`config.py`）
- `AdamW` / `CosineDecaySchedule`：优化器与调度器（`optimizer.py`）
- `TrainConfig`：训练流程编排，组合上述所有组件（`config.py`）

优势是各组件可独立替换（如换 SGD、换 RsqrtDecaySchedule），代价是配置分散在多个文件中，上手门槛更高。

**对跨框架对齐的影响**：

由于配置组织方式不同，将 OpenPI 的训练设置迁移到 LeRobot 时需要注意：
1. lr 在 OpenPI 属于 scheduler，在 LeRobot 属于 optimizer——找准来源
2. weight_decay 默认值差异显著（0.01 vs 1e-10）——需明确选择
3. OpenPI 的 freeze_filter 灵活性远超 LeRobot 的布尔开关——复杂冻结策略可能无法直接迁移
4. OpenPI 的具体 recipe（如 `pi05_libero`）会大幅覆盖默认值——对齐时应以目标 recipe 为准，而非默认值
