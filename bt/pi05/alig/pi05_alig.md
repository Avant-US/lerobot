# LeRobot pi0.5 与 OpenPI pi05_r1pro_chassis 对齐方案

> **日期**: 2026-04-08
> **目标**: 使 LeRobot 的 pi0.5 fine-tuning 在 R1 Pro chassis 数据上产生与 OpenPI `pi05_r1pro_chassis` 等价的训练结果
> **前置分析**: 基于 `pi05_diffanalyz.md` 的完整代码级 diff 分析
> **代码库**: OpenPI (`/mnt/r/share/lkx/pi/openpi`), LeRobot (`/home/Luogang/SRC/Robot/lerobot`)

---

## 0. 对齐目标与原则

**最终目标**：在 LeRobot 框架下，使用与 OpenPI `pi05_r1pro_chassis` 完全一致的超参数、数据处理流水线和训练策略，对 `lerobot/pi05_base` 预训练权重进行 fine-tuning，使训练过程（loss 曲线、梯度行为）和最终模型（推理动作输出）与 OpenPI PyTorch 训练等价。

**对齐原则**：
- 以 OpenPI 的 `pi05_r1pro_chassis` TrainConfig 为 ground truth
- 逐层对齐：配置 → 数据流水线 → 模型前向 → Loss → 优化器 → 推理
- 每层对齐后独立验证，再进入下一层

---

## 1. OpenPI pi05_r1pro_chassis 完整配置基线

**来源**: `/mnt/r/share/lkx/pi/openpi/src/openpi/training/config.py:1024-1042`

```python
TrainConfig(
    name="pi05_r1pro_chassis",
    model=Pi0Config(pi05=True),          # action_dim=32, action_horizon=50, max_token_len=200
    data=SimpleDataConfig(
        repo_id="r1_pro_data_convert_chassis",
        data_transforms=R1ProChassisInputs + R1ProChassisOutputs,
        model_transforms=ModelTransformFactory(
            default_prompt="Open the door with a downward-press handle, go through it, and enter the room."
        ),
        base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("actions",)),
    ),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    batch_size=64,
    # 以下均取默认值：
    # lr_schedule=CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-5, decay_steps=30000, decay_lr=2.5e-6)
    # optimizer=AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0)
    # ema_decay=0.99
    # freeze_filter=nnx.Nothing()  (全量 fine-tuning，无冻结)
    # seed=42
    # pytorch_training_precision="bfloat16"
)
```

### R1 Pro Chassis 动作空间 (23-dim)

**来源**: `/mnt/r/share/lkx/pi/openpi/src/openpi/policies/r1pro_chassis_policy.py`

| 索引 | 维度 | 含义 |
|------|------|------|
| [0:7] | 7 | left_arm (关节角度) |
| [7:14] | 7 | right_arm (关节角度) |
| [14] | 1 | left_gripper |
| [15] | 1 | right_gripper |
| [16:20] | 4 | torso |
| [20:23] | 3 | chassis_velocities (x, y, rotation) |

模型内部 padding 到 32 维（末尾补 9 个零）。

---

## 2. 差异清单与修复方案

### 2.1 P0 级别差异（致命 — 必须修复）

---

#### 2.1.1 Weight Decay: 1e-10 vs 0.01          @#2

| 框架 | 值 | 文件:行号 |
|------|-----|----------|
| **OpenPI** | `1e-10` | `openpi/src/openpi/training/optimizer.py:73` |
| **LeRobot** | `0.01` | `lerobot/src/lerobot/policies/pi05/configuration_pi05.py:88` |

**影响分析**：

Weight decay 为 0.01 意味着每步每个参数都被缩小 `0.01 × lr = 0.01 × 2.5e-5 = 2.5e-7` 的比例。对于 3.2B 参数的模型，这是一个显著的正则化力量。OpenPI 代码注释明确指出 `1e-10` 是 "a negligible value"，选择它而非 `0` 仅是为了避免某些框架的 OOM bug。

这意味着：
- **OpenPI 训练 ≈ 无正则化**：模型自由优化，仅靠 early stopping 和数据量防止过拟合
- **LeRobot 训练 = 强 L2 正则化**：模型权重被持续压缩，改变 loss landscape

这是**单一最大的超参数差异**，会直接导致：
1. 最终 loss 水平不同（带正则化的 loss 通常更高）
2. 模型参数 norm 不同（0.01 decay 会显著缩小参数 norm）
3. 泛化行为不同（正则化改变泛化特性）

**修复方案**：

**方案 A（推荐 — 修改默认值）**：
```python
# 文件: lerobot/src/lerobot/policies/pi05/configuration_pi05.py:88
# 修改前:
optimizer_weight_decay: float = 0.01
# 修改后:
optimizer_weight_decay: float = 1e-10
```

**方案 B（训练时覆盖）**：
在训练脚本中覆盖：
```python
config.optimizer_weight_decay = 1e-10
```

**方案 C（命令行覆盖）**：
```bash
lerobot-train --policy.optimizer_weight_decay=1e-10 ...
```

**推荐方案 A**，因为 OpenPI 的 weight_decay 对所有 pi05 配置都是 1e-10（默认值），修改 LeRobot 的默认值使两者完全对齐。

---

#### 2.1.2 Loss 截断: 全 32 维 vs 截断到 23 维

| 框架 | 行为 | 文件:行号 |
|------|------|----------|
| **OpenPI** | 在全 32 维（含 9 padding 维）上计算 MSE loss | `openpi/src/openpi/models/pi0.py:214` |
| **LeRobot** | 先截断到 23 维再计算 mean | `lerobot/src/lerobot/policies/pi05/modeling_pi05.py:1267-1268` |

**影响分析**：

OpenPI 的 loss 计算流程：
```python
# pi0.py:214 — JAX 主实现
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])  # v_t: [B, 50, 32]
return jnp.mean(jnp.square(v_t - u_t), axis=-1)  # 在 32 维上求均值 → [B, 50]

# train.py:150-151
chunked_loss = model.compute_loss(rng, observation, actions, train=True)
return jnp.mean(chunked_loss)  # 在 B 和 50 上求均值 → scalar
```

OpenPI PyTorch 的 loss 计算流程：
```python
# pi0_pytorch.py:374
return F.mse_loss(u_t, v_t, reduction="none")  # [B, 50, 32]，返回不 reduce
# 然后由调用方自行处理
```

LeRobot 的 loss 计算流程：
```python
# modeling_pi05.py:1264
losses = self.model.forward(images, img_masks, tokens, masks, actions)  # [B, 50, 32]
# modeling_pi05.py:1267-1268 — 截断！
original_action_dim = self.config.output_features[ACTION].shape[0]  # = 23
losses = losses[:, :, :original_action_dim]  # [B, 50, 23]
# modeling_pi05.py:1281
loss = losses.mean()  # 在 B, 50, 23 上求均值
```

**关键区别**：
1. **梯度流向不同**：OpenPI 的梯度流经 `action_out_proj` 的全部 32 个输出维度（包括 padding 维），LeRobot 只流经 23 个。这意味着 `action_out_proj` 权重矩阵的 [23:32] 列在 LeRobot 中不接收梯度。
2. **Loss 数值不同**：假设 padding 维度的 MSE 较小（因为 target 是 0），OpenPI 的 per-dim loss 会被这 9 个 "容易" 的维度拉低，导致总 loss 偏低。
3. **模型行为不同**：OpenPI 训练后模型学会了在 padding 维输出接近 0，这是额外的隐式约束。

**修复方案**：

**方案 A（推荐 — 添加配置开关）**：
```python
# 文件: lerobot/src/lerobot/policies/pi05/configuration_pi05.py
# 添加新配置项：
truncate_loss_to_action_dim: bool = True  # 默认保持向后兼容

# 文件: lerobot/src/lerobot/policies/pi05/modeling_pi05.py:1266-1268
# 修改前:
# Truncate losses to actual action dimensions
original_action_dim = self.config.output_features[ACTION].shape[0]
losses = losses[:, :, :original_action_dim]

# 修改后:
if getattr(self.config, 'truncate_loss_to_action_dim', True):
    original_action_dim = self.config.output_features[ACTION].shape[0]
    losses = losses[:, :, :original_action_dim]
```

R1 Pro chassis 训练时设置 `truncate_loss_to_action_dim=False`。

**方案 B（直接删除截断）**：
```python
# 文件: lerobot/src/lerobot/policies/pi05/modeling_pi05.py:1266-1268
# 直接注释或删除这两行：
# original_action_dim = self.config.output_features[ACTION].shape[0]
# losses = losses[:, :, :original_action_dim]
```

**推荐方案 A**，保持向后兼容的同时允许 OpenPI 对齐。

---

### 2.2 P1 级别差异（重要 — 应当修复）

---

#### 2.2.1 训练精度: bfloat16 vs float32

| 框架 | 默认值 | 文件:行号 |
|------|--------|----------|
| **OpenPI** | `"bfloat16"` | `openpi/src/openpi/training/config.py:488` |
| **LeRobot** | `"float32"` | `lerobot/src/lerobot/policies/pi05/configuration_pi05.py:34` |

**影响分析**：bfloat16 训练速度更快（约 2x）且内存占用更低（约 0.5x）。但数值精度不同——bfloat16 仅有 7 位有效数字（vs float32 的 24 位）。OpenPI 的 pi0.5 base model 是在 bfloat16 下预训练的，fine-tuning 也应使用 bfloat16 以保持一致。

**修复方案**：
```python
# 训练时设置：
config = PI05Config(dtype="bfloat16", ...)
```

---

#### 2.2.2 训练规模: batch_size 和 steps          @#2

| 参数 | OpenPI | LeRobot 默认 |
|------|--------|-------------|
| batch_size | 64 | 8 |
| num_train_steps | 30,000 | 100,000 |

**修复方案**：
```python
# 训练时设置：
batch_size = 64    # 若 GPU 内存不足，使用 gradient_accumulation_steps = 64 / actual_bs
steps = 30_000
```

如果使用单 GPU 且内存不足以支持 batch_size=64，可以使用梯度累积：
```python
# 例如实际 batch_size=8, gradient_accumulation_steps=8 → 等效 batch_size=64
actual_batch_size = 8
gradient_accumulation_steps = 64 // actual_batch_size  # = 8
```

---

#### 2.2.3 数据集 Key 映射  @#data_convert

**OpenPI 数据流**（`r1pro_chassis_policy.py`）：

```
LeRobot 数据集字段          →  R1ProChassisInputs 输出        →  模型输入
─────────────────────────────────────────────────────────────────────
data["state"]              →  inputs["state"]               →  observation.state
data["head_rgb"]           →  inputs["image"]["base_0_rgb"]       →  observation.images.base_0_rgb
data["left_wrist_rgb"]     →  inputs["image"]["left_wrist_0_rgb"] →  observation.images.left_wrist_0_rgb
data["right_wrist_rgb"]    →  inputs["image"]["right_wrist_0_rgb"]→  observation.images.right_wrist_0_rgb
data["actions"]            →  inputs["actions"]             →  actions
data["prompt"]             →  inputs["prompt"]              →  tokenized_prompt
```

**LeRobot 数据流**（`processor_pi05.py`）：

```
LeRobot 数据集字段              →  ProcessorStep 处理     →  模型输入
──────────────────────────────────────────────────────────────────
observation.state              →  normalize → discretize →  tokens (含 state)
observation.images.head_rgb    →  rename → normalize     →  images[0]
observation.images.left_wrist_rgb → rename → normalize  →  images[1]
observation.images.right_wrist_rgb→ rename → normalize  →  images[2]
action                         →  normalize → pad(32)    →  actions
task (str)                     →  prepend to state str   →  tokens (含 task)
```

**关键问题**：数据集中的相机 key 与 LeRobot PI05Policy 期望的 key 需要对齐。

数据集（由 `convert_r1pro_chassis_data.py` 创建）使用的 key：
- `observation.images.head_rgb`
- `observation.images.left_wrist_rgb`
- `observation.images.right_wrist_rgb`
- `observation.state`（23 维）
- `action`（23 维）
- `task`（文本 prompt）

PI05Policy 内部（`modeling_pi05.py:1185-1206`）直接使用 `batch` dict 中的 `observation.images.*` key 来构建 images list。

**修复方案**：

需要确保数据集 feature 名称与 PI05Policy 的 `input_features` 和 `output_features` 一致。在创建 PI05Config 时：

```python
from lerobot.configs.types import FeatureType, PolicyFeature

config = PI05Config(
    max_state_dim=32,
    max_action_dim=32,
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(23,)),
        "observation.images.head_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.left_wrist_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.right_wrist_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(23,)),
    },
)
```

**注意**：`shape=(23,)` 是原始维度，模型内部会 pad 到 `max_action_dim=32`。LeRobot 的 `NormalizerProcessorStep` 会基于 23 维的 norm_stats 进行归一化，然后模型 forward 中 `pad_vector` 补零到 32 维。

---

#### 2.2.4 Normalization Stats 对齐

**OpenPI**：
- 存储位置：`assets/r1_pro_data_convert_chassis/norm_stats.json`
- 格式：`{"state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}, "actions": {...}}`
- 键名：`"state"`, `"actions"`（23 维向量）
- 由 `compute_norm_stats.py` 预计算

**LeRobot**：
- 存储位置：数据集 `meta/stats.safetensors` 或 training 时传入
- 格式：`{"observation.state": {"mean": tensor, "std": tensor, "q01": tensor, "q99": tensor}, "action": {...}}`
- 键名：`"observation.state"`, `"action"`（23 维 tensor）
- 由 LeRobot 的 dataset stats 计算管线自动生成

**修复方案**：

**方案 A（推荐 — 使用 LeRobot 原生 stats）**：

如果 R1 Pro 数据集已经以 LeRobot 格式存在并包含 stats，直接使用即可。LeRobot 的 quantile 统计计算方式与 OpenPI 在数学上等价（两者都计算 q01/q99），结果应近似相同。

验证步骤：
```python
# 加载 OpenPI norm_stats
import json
with open("assets/r1_pro_data_convert_chassis/norm_stats.json") as f:
    openpi_stats = json.loads(f.read())

# 加载 LeRobot dataset stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("r1_pro_data_convert_chassis")
lerobot_stats = ds.meta.stats

# 对比
import numpy as np
for key_openpi, key_lerobot in [("state", "observation.state"), ("actions", "action")]:
    openpi_q01 = np.array(openpi_stats[key_openpi]["q01"])
    lerobot_q01 = lerobot_stats[key_lerobot]["q01"].numpy()
    print(f"{key_openpi} q01 diff: {np.max(np.abs(openpi_q01 - lerobot_q01))}")
```

如果差异 < 1e-3，可以直接使用 LeRobot stats。如果差异较大，使用方案 B。

**方案 B（转换 OpenPI stats）**：

编写转换脚本：
```python
import json
import torch

with open("assets/r1_pro_data_convert_chassis/norm_stats.json") as f:
    data = json.loads(f.read())
    openpi_stats = data["norm_stats"]

lerobot_stats = {
    "observation.state": {
        "mean": torch.tensor(openpi_stats["state"]["mean"], dtype=torch.float32),
        "std": torch.tensor(openpi_stats["state"]["std"], dtype=torch.float32),
        "q01": torch.tensor(openpi_stats["state"]["q01"], dtype=torch.float32),
        "q99": torch.tensor(openpi_stats["state"]["q99"], dtype=torch.float32),
    },
    "action": {
        "mean": torch.tensor(openpi_stats["actions"]["mean"], dtype=torch.float32),
        "std": torch.tensor(openpi_stats["actions"]["std"], dtype=torch.float32),
        "q01": torch.tensor(openpi_stats["actions"]["q01"], dtype=torch.float32),
        "q99": torch.tensor(openpi_stats["actions"]["q99"], dtype=torch.float32),
    },
}
```

---

#### 2.2.5 Quantile 归一化公式对比

| 步骤 | OpenPI (`transforms.py:141-145`) | LeRobot (`normalize_processor.py`) |
|------|----------------------------------|-------------------------------------|
| 归一化 | `(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0` | `2.0 * (x - q01) / max(q99 - q01, 1e-8) - 1.0` |
| 反归一化 | `(x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01` | `(x + 1.0) * max(q99 - q01, 1e-8) / 2.0 + q01` |

**差异**：
- OpenPI 用 `+ 1e-6`（additive），LeRobot 用 `max(denom, 1e-8)`（conditional）
- 当 `q99 ≠ q01`（实际数据几乎总是如此），两者数值等价
- 当 `q99 == q01`（某维度恒定），OpenPI 输出 `(x - q01) / 1e-6 * 2 - 1`，LeRobot 输出 `(x - q01) / 1e-8 * 2 - 1`，但这种情况极罕见且实际不影响训练

**结论**：无需修改，差异可忽略。

---

### 2.3 P2 级别差异（需要关注）

---

#### 2.3.1 EMA (Exponential Moving Average)

| 框架 | 支持 | 配置 |
|------|------|------|
| **OpenPI JAX** | 支持 | `ema_decay=0.99`（默认值，`pi05_r1pro_chassis` 使用） |
| **OpenPI PyTorch** | **不支持** | `train_pytorch.py` 明确注释 "EMA is not supported" |
| **LeRobot** | 不支持 | — |

**影响分析**：

`pi05_r1pro_chassis` 的默认 `ema_decay=0.99` 仅在 JAX 训练时生效。如果与 OpenPI **PyTorch** 训练对齐，则无 EMA 差异。如果与 OpenPI **JAX** 训练对齐，则 LeRobot 缺少 EMA，可能导致训练稳定性和最终性能差异。

EMA 的效果：维护一份参数的指数滑动平均，推理时使用 EMA 参数而非训练参数。效果是平滑训练噪声，通常提升 0.5-2% 性能。

**修复方案**：

对于与 OpenPI PyTorch 对齐：**无需修复**。

对于与 OpenPI JAX 对齐，需在 LeRobot 训练循环中添加：
```python
# 在训练循环外初始化
import copy
ema_model = copy.deepcopy(model)
ema_decay = 0.99

# 在每个训练步后
with torch.no_grad():
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

# 推理/评估时使用 ema_model
```

**建议**：先与 OpenPI PyTorch 对齐（无 EMA），验证基本对齐后再考虑添加 EMA。

---

#### 2.3.2 图像增强 (Data Augmentation)

| 增强类型 | OpenPI (`model.py:168-187`) | LeRobot |
|---------|----------------------------|---------|
| RandomCrop | 95%，非腕部相机 | 无 |
| Rotate | ±5°，非腕部相机 | 无 |
| ColorJitter | brightness=0.3, contrast=0.4, saturation=0.5 | 无 |
| 腕部相机 | 仅 ColorJitter | 无 |

**影响分析**：图像增强影响泛化能力而非训练收敛。不影响 loss 曲线的对齐验证，但影响最终部署性能。

**修复方案（低优先级）**：
```python
# 在 LeRobot 训练配置中启用图像增强：
dataset.image_transforms.enable = True
# 需要自定义 transform 以匹配 OpenPI 的增强参数
# 特别是区分腕部/非腕部相机的差异化处理
```

**建议**：先不启用图像增强进行对齐验证，确认核心流水线对齐后再添加。

---

#### 2.3.3 基础权重加载

| 框架 | 权重来源 | 格式 |
|------|---------|------|
| **OpenPI** | `gs://openpi-assets/checkpoints/pi05_base/params` | orbax (JAX) |
| **LeRobot** | `lerobot/pi05_base` (HuggingFace Hub) | safetensors |

**两者的 base 权重是否等价？**

是的。LeRobot 的 `lerobot/pi05_base` 是从 OpenPI 的 JAX checkpoint 转换而来，经过 `_fix_pytorch_state_dict_keys()` 的 key 映射。LeRobot 测试文件 `test_pi05_original_vs_lerobot.py` 验证了前向传播输出的一致性。

**修复方案**：
```python
# 直接使用 LeRobot 的预训练模型：
policy = PI05Policy.from_pretrained("lerobot/pi05_base")
```

无需额外的权重转换。

---

#### 2.3.4 Seed 与随机性          @#2

| 框架 | 默认 seed |
|------|----------|
| OpenPI | 42 |
| LeRobot | 1000 |

**修复方案**：训练时设置 `seed=42`。注意 PyTorch 和 JAX 的随机数生成器即使 seed 相同，产生的序列也不同，因此 loss 曲线只能在趋势和量级上对齐，不会逐步精确匹配。

---

### 2.4 P3 级别差异（优化项 — 可暂不处理）

---

#### 2.4.1 Image Padding 值

- **OpenPI**: 空相机 padding 为 `0.0`（黑色）
- **LeRobot**: 空相机 padding 为 `-1.0`

**影响**：R1 Pro chassis 始终提供全部 3 个相机，不存在空相机，因此此差异**不影响本用例**。

#### 2.4.2 State Padding 顺序

- **OpenPI**: 先 tokenize 23-dim state，再 pad state 到 32-dim
- **LeRobot**: PI05 processor 在 tokenize 前不 pad state（`Pi05PrepareStateTokenizerProcessorStep` 直接使用归一化后的 state）

**验证**：检查 `Pi05PrepareStateTokenizerProcessorStep.__call__()` (`processor_pi05.py:57-85`)，state 在 discretize 时使用的是归一化后的原始维度（23-dim），与 OpenPI 一致。**无差异**。

#### 2.4.3 KV Cache

- **OpenPI JAX**: 推理时使用 KV cache（prefix 只计算一次）
- **OpenPI PyTorch**: 无 KV cache
- **LeRobot**: 有 KV cache（`modeling_pi05.py:819-825`）

LeRobot 实际已实现 KV cache，与 OpenPI JAX 对齐。训练时不使用 KV cache（两者一致）。推理时 LeRobot 优于 OpenPI PyTorch。**无需修改**。

---

## 3. 完整训练配置对照表

| 参数 | OpenPI `pi05_r1pro_chassis` | LeRobot 需设置的值 | 默认对齐？ |
|------|---------------------------|-------------------|-----------|
| **model.action_dim** | 32 | `max_action_dim=32` | ✅ |
| **model.action_horizon** | 50 | `chunk_size=50` | ✅ |
| **model.max_token_len** | 200 | `tokenizer_max_length=200` | ✅ |
| **model.pi05** | True | 独立 PI05Policy | ✅ |
| **model.dtype** | bfloat16 | `dtype="bfloat16"` | ❌ 需设置 |
| **optimizer.lr** | 2.5e-5 | `optimizer_lr=2.5e-5` | ✅ |
| **optimizer.betas** | (0.9, 0.95) | `optimizer_betas=(0.9, 0.95)` | ✅ |
| **optimizer.eps** | 1e-8 | `optimizer_eps=1e-8` | ✅ |
| **optimizer.weight_decay** | **1e-10** | `optimizer_weight_decay=1e-10` | ❌ **P0** |
| **optimizer.clip_gradient_norm** | 1.0 | `optimizer_grad_clip_norm=1.0` | ✅ |
| **scheduler.warmup_steps** | 1,000 | `scheduler_warmup_steps=1000` | ✅ |
| **scheduler.peak_lr** | 2.5e-5 | `optimizer_lr=2.5e-5` | ✅ |
| **scheduler.decay_steps** | 30,000 | `scheduler_decay_steps=30000` | ✅ |
| **scheduler.decay_lr** | 2.5e-6 | `scheduler_decay_lr=2.5e-6` | ✅ |
| **batch_size** | 64 | 64 (需设置) | ❌ 需设置 |
| **num_train_steps** | 30,000 | `steps=30000` | ❌ 需设置 |
| **seed** | 42 | `seed=42` | ❌ 需设置 |
| **ema_decay** | 0.99 (JAX only) | 不支持 | ⚠️ PyTorch 无影响 |
| **freeze_filter** | Nothing (全量) | `freeze_vision_encoder=False, train_expert_only=False` | ✅ |
| **loss 截断** | 不截断 (32-dim) | `truncate_loss_to_action_dim=False` | ❌ **P0** |
| **quantile_norm** | True | `NormalizationMode.QUANTILES` | ✅ |
| **image_resolution** | (224, 224) | `image_resolution=(224, 224)` | ✅ |
| **num_cameras** | 3 | 3 | ✅ |
| **data_augmentation** | RandomCrop+Rotate+ColorJitter | 无 | ⚠️ P3 |

---

## 4. 训练脚本模板

基于 `/home/Luogang/SRC/Robot/lerobot/bt/sftpi05/train_pi05.py`，创建 R1 Pro chassis 对齐训练脚本：

```python
#!/usr/bin/env python
"""
R1 Pro Chassis pi0.5 fine-tuning script, aligned with OpenPI pi05_r1pro_chassis.
"""
from pathlib import Path

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

# ─── 1. 配置 (对齐 OpenPI pi05_r1pro_chassis) ───

config = PI05Config(
    # 模型架构（与 OpenPI 默认一致）
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    dtype="bfloat16",                          # P1: 匹配 OpenPI
    chunk_size=50,
    n_action_steps=50,
    max_state_dim=32,
    max_action_dim=32,
    tokenizer_max_length=200,
    num_inference_steps=10,

    # 归一化（与 OpenPI 一致）
    normalization_mapping={
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,
        "ACTION": NormalizationMode.QUANTILES,
    },

    # 优化器（P0: weight_decay 对齐！）
    optimizer_lr=2.5e-5,
    optimizer_betas=(0.9, 0.95),
    optimizer_eps=1e-8,
    optimizer_weight_decay=1e-10,              # P0: OpenPI 默认值！
    optimizer_grad_clip_norm=1.0,

    # 调度器
    scheduler_warmup_steps=1_000,
    scheduler_decay_steps=30_000,
    scheduler_decay_lr=2.5e-6,

    # 训练设置
    gradient_checkpointing=True,               # 节省内存以支持 BS=64
    freeze_vision_encoder=False,               # 全量 fine-tuning
    train_expert_only=False,                   # 全量 fine-tuning

    # P0: 不截断 loss（需要代码修改支持此选项）
    # truncate_loss_to_action_dim=False,

    # Features（R1 Pro chassis 23-dim）
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(23,)),
        "observation.images.head_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.left_wrist_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.right_wrist_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(23,)),
    },
)

# ─── 2. 加载预训练权重 ───
policy = PI05Policy.from_pretrained("lerobot/pi05_base", config=config)

# ─── 3. 训练参数 ───
# batch_size = 64
# steps = 30_000
# seed = 42
# dataset_repo_id = "r1_pro_data_convert_chassis"
# wandb_project = "pi05_r1pro_chassis_lerobot"

# ─── 4. 训练循环 ───
# 使用 LeRobot 的标准训练循环（lerobot-train），或自定义训练循环
# 确保使用 gradient_accumulation_steps 以达到等效 batch_size=64
```

**命令行等效**（如果使用 `lerobot-train`）：
```bash
lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.optimizer_weight_decay=1e-10 \
    --dataset.repo_id=r1_pro_data_convert_chassis \
    --steps=30000 \
    --batch_size=64 \
    --seed=42 \
    --gradient_accumulation_steps=8 \
    --log_freq=100 \
    --save_freq=1000
```

---

## 5. 代码修改清单

### 5.1 必须修改的文件

#### 文件 1: `lerobot/src/lerobot/policies/pi05/configuration_pi05.py`

**修改 1a: weight_decay 默认值（P0）**

```python
# Line 88
# Before:
optimizer_weight_decay: float = 0.01
# After:
optimizer_weight_decay: float = 1e-10
```

**修改 1b: 添加 loss 截断开关（P0）**

```python
# 在 optimizer/scheduler 配置区域后添加：
# Loss computation settings
truncate_loss_to_action_dim: bool = True  # If True, truncate loss to actual action dim (LeRobot default)
                                           # If False, compute loss on all max_action_dim dims (OpenPI behavior)
```

#### 文件 2: `lerobot/src/lerobot/policies/pi05/modeling_pi05.py`

**修改 2a: 支持 loss 截断开关（P0）**

```python
# Lines 1266-1268
# Before:
        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

# After:
        # Optionally truncate losses to actual action dimensions
        # OpenPI computes loss on all max_action_dim dims (including padding)
        # LeRobot defaults to truncating for backward compatibility
        if getattr(self.config, 'truncate_loss_to_action_dim', True):
            original_action_dim = self.config.output_features[ACTION].shape[0]
            losses = losses[:, :, :original_action_dim]
```

### 5.2 可选修改的文件

#### 文件 3: 训练脚本（新建）

**路径建议**: `/home/Luogang/SRC/Robot/lerobot/bt/sftpi05/train_r1pro_chassis.py`

内容见第 4 节的训练脚本模板。

#### 文件 4: Norm stats 验证/转换脚本（新建）

**路径建议**: `/mnt/r/share/lkx/pi/scripts/verify_norm_stats.py`

验证 OpenPI 和 LeRobot 的 norm stats 是否一致。

---

## 6. 验证策略

### 6.1 第一阶段: 静态对齐验证（无需训练）

#### 测试 1: 归一化对齐

```python
"""验证 OpenPI 和 LeRobot 对同一数据点的归一化结果一致。"""
import numpy as np
import torch

# 模拟一个 23-dim state
state = np.random.rand(23).astype(np.float32)
q01 = np.random.rand(23).astype(np.float32) * 0.1
q99 = q01 + np.random.rand(23).astype(np.float32) * 0.9 + 0.1

# OpenPI 归一化
openpi_normed = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

# LeRobot 归一化
state_t = torch.tensor(state)
q01_t = torch.tensor(q01)
q99_t = torch.tensor(q99)
denom = q99_t - q01_t
denom = torch.where(denom == 0, torch.tensor(1e-8), denom)
lerobot_normed = 2.0 * (state_t - q01_t) / denom - 1.0

# 对比
print(f"Max diff: {np.max(np.abs(openpi_normed - lerobot_normed.numpy()))}")
# 应当 < 1e-6
```

#### 测试 2: State 离散化对齐

```python
"""验证两者的 state 离散化（256-bin）结果一致。"""
import numpy as np

normed_state = np.random.uniform(-1, 1, 23).astype(np.float32)

# OpenPI (tokenizer.py:22-48)
openpi_disc = np.digitize(normed_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1

# LeRobot (processor_pi05.py:73)
lerobot_disc = np.digitize(normed_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

# 对比 (应完全一致)
assert np.array_equal(openpi_disc, lerobot_disc), f"Diff: {openpi_disc - lerobot_disc}"
```

#### 测试 3: Prompt 构建对齐

```python
"""验证两者构建的 prompt 字符串一致。"""
task = "Open the door with a downward-press handle, go through it, and enter the room."
state_str = " ".join(map(str, discretized_states))

# OpenPI (tokenizer.py:38-39)
openpi_prompt = f"Task: {task}, State: {state_str};\nAction: "

# LeRobot (processor_pi05.py:78-79)
cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
lerobot_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "

# 注意: LeRobot 额外做了 strip/replace
# 对于这个特定 prompt，两者应一致（无下划线、无换行）
assert openpi_prompt == lerobot_prompt
```

#### 测试 4: 前向传播对齐

```python
"""验证相同输入下两者的 loss 输出一致。"""
import torch

# 加载相同的 base weights
policy = PI05Policy.from_pretrained("lerobot/pi05_base")
policy.eval()

# 构造确定性输入
torch.manual_seed(42)
batch = {
    "observation.images.head_rgb": torch.randn(1, 3, 224, 224),
    "observation.images.left_wrist_rgb": torch.randn(1, 3, 224, 224),
    "observation.images.right_wrist_rgb": torch.randn(1, 3, 224, 224),
    "observation.state": torch.randn(1, 23),
    "action": torch.randn(1, 50, 23),
    # ... tokenized prompt
}

# 前向传播
loss, info = policy.forward(batch)
print(f"LeRobot loss: {loss.item()}")

# 与 OpenPI PyTorch 的 loss 对比（需同样的输入和权重）
# 差异应 < 1e-4 (float32) 或 < 1e-2 (bfloat16)
```

### 6.2 第二阶段: 训练对齐验证

#### 测试 5: 短期训练曲线对比

在两个框架下使用**相同的数据集**和**对齐的超参数**训练 100-500 步，对比 loss 曲线。

**期望结果**：
- 初始 loss 应接近（同一个 base model，同一份数据）
- Loss 下降趋势应一致（同一组超参数）
- 具体数值可能因随机种子不同而有偏差，但量级和形状应匹配

**关键监控指标**：
- `loss` (标量)
- `loss_per_dim` (23 维 or 32 维)
- `gradient_norm` (全局梯度范数)
- `param_norm` (参数总范数 — 受 weight_decay 影响最大)
- `learning_rate` (验证 scheduler 曲线)

#### 测试 6: 参数 Norm 监控

```python
# 每 100 步记录参数 norm
param_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
print(f"Step {step}: param_norm = {param_norm:.4f}")
```

如果 weight_decay 未对齐，param_norm 会在 OpenPI 中保持稳定但在 LeRobot 中持续下降。

### 6.3 第三阶段: 推理对齐验证

#### 测试 7: 推理动作对比

在两个框架下使用训练后的 checkpoint，对同一个观测帧进行推理，对比预测的 action chunk。

```python
# 使用相同的 noise seed
torch.manual_seed(0)
actions = policy.predict_action_chunk(obs_batch)
# 对比两个框架的 actions 输出
# 差异应 < 1e-2 (bfloat16 推理)
```

---

## 7. 实施步骤（按顺序执行）

### Step 1: 代码修改（30 分钟）
1. 修改 `configuration_pi05.py`: weight_decay 默认值 + loss 截断开关
2. 修改 `modeling_pi05.py`: 支持 loss 截断配置

### Step 2: 静态验证（1 小时）
1. 运行测试 1-3（归一化/离散化/prompt 对齐）
2. 运行测试 4（前向传播对齐）
3. 确认所有静态测试通过

### Step 3: Norm Stats 对齐（30 分钟）
1. 对比 OpenPI 和 LeRobot 的 norm stats
2. 如有差异，进行转换或重计算

### Step 4: 创建训练脚本（1 小时）
1. 基于模板创建 R1 Pro chassis 训练脚本
2. 确认数据集加载和 feature 映射正确

### Step 5: 短期训练验证（2-4 小时）
1. 在两个框架下各训练 100 步
2. 对比 loss 曲线和梯度 norm
3. 确认训练行为对齐

### Step 6: 完整训练（12-24 小时）
1. 在 LeRobot 下完成 30,000 步训练
2. 监控所有指标

### Step 7: 推理验证（1 小时）
1. 在测试数据上对比推理输出
2. 如有条件，在真实机器人上对比

---

## 8. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解方案 |
|------|--------|------|---------|
| Base weights 不完全等价 | 低 | 高 | 测试 4 验证前向传播一致性 |
| Norm stats 计算方法差异 | 中 | 中 | 使用 OpenPI 的原始 stats 或验证一致性 |
| bfloat16 数值精度差异 | 中 | 低 | 仅影响绝对值，不影响趋势 |
| 数据加载顺序差异 | 高 | 低 | 不影响收敛结果，仅影响逐步对比 |
| 图像增强缺失 | 确定 | 中 | 第二轮对齐中添加 |
| EMA 缺失 | 确定 | 中-低 | 与 OpenPI PyTorch 对齐时无影响 |
| 内存不足 (BS=64) | 中 | 高 | 使用 gradient checkpointing + 梯度累积 |
| transformers 版本差异 | 低 | 高 | 锁定 transformers 版本，避免 5.4.0+ |

---

## 9. 附录

### A. OpenPI pi05_r1pro_chassis 完整参数展开

```
name:                     pi05_r1pro_chassis
model.dtype:              bfloat16
model.paligemma_variant:  gemma_2b
model.action_expert_variant: gemma_300m
model.action_dim:         32
model.action_horizon:     50
model.max_token_len:      200
model.pi05:               True
model.discrete_state_input: True
model.pytorch_compile_mode: max-autotune

data.repo_id:             r1_pro_data_convert_chassis
data.prompt_from_task:    True
data.action_sequence_keys: ("actions",)
data.use_quantile_norm:   True  (auto for pi05)
data.default_prompt:      "Open the door with a downward-press handle, go through it, and enter the room."

optimizer.type:           AdamW
optimizer.b1:             0.9
optimizer.b2:             0.95
optimizer.eps:            1e-8
optimizer.weight_decay:   1e-10
optimizer.clip_gradient_norm: 1.0

lr_schedule.type:         CosineDecaySchedule
lr_schedule.warmup_steps: 1000
lr_schedule.peak_lr:      2.5e-5
lr_schedule.decay_steps:  30000
lr_schedule.decay_lr:     2.5e-6

ema_decay:                0.99 (JAX only)
freeze_filter:            nnx.Nothing() (全量 fine-tuning)
batch_size:               64
num_train_steps:          30000
seed:                     42
pytorch_training_precision: bfloat16

R1 Pro chassis specifics:
  native_action_dim:      23
  padded_action_dim:      32
  cameras:                head_rgb, left_wrist_rgb, right_wrist_rgb
  camera_mapping:         head→base_0, left_wrist→left_wrist_0, right_wrist→right_wrist_0
```

### B. LeRobot PI05Config 对齐后完整参数

```
paligemma_variant:        gemma_2b
action_expert_variant:    gemma_300m
dtype:                    bfloat16               ← 修改（从 float32）
chunk_size:               50
n_action_steps:           50
max_state_dim:            32
max_action_dim:           32
tokenizer_max_length:     200
num_inference_steps:      10
image_resolution:         (224, 224)

normalization_mapping:
  VISUAL:                 IDENTITY
  STATE:                  QUANTILES
  ACTION:                 QUANTILES

optimizer_lr:             2.5e-5
optimizer_betas:          (0.9, 0.95)
optimizer_eps:            1e-8
optimizer_weight_decay:   1e-10                  ← P0 修改（从 0.01）
optimizer_grad_clip_norm: 1.0

scheduler_warmup_steps:   1000
scheduler_decay_steps:    30000
scheduler_decay_lr:       2.5e-6

gradient_checkpointing:   True                   ← 新增
freeze_vision_encoder:    False
train_expert_only:        False
truncate_loss_to_action_dim: False               ← P0 新增
```

### C. 关键文件索引

| 文件 | 用途 | 修改？ |
|------|------|--------|
| `lerobot/policies/pi05/configuration_pi05.py` | PI05 配置定义 | ✅ 修改 weight_decay + 添加 loss 截断开关 |
| `lerobot/policies/pi05/modeling_pi05.py` | PI05 模型和 Policy | ✅ 修改 loss 截断逻辑 |
| `lerobot/policies/pi05/processor_pi05.py` | 数据处理流水线 | 无需修改 |
| `lerobot/policies/pi_gemma.py` | PiGemma/AdaRMS 实现 | 无需修改 |
| `openpi/training/config.py:1024-1042` | 参考: TrainConfig 定义 | 只读参考 |
| `openpi/policies/r1pro_chassis_policy.py` | 参考: data transform | 只读参考 |
| `openpi/training/optimizer.py` | 参考: optimizer 默认值 | 只读参考 |
| `openpi/transforms.py` | 参考: 归一化实现 | 只读参考 |
| `openpi/models/pi0.py:188-214` | 参考: JAX loss 计算 | 只读参考 |
| `openpi/models_pytorch/pi0_pytorch.py:317-374` | 参考: PyTorch loss 计算 | 只读参考 |
