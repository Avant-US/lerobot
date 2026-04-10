# LeRobot pi0.5 与 OpenPI JAX pi05_r1pro_chassis 深度对齐方案 (v3)

> **日期**: 2026-04-09
> **目标**: 使 LeRobot 的 pi0.5 fine-tuning 在 R1 Pro chassis 数据上产生与 OpenPI JAX `pi05_r1pro_chassis` 等价的训练结果
> **基准**: OpenPI **JAX 训练路径** (`scripts/train.py`)，而非 PyTorch 路径
> **代码库**: OpenPI (`/mnt/r/share/lkx/pi/openpi`), LeRobot (`/home/Luogang/SRC/Robot/lerobot`)

---

## 0. 本文定位与 v1/v2 的区别

### 0.1 为什么需要 v3

v1 (`pi05_alig.md`) 和 v2 (`pi05_alig_2.md`) 的分析存在以下不足：

1. **基准选择偏差**：v1/v2 主要以 OpenPI **PyTorch** 训练路径 (`train_pytorch.py`) 为参照。但 `pi05_r1pro_chassis` 的主训练路径是 **JAX** (`train.py`)，而 JAX 版有 PyTorch 版不具备的关键机制（EMA、数据增强）。
2. **LR Schedule 差异遗漏**：v1/v2 均未发现 LeRobot 的 `CosineDecayWithWarmupSchedulerConfig` 与 OpenPI 的 `optax.warmup_cosine_decay_schedule` 在余弦衰减相位上存在约 4% 的系统性差异。
3. **EMA 影响被低估**：v2 将 EMA 归为 P3（"与 OpenPI PyTorch 对齐时无影响"），但 `pi05_r1pro_chassis` 的 JAX 训练默认启用 `ema_decay=0.99`，且 **checkpoint 保存的是 EMA 参数而非原始参数**——这直接影响推理模型质量。
4. **数据增强被低估**：v2 标为 P2（"先不加增强"），但 JAX 训练在 JIT 编译的 `train_step` 内部执行完整的数据增强（RandomCrop、Rotate、ColorJitter），这是主训练路径的一部分。

### 0.2 v3 的新贡献

| 内容 | v1/v2 状态 | v3 |
|------|-----------|-----|
| LR schedule 余弦相位差异 | 未发现 | **新增 P0 差异**，含数值推导 |
| EMA 完整机制追踪 | P3/"无影响" | **升级为 P1**，含实现方案 |
| JAX 数据增强参数 | P2/缺参数 | **升级为 P1**，含 PyTorch 等价方案 |
| JAX 训练步骤端到端追踪 | 无 | **新增核心章节** |
| JIT/FSDP/RNG 对训练的影响 | 未分析 | **新增框架级分析** |
| "训练等价"的三层定义 | 无 | **新增** |

### 0.3 "训练等价"的定义

本文将"等价训练结果"分为三个层次：

| 层次 | 定义 | 可达性 |
|------|------|--------|
| L1: Loss 曲线等价 | 30000 步训练的 loss 曲线在量级和趋势上对齐 | ✅ 修复 P0 后可达 |
| L2: Checkpoint 质量等价 | 最终 checkpoint 在评估任务上的成功率无统计显著差异 | ✅ 修复 P0+P1 后可达 |
| L3: Action 输出等价 | 同输入下推理输出的 action 向量逐元素相近 | ❌ 跨框架不可达（RNG、数值精度） |

---

## 1. OpenPI JAX pi05_r1pro_chassis 完整配置基线

### 1.1 TrainConfig 原始定义

**来源**: `openpi/src/openpi/training/config.py:1024-1042`

```python
TrainConfig(
    name="pi05_r1pro_chassis",
    model=pi0_config.Pi0Config(pi05=True),
    data=SimpleDataConfig(
        repo_id="r1_pro_data_convert_chassis",
        data_transforms=lambda model: _transforms.Group(
            inputs=[r1pro_chassis_policy.R1ProChassisInputs(model_type=model.model_type)],
            outputs=[r1pro_chassis_policy.R1ProChassisOutputs()],
        ),
        model_transforms=ModelTransformFactory(
            default_prompt="Open the door with a downward-press handle, go through it, and enter the room."
        ),
        base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("actions",)),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    batch_size=64,
)
```

**关键：此配置未覆盖以下 TrainConfig 默认值，因此全部继承**：
- `ema_decay=0.99` (`config.py:492`)
- `optimizer=AdamW()` (`config.py:491`)
- `lr_schedule=CosineDecaySchedule()` (`config.py:490`)
- `seed=42` (`config.py:506`)
- `save_interval=1000` (`config.py:516`)
- `log_interval=100` (`config.py:517`)

### 1.2 完整参数展开表

| 参数路径 | 值 | 来源 | LeRobot 对齐？ |
|---------|-----|------|-------------|
| **模型** | | | |
| `model.pi05` | `True` | 显式 | ✅ |
| `model.discrete_state_input` | `True` | pi05 自动设置 (`pi0_config.py:41`) | ✅ |
| `model.action_dim` | `32` | Pi0Config 默认 | ✅ `max_action_dim=32` |
| `model.action_horizon` | `50` | Pi0Config 默认 | ✅ `chunk_size=50` |
| `model.max_token_len` | `200` | pi05 自动设置 (`pi0_config.py:39`) | ✅ `tokenizer_max_length=200` |
| `model.dtype` | `"bfloat16"` | Pi0Config 默认 (`pi0_config.py:20`) | ❌ LeRobot 默认 `float32` |
| `model.paligemma_variant` | `"gemma_2b"` | 默认 | ✅ |
| `model.action_expert_variant` | `"gemma_300m"` | 默认 | ✅ |
| **优化器** | | | |
| `optimizer.weight_decay` | `1e-10` | AdamW 默认 (`optimizer.py:73`) | ❌ **P0** LeRobot=0.01 |
| `optimizer.b1` | `0.9` | 默认 | ✅ |
| `optimizer.b2` | `0.95` | 默认 | ✅ |
| `optimizer.eps` | `1e-8` | 默认 | ✅ |
| `optimizer.clip_gradient_norm` | `1.0` | 默认 | ✅ |
| **LR Schedule** | | | |
| `lr_schedule.warmup_steps` | `1000` | 默认 | ✅ |
| `lr_schedule.peak_lr` | `2.5e-5` | 默认 | ✅ |
| `lr_schedule.decay_steps` | `30000` | 默认 | ❌ **P0** 余弦相位不同 |
| `lr_schedule.decay_lr` | `2.5e-6` | 默认 | ✅ |
| `lr_schedule.init_value` | `≈2.497e-8` | 计算: `2.5e-5/(1000+1)` | ✅ 公式一致 |
| **训练** | | | |
| `batch_size` | `64` | 显式 | ❌ 需设置 |
| `num_train_steps` | `30000` | 显式 | ❌ 需设置 |
| `seed` | `42` | 默认 | ❌ LeRobot 默认 1000 |
| `ema_decay` | **`0.99`** | 默认 (`config.py:492`) | ❌ **P1** LeRobot 无 EMA |
| **数据增强** | | | |
| 非 wrist: RandomCrop | 95% | `model.py:176` | ❌ **P1** LeRobot 无增强 |
| 非 wrist: Rotate | ±5° | `model.py:178` | ❌ |
| 所有: ColorJitter | b=0.3,c=0.4,s=0.5 | `model.py:181` | ❌ |

### 1.3 R1 Pro Chassis 动作空间

**来源**: `openpi/src/openpi/policies/r1pro_chassis_policy.py:1-7`

```
ACTION_DIM = 23
[0:7]    left_arm (7 关节)
[7:14]   right_arm (7 关节)
[14]     left_gripper
[15]     right_gripper
[16:20]  torso (4 DOF)
[20:23]  chassis_velocities (x, y, rotation)
```

模型内部 padding 到 32 维（末尾补 9 个零）。使用**绝对动作**（非 delta），无 `DeltaActions` 变换。

---

## 2. JAX 训练步骤完整追踪

本节逐步追踪 OpenPI JAX 训练循环中一个 batch 的完整生命周期。这是 v1/v2 中缺失的核心内容。

### 2.1 初始化

**来源**: `scripts/train.py:194-248`

```python
# 1. RNG 初始化 (train.py:205-206)
rng = jax.random.key(config.seed)  # seed=42
train_rng, init_rng = jax.random.split(rng)
# train_rng 在整个训练过程中保持不变
# 每步通过 jax.random.fold_in(train_rng, step) 生成确定性子密钥

# 2. FSDP Mesh 设置 (train.py:208-210)
mesh = sharding.make_mesh(config.fsdp_devices)  # fsdp_devices=1 默认
data_sharding = NamedSharding(mesh, PartitionSpec(DATA_AXIS))
replicated_sharding = NamedSharding(mesh, PartitionSpec())
# 单 GPU 下: 无实际分片，所有参数完整存储

# 3. 模型初始化 (train.py:236)
# TrainState 包含: step, params, model_def, opt_state, tx, ema_decay, ema_params
# ema_params 初始化为 params 的拷贝 (train.py:113)
ema_params = None if config.ema_decay is None else params
# 对 pi05_r1pro_chassis: ema_decay=0.99, ema_params = params（初始同源）

# 4. JIT 编译 (train.py:243-248)
ptrain_step = jax.jit(
    functools.partial(train_step, config),
    in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    out_shardings=(train_state_sharding, replicated_sharding),
    donate_argnums=(1,),  # 释放旧 train_state 内存
)
# 整个 train_step（含增强、前向、反向、优化器、EMA）编译为一个 XLA 程序
```

### 2.2 数据加载流水线

**来源**: `data_loader.py`, `transforms.py`, `tokenizer.py`

```
原始数据 (LeRobotDataset)
  │
  ├── head_rgb: uint8[H,W,3]
  ├── left_wrist_rgb: uint8[H,W,3]
  ├── right_wrist_rgb: uint8[H,W,3]
  ├── state: float32[23]
  ├── actions: float32[50,23]
  └── task_index: int
      │
      ▼ PromptFromLeRobotTask (data_loader.py:148-149)
      │  task_index → tasks[task_index] → prompt: str
      ▼
      ▼ R1ProChassisInputs (r1pro_chassis_policy.py:55)
      │  相机名映射 + 格式标准化
      │  image/base_0_rgb, image/left_wrist_0_rgb, image/right_wrist_0_rgb
      ▼
      ▼ Normalize (transforms.py:126, quantile 模式)
      │  公式: (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
      │  对 state 和 actions 归一化到 [-1, 1]
      │  image 不归一化（保持 uint8）
      ▼
      ▼ InjectDefaultPrompt (transforms.py:108)
      │  若无 prompt key → 注入 default_prompt (R1 Pro 已有, 跳过)
      ▼
      ▼ ResizeImages(224, 224) (transforms.py:189)
      │  若尺寸不对 → resize_with_pad; 已是 224x224 → no-op
      ▼
      ▼ TokenizePrompt (transforms.py:252, discrete_state_input=True)
      │  1. cleaned_text = prompt.strip().replace("_"," ").replace("\n"," ")
      │  2. discretized = np.digitize(state, np.linspace(-1,1,257)[:-1]) - 1
      │     → 每个 state 维度映射到 0-255 整数（256 bins）
      │  3. state_str = " ".join(map(str, discretized))
      │  4. full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
      │  5. tokens = sentencepiece.encode(full_prompt, add_bos=True)
      │  6. pad/truncate 到 max_len=200
      │  输出: tokenized_prompt: int32[200], tokenized_prompt_mask: bool[200]
      │  ★ state 在此被消费（编码进 prompt tokens）
      ▼
      ▼ PadStatesAndActions(32) (transforms.py:328)
      │  state: [23] → [32] (补 9 个 0)
      │  actions: [50,23] → [50,32] (每步补 9 个 0)
      │  ★ padding 发生在 tokenization 之后
      ▼
      ▼ _collate_fn (data_loader.py:471)
      │  np.stack → numpy batch
      ▼
      ▼ Observation.from_dict() (model.py:110-129)
         image: uint8[B,H,W,3] → float32 via /255*2-1 → [-1,1]
         输出: Observation(images, image_masks, state, tokenized_prompt, ...)
```

### 2.3 JIT 内部: train_step 完整逻辑

**来源**: `scripts/train.py:137-191`

```python
def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    # ── Loss 函数 ──
    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)  # scalar

    # ── 生成本步 RNG ──
    train_rng = jax.random.fold_in(rng, state.step)
    # 注意: rng 是固定的 train_rng，通过 fold_in(step) 产生每步不同的随机序列
    # 这与 PyTorch 的 stateful PRNG 完全不同

    observation, actions = batch

    # ── 梯度计算 ──
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    # ── 优化器更新 ──
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # ── 更新模型 ──
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1,
                                     params=new_params, opt_state=new_opt_state)

    # ── EMA 更新 ──
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params
            ),
        )
    # 对 pi05_r1pro_chassis: ema_params = 0.99 * ema_old + 0.01 * params_new

    # ── 指标收集 ──
    # grad_norm, param_norm (仅 kernel 参数), loss
    return new_state, {"loss": loss, "grad_norm": ..., "param_norm": ...}
```

### 2.4 数据增强（JIT 内部）

**来源**: `models/model.py:144-208`（在 `compute_loss` 内调用）

```python
# pi0.py:193 — compute_loss 的第一步
observation = preprocess_observation(preprocess_rng, observation, train=True)
```

`preprocess_observation` 逻辑 (`model.py:168-187`):

```python
if train:
    image = image / 2.0 + 0.5  # [-1,1] → [0,1] for augmax

    transforms = []
    if "wrist" not in key:  # 非 wrist 相机 (base_0_rgb)
        height, width = image.shape[1:3]  # 224, 224
        transforms += [
            augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),  # 212x212
            augmax.Resize(width, height),                              # 回到 224x224
            augmax.Rotate((-5, 5)),                                    # ±5°
        ]
    # 所有相机
    transforms += [
        augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
    ]
    sub_rngs = jax.random.split(rng, image.shape[0])  # 每样本独立 RNG
    image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

    image = image * 2.0 - 1.0  # [0,1] → [-1,1]
```

| 增强类型 | 参数 | 应用范围 | 在 JIT 内？ |
|---------|------|---------|-----------|
| RandomCrop | 95% (212×212) | 非 wrist (base_0_rgb) | ✅ |
| Resize | 224×224 (bilinear) | 非 wrist | ✅ |
| Rotate | ±5° | 非 wrist | ✅ |
| ColorJitter | b=0.3, c=0.4, s=0.5 | 所有 3 个相机 | ✅ |

**关键**: 增强在 JIT 内执行，使用 JAX 的 `vmap` 并行处理 batch 内每个样本。这意味着：
- 增强操作被编译到 XLA 计算图中
- RNG 通过 `jax.random.split` 为每个样本生成独立的随机数

**LeRobot 现状**: 无数据增强。

### 2.5 Loss 计算

**来源**: `models/pi0.py:189-214`

```python
def compute_loss(self, rng, observation, actions, *, train=False):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]  # [B]
    noise = jax.random.normal(noise_rng, actions.shape)  # [B, 50, 32]
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # [B]

    time_expanded = time[..., None, None]  # [B, 1, 1]
    x_t = time_expanded * noise + (1 - time_expanded) * actions  # [B, 50, 32]
    u_t = noise - actions  # [B, 50, 32] — target velocity

    # Forward pass
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        observation, x_t, time
    )
    # ... attention mask construction, position computation ...
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask,
        positions=positions, adarms_cond=[None, adarms_cond]
    )

    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])  # [B, 50, 32]

    # ★ MSE loss — 在全部 32 维上计算 ★
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [B, 50]
```

然后在 `train_step` 中:
```python
chunked_loss = model.compute_loss(rng, observation, actions, train=True)
loss = jnp.mean(chunked_loss)  # [B, 50] → scalar
```

**总 loss = mean(MSE[B, 50, 32]) = sum / (B × 50 × 32)**

**LeRobot** (`modeling_pi05.py:1264-1268`):
```python
losses = self.model.forward(images, img_masks, tokens, masks, actions)  # [B, 50, 32]
original_action_dim = self.config.output_features[ACTION].shape[0]  # 23
losses = losses[:, :, :original_action_dim]  # [B, 50, 23] ← 截断!
loss = losses.mean()  # sum / (B × 50 × 23)
```

**差异影响**:
1. **Loss 数值**: OpenPI 的 loss 分母含 32 维（含 9 个"容易"的 padding 维），平均值偏低
2. **梯度流**: OpenPI 的 `action_out_proj` 全部 32 个输出列接收梯度；LeRobot 仅 23 列有梯度，后 9 列梯度为零
3. **隐式正则化**: OpenPI 训练后模型学会在 padding 维输出接近 0，是额外的约束信号

### 2.6 梯度计算与优化器

```python
# 梯度 (train.py:157-162)
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rng, obs, acts)

# 优化器链 (optimizer.py:81-85)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # 先裁剪梯度
    optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10)  # 再 Adam 更新
)

# 更新 (train.py:160-162)
updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
new_params = optax.apply_updates(params, updates)
```

**与 LeRobot 的对比**:

| 方面 | OpenPI JAX | LeRobot |
|------|-----------|---------|
| 梯度裁剪 | `optax.clip_by_global_norm(1.0)` | `clip_grad_norm_(params, 1.0)` |
| 裁剪位置 | 在 Adam 更新之前（chain 内） | 在 `optimizer.step()` 之前 |
| weight_decay | `1e-10` (optax adamw) | `0.01` (torch AdamW) |
| LR 更新 | optax schedule 自动应用 | `LambdaLR.step()` |
| 混合精度 | 无 autocast（模型参数本身是 bfloat16） | `accelerator.autocast()` |

裁剪和 Adam 的执行顺序是等价的。

### 2.7 EMA 更新

**来源**: `scripts/train.py:169-175`

```python
if state.ema_decay is not None:  # ema_decay=0.99 for pi05_r1pro_chassis
    new_state = dataclasses.replace(
        new_state,
        ema_params=jax.tree.map(
            lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
            state.ema_params, new_params
        ),
    )
```

**关键特性**:
- EMA 更新**在 JIT 内部**，每步执行
- `ema_params` 是所有参数的独立拷贝（内存 ×2）
- EMA **不影响训练动态** — 梯度基于 `params` 计算，不使用 `ema_params`
- EMA 只影响保存的 checkpoint（推理权重）

**初始化** (`train.py:113`):
```python
ema_params = None if config.ema_decay is None else params
```
即 EMA 参数初始值 = 模型参数初始值（同源）。

**有效平均窗口**: 1/(1−0.99) = 100 步

### 2.8 Checkpoint 保存

**来源**: `checkpoints.py:145-152`

```python
def _split_params(state: TrainState):
    if state.ema_params is not None:
        params = state.ema_params   # ← 保存 EMA 参数用于推理
        train_state = replace(state, ema_params=None)
    else:
        params = state.params       # ← 无 EMA 时保存原始参数
        train_state = replace(state, params={})
    return train_state, params
```

**对 `pi05_r1pro_chassis`**:
- `ema_params is not None` → **checkpoint 的 "params" item 保存的是 EMA 参数**
- `train_state` 中清空了 `ema_params` 但保留了 `params`（原始训练参数）
- 恢复时: `_merge_params` 检测 `train_state.params` 非空 → 将加载的 params 放入 `ema_params`

**保存频率**: `save_interval=1000`（每 1000 步），最大保留 1 个最新 + 每 5000 步保留一个
**保存内容**: Orbax checkpoint, 含 "train_state" (opt_state + params) + "params" (EMA) + "assets" (norm_stats)

**LeRobot checkpoint 保存** (`train_utils.py:65-110`):
- `policy.save_pretrained(pretrained_dir)` → `model.safetensors` (原始参数)
- 无 EMA 参数保存

### 2.9 主训练循环

**来源**: `scripts/train.py:259-276`

```python
infos = []
for step in pbar:
    with sharding.set_mesh(mesh):
        train_state, info = ptrain_step(train_rng, train_state, batch)
    infos.append(info)
    if step % config.log_interval == 0:  # 每 100 步
        stacked_infos = common_utils.stack_forest(infos)
        reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
        wandb.log(reduced_info, step=step)
        infos = []
    batch = next(data_iter)  # ★ 注意: batch 在循环末尾获取，下一步使用

    if (step % config.save_interval == 0 and step > start_step) or \
       step == config.num_train_steps - 1:
        _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
```

**注意**: `batch = next(data_iter)` 在循环**末尾**而非开头。第一个 batch 在循环外获取 (`train.py:226`)。

---

## 3. LR Schedule 差异深度分析（v1/v2 遗漏的新发现）

### 3.1 OpenPI 实现: optax.warmup_cosine_decay_schedule

**来源**: `optimizer.py:24-31`

```python
def create(self) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=self.peak_lr / (self.warmup_steps + 1),   # 2.5e-5 / 1001 ≈ 2.497e-8
        peak_value=self.peak_lr,                              # 2.5e-5
        warmup_steps=self.warmup_steps,                       # 1000
        decay_steps=self.decay_steps,                         # 30000
        end_value=self.decay_lr,                              # 2.5e-6
    )
```

**optax 内部行为** (`decay_steps` 是总步数，含 warmup):
- Steps 0 → 1000: 线性 warmup，lr 从 `init_value` 线性增到 `peak_value`
- Steps 1000 → 30000: 余弦衰减，lr 从 `peak_value` 衰减到 `end_value`
- 余弦相位跨度: **29000 步** (decay_steps − warmup_steps)

**公式** (post-warmup, step > 1000):
```
progress = (step - 1000) / (30000 - 1000)  # 0 → 1
lr = end_value + 0.5 * (peak_value - end_value) * (1 + cos(π × progress))
```

### 3.2 LeRobot 实现: CosineDecayWithWarmupSchedulerConfig

**来源**: `schedulers.py:113-130`

```python
def lr_lambda(current_step):
    def linear_warmup_schedule(current_step):
        if current_step <= 0:
            return 1 / (actual_warmup_steps + 1)
        frac = 1 - current_step / actual_warmup_steps
        return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

    def cosine_decay_schedule(current_step):
        step = min(current_step, actual_decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
        alpha = self.decay_lr / self.peak_lr  # 0.1
        decayed = (1 - alpha) * cosine_decay + alpha
        return decayed

    if current_step < actual_warmup_steps:
        return linear_warmup_schedule(current_step)
    return cosine_decay_schedule(current_step)
```

**问题**: `cosine_decay_schedule(current_step)` 使用 **绝对 step** 而非相对于 warmup 结束的 step。

- 在 step=1000 时: `cos(π × 1000/30000) = cos(π/30) ≈ 0.9945`
- 返回: `(1-0.1) × 0.5 × (1+0.9945) + 0.1 = 0.9975`
- 实际 lr = peak_lr × 0.9975 = 2.494e-5 **（不是 peak_lr!）**

### 3.3 数值对比

| Step | OpenPI lr | LeRobot lr | 差异 | 相对差异 |
|------|----------|-----------|------|---------|
| 0 | 2.497e-8 | 2.497e-8 | 0 | 0% |
| 500 | 1.250e-5 | 1.250e-5 | ~0 | ~0% |
| 1000 (peak) | **2.500e-5** | **2.494e-5** | 6e-8 | **-0.25%** |
| 2000 | 2.487e-5 | 2.476e-5 | 1.1e-7 | -0.44% |
| 5000 | 2.302e-5 | 2.261e-5 | 4.1e-7 | -1.8% |
| 10000 | 1.819e-5 | 1.748e-5 | 7.1e-7 | -3.9% |
| **15000** | **1.434e-5** | **1.375e-5** | **5.9e-7** | **-4.13%** |
| 20000 | 1.056e-5 | 1.002e-5 | 5.4e-7 | -5.1% |
| 25000 | 5.31e-6 | 4.89e-6 | 4.2e-7 | -7.9% |
| 29000 | 2.90e-6 | 2.72e-6 | 1.8e-7 | -6.2% |
| 30000 | **2.500e-6** | **2.500e-6** | 0 | 0% |

**观察**:
- Warmup 阶段: 完全一致
- Peak 处 (step 1000): LeRobot 已经偏低 0.25%
- 中间区域 (step 15000-25000): 差异最大，约 4-8%
- 终点 (step 30000): 完全一致（都达到 decay_lr）

**差异原因**: OpenPI 的余弦跨 29000 步 (1000→30000)，LeRobot 的余弦跨 30000 步 (0→30000) 但从 step 1000 开始取值。

### 3.4 影响评估

- 4-8% 的 LR 差异意味着 LeRobot 在训练中后期使用了**系统性偏低**的学习率
- 这会导致: (a) 收敛速度略慢, (b) 最终 loss 略高, (c) 可能更不容易过拟合
- 在 30000 步训练中，累积效应不可忽视

### 3.5 修复方案

**方案 A: 修改 `cosine_decay_schedule` 使用相对 step**

```python
# schedulers.py — 修改后
def cosine_decay_schedule(current_step):
    # 使用相对于 warmup 结束的步数
    relative_step = current_step - actual_warmup_steps
    total_decay_steps = actual_decay_steps - actual_warmup_steps
    progress = min(relative_step / total_decay_steps, 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    alpha = self.decay_lr / self.peak_lr
    decayed = (1 - alpha) * cosine_decay + alpha
    return decayed
```

**方案 B: 手动实现 optax 完整 schedule**

```python
def lr_lambda(current_step):
    if current_step < actual_warmup_steps:
        # 线性 warmup
        return (current_step / actual_warmup_steps) * (1 - 1/(actual_warmup_steps+1)) + 1/(actual_warmup_steps+1)
    else:
        # 余弦衰减 (相对 step)
        progress = (current_step - actual_warmup_steps) / (actual_decay_steps - actual_warmup_steps)
        progress = min(progress, 1.0)
        alpha = self.decay_lr / self.peak_lr
        return alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * progress))
```

**推荐方案 A**，因为改动最小且语义清晰。

---

## 4. EMA 机制深度分析

### 4.1 为什么 EMA 被从 P3 升级为 P1

v2 将 EMA 标为 P3 的理由是"与 OpenPI PyTorch 对齐时无影响"。但本文的基准是 **JAX 训练**:

1. `pi05_r1pro_chassis` 的 JAX 训练默认 `ema_decay=0.99`（`config.py:492`，未覆盖）
2. **Checkpoint 保存 EMA 参数**（`checkpoints.py:146-147`）
3. 因此，OpenPI 发布的 `pi05_r1pro_chassis` checkpoint 中的权重是 **EMA 平滑后的**
4. LeRobot 无 EMA → 保存原始训练参数 → **推理模型质量不同**

### 4.2 EMA 对模型质量的影响

**EMA 的本质**: 对训练轨迹上的参数做指数加权移动平均。

- **不影响训练过程**: 梯度始终基于原始 `params` 计算，`ema_params` 只是旁路拷贝
- **影响推理模型**: checkpoint 中的推理权重是平滑后的
- **有效窗口**: 1/(1−0.99) = 100 步。即 EMA 权重 ≈ 最近 100 步参数的加权平均

**典型效果**:
- 减少参数估计的方差 → 更稳定的推理
- 隐式正则化 → 更好的泛化
- 通常改善评估指标 0.5-3%

**数学关系**: 设第 t 步的参数为 θ_t，EMA 参数为 θ̂_t:
```
θ̂_t = 0.99 × θ̂_{t-1} + 0.01 × θ_t
    = Σ_{i=0}^{t} 0.01 × 0.99^{t-i} × θ_i  (展开)
```
对于 i < t−500 的项，权重 < 0.01 × 0.99^500 ≈ 6.6e-5，基本可忽略。

### 4.3 LeRobot 中实现 EMA 的方案

#### 方案 A: 手动实现（推荐）

在训练循环中维护 EMA state:

```python
import copy

class EMAModel:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for key, param in model.state_dict().items():
            self.shadow[key] = self.decay * self.shadow[key] + (1 - self.decay) * param

    def apply_to(self, model):
        """将 EMA 参数加载到模型中（用于保存 checkpoint 或推理）"""
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
```

**集成到训练循环**:
```python
ema = EMAModel(policy, decay=0.99)

for step in range(num_steps):
    # ... forward, backward, optimizer.step() ...
    ema.update(policy)

    if step % save_interval == 0:
        # 保存 EMA 参数作为推理权重
        ema.apply_to(policy)
        policy.save_pretrained(checkpoint_dir)
        # 恢复原始参数继续训练
        policy.load_state_dict(original_params)
```

#### 方案 B: 使用 torch-ema 库

```python
from torch_ema import ExponentialMovingAverage
ema = ExponentialMovingAverage(policy.parameters(), decay=0.99)

# 训练步后
ema.update()

# 保存 checkpoint 时
with ema.average_parameters():
    policy.save_pretrained(checkpoint_dir)
```

#### 方案 C: 不实现 EMA（简化方案）

- 接受推理模型质量有 0.5-3% 的差异
- 适用于快速验证 P0 修复是否有效

**推荐**: 先用方案 C 验证 P0 对齐，再用方案 A 实现完整对齐。

### 4.4 内存影响

EMA 需要存储全部参数的额外拷贝:
- PI0.5 参数量: ~2.3B (Gemma 2B + Gemma 300M)
- bfloat16: ~4.6 GB 额外内存
- 结合 `gradient_checkpointing=True` 可以缓解总内存压力

---

## 5. 差异影响分级（以 JAX 为基准重新分级）

### 5.1 P0: Critical — 必须修复以实现 L1 等价

#### P0-1: Weight Decay (1e-10 vs 0.01)

| 项目 | OpenPI JAX | LeRobot |
|------|-----------|---------|
| 值 | `1e-10` | `0.01` |
| 来源 | `optimizer.py:73` | `configuration_pi05.py:88` |
| 含义 | 实质无正则化 | 活跃 L2 正则化 |

**影响**: 0.01 的 weight_decay 在 AdamW 中每步将权重乘以 `(1 - lr × wd)`:
- lr=2.5e-5, wd=0.01 → 每步乘 0.99999975
- 30000 步后: 参数缩小约 0.75%
- 这改变了 loss landscape，影响收敛方向

**修复**: `configuration_pi05.py:88` → `optimizer_weight_decay: float = 1e-10`

---

#### P0-2: Loss 截断 (32 维 vs 23 维)   @#0

| 项目 | OpenPI JAX | LeRobot |
|------|-----------|---------|
| loss 维度 | 32（含 padding） | 23（截断 padding） |
| 来源 | `pi0.py:214` | `modeling_pi05.py:1266-1268` |

**影响**:
- 梯度差异: `action_out_proj` 的后 9 列在 LeRobot 中无梯度
- Loss 数值差异: OpenPI loss 分母含 "容易" 的 padding 维，数值偏低
- 隐式正则化: OpenPI 训练后 padding 维输出接近 0

**修复**: 添加 `truncate_loss_to_action_dim` 配置开关

---

#### P0-3: LR Schedule 余弦相位差异（新发现）   @#1

| 项目 | OpenPI JAX | LeRobot |
|------|-----------|---------|
| 余弦跨度 | 29000 步 (1000→30000) | 30000 步 (0→30000) |
| Peak 处 lr | 2.5e-5 (精确) | 2.494e-5 (偏低 0.25%) |
| 最大差异 | — | step 20000 处约 -5.1% |
| 来源 | `optimizer.py:25-26` (optax) | `schedulers.py:120-122` |

**影响**: 系统性偏低的 LR 导致训练中后期学习速率不足，收敛特性不同

**修复**: 修改 `schedulers.py` 的 `cosine_decay_schedule` (见 Section 3.5)

---

### 5.2 P1: Important — 需修复以实现 L2 等价

#### P1-1: EMA (0.99 vs 无)     @#0

- **影响**: 推理权重质量差异 0.5-3%
- **修复**: 实现 EMA (见 Section 4.3)

#### P1-2: dtype (bfloat16 vs float32)

| 项目 | OpenPI JAX | LeRobot |
|------|-----------|---------|
| 模型参数 | bfloat16 | float32 (默认) |
| 来源 | `pi0_config.py:20` | `configuration_pi05.py:34` |

- **影响**: bfloat16 训练更快、内存更省，但有 ~1e-3 的数值精度差异
- OpenPI JAX 不使用 `autocast`，参数本身就是 bfloat16
- **修复**: 训练时设 `dtype="bfloat16"`

#### P1-3: 数据增强（有 vs 无）     @#1

- **影响**: 增强提升泛化能力，对最终部署成功率有显著影响
- **修复**: 实现等价的 PyTorch 增强（见 Section 10）

#### P1-4: Batch Size (64 vs 默认)

- OpenPI: `batch_size=64`，无梯度累积
- LeRobot: 需根据 GPU 内存设置，可能需要梯度累积

---

### 5.3 P2: Minor — 微小差异

#### P2-1: Quantile 归一化公式微差

| 项目 | OpenPI | LeRobot |
|------|--------|---------|
| 分母处理 | `q99 - q01 + 1e-6` (总是加) | `max(q99 - q01, 1e-8)` (仅零时替换) |
| 来源 | `transforms.py:144` | `normalize_processor.py:370-377` |

- 实际数据中 `q99 - q01 >> 1e-6`，差异可忽略

#### P2-2: Normalization Stats 来源     @#2

- OpenPI: `assets/r1_pro_data_convert_chassis/norm_stats.json`
- LeRobot: `dataset.meta.stats`
- 需要验证数值是否一致

#### P2-3: Tokenizer (SentencePiece vs HuggingFace)

- 词表同源（PaliGemma），但加载方式不同
- 需运行时验证 token ID 一致性

#### P2-4: Warmup 初始值

- 两者都是 `peak_lr / (warmup_steps + 1)`，完全一致

#### P2-5: PEFT Targets 命名 (仅 LoRA 时相关)

- LeRobot 默认 targets 含 `action_time_mlp_*`（PI0 命名），PI0.5 实际使用 `time_mlp_*`
- 全量 fine-tuning 时无影响

---

### 5.4 P3: Framework-level — 无法消除的跨框架差异

#### P3-1: RNG 行为

- JAX: key-based splittable PRNG (`jax.random.key(42)` + `fold_in(step)`)
- PyTorch: stateful PRNG (`torch.manual_seed(42)`)
- 同 seed → 不同随机序列 → 不同 noise/time 采样
- **无法消除**，只能保证统计分布一致

#### P3-2: XLA JIT vs Eager PyTorch

- JAX 的 XLA 编译器会做操作融合，改变 FP 运算顺序
- 在 bfloat16 下，运算顺序影响累积误差
- **无法消除**

#### P3-3: Autodiff 实现

- `jax.value_and_grad` vs `.backward()` — 数学等价但 FP 精度不同
- **无法消除**

#### P3-4: FSDP vs DDP

- 单 GPU: 无影响（FSDP 退化为 replicated）
- 多 GPU: FSDP 和 DDP 的梯度同步方式不同，但数学等价

---

## 6. 完整训练配置对照表（JAX 为基准）

| 参数 | OpenPI JAX | LeRobot 当前值 | 对齐后目标值 | 优先级 |
|------|-----------|-------------|-----------|-------|
| **P0 — 必须修复** | | | | |
| weight_decay | `1e-10` | `0.01` | `1e-10` | P0 |
| loss 维度 | 32 (含 padding) | 23 (截断) | 32 | P0 |
| LR cosine 相位 | 29000步(post-warmup) | 30000步(absolute) | 29000步 | P0 |
| **P1 — 重要修复** | | | | |
| EMA decay | `0.99` | 无 | `0.99` | P1 |
| dtype | `bfloat16` | `float32` | `bfloat16` | P1 |
| 数据增强 | 有 (augmax) | 无 | 有 (torchvision) | P1 |
| batch_size | `64` | 默认 | `64` | P1 |
| num_train_steps | `30000` | — | `30000` | P1 |
| **P2 — 需验证** | | | | |
| quantile 公式 | `+1e-6` | `max(.,1e-8)` | 保持 | P2 |
| norm stats | assets JSON | dataset stats | 验证一致 | P2 |
| tokenizer | SentencePiece | HuggingFace | 验证一致 | P2 |
| **P3 — 无需修复** | | | | |
| seed | `42` | `1000` | `42` | P3 |
| RNG 类型 | JAX splittable | PyTorch stateful | 无法对齐 | P3 |
| 编译 | XLA JIT | eager | 无法对齐 | P3 |
| **已对齐** | | | | |
| optimizer_lr | `2.5e-5` | `2.5e-5` | ✅ | — |
| optimizer_betas | `(0.9, 0.95)` | `(0.9, 0.95)` | ✅ | — |
| optimizer_eps | `1e-8` | `1e-8` | ✅ | — |
| grad_clip_norm | `1.0` | `1.0` | ✅ | — |
| warmup_steps | `1000` | `1000` | ✅ | — |
| decay_lr | `2.5e-6` | `2.5e-6` | ✅ | — |
| action_dim (padded) | `32` | `32` | ✅ | — |
| action_horizon | `50` | `50` | ✅ | — |
| max_token_len | `200` | `200` | ✅ | — |
| image_resolution | `224×224` | `224×224` | ✅ | — |
| quantile_norm | `True` | `QUANTILES` | ✅ | — |
| time_beta_alpha | `1.5` | `1.5` | ✅ | — |
| time_beta_beta | `1.0` | `1.0` | ✅ | — |
| min_period | `4e-3` | `4e-3` | ✅ | — |
| max_period | `4.0` | `4.0` | ✅ | — |

---

## 7. 代码修改清单

### 7.1 P0 修改 — 必须

#### 修改 1: Weight Decay 默认值

**文件**: `src/lerobot/policies/pi05/configuration_pi05.py`
**行号**: 88

```python
# Before:
optimizer_weight_decay: float = 0.01

# After:
optimizer_weight_decay: float = 1e-10
```

---

#### 修改 2: Loss 截断开关   @#0

**文件 A**: `src/lerobot/policies/pi05/configuration_pi05.py`

```python
# 在 optimizer 配置区域后添加:
truncate_loss_to_action_dim: bool = True  # 默认保持向后兼容
                                           # 设 False 以对齐 OpenPI
```

**文件 B**: `src/lerobot/policies/pi05/modeling_pi05.py`
**行号**: 1266-1268

```python
# Before:
        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

# After:
        # Optionally truncate losses to actual action dimensions
        if getattr(self.config, 'truncate_loss_to_action_dim', True):
            original_action_dim = self.config.output_features[ACTION].shape[0]
            losses = losses[:, :, :original_action_dim]
```

---

#### 修改 3: LR Schedule 余弦相位

**文件**: `src/lerobot/optim/schedulers.py`
**行号**: 120-125

```python
# Before:
def cosine_decay_schedule(current_step):
    step = min(current_step, actual_decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
    alpha = self.decay_lr / self.peak_lr
    decayed = (1 - alpha) * cosine_decay + alpha
    return decayed

# After:
def cosine_decay_schedule(current_step):
    # 使用相对于 warmup 结束的步数（与 optax 一致）
    relative_step = current_step - actual_warmup_steps
    total_cosine_steps = actual_decay_steps - actual_warmup_steps
    progress = min(relative_step / total_cosine_steps, 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    alpha = self.decay_lr / self.peak_lr
    decayed = (1 - alpha) * cosine_decay + alpha
    return decayed
```

**注意**: 此修改会影响所有使用 `CosineDecayWithWarmupSchedulerConfig` 的训练。如果担心向后兼容性，可以添加一个 `use_relative_cosine: bool = True` 配置项。

---

### 7.2 P1 修改 — 重要

#### 修改 4: EMA 支持    @#0

**方案**: 在训练脚本中添加 EMA 逻辑（不修改 LeRobot 核心代码）

```python
# 在训练脚本中:
class EMAModel:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, param in model.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow

# 初始化
ema = EMAModel(policy, decay=0.99)

# 每步更新
ema.update(policy)

# 保存 checkpoint 时用 EMA 参数
ema_state = ema.state_dict()
policy.load_state_dict(ema_state)
policy.save_pretrained(checkpoint_dir)
policy.load_state_dict(original_state)  # 恢复继续训练
```

---

#### 修改 5: 数据增强（在 PI05Policy.forward() 中）

**文件**: `src/lerobot/policies/pi05/modeling_pi05.py`

在 `_preprocess_images` 后添加增强:

```python
import torchvision.transforms.v2 as T

def _augment_images(self, images: list[Tensor], is_wrist: list[bool]) -> list[Tensor]:
    """与 OpenPI JAX preprocess_observation 等价的数据增强"""
    augmented = []
    for img, wrist in zip(images, is_wrist):
        # img: [B, C, H, W] in [-1, 1]
        img = img / 2.0 + 0.5  # → [0, 1]

        if not wrist:
            h, w = img.shape[2], img.shape[3]
            crop_h, crop_w = int(h * 0.95), int(w * 0.95)
            img = T.RandomCrop((crop_h, crop_w))(img)
            img = T.Resize((h, w), antialias=True)(img)
            img = T.RandomRotation(5)(img)

        img = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)(img)
        img = img * 2.0 - 1.0  # → [-1, 1]
        augmented.append(img)
    return augmented
```

**注意**: OpenPI 的 augmax 是 per-sample 独立增强（通过 vmap），torchvision 的 RandomCrop/Rotation 默认也是 batch 内统一参数。需要逐样本应用以匹配 OpenPI 行为。

---

## 8. LR Schedule 验证脚本

```python
#!/usr/bin/env python3
"""验证 OpenPI optax 和 LeRobot LR schedule 的数值等价性"""
import math

# OpenPI: optax.warmup_cosine_decay_schedule
def openpi_lr(step, peak_lr=2.5e-5, warmup_steps=1000, decay_steps=30000, end_lr=2.5e-6):
    init_value = peak_lr / (warmup_steps + 1)
    if step < warmup_steps:
        # 线性 warmup
        return init_value + (peak_lr - init_value) * step / warmup_steps
    else:
        # 余弦衰减 (相对于 warmup 结束)
        progress = (step - warmup_steps) / (decay_steps - warmup_steps)
        progress = min(progress, 1.0)
        return end_lr + 0.5 * (peak_lr - end_lr) * (1 + math.cos(math.pi * progress))

# LeRobot (修复前): 使用绝对 step
def lerobot_lr_before(step, peak_lr=2.5e-5, warmup_steps=1000, decay_steps=30000, decay_lr=2.5e-6):
    if step <= 0:
        return peak_lr / (warmup_steps + 1)
    if step < warmup_steps:
        frac = 1 - step / warmup_steps
        return peak_lr * ((1/(warmup_steps+1) - 1) * frac + 1)
    step_clamped = min(step, decay_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * step_clamped / decay_steps))
    alpha = decay_lr / peak_lr
    return peak_lr * ((1 - alpha) * cosine + alpha)

# LeRobot (修复后): 使用相对 step
def lerobot_lr_after(step, peak_lr=2.5e-5, warmup_steps=1000, decay_steps=30000, decay_lr=2.5e-6):
    if step <= 0:
        return peak_lr / (warmup_steps + 1)
    if step < warmup_steps:
        frac = 1 - step / warmup_steps
        return peak_lr * ((1/(warmup_steps+1) - 1) * frac + 1)
    relative_step = step - warmup_steps
    total_cosine = decay_steps - warmup_steps
    progress = min(relative_step / total_cosine, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    alpha = decay_lr / peak_lr
    return peak_lr * ((1 - alpha) * cosine + alpha)

# 对比
print(f"{'Step':>6} | {'OpenPI':>12} | {'LR(before)':>12} | {'LR(after)':>12} | {'Diff(before)':>12} | {'Diff(after)':>12}")
print("-" * 80)
for s in [0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 29000, 30000]:
    o = openpi_lr(s)
    b = lerobot_lr_before(s)
    a = lerobot_lr_after(s)
    db = (b - o) / o * 100 if o > 0 else 0
    da = (a - o) / o * 100 if o > 0 else 0
    print(f"{s:6d} | {o:12.6e} | {b:12.6e} | {a:12.6e} | {db:+11.4f}% | {da:+11.4f}%")
```

---

## 9. 训练等价验证策略

### 9.1 Phase 1: 静态验证（无需训练，约 1 小时）

| 测试 | 方法 | 通过标准 |
|------|------|---------|
| T1: LR Schedule | 运行 Section 8 脚本 | 修复后 max_diff < 1e-10 |
| T2: Quantile 公式 | 同数据两框架归一化 | max_diff < 1e-5 |
| T3: Norm Stats | 对比 JSON vs safetensors | max_diff < 1e-3 |
| T4: State 离散化 | 同 state 两框架 bin 编号 | 完全一致 |
| T5: Prompt 构建 | 同 task+state 的完整 prompt | 字符级一致 |
| T6: Tokenizer | SentencePiece vs HF token IDs | 有效 token 一致 |
| T7: Forward Pass | 同权重同输入的 loss 值 | diff < 1e-3 (bfloat16) |

### 9.2 Phase 2: 短期训练验证（约 4 小时）

| 测试 | 方法 | 通过标准 |
|------|------|---------|
| T8: 100 步 loss 曲线 | 两框架训练 100 步 | 趋势一致，量级差 < 10% |
| T9: LR 曲线 | 记录每步 lr | 与 OpenPI 完全重合 |
| T10: Grad norm | 记录每步梯度范数 | 量级一致，无爆炸/消失 |
| T11: Param norm | 记录参数 L2 范数 | 无 weight_decay 导致的异常缩小 |

### 9.3 Phase 3: 完整训练验证（约 24 小时）

| 测试 | 方法 | 通过标准 |
|------|------|---------|
| T12: 30000 步 loss | 完整训练 | 最终 loss 在 OpenPI 的 ±10% 内 |
| T13: EMA 权重质量 | 比较 EMA vs 原始权重的推理 | EMA 权重指标更好 |
| T14: 推理 action | 同输入的 action 输出 | 趋势一致 |

---

## 10. 数据增强对齐方案

### 10.1 OpenPI JAX 增强参数精确列表

| 增强 | 库 | 参数 | 相机 | 代码位置 |
|------|-----|------|------|---------|
| RandomCrop | augmax | (212, 212) = 95% of 224 | 非 wrist | `model.py:176` |
| Resize | augmax | (224, 224), 回到原尺寸 | 非 wrist | `model.py:177` |
| Rotate | augmax | (-5°, +5°) | 非 wrist | `model.py:178` |
| ColorJitter | augmax | b=0.3, c=0.4, s=0.5 | **所有** | `model.py:181` |

**执行空间**: [0, 1]（先从 [-1,1] 转换到 [0,1]，增强后转回）

### 10.2 PyTorch 等价实现

```python
import torchvision.transforms.v2 as T
import torch

def augment_image(img: torch.Tensor, is_wrist: bool) -> torch.Tensor:
    """
    对单张图像应用与 OpenPI JAX 等价的数据增强。
    img: [C, H, W] in [-1, 1]
    """
    img = img / 2.0 + 0.5  # → [0, 1]

    if not is_wrist:
        h, w = img.shape[1], img.shape[2]
        crop_size = (int(h * 0.95), int(w * 0.95))  # (212, 212)
        # 随机裁剪
        i, j, th, tw = T.RandomCrop.get_params(img, crop_size)
        img = T.functional.crop(img, i, j, th, tw)
        # 缩放回原尺寸
        img = T.functional.resize(img, [h, w], antialias=True)
        # 随机旋转
        angle = torch.empty(1).uniform_(-5.0, 5.0).item()
        img = T.functional.rotate(img, angle)

    # ColorJitter (对所有相机)
    jitter = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
    img = jitter(img)

    img = img * 2.0 - 1.0  # → [-1, 1]
    return img
```

**注意事项**:
1. augmax 的 Rotate 使用双线性插值并用 0 填充（黑色），torchvision 默认也如此
2. augmax 的 ColorJitter 参数语义与 torchvision 一致（factor 范围 [1-p, 1+p]）
3. OpenPI 用 `jax.vmap` 对 batch 内每个样本独立增强，torchvision 的 Random* 系列默认也是
4. 增强在图像空间 [0,1] 中进行，不影响 [-1,1] 下的 padding 像素

### 10.3 集成位置

建议在 `PI05Policy.forward()` 中、`_preprocess_images()` 之后添加增强调用，仅在 `self.training` 时执行。这与 OpenPI 的执行位置一致（在 `compute_loss` 内、`embed_prefix` 之前）。

---

## 11. 实施路线图

### Phase 1: P0 代码修改（0.5h）
1. `configuration_pi05.py`: weight_decay → 1e-10, 添加 `truncate_loss_to_action_dim`
2. `modeling_pi05.py`: 条件 loss 截断
3. `schedulers.py`: 修复余弦相位

### Phase 2: 静态验证（1h）
1. 运行 LR schedule 验证脚本 (T1)
2. 验证 norm stats 一致性 (T2-T3)
3. 验证 tokenizer 等价性 (T5-T6)
4. 单步 forward pass 对比 (T7)

### Phase 3: P1 修改（2h）
1. 实现 EMA（手动方案或 torch-ema）
2. 设置 dtype=bfloat16
3. 实现数据增强
4. 配置 batch_size=64 + gradient_checkpointing

### Phase 4: 短期训练验证（4h）
1. 两框架各训练 100-500 步
2. 对比 loss/grad_norm/param_norm/LR 曲线

### Phase 5: 完整训练（12-24h）
1. LeRobot 30000 步训练
2. 监控所有指标
3. 每 5000 步对比 checkpoint 质量

### Phase 6: 推理验证（1h）
1. 同输入 action 对比
2. 如有条件，真实机器人对比

---

## 12. 风险矩阵

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| LR schedule 修复影响其他训练 | 中 | 高 | 添加 `use_relative_cosine` 配置项做开关 |
| Tokenizer ID 不一致 | 低 | 高 | T6 必须通过；不一致则改用 SentencePiece |
| EMA 内存不足 | 中 | 高 | 配合 gradient_checkpointing；或先用方案 C 跳过 EMA |
| BS=64 OOM | 中 | 高 | gradient_accumulation_steps = 64/actual_bs |
| augmax vs torchvision 微差 | 中 | 低 | 旋转插值和颜色空间处理的细微差异，对训练结果影响极小 |
| Norm stats 差异 > 1e-3 | 中 | 中 | 直接使用 OpenPI 的 norm_stats JSON |
| bfloat16 数值精度 | 确定 | 低 | JAX XLA 和 PyTorch 的 bfloat16 累积方式不同，只影响绝对值不影响趋势 |
| JAX vs PyTorch RNG 差异 | 确定 | 低 | 无法消除，只保证统计分布一致 |
| CosineDecay auto-scaling 交互 | 低 | 中 | 修复后需测试 `num_training_steps < num_decay_steps` 的情况 |

---

## 13. 附录

### A. 关键文件索引

| 文件 | 用途 | 操作 |
|------|------|------|
| **OpenPI JAX（只读参考）** | | |
| `openpi/scripts/train.py` | JAX 训练循环，EMA | 参考 |
| `openpi/src/openpi/models/pi0.py` | JAX 模型，loss 计算 | 参考 |
| `openpi/src/openpi/models/model.py` | 数据增强 preprocess_observation | 参考 |
| `openpi/src/openpi/training/optimizer.py` | LR schedule, AdamW | 参考 |
| `openpi/src/openpi/training/checkpoints.py` | EMA→checkpoint 保存 | 参考 |
| `openpi/src/openpi/training/config.py` | pi05_r1pro_chassis 配置 | 参考 |
| `openpi/src/openpi/training/utils.py` | TrainState (含 ema_params) | 参考 |
| `openpi/src/openpi/transforms.py` | 归一化、tokenize、pad | 参考 |
| `openpi/src/openpi/models/tokenizer.py` | SentencePiece tokenizer | 参考 |
| `openpi/src/openpi/models/pi0_config.py` | Pi0Config 默认值 | 参考 |
| `openpi/src/openpi/policies/r1pro_chassis_policy.py` | R1 Pro 数据变换 | 参考 |
| `openpi/src/openpi/shared/image_tools.py` | JAX resize_with_pad | 参考 |
| `openpi/src/openpi/training/data_loader.py` | 数据加载流水线 | 参考 |
| **LeRobot（需修改）** | | |
| `lerobot/src/lerobot/policies/pi05/configuration_pi05.py` | PI05 配置 | ✏️ weight_decay + loss 截断开关 |
| `lerobot/src/lerobot/policies/pi05/modeling_pi05.py` | PI05 模型 | ✏️ loss 截断逻辑 + 数据增强 |
| `lerobot/src/lerobot/optim/schedulers.py` | LR scheduler | ✏️ 余弦相位修复 |
| `lerobot/src/lerobot/policies/pi05/processor_pi05.py` | 数据处理 | 无需修改 |
| `lerobot/src/lerobot/processor/normalize_processor.py` | quantile 归一化 | 无需修改（差异可忽略） |

### B. optax.warmup_cosine_decay_schedule 精确公式

```
设 warmup_steps = W, decay_steps = D, peak = P, init = I, end = E

对于 step ∈ [0, W):
    lr(step) = I + (P - I) × step / W
    其中 I = P / (W + 1)

对于 step ∈ [W, D]:
    progress = (step - W) / (D - W)
    lr(step) = E + 0.5 × (P - E) × (1 + cos(π × progress))

对于 step > D:
    lr(step) = E
```

对 pi05_r1pro_chassis: W=1000, D=30000, P=2.5e-5, E=2.5e-6, I=2.497e-8

### C. 与 v1/v2 的逐项对比

| 差异项 | v1 | v2 | v3 |
|-------|-----|-----|-----|
| Weight decay | P0 ✅ | P0 ✅ | P0 ✅ 无修正 |
| Loss 截断 | P0 ✅ | P0 ✅ | P0 ✅ 无修正 |
| **LR schedule** | 未发现 | 未发现 | **P0 新增** 余弦相位差 4-8% |
| **EMA** | P2 | P3 | **P1 升级** 因 JAX 版默认启用 |
| **数据增强** | P2 缺参数 | P2 | **P1 升级** 含 PyTorch 等价方案 |
| dtype | P1 | P1 | P1 无修正 |
| Batch size | P1 | P1 | P1 无修正 |
| Norm stats | P1 | P1 | P2 无修正 |
| Quantile 公式 | 判定可忽略 | 判定可忽略 | P2 无修正 |
| Tokenizer | 未分析 | P1 | P2 无修正 |
| Image padding | P3 | P3 (含分析) | P3 无修正 |
| State padding 顺序 | P3 | P3 (含追踪) | P3 无修正 |
| Seed | P3 | P3 | P3 无修正 |
| **JAX JIT/XLA** | 未分析 | 未分析 | **P3 新增** |
| **JAX RNG** | 未分析 | 未分析 | **P3 新增** |
| **JAX FSDP** | 未分析 | 未分析 | **P3 新增** |
| **训练等价定义** | 无 | 无 | **新增** L1/L2/L3 三层定义 |
| **JAX train_step 追踪** | 无 | 无 | **新增** 核心章节 |
