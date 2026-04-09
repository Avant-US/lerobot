# LeRobot pi0.5 与 OpenPI pi05_r1pro_chassis 深度对齐方案 (v2)

> **日期**: 2026-04-08
> **目标**: 使 LeRobot 的 pi0.5 fine-tuning 在 R1 Pro chassis 数据上产生与 OpenPI `pi05_r1pro_chassis` 等价的训练结果
> **前置分析**: 基于 `pi05_diffanalyz.md`、`mdldiff.md`、`pi05_alig.md` 的代码级分析，并在其基础上做了更广更深的补充
> **代码库**: OpenPI (`/mnt/r/share/lkx/pi/openpi`), LeRobot (`/home/Luogang/SRC/Robot/lerobot`)

---

## 0. 与 pi05_alig.md (v1) 的主要区别

本文档在 v1 基础上做了以下扩展和修正：

1. **新增数据流水线端到端逐步追踪**（第 3 节），精确到每一步的输入输出形状、数据类型和值域
2. **新增 tokenizer 等价性分析**（第 4.6 节），对比 SentencePiece vs HuggingFace tokenizer 的 token ID 输出
3. **新增图像预处理全链路分析**（第 4.7 节），发现 `resize_with_pad_torch` 存在 clamp 范围差异但对 R1 Pro 无实际影响
4. **新增数据增强详细参数**（第 4.8 节），包括 OpenPI PyTorch 版的完整增强实现
5. **新增 LR scheduler 初始值差异**（第 4.9 节），OpenPI 使用 `peak_lr / (warmup_steps + 1)` 作为 warmup 起始值
6. **新增 PEFT target 不匹配分析**（第 4.10 节），LeRobot 默认 PEFT targets 包含 pi0 残留命名
7. **新增 prompt_from_task 等价性分析**（第 4.11 节），两者提取 prompt 的机制不同
8. **新增 OpenPI PyTorch 训练循环的精确行为**（第 4.12 节），无 autocast、无梯度累积、手动 LR 更新
9. **修正图像 padding 值分析** — v1 标为 P3（不影响），实际分析后确认对 R1 Pro chassis 无影响，但原因更精确
10. **新增归一化-离散化-padding 的精确执行顺序对比**（第 3 节）

---

## 1. OpenPI pi05_r1pro_chassis 完整配置基线

**来源**: `openpi/src/openpi/training/config.py:1024-1042`

```python
TrainConfig(
    name="pi05_r1pro_chassis",
    model=Pi0Config(pi05=True),
    data=SimpleDataConfig(
        repo_id="r1_pro_data_convert_chassis",
        data_transforms=lambda model: Group(
            inputs=[R1ProChassisInputs(model_type=model.model_type)],
            outputs=[R1ProChassisOutputs()],
        ),
        model_transforms=ModelTransformFactory(
            default_prompt="Open the door with a downward-press handle, go through it, and enter the room."
        ),
        base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("actions",)),
    ),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    batch_size=64,
)
```

### 1.1 展开后的完整参数表

| 参数路径 | 值 | 来源 |
|---------|-----|------|
| **模型** | | |
| `model.pi05` | `True` | 显式设置 |
| `model.discrete_state_input` | `True` | 自动跟随 pi05 (`pi0_config.py:41`) |
| `model.action_dim` | `32` | Pi0Config 默认值 |
| `model.action_horizon` | `50` | Pi0Config 默认值 |
| `model.max_token_len` | `200` | pi05=True 时自动设为 200 (`pi0_config.py:39`) |
| `model.dtype` | `"bfloat16"` | Pi0Config 默认值 (`pi0_config.py:20`) |
| `model.paligemma_variant` | `"gemma_2b"` | Pi0Config 默认值 |
| `model.action_expert_variant` | `"gemma_300m"` | Pi0Config 默认值 |
| `model.pytorch_compile_mode` | `"max-autotune"` | Pi0Config 默认值 (`pi0_config.py:35`) |
| **数据** | | |
| `data.repo_id` | `"r1_pro_data_convert_chassis"` | 显式设置 |
| `data.prompt_from_task` | `True` | 显式设置 |
| `data.action_sequence_keys` | `("actions",)` | 显式设置 |
| `data.use_quantile_norm` | `True` | 自动: `model_type != PI0` (`config.py:189`) |
| `data.default_prompt` | `"Open the door with..."` | 显式设置（fallback 用） |
| **优化器** | | |
| `optimizer` (类型) | `AdamW` | TrainConfig 默认值 (`config.py:491`) |
| `optimizer.b1` | `0.9` | AdamW 默认值 (`optimizer.py:69`) |
| `optimizer.b2` | `0.95` | AdamW 默认值 (`optimizer.py:70`) |
| `optimizer.eps` | `1e-8` | AdamW 默认值 (`optimizer.py:71`) |
| `optimizer.weight_decay` | `1e-10` | AdamW 默认值 (`optimizer.py:73`) |
| `optimizer.clip_gradient_norm` | `1.0` | AdamW 默认值 (`optimizer.py:74`) |
| **学习率** | | |
| `lr_schedule` (类型) | `CosineDecaySchedule` | TrainConfig 默认值 (`config.py:490`) |
| `lr_schedule.warmup_steps` | `1000` | CosineDecaySchedule 默认值 (`optimizer.py:19`) |
| `lr_schedule.peak_lr` | `2.5e-5` | CosineDecaySchedule 默认值 (`optimizer.py:20`) |
| `lr_schedule.decay_steps` | `30000` | CosineDecaySchedule 默认值 (`optimizer.py:21`) |
| `lr_schedule.decay_lr` | `2.5e-6` | CosineDecaySchedule 默认值 (`optimizer.py:22`) |
| `lr_schedule.init_value` | `2.5e-5 / 1001 ≈ 2.497e-8` | 计算值 (`optimizer.py:26`) |
| **训练** | | |
| `batch_size` | `64` | 显式设置 |
| `num_train_steps` | `30000` | 显式设置 |
| `seed` | `42` | TrainConfig 默认值 (`config.py:506`) |
| `ema_decay` | `0.99` | TrainConfig 默认值（仅 JAX） |
| `pytorch_training_precision` | `"bfloat16"` | TrainConfig 默认值 (`config.py:488`) |
| `freeze_filter` | `nnx.Nothing()` | 默认值（全量 fine-tuning，不冻结）|
| `num_workers` | `2` | TrainConfig 默认值 (`config.py:511`) |

### 1.2 R1 Pro Chassis 动作空间

**来源**: `openpi/src/openpi/policies/r1pro_chassis_policy.py:1-7`

| 索引 | 维度 | 含义 |
|------|------|------|
| [0:7] | 7 | left_arm 关节角度 |
| [7:14] | 7 | right_arm 关节角度 |
| [14] | 1 | left_gripper |
| [15] | 1 | right_gripper |
| [16:20] | 4 | torso |
| [20:23] | 3 | chassis_velocities (x, y, rotation) |

原始 23 维，模型内部 padding 到 32 维（末尾补 9 个零）。

**R1 Pro chassis 使用绝对动作（非 delta）**：`R1ProChassisInputs` 不包含任何 `DeltaActions` 变换（对比 Aloha 配置中显式使用的 `DeltaActions`），actions 直接透传。

---

## 2. LeRobot PI05Config 当前默认值

**来源**: `lerobot/src/lerobot/policies/pi05/configuration_pi05.py`

| 参数 | LeRobot 默认值 | OpenPI 对应值 | 对齐？ |
|------|--------------|-------------|--------|
| `paligemma_variant` | `"gemma_2b"` | `"gemma_2b"` | ✅ |
| `action_expert_variant` | `"gemma_300m"` | `"gemma_300m"` | ✅ |
| `dtype` | `"float32"` | `"bfloat16"` | ❌ |
| `chunk_size` | `50` | `50` (action_horizon) | ✅ |
| `n_action_steps` | `50` | N/A (客户端控制) | ✅ |
| `max_state_dim` | `32` | `32` (action_dim) | ✅ |
| `max_action_dim` | `32` | `32` (action_dim) | ✅ |
| `tokenizer_max_length` | `200` | `200` | ✅ |
| `num_inference_steps` | `10` | `10` | ✅ |
| `image_resolution` | `(224, 224)` | `(224, 224)` | ✅ |
| `normalization_mapping` | `QUANTILES` for STATE/ACTION | `use_quantile_norm=True` | ✅ |
| `optimizer_lr` | `2.5e-5` | `2.5e-5` (peak_lr) | ✅ |
| `optimizer_betas` | `(0.9, 0.95)` | `(0.9, 0.95)` | ✅ |
| `optimizer_eps` | `1e-8` | `1e-8` | ✅ |
| `optimizer_weight_decay` | **`0.01`** | **`1e-10`** | ❌ **P0** |
| `optimizer_grad_clip_norm` | `1.0` | `1.0` | ✅ |
| `scheduler_warmup_steps` | `1000` | `1000` | ✅ |
| `scheduler_decay_steps` | `30000` | `30000` | ✅ |
| `scheduler_decay_lr` | `2.5e-6` | `2.5e-6` | ✅ |
| `freeze_vision_encoder` | `False` | 无冻结 | ✅ |
| `train_expert_only` | `False` | 无冻结 | ✅ |
| `gradient_checkpointing` | `False` | 可选 | ⚠️ 内存相关 |
| `compile_model` | `False` | `"max-autotune"` | ⚠️ 性能相关 |
| `time_sampling_beta_alpha` | `1.5` | `1.5` (硬编码) | ✅ |
| `time_sampling_beta_beta` | `1.0` | `1.0` (硬编码) | ✅ |
| `min_period` | `4e-3` | `4e-3` | ✅ |
| `max_period` | `4.0` | `4.0` | ✅ |

---

## 3. 数据流水线端到端对比（核心章节）

这是本文档相比 v1 最重要的新增内容。逐步追踪一个 R1 Pro chassis 数据样本在两个框架中的完整处理过程。

### 3.1 OpenPI 数据流水线

```
┌─────────────────────────────────────────────────────────┐
│ Step 0: LeRobot 数据集加载                                │
│ create_torch_dataset() → LeRobotDataset(                │
│   repo_id="r1_pro_data_convert_chassis",                │
│   delta_timestamps={"actions": [0, 1/fps, ..., 49/fps]} │
│ )                                                       │
│ 输出: {                                                  │
│   "head_rgb": uint8[H,W,3],                             │
│   "left_wrist_rgb": uint8[H,W,3],                       │
│   "right_wrist_rgb": uint8[H,W,3],                      │
│   "state": float32[23],                                  │
│   "actions": float32[50,23],                             │
│   "task_index": int,                                     │
│ }                                                       │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: PromptFromLeRobotTask (data_loader.py:149)       │
│ 条件: prompt_from_task=True                              │
│ 行为: task_index → dataset.meta.tasks[task_index] → str  │
│ 输出: data["prompt"] = "具体任务描述"                      │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: R1ProChassisInputs (r1pro_chassis_policy.py:55)  │
│ 行为:                                                    │
│   - _parse_image(): CHW→HWC, float→uint8 if needed      │
│   - head_rgb → image/base_0_rgb (uint8[224,224,3])       │
│   - left_wrist_rgb → image/left_wrist_0_rgb              │
│   - right_wrist_rgb → image/right_wrist_0_rgb            │
│   - state → state (float32[23]) 直接透传                  │
│   - actions → actions (float32[50,23]) 直接透传            │
│   - image_mask: all True                                 │
│ 输出: {"image": {3 cameras}, "state": [23],              │
│        "actions": [50,23], "prompt": str}                │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Normalize (transforms.py:126, quantile模式)       │
│ 行为:                                                    │
│   - 加载 assets/r1_pro_data_convert_chassis/norm_stats    │
│   - state: (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0  │
│   - actions: 同上公式                                     │
│   - image: 不归一化（norm_stats 中无 image key）           │
│ 关键: 归一化 ONLY 对 state 和 actions，不对 image          │
│ 输出: state ∈ [-1,1], actions ∈ [-1,1], image 仍是 uint8  │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: InjectDefaultPrompt (transforms.py:108)          │
│ 行为: 如果 data 中无 "prompt" key，注入 default_prompt    │
│ 对 R1 Pro: prompt_from_task=True 已提供 prompt，此步跳过  │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: ResizeImages(224, 224) (transforms.py:189)       │
│ 行为: resize_with_pad(image, 224, 224)                   │
│   - image 此时仍为 uint8                                 │
│   - 若已是 224x224 → 不做任何操作                         │
│   - 若不是 → 缩放保持宽高比，uint8 padding=0 (黑色)       │
│ 对 R1 Pro: 图像通常已是 224x224，此步为 no-op             │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 6: TokenizePrompt (transforms.py:252)               │
│ 行为:                                                    │
│   discrete_state_input=True (pi05 模式):                 │
│   1. cleaned_text = prompt.strip().replace("_"," ").     │
│      replace("\n"," ")                                   │
│   2. discretized = np.digitize(state,                    │
│      bins=np.linspace(-1,1,257)[:-1]) - 1                │
│      → 256 bins, 每个 state 维度映射到 0-255 整数         │
│   3. state_str = " ".join(map(str, discretized))         │
│   4. full_prompt = f"Task: {cleaned_text}, State:        │
│      {state_str};\nAction: "                             │
│   5. tokens = sentencepiece.encode(full_prompt,           │
│      add_bos=True)                                       │
│   6. pad/truncate 到 max_len=200                         │
│                                                          │
│ 输出: tokenized_prompt: int32[200],                      │
│       tokenized_prompt_mask: bool[200]                    │
│ 注意: state 在这里被消费（编码进 prompt tokens）           │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 7: PadStatesAndActions(32) (transforms.py:328)      │
│ 行为:                                                    │
│   - state: [23] → pad_to_dim → [32] (末尾补 9 个 0)     │
│   - actions: [50,23] → pad_to_dim → [50,32]             │
│ 注意: state padding 发生在 tokenization 之后              │
│       tokenization 使用的是未 padding 的 23-dim state     │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 8: 批量拼接 (_collate_fn, data_loader.py:475)       │
│ 行为: np.stack → numpy batch                             │
│ 输出: {                                                  │
│   "image/base_0_rgb": uint8[B,224,224,3],                │
│   "image/left_wrist_0_rgb": uint8[B,224,224,3],          │
│   "image/right_wrist_0_rgb": uint8[B,224,224,3],         │
│   "image_mask/...": bool[B],                             │
│   "state": float32[B,32],                                │
│   "actions": float32[B,50,32],                           │
│   "tokenized_prompt": int32[B,200],                      │
│   "tokenized_prompt_mask": bool[B,200],                  │
│ }                                                       │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 9: Observation.from_dict() (model.py:110)           │
│ 行为:                                                    │
│   - uint8 image → float32: image / 255.0 * 2.0 - 1.0    │
│     即映射到 [-1, 1]                                     │
│   - PyTorch: 同时做 permute [B,H,W,C] → [B,C,H,W]       │
│ 输出: Observation 对象                                    │
│   images: float32[B,C,H,W] ∈ [-1,1]                     │
│   state: float32[B,32] ∈ [-1,1]（前 23 维）+ 0（后 9 维）│
│   tokenized_prompt: int32[B,200]                         │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 10: preprocess_observation_pytorch()                 │
│ (preprocessing_pytorch.py:20)                            │
│ 行为:                                                    │
│   - 若 image 尺寸不是 224x224 → resize_with_pad_torch    │
│     (此处 image 已在 [-1,1]，clip 到 [-1,1])             │
│   - 若 train=True → 数据增强:                            │
│     a. 转到 [0,1]: image / 2.0 + 0.5                    │
│     b. 非 wrist 相机: RandomCrop(95%) + Resize + Rotate  │
│     c. 所有相机: ColorJitter(b=0.3, c=0.4, s=0.5)        │
│     d. 转回 [-1,1]: image * 2.0 - 1.0                   │
│ 输出: 增强后的 images ∈ [-1,1]                            │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 11: PI0Pytorch.forward() (pi0_pytorch.py:317)       │
│ 行为:                                                    │
│   1. noise ~ N(0,1), shape [B,50,32]                     │
│   2. time ~ Beta(1.5, 1.0) * 0.999 + 0.001              │
│   3. x_t = time * noise + (1-time) * actions             │
│   4. u_t = noise - actions (target velocity)             │
│   5. embed_prefix(images, tokenized_prompt)              │
│   6. embed_suffix(x_t, time)                             │
│      - sinusoidal_pos_emb(time)                          │
│      - time_mlp_in → SiLU → time_mlp_out → SiLU         │
│      - action_in_proj(x_t)                               │
│      - adarms_cond = time_emb                            │
│   7. PaliGemmaWithExpert forward (18 层)                  │
│   8. v_t = action_out_proj(suffix_out[-50:])             │
│   9. loss = MSE(u_t, v_t, reduction="none")              │
│      → [B, 50, 32]                                       │
│  10. total_loss = loss.mean() → scalar                   │
│ 关键: loss 在全部 32 维上计算（包含 padding 维）           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 LeRobot 数据流水线

```
┌─────────────────────────────────────────────────────────┐
│ Step 0: LeRobot 数据集加载                                │
│ LeRobotDataset(                                          │
│   repo_id="r1_pro_data_convert_chassis",                 │
│   delta_timestamps={"action": [0, 1/fps, ..., 49/fps]}  │
│ )                                                        │
│ 注意: key 是 "action" (单数)，非 OpenPI 的 "actions"       │
│ 输出: {                                                  │
│   "observation.images.head_rgb": float32[3,H,W] ∈ [0,1],│
│   "observation.images.left_wrist_rgb": float32[3,H,W],  │
│   "observation.images.right_wrist_rgb": float32[3,H,W], │
│   "observation.state": float32[23],                      │
│   "action": float32[50,23],                              │
│   "task": str,                                           │
│ }                                                       │
│ 注意: 图像已是 float32 [0,1]，CHW 格式                    │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: RenameObservationsProcessorStep (空映射)          │
│ 行为: 无操作（兼容性占位）                                │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: AddBatchDimensionProcessorStep                    │
│ 行为: 所有 tensor 增加 batch 维度                         │
│ 输出: state: [1,23], action: [1,50,23], images: [1,3,H,W]│
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: NormalizerProcessorStep (processor_pi05.py:134)  │
│ 行为:                                                    │
│   - VISUAL: IDENTITY (不归一化)                           │
│   - STATE: QUANTILES → 2*(x-q01)/max(q99-q01,1e-8) - 1  │
│   - ACTION: QUANTILES → 同上                             │
│ 注意: 图像保持 [0,1]，不做归一化                           │
│       state/action 归一化到 [-1,1]                        │
│ 与 OpenPI 的区别:                                         │
│   - 公式微差: max(denom,1e-8) vs (denom+1e-6)            │
│   - 实际数据中 denom >> 1e-6，差异可忽略                   │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: Pi05PrepareStateTokenizerProcessorStep           │
│ (processor_pi05.py:57)                                   │
│ 行为:                                                    │
│   1. state_np = state.cpu().numpy()  (23 维，已归一化)    │
│   2. discretized = np.digitize(state_np,                 │
│      bins=np.linspace(-1,1,257)[:-1]) - 1                │
│   3. cleaned_text = task.strip().replace("_"," ").       │
│      replace("\n"," ")                                   │
│   4. state_str = " ".join(map(str, discretized[i]))      │
│   5. full_prompt = f"Task: {cleaned_text}, State:        │
│      {state_str};\nAction: "                             │
│   6. 存入 complementary_data["task"]                     │
│ 关键对比:                                                 │
│   - prompt 模板与 OpenPI 完全一致 ✓                       │
│   - 离散化逻辑与 OpenPI 完全一致 ✓                        │
│   - 使用的是未 padding 的 23-dim state ✓                  │
│     (与 OpenPI 一致: TokenizePrompt 在 Pad 之前)          │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: TokenizerProcessorStep (processor_pi05.py:140)   │
│ 行为:                                                    │
│   - tokenizer_name="google/paligemma-3b-pt-224"          │
│   - max_length=200, padding_side="right",                │
│     padding="max_length"                                 │
│   - 加载 HuggingFace AutoTokenizer                       │
│   - 对 full_prompt 做 tokenization → token IDs           │
│ 输出:                                                    │
│   observation.language_tokens: int[1,200]                 │
│   observation.language_attention_mask: int[1,200]         │
│ 关键问题:                                                 │
│   OpenPI 用 SentencePiece (paligemma_tokenizer.model)    │
│   LeRobot 用 HuggingFace (google/paligemma-3b-pt-224)   │
│   → 词表同源，但加载方式不同，token ID 需验证等价         │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 6: DeviceProcessorStep                              │
│ 行为: 所有 tensor 移到目标设备 (GPU)                      │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 7: 训练 DataLoader 批量拼接                          │
│ (由 PyTorch DataLoader 完成)                              │
│ 输出: {                                                  │
│   "observation.images.head_rgb": float32[B,3,224,224],   │
│   "observation.state": float32[B,23],                    │
│   "observation.language_tokens": int[B,200],             │
│   "observation.language_attention_mask": int[B,200],     │
│   "action": float32[B,50,23],                            │
│ }                                                       │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 8: PI05Policy._preprocess_images()                  │
│ (modeling_pi05.py:1148)                                  │
│ 行为:                                                    │
│   - 图像此时是 float32 [0,1]，CHW 格式                    │
│   - permute [B,C,H,W] → [B,H,W,C]                       │
│   - 若尺寸不是 224x224 → resize_with_pad_torch            │
│     (此处 clamp 到 [0,1]，pad 用 0.0)                     │
│   - 归一化: img * 2.0 - 1.0 → [-1,1]                    │
│   - permute 回 [B,C,H,W]                                │
│ 关键:                                                     │
│   - 无数据增强（LeRobot 默认不做训练时增强）              │
│   - 有效像素: [0,1] → [-1,1] ✓ 与 OpenPI 一致           │
│   - padding 像素: 0.0 → 0*2-1 = -1.0 ✓ 与 OpenPI 一致   │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 9: PI05Policy.forward() (modeling_pi05.py:1248)     │
│ 行为:                                                    │
│   1. prepare_action: pad_vector(action, 32)              │
│      → [B,50,23] → [B,50,32]                            │
│   2. self.model.forward(images, img_masks, tokens,       │
│      masks, actions)                                     │
│      (同 OpenPI 的 flow matching 逻辑)                    │
│   3. losses: [B,50,32]                                   │
│   4. ★截断★: losses = losses[:,:,:23]                    │
│      (modeling_pi05.py:1267-1268)                        │
│   5. loss = losses.mean() → scalar                       │
│ 关键差异: loss 只在 23 维上计算，不含 padding 维           │
└─────────────────────────────────────────────────────────┘
```

### 3.3 流水线顺序对比总结

| 处理步骤 | OpenPI 顺序 | LeRobot 顺序 | 一致？ |
|---------|-----------|-------------|--------|
| 数据集加载 | LeRobotDataset | LeRobotDataset | ✅ |
| Prompt 提取 | PromptFromLeRobotTask | 数据集直接提供 task | ⚠️ 需验证 |
| 相机映射 | R1ProChassisInputs | config.input_features | ✅ 语义等价 |
| 归一化 | Normalize (quantile) | NormalizerProcessorStep | ✅ 公式微差可忽略 |
| State 离散化 | TokenizePrompt (内含) | Pi05PrepareStateTokenizer | ✅ 逻辑一致 |
| Tokenization | SentencePiece | HuggingFace AutoTokenizer | ⚠️ 需验证 |
| State/Action Padding | PadStatesAndActions(32) | pad_vector in forward() | ✅ 时序一致 |
| 图像转 float32 | Observation.from_dict | DataLoader (已是 float32) | ✅ 最终值域一致 |
| 图像→[-1,1] | uint8/255*2-1 | float32*2-1 | ✅ 数值等价 |
| 数据增强 | 有 (train=True) | 无 | ❌ **差异** |
| Loss 维度 | 32 维 (含 padding) | 23 维 (截断) | ❌ **P0 差异** |

---

## 4. 差异清单与修复方案

### 4.1 P0: Weight Decay (1e-10 vs 0.01) — ⚠️ 必须修复

**OpenPI**: `optimizer.py:73` → `weight_decay: float = 1e-10`
**LeRobot**: `configuration_pi05.py:88` → `optimizer_weight_decay: float = 0.01`

**影响**: 这是**最大的超参数差异**。OpenPI 代码注释明确说："Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value." 即 OpenPI 实质上不使用 weight decay。LeRobot 的 0.01 是标准 L2 正则化，会持续缩小参数 norm，改变 loss landscape 和泛化特性。

**修复**:
```python
# 在训练配置中覆盖
config.optimizer_weight_decay = 1e-10
```

或修改 `configuration_pi05.py:88` 的默认值为 `1e-10`。

---

### 4.2 P0: Loss 截断 (32 维 vs 23 维) — ⚠️ 必须修复

**OpenPI**: `pi0.py:214` / `pi0_pytorch.py:374`
```python
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
return F.mse_loss(u_t, v_t, reduction="none")  # [B, 50, 32] — 全 32 维
# 然后由调用方 loss.mean() 对全部维度求均值
```

**LeRobot**: `modeling_pi05.py:1266-1268`
```python
original_action_dim = self.config.output_features[ACTION].shape[0]  # 23
losses = losses[:, :, :original_action_dim]  # [B, 50, 32] → [B, 50, 23]
```

**影响**:
1. **梯度差异**: OpenPI 的 `action_out_proj` 权重矩阵全部 32 列都接收梯度；LeRobot 的 [23:32] 列不接收梯度
2. **Loss 数值差异**: padding 维的 target 是 0，通常较"容易"，OpenPI 的 per-dim loss 被这些维度拉低
3. **隐式约束差异**: OpenPI 训练后模型学会在 padding 维输出接近 0，这是额外的正则化信号

**修复**: 在 `modeling_pi05.py` 中添加开关，训练 R1 Pro chassis 时禁用截断：

```python
# configuration_pi05.py — 添加配置项
truncate_loss_to_action_dim: bool = True  # 默认保持向后兼容

# modeling_pi05.py:1266-1268 — 条件截断
if getattr(self.config, 'truncate_loss_to_action_dim', True):
    original_action_dim = self.config.output_features[ACTION].shape[0]
    losses = losses[:, :, :original_action_dim]
```

对齐训练时设 `truncate_loss_to_action_dim=False`。

---

### 4.3 P1: 训练精度 (bfloat16 vs float32)

**OpenPI**: `config.py:488` → `pytorch_training_precision: "bfloat16"`
**LeRobot**: `configuration_pi05.py:34` → `dtype: "float32"`（默认值，非运行时值）

**影响**: bfloat16 训练速度快 ~2x，内存省 ~50%。pi05_base 权重是 bfloat16 预训练的，fine-tuning 也应使用 bfloat16。

**关键细节**: OpenPI PyTorch 训练**不使用 `torch.autocast`**。它直接设置 `model_cfg.dtype = config.pytorch_training_precision`，使模型参数本身存储为 bfloat16，所有计算在 bfloat16 下进行（见 `train_pytorch.py:407`）。

**修复**: 训练时设 `dtype="bfloat16"`。

---

### 4.4 P1: Batch Size 和步数

**OpenPI**: `batch_size=64, num_train_steps=30000`
**LeRobot**: 默认 `batch_size=8`（通常由训练脚本设置）

**影响**: Batch size 直接影响梯度估计的方差和有效学习率。OpenPI PyTorch 训练**不使用梯度累积**（每步直接 `loss.backward() → optim.step()`）。

**修复**: 使用 `batch_size=64`，若单 GPU 内存不足，使用 `gradient_accumulation_steps = 64 / actual_batch_size`。注意 OpenPI 原生不做梯度累积，所以用梯度累积模拟大 batch 可能有微小差异（BN 层行为等），但 pi05 不含 BN，差异可忽略。

---

### 4.5 P1: Normalization Stats 来源

**OpenPI**: 从 `assets/r1_pro_data_convert_chassis/norm_stats.json` 加载
**LeRobot**: 从 `dataset.meta.stats` 加载

**关键问题**: 两者计算 quantile 的方式和精度是否一致？

**验证脚本**:
```python
import json, torch, numpy as np

# 加载 OpenPI stats
with open("assets/r1_pro_data_convert_chassis/norm_stats.json") as f:
    openpi_stats = json.load(f)

# 加载 LeRobot stats
from safetensors.torch import load_file
lerobot_stats = load_file("path/to/dataset/meta/stats.safetensors")

# 对比
for openpi_key, lerobot_key in [("state", "observation.state"), ("actions", "action")]:
    for stat in ["q01", "q99", "mean", "std"]:
        openpi_val = np.array(openpi_stats[openpi_key][stat])
        lerobot_val = lerobot_stats[f"{lerobot_key}/{stat}"].numpy()
        diff = np.max(np.abs(openpi_val - lerobot_val))
        print(f"{openpi_key}.{stat} max diff: {diff:.8f}")
```

如果差异 < 1e-3，可直接使用 LeRobot stats。否则需要将 OpenPI stats 转换后传入。

---

### 4.6 P1: Tokenizer 等价性 — SentencePiece vs HuggingFace

**OpenPI**: `tokenizer.py:18-20`
```python
path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model")
self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
tokens = self._tokenizer.encode(full_prompt, add_bos=True)
```

**LeRobot**: `processor_pi05.py:140-145`
```python
TokenizerProcessorStep(
    tokenizer_name="google/paligemma-3b-pt-224",
    max_length=200,
    padding_side="right",
    padding="max_length",
)
```

**分析**:
- `google/paligemma-3b-pt-224` 的 HuggingFace tokenizer 内部也使用 SentencePiece，词表来源相同
- 但需要验证两个关键点：
  1. **BOS token**: OpenPI 显式 `add_bos=True`，HuggingFace tokenizer 通常也自动添加 BOS
  2. **Token ID 一致性**: 同一字符串是否产生相同的 token ID 序列
  3. **Padding 策略**: OpenPI 用 `False`(=0) padding，LeRobot 用 `pad_token_id` padding

**验证脚本**:
```python
import sentencepiece, transformers

# OpenPI tokenizer
sp = sentencepiece.SentencePieceProcessor()
sp.Load("paligemma_tokenizer.model")
openpi_tokens = sp.encode("Task: pick up, State: 128 64;\nAction: ", add_bos=True)

# LeRobot tokenizer
hf_tok = transformers.AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
hf_result = hf_tok("Task: pick up, State: 128 64;\nAction: ", 
                    return_tensors="pt", max_length=200, padding="max_length")
lerobot_tokens = hf_result["input_ids"][0].tolist()

# 对比非 padding 部分
openpi_len = len(openpi_tokens)
print(f"OpenPI tokens ({openpi_len}): {openpi_tokens}")
print(f"LeRobot tokens (first {openpi_len}): {lerobot_tokens[:openpi_len]}")
assert openpi_tokens == lerobot_tokens[:openpi_len], "Token mismatch!"
```

**风险**: 如果 HuggingFace tokenizer 的 BOS/padding 行为与 SentencePiece 不完全一致，会导致 prompt 编码不同，进而影响模型输出。这是一个必须在实验前验证的项目。

---

### 4.7 P2: 图像预处理链路

**结论**: 对 R1 Pro chassis **无实际影响**，但需要理解原因。

**OpenPI 链路**:
1. 图像始终以 uint8 处理到 `Observation.from_dict`
2. `from_dict`: `uint8 / 255.0 * 2.0 - 1.0` → float32 [-1, 1]
3. `resize_with_pad` (若需要): float32 clamp [-1, 1], pad -1.0
4. 增强: [-1,1] → [0,1] → 增强 → [0,1] → [-1,1]

**LeRobot 链路**:
1. 图像从 dataset 加载就是 float32 [0, 1], CHW 格式
2. `_preprocess_images`: permute CHW → HWC
3. `resize_with_pad_torch` (若需要): float32 clamp [0, 1], pad 0.0
4. 归一化: `img * 2.0 - 1.0` → [-1, 1]
5. permute 回 CHW

**两个 `resize_with_pad_torch` 的区别**:
- OpenPI (`image_tools.py:101`): `clamp(-1.0, 1.0)`, pad `-1.0`
- LeRobot (`modeling_pi05.py:199`): `clamp(0.0, 1.0)`, pad `0.0`

这不是 bug，而是因为两处的输入值域不同：
- OpenPI 此时图像已在 [-1, 1]
- LeRobot 此时图像在 [0, 1]，之后才转到 [-1, 1]

**最终结果**:
- 有效像素: 两者都在 [-1, 1]，数值等价
- Padding 像素: OpenPI 直接 -1.0；LeRobot 0.0 经 `*2-1` 变为 -1.0。等价。
- **前提**: 图像已经是 224x224（R1 Pro 通常如此），则 resize_with_pad 不被调用，完全无差异。

---

### 4.8 P2: 数据增强

**OpenPI PyTorch 实现** (`preprocessing_pytorch.py:52-142`):

| 增强类型 | 参数 | 应用范围 |
|---------|------|---------|
| RandomCrop | 95% 宽高 | 非 wrist 相机 (base_0_rgb) |
| Resize 回原尺寸 | bilinear | 非 wrist 相机 |
| Random Rotate | ±5° | 非 wrist 相机 |
| Brightness | factor ∈ [0.7, 1.3] | 所有相机 |
| Contrast | factor ∈ [0.6, 1.4] | 所有相机 |
| Saturation | factor ∈ [0.5, 1.5] | 所有相机 |

**LeRobot**: 默认不做数据增强。

**影响**: 数据增强主要影响泛化能力而非训练收敛。对齐 loss 曲线时可以先不加增强（两框架在不增强时更容易对齐）。部署性能可能受影响。

**修复优先级**: 低。先完成核心对齐验证后再添加。

---

### 4.9 P2: LR Scheduler 初始值

**OpenPI**: `optimizer.py:26`
```python
init_value=self.peak_lr / (self.warmup_steps + 1)  # 2.5e-5 / 1001 ≈ 2.497e-8
```

**LeRobot**: 需要检查 `CosineDecayWithWarmupSchedulerConfig` 的实现是否使用相同的初始值。

如果 LeRobot 的 warmup 从 0 开始而非 `peak_lr / (warmup_steps + 1)` 开始，前几步的学习率会有微小差异。差异量级约 `2.5e-8`，实际影响可忽略。

---

### 4.10 P2: PEFT/LoRA Target Mismatch

**LeRobot 默认 PEFT targets** (`modeling_pi05.py` 中 `_get_default_peft_targets()`):
```python
target_modules = r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|
                     model\.(state_proj|action_in_proj|action_out_proj|
                     action_time_mlp_in|action_time_mlp_out))"
```

**问题**: 包含 `state_proj` 和 `action_time_mlp_*`，但 PI0.5 实际使用的是 `time_mlp_in` 和 `time_mlp_out`，且没有 `state_proj`。

**影响**: 如果做全量 fine-tuning（`pi05_r1pro_chassis` 的默认行为），此差异无影响。仅在使用 LoRA fine-tuning 时才会导致目标模块匹配失败。

**修复**: 若计划使用 LoRA，需修正 PEFT targets:
```python
target_modules = r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|
                     model\.(action_in_proj|action_out_proj|
                     time_mlp_in|time_mlp_out))"
```

---

### 4.11 P2: prompt_from_task 等价性

**OpenPI**: `data_loader.py:148-149`
```python
if data_config.prompt_from_task:
    dataset = TransformedDataset(dataset, [PromptFromLeRobotTask(dataset_meta.tasks)])
```
`PromptFromLeRobotTask` (`transforms.py:310-324`) 从 `data["task_index"]` 查 `dataset.meta.tasks` 字典得到 prompt 字符串。

**LeRobot**: 直接从数据集的 `"task"` 字段读取字符串。

**分析**: 两者最终都会得到相同的 prompt 字符串，因为 LeRobot 数据集的 `task` 字段就是 `tasks[task_index]` 的解引用结果。等价。

---

### 4.12 P2: OpenPI PyTorch 训练循环精确行为

根据 `train_pytorch.py:509-556` 的精确分析：

```python
# 关键行为：
for observation, actions in loader:
    # 1. 手动设置 LR（非 PyTorch scheduler）
    for pg in optim.param_groups:
        pg["lr"] = lr_schedule(global_step)  # optax CosineDecaySchedule
    
    # 2. Forward — 无 autocast wrapper
    losses = model(observation, actions)     # [B, 50, 32]
    loss = losses.mean()                     # scalar
    
    # 3. Backward — 无 GradScaler
    loss.backward()
    
    # 4. Gradient clipping — clip THEN step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 5. Optimizer step
    optim.step()
    optim.zero_grad(set_to_none=True)
```

**与 LeRobot 的关键区别**:
1. **无 `torch.autocast`**: OpenPI 在 bfloat16 下直接计算，不使用混合精度。模型参数本身是 bfloat16。
2. **无梯度累积**: 每个 micro-batch 都是一个完整的训练步。
3. **手动 LR 更新**: 不使用 `scheduler.step()`，而是直接设 `param_groups["lr"]`。
4. **Gradient clipping 在 optim.step() 之前**: 这与标准 PyTorch 实践一致。

---

### 4.13 P3: EMA

**OpenPI JAX**: `ema_decay=0.99`，每步更新 EMA 参数
**OpenPI PyTorch**: **不支持**（`train_pytorch.py:499` 明确 log "EMA is not supported"）
**LeRobot**: 不支持

**结论**: 与 OpenPI PyTorch 对齐时，EMA 差异不存在。与 JAX 对齐时，需额外实现 EMA。建议先与 PyTorch 对齐。

---

### 4.14 P3: torch.compile

**OpenPI**: `pytorch_compile_mode="max-autotune"`，编译 `sample_actions`（推理时使用）
**LeRobot**: 默认 `compile_model=False`

**影响**: `torch.compile` 仅影响推理速度，不影响训练行为和模型权重。训练对齐无需关注此项。

---

### 4.15 P3: Seed

**OpenPI**: `seed=42`
**LeRobot**: `seed=1000`（默认）

**修复**: 训练时设 `seed=42`。但 JAX 和 PyTorch 的随机数生成器即使 seed 相同也不会产生相同序列，因此 loss 曲线只能在趋势和量级上对齐。

---

## 5. 完整训练配置对照表

| 参数 | OpenPI `pi05_r1pro_chassis` | LeRobot 需设置的值 | 默认对齐？ | 优先级 |
|------|---------------------------|-------------------|-----------|-------|
| weight_decay | **1e-10** | `1e-10` | ❌ | P0 |
| loss 截断 | **不截断 (32-dim)** | `truncate_loss_to_action_dim=False` | ❌ | P0 |
| dtype | **bfloat16** | `"bfloat16"` | ❌ | P1 |
| batch_size | **64** | `64` | ❌ 需设置 | P1 |
| num_train_steps | **30000** | `30000` | ❌ 需设置 | P1 |
| norm_stats | **asset 文件** | 验证后使用 dataset stats | ⚠️ 需验证 | P1 |
| tokenizer | SentencePiece | HuggingFace | ⚠️ 需验证 | P1 |
| action_dim | 32 | `max_action_dim=32` | ✅ | — |
| action_horizon | 50 | `chunk_size=50` | ✅ | — |
| max_token_len | 200 | `tokenizer_max_length=200` | ✅ | — |
| optimizer_lr | 2.5e-5 | `optimizer_lr=2.5e-5` | ✅ | — |
| optimizer_betas | (0.9, 0.95) | `optimizer_betas=(0.9, 0.95)` | ✅ | — |
| optimizer_eps | 1e-8 | `optimizer_eps=1e-8` | ✅ | — |
| optimizer_clip | 1.0 | `optimizer_grad_clip_norm=1.0` | ✅ | — |
| warmup_steps | 1000 | `scheduler_warmup_steps=1000` | ✅ | — |
| decay_steps | 30000 | `scheduler_decay_steps=30000` | ✅ | — |
| decay_lr | 2.5e-6 | `scheduler_decay_lr=2.5e-6` | ✅ | — |
| quantile_norm | True | `QUANTILES` | ✅ | — |
| image_resolution | 224x224 | `(224, 224)` | ✅ | — |
| freeze | 无 | `False / False` | ✅ | — |
| seed | 42 | `seed=42` | ❌ 需设置 | P3 |
| 数据增强 | 有 | 无 | ❌ | P2 |
| EMA | JAX only | 不支持 | ⚠️ PyTorch 无影响 | P3 |
| compile | max-autotune | 不编译 | N/A | P3 |

---

## 6. 代码修改清单

### 6.1 必须修改 — `configuration_pi05.py`

**修改 1: weight_decay 默认值 (P0)**
```python
# 文件: src/lerobot/policies/pi05/configuration_pi05.py
# 行 88
# Before:
optimizer_weight_decay: float = 0.01
# After:
optimizer_weight_decay: float = 1e-10
```

**修改 2: 添加 loss 截断开关 (P0)**
```python
# 在 optimizer/scheduler 配置区域后添加:
truncate_loss_to_action_dim: bool = True  # If True, truncate loss to actual action dim
                                           # Set False for OpenPI-aligned training
```

### 6.2 必须修改 — `modeling_pi05.py`

**修改 3: 支持 loss 截断开关 (P0)**
```python
# 文件: src/lerobot/policies/pi05/modeling_pi05.py
# 行 1266-1268
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

### 6.3 可选修改 — PEFT targets

```python
# 文件: src/lerobot/policies/pi05/modeling_pi05.py
# _get_default_peft_targets() 方法中
# 将 action_time_mlp_in|action_time_mlp_out 改为 time_mlp_in|time_mlp_out
# 移除 state_proj
```

---

## 7. 训练脚本模板

```python
#!/usr/bin/env python
"""
R1 Pro Chassis pi0.5 fine-tuning script, aligned with OpenPI pi05_r1pro_chassis.
"""
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

# ─── 配置 (对齐 OpenPI pi05_r1pro_chassis) ───

config = PI05Config(
    # 模型架构
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    dtype="bfloat16",
    chunk_size=50,
    n_action_steps=50,
    max_state_dim=32,
    max_action_dim=32,
    tokenizer_max_length=200,
    num_inference_steps=10,

    # 归一化
    normalization_mapping={
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,
        "ACTION": NormalizationMode.QUANTILES,
    },

    # 优化器 — weight_decay 对齐!
    optimizer_lr=2.5e-5,
    optimizer_betas=(0.9, 0.95),
    optimizer_eps=1e-8,
    optimizer_weight_decay=1e-10,  # OpenPI 默认值
    optimizer_grad_clip_norm=1.0,

    # 调度器
    scheduler_warmup_steps=1_000,
    scheduler_decay_steps=30_000,
    scheduler_decay_lr=2.5e-6,

    # 训练设置
    gradient_checkpointing=True,
    freeze_vision_encoder=False,
    train_expert_only=False,

    # Loss 截断 — 对齐 OpenPI 全维度 loss
    truncate_loss_to_action_dim=False,

    # Features
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

# ─── 加载预训练权重 ───
policy = PI05Policy.from_pretrained("lerobot/pi05_base", config=config)

# ─── 训练参数 ───
# batch_size = 64  (或 actual_bs=8, gradient_accumulation_steps=8)
# steps = 30_000
# seed = 42
# dataset_repo_id = "r1_pro_data_convert_chassis"
```

---

## 8. 验证策略

### 8.1 第一阶段: 静态验证（无需训练）

| 测试 | 验证内容 | 预期结果 |
|------|---------|---------|
| T1: 归一化对齐 | quantile 公式数值对比 | max_diff < 1e-6 |
| T2: Norm stats 对比 | OpenPI vs LeRobot q01/q99 | max_diff < 1e-3 |
| T3: State 离散化 | 256-bin 输出对比 | 完全一致 |
| T4: Prompt 构建 | 完整 prompt 字符串对比 | 字符级一致 |
| T5: **Tokenizer 对比** | SentencePiece vs HF token IDs | 有效 token 一致 |
| T6: 前向传播 | 同权重同输入的 loss 对比 | diff < 1e-4 (float32) |

### 8.2 第二阶段: 训练验证

| 测试 | 验证内容 | 预期结果 |
|------|---------|---------|
| T7: 100 步 loss 曲线 | 两框架 loss 趋势对比 | 同一量级，同一趋势 |
| T8: 参数 norm 监控 | weight_decay 影响 | 无明显 norm 衰减 |
| T9: 梯度 norm | 训练稳定性 | 无爆炸/消失 |
| T10: LR 曲线 | scheduler 行为 | 两框架 LR 曲线重合 |

### 8.3 第三阶段: 推理验证

| 测试 | 验证内容 | 预期结果 |
|------|---------|---------|
| T11: 推理 action 对比 | 同 checkpoint 同输入的 action 输出 | diff < 1e-2 (bfloat16) |
| T12: 真实部署 | 机器人上执行 | 行为一致 |

---

## 9. 实施路线图

### Phase 1: 代码修改 (0.5h)
1. 修改 `configuration_pi05.py`: weight_decay + loss 截断开关
2. 修改 `modeling_pi05.py`: 条件 loss 截断

### Phase 2: 静态验证 (2h)
1. 运行 T1-T5 (归一化、norm stats、离散化、prompt、tokenizer)
2. 运行 T6 (前向传播对比)
3. 确认所有静态测试通过

### Phase 3: 训练配置 (1h)
1. 创建 R1 Pro chassis 训练脚本
2. 确认数据集加载和 feature 映射
3. 验证 norm stats 一致性

### Phase 4: 短期训练验证 (4h)
1. 两框架各训练 100-500 步
2. 对比 loss / grad_norm / param_norm / LR 曲线
3. 确认训练行为对齐

### Phase 5: 完整训练 (12-24h)
1. LeRobot 下完成 30,000 步训练
2. 监控所有指标

### Phase 6: 推理验证 (1h)
1. 测试数据上对比推理输出
2. 如有条件，真实机器人对比

---

## 10. 风险矩阵

| 风险 | 可能性 | 影响 | 缓解方案 |
|------|--------|------|---------|
| Tokenizer ID 不一致 | 低 | **高** | 验证 T5 必须通过；若不一致则改用 SentencePiece |
| Norm stats 差异 > 1e-3 | 中 | 中 | 使用 OpenPI 的原始 stats |
| bfloat16 数值精度差异 | 中 | 低 | 仅影响绝对值，不影响趋势 |
| 数据加载顺序差异 | 高 | 低 | 不影响收敛，仅影响逐步对比 |
| 无数据增强导致泛化差异 | 确定 | 中 | 第二轮对齐中添加 |
| EMA 缺失 | 确定 | 中-低 | 与 OpenPI PyTorch 对齐时无影响 |
| BS=64 内存不足 | 中 | 高 | gradient_checkpointing + 梯度累积 |
| CosineDecay 初始值差异 | 低 | 低 | 前几步 LR 差 ~2.5e-8，可忽略 |

---

## 11. 附录

### A. 关键文件索引

| 文件 | 用途 | 修改？ |
|------|------|--------|
| `lerobot/policies/pi05/configuration_pi05.py` | PI05 配置 | ✅ weight_decay + loss 截断 |
| `lerobot/policies/pi05/modeling_pi05.py` | PI05 模型 | ✅ loss 截断逻辑 |
| `lerobot/policies/pi05/processor_pi05.py` | 数据处理 | 无需修改 |
| `lerobot/policies/pi_gemma.py` | AdaRMS 实现 | 无需修改 |
| `openpi/training/config.py:1024-1042` | TrainConfig | 只读参考 |
| `openpi/policies/r1pro_chassis_policy.py` | Data transform | 只读参考 |
| `openpi/training/optimizer.py` | Optimizer 默认值 | 只读参考 |
| `openpi/transforms.py:114-181` | 归一化实现 | 只读参考 |
| `openpi/models/tokenizer.py:14-48` | Tokenizer | 只读参考 |
| `openpi/models/model.py:110-208` | 图像预处理 | 只读参考 |
| `openpi/models_pytorch/preprocessing_pytorch.py` | PyTorch 数据增强 | 只读参考 |
| `openpi/shared/image_tools.py` | resize_with_pad | 只读参考 |
| `openpi/training/data_loader.py` | 数据加载 | 只读参考 |
| `openpi/scripts/train_pytorch.py:500-556` | PyTorch 训练循环 | 只读参考 |

### B. 与 pi05_alig.md (v1) 的逐项对比

| 差异项 | v1 分析 | v2 补充/修正 |
|-------|---------|-------------|
| Weight decay | ✅ P0, 分析充分 | 无修正 |
| Loss 截断 | ✅ P0, 分析充分 | 补充了梯度流向的精确影响 |
| 训练精度 | ✅ P1 | 补充: OpenPI PyTorch 无 autocast |
| Batch size | ✅ P1 | 补充: OpenPI PyTorch 无梯度累积 |
| 数据 key 映射 | ✅ P1 | 补充: "actions" vs "action" 差异在框架层面处理 |
| Norm stats | ✅ P1 | 无修正 |
| Quantile 公式 | ✅ 正确判定可忽略 | 无修正 |
| EMA | ✅ P2 | 无修正 |
| 数据增强 | ✅ P2, 但缺参数 | **补充**: 完整的 PyTorch 增强参数 |
| 基础权重 | ✅ P2 | 无修正 |
| Seed | ✅ P3 | 无修正 |
| Image padding 值 | ✅ P3, 判定正确 | **补充**: 精确的 clamp 范围分析 |
| State padding 顺序 | ✅ P3, 判定正确 | **补充**: 端到端流水线追踪确认 |
| KV cache | ✅ P3 | 无修正 |
| **Tokenizer 等价性** | 未分析 | **新增** P1 |
| **图像全链路** | 未分析 | **新增** P2 |
| **LR init_value** | 未分析 | **新增** P2 |
| **PEFT targets** | 未分析 | **新增** P2 |
| **prompt_from_task** | 未分析 | **新增** P2 |
| **训练循环精确行为** | 未分析 | **新增** P2 |
| **数据流端到端追踪** | 未提供 | **新增** 核心章节 |
