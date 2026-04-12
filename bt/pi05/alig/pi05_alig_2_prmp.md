# Pi0.5 Instruct Prompt 对齐分析：OpenPI vs LeRobot 完整数据流对比

> **日期**: 2026-04-11
> **目标**: 深入追踪 OpenPI 与 LeRobot 在 Pi0.5 训练中 instruct prompt（任务文本）的完整处理流程，纠正 `pi05_alig_2.md` 4.11 节"等价"结论的不完整性
> **前置文档**: `pi05_alig_2.md` 第 4.11 节（prompt_from_task 等价性）、`data_convert.md` 第 9 节（Norm Stats 对齐）
> **代码库**: OpenPI (`/mnt/r/share/lkx/pi/openpi`), LeRobot (`/home/Luogang/SRC/Robot/lerobot`)

---

## 1. 背景与动机

`pi05_alig_2.md` 4.11 节（line 713-725）对 prompt 处理的分析如下：

> **OpenPI**: `data_loader.py:148-149`
> ```python
> if data_config.prompt_from_task:
>     dataset = TransformedDataset(dataset, [PromptFromLeRobotTask(dataset_meta.tasks)])
> ```
> `PromptFromLeRobotTask` (`transforms.py:310-324`) 从 `data["task_index"]` 查 `dataset.meta.tasks` 字典得到 prompt 字符串。
>
> **LeRobot**: 直接从数据集的 `"task"` 字段读取字符串。
>
> **分析**: 两者最终都会得到相同的 prompt 字符串，因为 LeRobot 数据集的 `task` 字段就是 `tasks[task_index]` 的解引用结果。**等价**。

该分析仅比较了**任务文本来源**（task_index 查找 vs 直接读取 task 字段），但忽略了从原始文本到最终 token ID 之间的完整处理链。实际上两个框架中 prompt 会经历清洗、格式构建、state 嵌入、tokenization 等多个变换步骤，且 OpenPI 存在 `InjectDefaultPrompt` 机制。

本文档追踪 prompt 从数据集到最终 token ID 的**完整路径**，逐步对比每个变换步骤。

---

## 2. OpenPI Prompt 完整数据流

以下追踪 `pi05_r1pro_chassis` 配置下，prompt 从数据集加载到模型输入的 7 个步骤。

### Step 1: 数据集加载 — PromptFromLeRobotTask

**文件**: `openpi/src/openpi/training/data_loader.py:148-149`

```python
if data_config.prompt_from_task:
    dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
```

`pi05_r1pro_chassis` 配置中 `prompt_from_task=True`（`config.py:1035`），所以此 transform 会执行。

**Transform 实现**: `openpi/src/openpi/transforms.py:309-324`

```python
@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')
        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")
        return {**data, "prompt": prompt}
```

**行为**: 读取 `data["task_index"]`（整数），在 `dataset_meta.tasks` 字典中查找对应的任务文本字符串，添加到 data dict 的 `"prompt"` key。

对于 `r1_pro_data_convert_chassis` 数据集，`tasks = {0: "Open the door with a downward-press handle, go through it, and enter the room."}`。

**输出**: `data["prompt"] = "Open the door with a downward-press handle, go through it, and enter the room."`

### Step 2: 策略数据变换 — R1ProChassisInputs

**文件**: `openpi/src/openpi/policies/r1pro_chassis_policy.py:55-80`

```python
def __call__(self, data: dict) -> dict:
    # ... 图像和 state 处理 ...
    inputs = {
        "state": np.asarray(data["state"], dtype=np.float32),
        "image": { ... },
        "image_mask": { ... },
    }
    if "actions" in data:
        inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
    if "prompt" in data:
        inputs["prompt"] = data["prompt"]  # line 77-78: 原样透传
    return inputs
```

**行为**: prompt 原样透传，无任何修改。

### Step 3: 归一化 — Normalize

**文件**: `openpi/src/openpi/transforms.py:114-118`, 应用位置 `data_loader.py:188`

```python
@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
```

**行为**: 仅归一化 `state` 和 `actions` 字段（根据 norm_stats keys）。不碰 `prompt`。

归一化后的 state 将在后续 Step 6 中用于离散化。

### Step 4: InjectDefaultPrompt — **训练时 NO-OP**

**文件**: `openpi/src/openpi/transforms.py:104-111`

```python
@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:  # 关键条件
            data["prompt"] = np.asarray(self.prompt)
        return data
```

**配置**: `default_prompt = "Open the door with a downward-press handle, go through it, and enter the room."` (`config.py:1033`)

**训练时行为**: 由于 Step 1 已经将 prompt 添加到 data dict，`"prompt" not in data` 为 `False`，所以 **InjectDefaultPrompt 不执行任何操作**。

**推理时行为**: 如果调用方未提供 `"prompt"` key（如通过 WebSocket 发送的 obs dict 没有包含 prompt），则注入 default_prompt。详见 5.2 节。

### Step 5: ResizeImages

不影响 prompt。

### Step 6: TokenizePrompt → PaligemmaTokenizer.tokenize — **核心变换**

**文件**: `openpi/src/openpi/transforms.py:247-266`

```python
@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False  # Pi0.5: True

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None
        if not isinstance(prompt, str):
            prompt = prompt.item()  # numpy array → Python string
        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
```

**PaligemmaTokenizer.tokenize** — `openpi/src/openpi/models/tokenizer.py:22-48`:

```python
def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")     # (a) 文本清洗
    if state is not None:
        discretized_state = np.digitize(                                    # (b) State 离散化
            state, bins=np.linspace(-1, 1, 256 + 1)[:-1]
        ) - 1
        state_str = " ".join(map(str, discretized_state))                   # (c) State → 字符串
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: " # (d) 模板组装
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)          # (e) SentencePiece 编码
    else:
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

    tokens_len = len(tokens)
    if tokens_len < self._max_len:                                          # (f) 填充到 max_len
        padding = [False] * (self._max_len - tokens_len)
        mask = [True] * tokens_len + padding
        tokens = tokens + padding
    else:
        tokens = tokens[: self._max_len]                                    # (g) 截断
        mask = [True] * self._max_len
    return np.asarray(tokens), np.asarray(mask)
```

**关键操作分解**:

| 子步骤 | 代码 | 行为 |
|--------|------|------|
| (a) 文本清洗 | `prompt.strip().replace("_", " ").replace("\n", " ")` | 去首尾空白，下划线→空格，换行→空格 |
| (b) State 离散化 | `np.digitize(state, np.linspace(-1,1,257)[:-1]) - 1` | 归一化后的 23 维 state → 256 个 bin 的整数索引 (0-255) |
| (c) State 字符串化 | `" ".join(map(str, discretized_state))` | 如 `"128 64 192 ..."` |
| (d) 模板组装 | `f"Task: {text}, State: {bins};\nAction: "` | 固定格式模板 |
| (e) SentencePiece 编码 | `self._tokenizer.encode(full_prompt, add_bos=True)` | 添加 BOS token + 编码为 token ID 序列 |
| (f) 填充 | `[False] * (max_len - tokens_len)` | 用 `False` (=0) 填充到 200 |
| (g) 截断 | `tokens[:max_len]` | 超过 200 则截断 |

**max_len**: 200（`Pi0Config.__post_init__` 中 `pi05=True` 时自动设为 200）

**Tokenizer 模型**: `gs://big_vision/paligemma_tokenizer.model`（SentencePiece）

### Step 7: PadStatesAndActions

不影响已 tokenize 的 prompt。

---

## 3. LeRobot Prompt 完整数据流

以下追踪 LeRobot Pi0.5 训练中 prompt 的 6 个处理步骤。

### Step 1: 数据集加载 — LeRobotDataset.__getitem__

**文件**: `lerobot/src/lerobot/datasets/lerobot_dataset.py:1110-1111`

```python
task_idx = item["task_index"].item()
item["task"] = self.meta.tasks.iloc[task_idx].name
```

**行为**: 从 `meta/tasks.parquet`（v3.0 格式）中读取任务文本。`self.meta.tasks` 是一个 pandas DataFrame，其 **index** 是任务文本字符串，`.name` 属性返回该索引值。

**输出**: `item["task"] = "Open the door with a downward-press handle, go through it, and enter the room."`

### Step 2: Batch 转换 — _extract_complementary_data

**文件**: `lerobot/src/lerobot/processor/converters.py:157-176`

```python
def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    task_key = {"task": batch["task"]} if "task" in batch else {}
    # ...
    return {**pad_keys, **task_key, **subtask_key, ...}
```

**行为**: 将 `"task"` key 从 batch 提取到 `complementary_data` 字典中。

### Step 3: AddBatchDimensionProcessorStep

**文件**: `lerobot/src/lerobot/processor/batch_processor.py:160-177`

```python
if "task" in complementary_data:
    task_value = complementary_data["task"]
    if isinstance(task_value, str):
        complementary_data["task"] = [task_value]
```

**行为**: 将单个字符串包装为列表 `["task string"]`，以支持 batch 处理。

### Step 4: NormalizerProcessorStep

**文件**: `lerobot/src/lerobot/processor/normalize_processor.py`

**行为**: 使用 QUANTILES 模式归一化 `observation.state` 和 `action` 到 [-1, 1] 范围。不碰 task 文本。

归一化后的 state 将在 Step 5 中用于离散化。

### Step 5: Pi05PrepareStateTokenizerProcessorStep — **核心变换**

**文件**: `lerobot/src/lerobot/policies/pi05/processor_pi05.py:47-85`

```python
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)

        # State 离散化
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Prompt 构建
        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")     # (a) 文本清洗
            state_str = " ".join(map(str, discretized_states[i]))               # (b) State 字符串化
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: " # (c) 模板组装
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition
```

**行为**: 与 OpenPI `PaligemmaTokenizer.tokenize` 的 (a)-(d) 子步骤完全对应：

| 子步骤 | LeRobot (`processor_pi05.py`) | OpenPI (`tokenizer.py`) | 一致？ |
|--------|------|------|--------|
| 文本清洗 | `task.strip().replace("_"," ").replace("\n"," ")` (line 77) | `prompt.strip().replace("_"," ").replace("\n"," ")` (line 23) | **是** |
| State 离散化 | `np.digitize(state_np, np.linspace(-1,1,257)[:-1]) - 1` (line 73) | `np.digitize(state, np.linspace(-1,1,257)[:-1]) - 1` (line 26) | **是** |
| State 字符串 | `" ".join(map(str, discretized_states[i]))` (line 78) | `" ".join(map(str, discretized_state))` (line 27) | **是** |
| 模板格式 | `f"Task: {cleaned_text}, State: {state_str};\nAction: "` (line 79) | `f"Task: {cleaned_text}, State: {state_str};\nAction: "` (line 28) | **是** |

注释 (line 71) 明确注明此代码参考自 OpenPI: `# Discretize into 256 bins (see openpi PaligemmaTokenizer.tokenize())`

### Step 6: TokenizerProcessorStep — HuggingFace Tokenize

**文件**: `lerobot/src/lerobot/processor/tokenizer_processor.py:111, 170-205, 253-268`

```python
# 初始化 (line 111)
self.input_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

# 获取 task 文本 (line 183-188)
task = self.get_task(self.transition)
tokenized_prompt = self._tokenize_text(task)

# 实际 tokenize (line 253-268)
def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
    return self.input_tokenizer(
        text,
        max_length=self.max_length,           # 200
        truncation=self.truncation,            # True
        padding=self.padding,                  # "max_length"
        return_tensors=self.return_tensors,     # "pt"
    )

# 输出 (line 204-205)
new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)
```

**行为**: 使用 HuggingFace `AutoTokenizer` (PaliGemma-3B) 对 Step 5 构建的 full_prompt 进行 tokenize。padding 到 200 tokens，右侧填充。

---

## 4. 逐维度对比表

| 对比维度 | OpenPI | LeRobot | 一致？ | 备注 |
|---------|--------|---------|--------|------|
| **任务文本来源** | `tasks[task_index]` dict 查找 (`transforms.py:320-321`) | `meta.tasks.iloc[task_idx].name` DataFrame 查找 (`lerobot_dataset.py:1110-1111`) | **是** | 同一 `tasks.jsonl` |
| **InjectDefaultPrompt** | 有 (训练 NO-OP, 推理生效) (`transforms.py:104-111`) | **无此机制** | **否** | 推理差异，见 5.2 |
| **文本清洗** | `strip().replace("_"," ").replace("\n"," ")` (`tokenizer.py:23`) | 同左 (`processor_pi05.py:77`) | **是** | 完全相同 |
| **State 离散化公式** | `np.digitize(state, np.linspace(-1,1,257)[:-1])-1` (`tokenizer.py:26`) | 同左 (`processor_pi05.py:73`) | **是** | |
| **State 离散化输入** | 归一化后的 state (23 维) | 同左 | **条件** | 需 norm stats 一致 |
| **Prompt 格式模板** | `f"Task: {text}, State: {bins};\nAction: "` (`tokenizer.py:28`) | 同左 (`processor_pi05.py:79`) | **是** | 完全相同 |
| **Tokenizer 实现** | SentencePiece 直接加载 (`tokenizer.py:18-20`) | HuggingFace AutoTokenizer (`tokenizer_processor.py:111`) | **同源** | 需验证，见 5.6 |
| **BOS token** | `encode(prompt, add_bos=True)` (`tokenizer.py:29`) | HuggingFace 默认行为 | **需验证** | |
| **Padding 值** | `False` (= int 0) (`tokenizer.py:36`) | `pad_token_id` (HuggingFace) | **需验证** | |
| **Padding 方向** | 右侧 (短序列末尾补) | `padding_side="right"` | **是** | |
| **max_len** | 200 (`pi0_config.py` pi05 模式) | 200 (`configuration_pi05.py:tokenizer_max_length`) | **是** | |
| **输出 Key** | `tokenized_prompt`, `tokenized_prompt_mask` | `observation.language.tokens`, `observation.language.attention_mask` | **不同** | 功能等价 |
| **State padding 时序** | tokenize 前 state 未 pad (用原始 23 维) | 同左 (PadStatesAndActions 在 tokenize 之后) | **是** | |

---

## 5. 差异深度分析

### 5.1 任务文本来源 — 等价

两个框架都从同一 `tasks.jsonl` 文件读取任务文本。对于 `r1_pro_data_convert_chassis` 数据集，只有 1 个 task:

```json
{"task_index": 0, "task": "Open the door with a downward-press handle, go through it, and enter the room."}
```

OpenPI 通过 `PromptFromLeRobotTask` 用 `dict[int, str]` 查找，LeRobot 通过 `DataFrame.iloc[idx].name` 查找。两者得到的原始字符串完全相同。

`pi05_alig_2.md` 4.11 节的这部分结论**正确**。

### 5.2 InjectDefaultPrompt — 训练 NO-OP，推理不等价

#### 5.2.1 机制说明

`InjectDefaultPrompt` 是 OpenPI 独有的 transform（`transforms.py:104-111`）。其核心逻辑：

```python
if self.prompt is not None and "prompt" not in data:
    data["prompt"] = np.asarray(self.prompt)
```

**只在 `"prompt"` key 不存在时才注入 default_prompt。**

#### 5.2.2 训练时的位置与行为

**训练 transform 链** (`data_loader.py:183-191`):

```
repack_transforms → data_transforms → Normalize → model_transforms
                    (R1ProChassisInputs)            (InjectDefaultPrompt → ResizeImages → TokenizePrompt → Pad)
```

注意 `InjectDefaultPrompt` 在 `model_transforms` 内部（`config.py:132`），位于 Normalize 之后。

但 `PromptFromLeRobotTask` 在数据集创建阶段已执行（`data_loader.py:148-149`），在所有 transform 之前。所以当 `InjectDefaultPrompt` 运行时，`"prompt"` 已存在 → **训练时是 NO-OP**。

#### 5.2.3 推理时的位置与行为

**推理 transform 链** (`policy_config.py:75-83`):

```python
transforms=[
    *repack_transforms.inputs,
    transforms.InjectDefaultPrompt(default_prompt),    # ← 位置 1: policy_config 层
    *data_config.data_transforms.inputs,
    transforms.Normalize(norm_stats, ...),
    *data_config.model_transforms.inputs,              # ← 位置 2: ModelTransformFactory 内部
],
```

推理时**没有 `PromptFromLeRobotTask`**（没有数据集可查询）。如果调用方的 obs dict 不包含 `"prompt"` key，`InjectDefaultPrompt` 会注入 `default_prompt`。

**结论**: OpenPI 推理有 fallback prompt 机制，LeRobot **没有**（会直接抛出 `ValueError("No task found in complementary data")`）。

#### 5.2.4 各 R1 Pro 配置的 default_prompt

| 配置名 | default_prompt | 与数据集 task 0 一致？ |
|--------|---------------|---------------------|
| `pi05_r1pro_chassis` | `"Open the door with a downward-press handle, go through it, and enter the room."` | **是** |
| `pi05_r1pro_chassis_all` | `"open the door handle"` | **否** — 大小写和内容都不同 |
| `pi05_r1pro_open_door` | `"open the door handle"` | 需查看对应数据集 |
| `pi05_r1pro_open_door_lora` | `"open the door handle"` | 需查看对应数据集 |

`pi05_r1pro_chassis_all` 的 `default_prompt` 与 `r1_pro_data_convert_chassis` 的 task 0 文本**不一致**。推理时若不提供 prompt，会使用 `"open the door handle"` 而非训练时实际使用的 `"Open the door with a downward-press handle, go through it, and enter the room."`。

### 5.3 文本清洗 — 完全一致

两个框架应用完全相同的清洗操作：

| 操作 | OpenPI (`tokenizer.py:23`) | LeRobot (`processor_pi05.py:77`) |
|------|---------------------------|----------------------------------|
| 去首尾空白 | `.strip()` | `.strip()` |
| 下划线→空格 | `.replace("_", " ")` | `.replace("_", " ")` |
| 换行→空格 | `.replace("\n", " ")` | `.replace("\n", " ")` |

对于 `"Open the door with a downward-press handle, go through it, and enter the room."` 这个特定文本，清洗操作无任何变化（无下划线、无换行、无首尾空白）。

### 5.4 State 离散化 — 逻辑一致，数值受 norm stats 影响

#### 5.4.1 离散化算法完全一致

两个框架使用相同的公式：

```python
discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
```

将 [-1, 1] 范围均匀划分为 256 个 bin，每个 bin 宽度 = 2/256 ≈ 0.0078。归一化后的 state 值映射到 0-255 的整数。

#### 5.4.2 数值依赖归一化结果

离散化的输入是**归一化后**的 state。两个框架的归一化公式：

| | OpenPI (`transforms.py:141`) | LeRobot (`normalize_processor.py`) |
|-|-------|---------|
| 公式 | `(x - q01) / (q99 - q01 + 1e-6) * 2 - 1` | `2 * (x - q01) / max(q99 - q01, 1e-8) - 1` |
| q01/q99 来源 | `norm_stats.json` (64 episodes) | `stats.json` (取决于转换模式) |

如果 q01/q99 不一致（如使用采样 10 episodes 计算 vs 全量 64 episodes），归一化后的 state 不同 → 离散化 bin 不同 → prompt 的 State 部分不同 → **最终 token 序列不同**。

`data_convert.md` 第 9.6 节已量化了这个偏差：max_diff = 0.17（维度 9），可导致 **24 bins 偏差**。

**解决方案**: 使用 `--norm-stats-path` 导入 OpenPI 的 norm_stats.json，消除 q01/q99 差异。导入后 state 离散化结果**完全一致**。

### 5.5 Prompt 格式模板 — 完全一致

两个框架使用完全相同的 f-string 模板：

```python
f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

示例输出（假设 state 全为 128）：

```
Task: Open the door with a downward-press handle, go through it, and enter the room., State: 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128;\nAction: 
```

### 5.6 Tokenizer 实现 — 同源不同加载方式

#### 5.6.1 加载方式对比

| | OpenPI | LeRobot |
|-|--------|---------|
| 库 | `sentencepiece.SentencePieceProcessor` | `transformers.AutoTokenizer` |
| 模型来源 | `gs://big_vision/paligemma_tokenizer.model` | `google/paligemma-3b-pt-224` (HuggingFace Hub) |
| 底层模型 | PaliGemma SentencePiece | PaliGemma SentencePiece (HuggingFace 封装) |

两者底层使用**相同的 SentencePiece 模型**。HuggingFace 的 `AutoTokenizer` 在加载 `google/paligemma-3b-pt-224` 时，内部也使用 SentencePiece。

#### 5.6.2 编码行为对比

| | OpenPI | LeRobot |
|-|--------|---------|
| 编码调用 | `self._tokenizer.encode(full_prompt, add_bos=True)` | `self.input_tokenizer(text, max_length=200, ...)` |
| BOS token | 显式 `add_bos=True` | HuggingFace 默认行为（PaliGemma 默认 `add_bos_token=True`） |
| 填充值 | `False` (= int 0) | `pad_token_id`（PaliGemma 的 pad_token_id = 0） |
| 返回类型 | `np.ndarray` | `torch.Tensor` |

**BOS token**: 两者都会添加 BOS token（token ID 2 for PaliGemma），行为应一致。

**填充值**: OpenPI 用 `False` (Python 中 `int(False) == 0`)，LeRobot 用 `pad_token_id`。PaliGemma 的 `pad_token_id = 0`，所以两者填充值相同（都是 0）。

**Token ID 序列**: 对于相同的输入文本，SentencePiece 的 `encode()` 和 HuggingFace 的 `__call__()` 应产生相同的 token ID 序列。但这需要实验验证（如 HuggingFace 可能有额外的 pre-tokenization 步骤或不同的 normalization 处理）。

#### 5.6.3 需要实验验证的点

```python
# 验证脚本示例
import sentencepiece
from transformers import AutoTokenizer

# OpenPI 方式
sp = sentencepiece.SentencePieceProcessor(model_file="paligemma_tokenizer.model")
tokens_openpi = sp.encode("Task: test, State: 128 64;\nAction: ", add_bos=True)

# LeRobot 方式
hf = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
tokens_lerobot = hf("Task: test, State: 128 64;\nAction: ")["input_ids"]

# 比较
assert tokens_openpi == tokens_lerobot[0].tolist(), "Token 序列不一致!"
```

### 5.7 Token 最大长度与填充 — 一致

| | OpenPI | LeRobot |
|-|--------|---------|
| max_length | 200 (`pi0_config.py`, pi05 时 `max_token_len=200`) | 200 (`configuration_pi05.py:tokenizer_max_length=200`) |
| padding 方向 | 右侧 | `padding_side="right"` |
| truncation | 截断到 max_len (`tokenizer.py:45`) | `truncation=True` |

---

## 6. 实际数据集中的任务文本

### 6.1 原始数据集

**`r1_pro_data_convert_chassis/meta/tasks.jsonl`**:

```json
{"task_index": 0, "task": "Open the door with a downward-press handle, go through it, and enter the room."}
```

1 个 task，所有 64 个 episodes 都使用 task_index 0。

**`r1_pro_test_data/meta/tasks.jsonl`**:

```json
{"task_index": 0, "task": "Open the door with a downward-press handle, go through it, and enter the room."}
{"task_index": 1, "task": "open the door handle"}
```

2 个 task，不同 episodes 可能使用不同的 task。

### 6.2 OpenPI 配置的 default_prompt 对照

| 配置 | default_prompt | 对应数据集 | 匹配？ |
|------|---------------|-----------|--------|
| `pi05_r1pro_chassis` (line 1033) | `"Open the door with a downward-press handle, go through it, and enter the room."` | `r1_pro_data_convert_chassis` task 0 | **完全匹配** |
| `pi05_r1pro_chassis_all` (line 1052) | `"open the door handle"` | `r1_pro_data_convert_chassis_all` | **不匹配 chassis task 0** |

### 6.3 影响分析

对于 `pi05_r1pro_chassis`:
- **训练**: 使用数据集 task 0 文本（`PromptFromLeRobotTask`）→ 与 default_prompt 完全一致 → 无影响
- **推理 (有 prompt)**: 使用用户提供的 prompt → 与训练文本一致性取决于用户
- **推理 (无 prompt)**: 注入 default_prompt → 与训练文本一致 ✓

对于 `pi05_r1pro_chassis_all`:
- **训练**: 使用数据集 task 文本 → 可能是长版本
- **推理 (无 prompt)**: 注入 `"open the door handle"` → **与训练时使用的长版本 prompt 不一致** ⚠️

---

## 7. 结论

### 7.1 综合判定

| 维度 | 训练阶段 | 推理阶段 |
|------|---------|---------|
| 任务文本来源 | **等价** ✅ | N/A (推理无数据集) |
| InjectDefaultPrompt | **等价** (NO-OP) ✅ | **不等价** ❌ |
| 文本清洗 | **等价** ✅ | **等价** ✅ |
| State 离散化逻辑 | **等价** ✅ | **等价** ✅ |
| State 离散化数值 | **条件等价** ⚠️ (需 norm stats 对齐) | 同左 |
| Prompt 格式模板 | **等价** ✅ | **等价** ✅ |
| Tokenizer 输出 | **大概率等价** ⚠️ (需实验验证) | 同左 |

**训练阶段**: **条件等价**。条件为：
1. Norm stats 对齐（使用 `--norm-stats-path` 导入可满足）
2. Tokenizer 输出验证（同源 SentencePiece 模型，大概率一致）

**推理阶段**: **不等价**。OpenPI 有 `InjectDefaultPrompt` fallback 机制，LeRobot 没有。

### 7.2 对 pi05_alig_2.md 4.11 节的修正

原结论"等价"不完整。建议修正为：

> **4.11 P2: prompt_from_task 等价性（修正）**
>
> **结论**: 训练阶段**条件等价**，推理阶段**不等价**。
>
> **训练等价条件**:
> 1. 文本来源等价 ✅ — 同一 tasks.jsonl
> 2. 文本清洗等价 ✅ — 相同的 strip/replace 操作
> 3. 格式模板等价 ✅ — 相同的 f-string
> 4. State 离散化等价 ✅ — 相同的 np.digitize 逻辑（但数值依赖 norm stats 对齐）
> 5. Tokenizer 等价 ⚠️ — 同源 SentencePiece 模型，需实验验证
>
> **推理差异**:
> - OpenPI: `InjectDefaultPrompt` 提供 fallback prompt
> - LeRobot: 无此机制，必须显式提供 task
>
> 详细分析见 `pi05_alig_2_prmp.md`。

### 7.3 行动建议

| 优先级 | 行动 | 说明 |
|--------|------|------|
| P1 | 确保 norm stats 对齐 | 使用 `--norm-stats-path` 导入 OpenPI 的 norm_stats.json |
| P2 | 实验验证 tokenizer 等价性 | 对相同 prompt 比较 SentencePiece 和 HuggingFace 的 token ID |
| P3 | 评估是否在 LeRobot 推理中添加 default prompt | 如需与 OpenPI 推理行为一致 |
