# LeRobot PI0.5 与 OpenPI PI0.5 差异梳理

本文把前面对 `lerobot/pi05_base` 与 OpenPI 官方 `gs://openpi-assets/checkpoints/pi05_base/` 的分析重新归类，重点回答两个问题：

1. `LeRobot` 的 PI0.5 和 `OpenPI` 官方 PI0.5 到底差在哪里。
2. 如果要把 `LeRobot` 的 PI0.5 尽量对齐到 `OpenPI` 官方行为，应该怎么做。

## 1. 先分清对比对象

这次对比里最容易混淆的是：`OpenPI` 的 `pi05_base` 和 `LeRobot` 的 `lerobot/pi05_base` 并不是同一种“配置对象”。

- `OpenPI` 的 `pi05_base` 主要是基础权重和配套资产：
  - `gs://openpi-assets/checkpoints/pi05_base/params`
  - `gs://openpi-assets/checkpoints/pi05_base/assets`
- 它本身不是一个单独命名的 `TrainConfig`。实际运行行为要结合：
  - `src/openpi/models/pi0_config.py`
  - `src/openpi/training/config.py`
  - `src/openpi/transforms.py`
  - 以及具体下游 recipe，比如 `pi05_libero`、`pi05_droid`、`pi05_aloha`
- `LeRobot` 的 `lerobot/pi05_base` 是一个 Hugging Face 模型仓库，直接打包了：
  - `config.json`
  - `policy_preprocessor.json`
  - `policy_postprocessor.json`
  - `model.safetensors`

因此，讨论差异时最好分成三层：

1. `OpenPI` 默认 `Pi0Config(pi05=True)` 的模型行为
2. `OpenPI` 官方下游 recipe 的覆盖行为
3. `LeRobot` 的 `lerobot/pi05_base` 及其 processor/runtime 封装行为

## 2. 哪些地方基本一致

先说结论：`LeRobot` 的 PI0.5 不是另一套新模型，核心网络结构基本是对 `OpenPI` PI0.5 的移植。

| 项目 | LeRobot `lerobot/pi05_base` | OpenPI 默认 `Pi0Config(pi05=True)` | 结论 |
|---|---|---|---|
| VLM 变体 | `gemma_2b` | `gemma_2b` | 一致 |
| Action expert 变体 | `gemma_300m` | `gemma_300m` | 一致 |
| state/action 维度 | `32 / 32` | `32 / 32` | 一致 |
| 默认 action horizon | `chunk_size=50` | `action_horizon=50` | 基本一致 |
| 图像分辨率 | `224x224` | `224x224` | 一致 |
| token 长度 | `200` | `200` | 一致 |
| PI0.5 结构特征 | `time_mlp_*` + AdaRMS | `time_mlp_*` + AdaRMS | 一致 |
| state_proj | 无 | 无 | 一致 |
| flow-matching 推理步数 | `10` | `10` | 一致 |

所以，差异主要不在主干网络，而在：

- 配置承载方式
- processor / tokenizer / norm stats
- 数据集变换
- 运行时 API 封装
- 官方下游 recipe 的覆盖逻辑

## 3. 差异重新归类

### 3.1 配置来源与“对象边界”不同

这是最本质的差异。

- `OpenPI` 的 `pi05_base` 更像“基础权重 + 资产目录”
- `LeRobot` 的 `lerobot/pi05_base` 更像“可直接 from_pretrained 的完整模型包”

这会带来一个直接后果：

- 在 `OpenPI` 里，你不能只盯着 `pi05_base/params` 就断言实际行为，必须再看当前 `TrainConfig`
- 在 `LeRobot` 里，你很容易把 `config.json` + `policy_preprocessor.json` 误以为“已经完整表达了 OpenPI 官方行为”

实际上并没有这么简单。

### 3.2 tokenizer、prompt 和 state 输入方式不完全一样

这一类差异最容易影响“输入是否真的对齐”。

#### LeRobot 的固定行为

`LeRobot` 的 `src/lerobot/policies/pi05/processor_pi05.py` 里，PI0.5 的 prompt 预处理固定是：

- 先假定 `state` 已经归一化到 `[-1, 1]`
- 再把 `state` 离散成 256 个 bin
- 然后构造成：

```text
Task: {cleaned_text}, State: {bin0 bin1 ...};\nAction:
```

这代表 `LeRobot` 当前实现里，PI0.5 默认总是走“state 拼进文本”的路径。

#### OpenPI 默认行为

`OpenPI` 默认 `Pi0Config(pi05=True)` 时：

- `discrete_state_input=True`
- 行为和上面的格式是对齐的

#### OpenPI 官方 recipe 行为

但 `OpenPI` 的官方 `pi05_libero` recipe 并不是这样：

- `pi05_libero` 显式设为 `discrete_state_input=False`
- 这意味着它不会走“state 拼进 prompt”的默认 PI0.5 输入形式

所以要分清：

- 如果对齐目标是“OpenPI 默认 PI0.5 base”，那 `LeRobot` 这部分总体方向是对的
- 如果对齐目标是“OpenPI 官方 `pi05_libero`”，那 `LeRobot` 当前 prompt 行为就不等价

#### tokenizer 加载入口不同

- `LeRobot` 通过 `google/paligemma-3b-pt-224` 用 HF tokenizer 加载
- `OpenPI` 通过 `gs://big_vision/paligemma_tokenizer.model` 直接加载 SentencePiece

这两条路径的加载方式不同，但 `google/paligemma-3b-pt-224` 本身也包含 `tokenizer.model`，所以底层词表很可能同源。更准确的说法是：

- “tokenizer 加载入口不同”
- 但“不一定代表 token id 必然不同”

### 3.3 归一化与 assets 管理方式差异很大

这是目前 `LeRobot` 和 `OpenPI` 行为最不容易真正对齐的一层。

#### OpenPI 的做法

在 `OpenPI` 中：

- `PI05` 默认使用 quantile normalization
- norm stats 通过 `assets_dir + asset_id` 显式加载
- 也就是说，权重之外的“输入输出尺度语义”是由 `assets` 补齐的

#### LeRobot 的做法

在 `LeRobot` 中：

- `PI05Config` 也把 `STATE/ACTION` 默认设为 `QUANTILES`
- 但 `lerobot/pi05_base` 仓库里并没有像 `OpenPI` 那样显式附带 `assets/<asset_id>/...`
- 同时 `policy_preprocessor.json` / `policy_postprocessor.json` 里也没有对应的 state file

这导致一个很关键的现象：

- 如果你只是直接加载 `lerobot/pi05_base` 及其保存好的 processor 配置，processor 里不一定真的有可用的 quantile stats
- 结果就是“配置上写着 QUANTILES，实际运行时未必真的做了等价的 quantile 归一化/反归一化”

#### 本目录当前训练脚本的现实意义

`bt/pi05/train_pi05_local.py` 已经绕过了这个坑的一部分：

- 它显式加载 tokenizer
- 它显式把 `dataset.meta.stats` 传给 `NormalizerProcessorStep` / `UnnormalizerProcessorStep`

因此，`bt/pi05/train_pi05_local.py` 的 processor 行为其实比“直接用 `lerobot/pi05_base` 自带 processor”更接近 OpenPI 的“有 stats 的真实运行方式”。

但要注意：

- 这里用的是当前数据集的 `dataset.meta.stats`
- 不一定等于 `OpenPI pi05_base assets` 里原始配套的 norm stats

所以它是“更接近”，不是“严格等价”。

### 3.4 运行时封装和 API 行为不完全一样

#### LeRobot 多了 `n_action_steps` 的执行封装

`LeRobot` 在 `PI05Policy.select_action()` 里引入了 action queue：

- 先预测一整个 chunk
- 再按 `n_action_steps` 分批出队执行

而 `OpenPI` 更接近底层模型 API：

- `sample_actions()` 返回整个 action chunk
- 后续怎么截断、怎么映射回环境动作空间，更多是由上层 transform / policy wrapper 决定

因此：

- `LeRobot` 更偏“可直接接到 LeRobot 生态里跑”
- `OpenPI` 更偏“底层模型 + 外部数据/策略封装”

#### 默认精度和 compile 行为不同

- `LeRobot` 发布的 `lerobot/pi05_base/config.json` 里默认 `dtype=float32`
- `OpenPI` 默认 `Pi0Config(pi05=True)` 是 `dtype=bfloat16`

另外：

- `OpenPI` 默认 `pytorch_compile_mode="max-autotune"`，通常会 compile `sample_actions`
- `LeRobot` 里则多了显式开关 `compile_model`，默认关闭

这类差异通常不会改变模型“是什么”，但会影响：

- 显存
- 速度
- 某些数值细节

### 3.5 OpenPI 官方下游 recipe 与 LeRobot 默认行为不是一回事

很多“看起来是 PI0.5 差异”的问题，本质上其实是“下游 recipe 差异”。

#### `pi05_libero`

官方 `pi05_libero` 的重要覆盖包括：

- `action_horizon=10`
- `discrete_state_input=False`
- `extra_delta_transform=False`
- 输出按 `LiberoOutputs` 裁成前 7 维

而 `LeRobot` 默认 `pi05_base` 是：

- `chunk_size=50`
- 固定 state-in-text
- 没有直接内置 `LiberoOutputs` 这一层 recipe 语义

所以 `LeRobot pi05_base != OpenPI pi05_libero`。

#### `pi05_droid`

官方 `pi05_droid` 还会覆盖：

- `action_horizon=15`
- DROID 专用输入/输出 transform
- `prompt_from_task=True`

这也不是 `LeRobot pi05_base` 默认就会自动带上的行为。

#### Aloha / delta action

`OpenPI` 在 Aloha、DROID 等配置中，会把 delta-action / absolute-action 的逻辑放在 data transforms 里：

- 哪些维度做 delta
- gripper 是否保持 absolute
- 是否需要 `AbsoluteActions` 还原

这些都不是 `pi05_base` 单个 checkpoint 自己能表达出来的。

### 3.6 加载兼容层和微调目标还有移植痕迹

#### checkpoint 加载兼容逻辑

`LeRobot` 的 `PI05Policy.from_pretrained()` 里带有一层显式兼容修补：

- 把 `action_time_mlp_*` 映射到 `time_mlp_*`
- 跳过 `state_proj.*`

这说明：

- `LeRobot` 并不是“OpenPI 原样搬过来直接加载”
- 而是“做过结构兼容和权重键名适配的移植版”

#### 默认 PEFT target 仍有 PI0 命名残留

`LeRobot` 的 `PI05` 默认 PEFT target regex 里，仍包含：

- `state_proj`
- `action_time_mlp_in`
- `action_time_mlp_out`

但 `PI05` 实际用的是：

- `time_mlp_in`
- `time_mlp_out`
- 而且没有 `state_proj`

这意味着：

- 如果你直接复用默认 PEFT target 做 PI0.5 微调
- 它与 PI0.5 真实模块名并不是完全对齐的

## 4. 把差异压缩成一句话

如果问题是：

- “`LeRobot` 的 PI0.5 和 `OpenPI` 的 PI0.5 是不是两套不同模型？”

答案是：

- 不是。核心网络结构和大部分模型超参基本同源。

如果问题是：

- “它们的实际输入/输出行为是不是 1:1 完全一致？”

答案是：

- 也不是。最主要的不一致来自 processor、norm stats、数据 transforms、以及官方下游 recipe 覆盖。

## 5. 如何将 LeRobot 的 PI0.5 与 OpenPI 的 PI0.5 对齐

先说结论：不要笼统地说“对齐 OpenPI PI0.5”，而要先选定目标。

### 5.1 先明确你要对齐哪个目标

| 对齐目标 | 典型用途 | 关键设置 |
|---|---|---|
| OpenPI 默认 PI0.5 base | 复现基础模型默认行为 | `pi05=True`、`action_horizon=50`、`discrete_state_input=True`、quantile norm |
| OpenPI `pi05_libero` | 对齐 LIBERO 官方 recipe | `action_horizon=10`、`discrete_state_input=False`、`extra_delta_transform=False`、输出裁前 7 维 |
| OpenPI `pi05_droid` | 对齐 DROID 官方 recipe | `action_horizon=15`、DROID transforms、`prompt_from_task=True` |

如果目标不先定，后面很多“该不该拼 state 到 prompt”“该不该 10 步还是 50 步”“该不该裁 7 维”都没有唯一答案。

### 5.2 推荐的对齐顺序

建议按下面顺序推进，而不是一开始就改很多代码：

1. 先固定对齐目标：默认 PI0.5 base、`pi05_libero`、还是 `pi05_droid`
2. 先对齐 processor 与 norm stats
3. 再对齐 prompt/state 输入形式
4. 再对齐 horizon、输出裁剪、delta-action 规则
5. 最后用端到端对照测试验证

推荐测试入口：

- `tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py`

### 5.3 如果目标是“对齐 OpenPI 默认 PI0.5 base”

这是最容易的一条路径。

#### 需要对齐的点

- 维持 `chunk_size=50`
- 维持 `max_state_dim=32`
- 维持 `max_action_dim=32`
- 保持“state 离散后拼进 prompt”
- 使用 quantile normalization
- 尽量使用与 OpenPI 同源的 tokenizer 资产
- 训练/推理时使用 `bfloat16`

#### 落地建议

1. 不要直接依赖 `lerobot/pi05_base` 自带的 processor JSON 作为唯一事实来源  
   原因：它没有完整表达 OpenPI 的 assets/norm stats 语义。

2. 像 `bt/pi05/train_pi05_local.py` 一样，显式构造 processor  
   关键点是：
   - 显式加载 tokenizer
   - 显式给 normalizer/unnormalizer 传 stats

3. 把对齐目标里使用的 stats 来源固定下来  
   两种做法：
   - 用 OpenPI `pi05_base/assets` 对应的 norm stats
   - 或者明确约定“本实验以当前数据集 quantile stats 为准”

4. 把默认精度和 compile 习惯改到更接近 OpenPI  
   - `dtype=bfloat16`
   - 推理 compile 打开

#### 当前本目录里已经做对的部分

`bt/pi05/train_pi05_local.py` 已经做了两件有利于对齐的事：

- 显式加载 `google/paligemma-3b-pt-224`
- 显式把 `dataset.meta.stats` 注入 pre/postprocessor

所以它比“直接用 Hub 里保存好的 processor 配置”更接近 OpenPI。

### 5.4 如果目标是“对齐 OpenPI `pi05_libero`”

这条路径不是简单改几个训练超参就够了，因为它涉及输入格式差异。

#### 必须改的点

1. `action_horizon` 要从 50 改成 10  
   在 `LeRobot` 侧通常对应：
   - `chunk_size=10`
   - `n_action_steps` 根据你的执行策略决定是否也取 10

2. 需要支持 `discrete_state_input=False`  
   这是当前 `LeRobot` PI0.5 最大的结构性不一致之一。

3. 保持 `extra_delta_transform=False`  
   对于 `pi05_libero`，官方配置明确不再额外叠一层 delta transform。

4. 输出动作要与 LIBERO 环境期望一致  
   官方 `LiberoOutputs` 会把 pad 后动作裁到前 7 维。

#### 代码层面的建议改动

最关键的是给 `LeRobot` 的 PI0.5 增加一个显式开关，例如：

- 在 `src/lerobot/policies/pi05/configuration_pi05.py` 增加 `discrete_state_input: bool = True`
- 在 `src/lerobot/policies/pi05/processor_pi05.py` 里根据这个开关支持两条路径：
  - `True`: `Task: ..., State: ...;\nAction: `
  - `False`: 只保留文本 prompt，不把 state 拼进文本

如果要更严格对齐 `pi05_libero`，还需要在 LIBERO 对接层补齐等价的输入/输出 transform。

### 5.5 如果目标是“对齐 OpenPI `pi05_droid`”

这一条重点不在模型，而在数据和环境接口。

#### 需要对齐的点

- `action_horizon=15`
- 使用 DROID 专用输入/输出 transform
- `prompt_from_task=True`
- 若 action space 是 joint position，要对齐官方的 delta/absolute 变换规则

这意味着：

- 不能只改 `PI05Config`
- 还要补 DROID 对应的数据 repack / data transforms

### 5.6 建议优先改哪些文件

| 文件 | 作用 | 对齐价值 |
|---|---|---|
| `src/lerobot/policies/pi05/configuration_pi05.py` | 加入 `discrete_state_input`、明确对齐目标相关配置 | 高 |
| `src/lerobot/policies/pi05/processor_pi05.py` | 决定 prompt、state 离散化、tokenizer 路径 | 最高 |
| `src/lerobot/policies/factory.py` | 处理从 Hub 加载 processor 时的 stats 覆盖问题 | 高 |
| `bt/pi05/train_pi05_local.py` | 当前本地实验入口，可先在这里验证对齐策略 | 高 |
| `tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py` | 做 OpenPI vs LeRobot 对照验证 | 高 |

### 5.7 最务实的一条实施路线

如果目标是“先把本地实验尽量拉近 OpenPI”，建议按这个顺序做：

1. 保持使用 `bt/pi05/train_pi05_local.py`
2. 明确本次要对齐的是：
   - OpenPI 默认 PI0.5 base
   - 或 `pi05_libero`
3. 显式固定 tokenizer 与 stats 来源
4. 如果目标是 `pi05_libero`，先改 `processor_pi05.py` 支持 `discrete_state_input=False`
5. 把 `chunk_size` / `n_action_steps` / 输出裁剪改到目标 recipe
6. 跑 `tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py` 或等价对照脚本

## 6. 对当前 `bt/pi05` 目录的直接判断

结合当前目录里的 `bt/pi05/train_pi05_local.py`，可以把它的位置概括成一句话：

- 它已经比“直接加载 `lerobot/pi05_base` 自带 processor”更接近 OpenPI
- 但它仍然没有完全对齐 `OpenPI pi05_libero`

原因是：

- 它显式加载了 tokenizer
- 它显式给 processor 注入了 `dataset.meta.stats`
- 但它仍然走 `Pi05PrepareStateTokenizerProcessorStep`
- 也就是仍然默认把 state 离散后拼进 prompt
- 同时 `PI05Config` 仍然默认是 50-step 语义，而不是 `pi05_libero` 的 10-step 语义

## 7. 最后一句话总结

最准确的说法不是：

- “`LeRobot` 的 PI0.5 和 `OpenPI` 的 PI0.5 不一样”

而是：

- “`LeRobot` 的 PI0.5 是 `OpenPI` PI0.5 的移植版，核心模型基本同源；真正不完全一致的地方，主要集中在 processor、norm stats、下游 recipe 覆盖、以及 LeRobot 自己的运行时封装。”

如果后续要真正做到“对齐”，第一步永远不是改代码，而是先回答：

- 你到底要对齐 OpenPI 默认 PI0.5 base，还是要对齐 `pi05_libero` / `pi05_droid` 这样的官方具体 recipe。
