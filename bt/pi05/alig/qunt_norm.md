# Quantile 归一化分析

本文结合 `pi05_alig.md`、`pi05_alig_3.md`、OpenPI 源码和 LeRobot 源码，回答 4 个问题：

1. `Quantile 归一化` 是什么
2. 它在训练/推理里有什么作用
3. OpenPI 和 LeRobot 在这方面有什么差异
4. 这些差异到底能不能忽略，哪些能忽略，哪些不能

---

## 1. Quantile 归一化是什么意思

`Quantile 归一化` 不是均值方差归一化（mean/std, z-score），而是用分位数区间把数值线性映射到 `[-1, 1]`。

这里用到的两个关键统计量是：

- `q01`: 第 1 百分位
- `q99`: 第 99 百分位

其核心思想是：

- 把“绝大多数正常数据”的范围近似看作 `[q01, q99]`
- 再把这个区间映射到 `[-1, 1]`
- 比起直接用 `min/max`，它对极端异常值更稳健
- 比起 `mean/std`，它不要求数据大致服从高斯分布

### 1.1 OpenPI 公式

OpenPI 在 `openpi/src/openpi/transforms.py` 中这样实现：

```141:145:/mnt/r/share/lkx/pi/openpi/src/openpi/transforms.py
    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

对应反归一化：

```175:181:/mnt/r/share/lkx/pi/openpi/src/openpi/transforms.py
    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
```

也就是：

```text
normalize:   y = 2 * (x - q01) / (q99 - q01 + 1e-6) - 1
unnormalize: x = (y + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

### 1.2 LeRobot 公式

LeRobot 在 `src/lerobot/processor/normalize_processor.py` 中这样实现：

```362:377:/home/Luogang/SRC/Robot/lerobot/src/lerobot/processor/normalize_processor.py
        if norm_mode == NormalizationMode.QUANTILES:
            q01 = stats.get("q01", None)
            q99 = stats.get("q99", None)
            if q01 is None or q99 is None:
                raise ValueError(
                    "QUANTILES normalization mode requires q01 and q99 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                )

            denom = q99 - q01
            # Avoid division by zero by adding epsilon when quantiles are identical
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q01
            return 2.0 * (tensor - q01) / denom - 1.0
```

也就是：

```text
denom = q99 - q01
if denom == 0:
    denom = 1e-8

normalize:   y = 2 * (x - q01) / denom - 1
unnormalize: x = (y + 1) * denom / 2 + q01
```

---

## 2. 它有什么作用

### 2.1 作用一：把 state/action 压到稳定数值范围

机器人状态和动作各维度的量纲差别通常很大，比如：

- 关节角可能是弧度
- gripper 可能是开合量
- chassis 可能是速度
- 某些维度变化很小，某些维度变化很大

如果直接把原始数值送进模型：

- 大量纲维度会主导梯度
- 小量纲维度容易被淹没
- 优化器更难找到稳定步长

Quantile 归一化把大部分有效数据压到 `[-1, 1]`，能让：

- 不同维度的数值尺度更接近
- tokenization / MLP / diffusion head 更稳定
- 训练初期更不容易出现某几维梯度过大

### 2.2 作用二：对异常值更稳健

比起 `min/max`，分位数不太受极端 outlier 影响。

例如一个 action 维度大部分都在 `[-0.1, 0.1]`，但偶尔有一个坏样本到 `5.0`：

- `min/max` 会把整个缩放区间拉得很大
- `q01/q99` 往往不会被这个单点严重破坏

这也是它适合真实机器人数据的原因之一。

### 2.3 作用三：为离散化或后续解码提供更稳定输入

PI0.5/PI05 里有一条重要链路：先把 state 归一化，再做离散化或 tokenizer 相关处理。LeRobot 的 PI05 processor 代码写得很明确：

```132:139:/home/Luogang/SRC/Robot/lerobot/src/lerobot/policies/pi05/processor_pi05.py
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
```

这说明它不只是“让数值好看一点”，而是直接影响后续离散化语义。

---

## 3. OpenPI 和 LeRobot 分别怎么用它

## 3.1 OpenPI 数据流

OpenPI 先计算统计量，只针对 `state` 和 `actions`：

```102:113:/mnt/r/share/lkx/pi/openpi/scripts/compute_norm_stats.py
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
```

训练时把 `Normalize` 插进数据管线：

```183:190:/mnt/r/share/lkx/pi/openpi/src/openpi/training/data_loader.py
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )
```

推理时在输入侧 `Normalize`、输出侧 `Unnormalize`：

```75:87:/mnt/r/share/lkx/pi/openpi/src/openpi/policies/policy_config.py
    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
```

此外，OpenPI 不是所有模型都默认用 quantile。配置里写得很直接：

```181:189:/mnt/r/share/lkx/pi/openpi/src/openpi/training/config.py
    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )
```

这意味着：

- `PI0` 默认不是 quantile
- `PI05` 默认是 quantile

## 3.2 LeRobot 数据流

LeRobot 的统计量来自 dataset metadata：

```54:55:/home/Luogang/SRC/Robot/lerobot/src/lerobot/datasets/utils.py
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
```

dataset 加载时会把它读到 `dataset.meta.stats`：

```163:169:/home/Luogang/SRC/Robot/lerobot/src/lerobot/datasets/lerobot_dataset.py
    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks = load_tasks(self.root)
        self.subtasks = load_subtasks(self.root)
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)
```

PI05 默认配置是：

```66:72:/home/Luogang/SRC/Robot/lerobot/src/lerobot/policies/pi05/configuration_pi05.py
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )
```

也就是：

- `VISUAL`: 不走 quantile
- `STATE`: 走 quantile
- `ACTION`: 走 quantile

processor 构建时，输入先做 `NormalizerProcessorStep`，输出再做 `UnnormalizerProcessorStep`：

```134:152:/home/Luogang/SRC/Robot/lerobot/src/lerobot/policies/pi05/processor_pi05.py
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
```

还可以直接从 dataset 构造 normalizer：

```412:439:/home/Luogang/SRC/Robot/lerobot/src/lerobot/processor/normalize_processor.py
    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        normalize_observation_keys: set[str] | None = None,
        eps: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> NormalizerProcessorStep:
        ...
        return cls(
            features=features,
            norm_map=norm_map,
            stats=dataset.meta.stats,
```

### 3.3 图像是否也做 quantile

不是。

这一点必须单独说清楚，因为很多人看到“归一化”会误以为 image 也走相同公式。

OpenPI 计算 norm stats 时只处理 `state/actions`，不处理 image：

```102:103:/mnt/r/share/lkx/pi/openpi/scripts/compute_norm_stats.py
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
```

LeRobot 的 PI05 也明确把 `VISUAL` 设成了 `IDENTITY`：

```68:70:/home/Luogang/SRC/Robot/lerobot/src/lerobot/policies/pi05/configuration_pi05.py
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
```

所以这里讨论的 Quantile 归一化，核心是 **state/action**，不是图像。

---

## 4. 为什么 PI05 更偏向 quantile，而不是 mean/std

这是一个“数据分布假设”的问题。

`mean/std` 更适合：

- 分布接近单峰
- 没有太强长尾
- 异常值不多

机器人状态和动作往往不是这种分布：

- 某些关节维度长期集中在狭窄范围
- 某些 gripper 维度会堆在两个端点
- 某些动作维度可能有明显截断或多峰分布

在这种情况下：

- `mean/std` 受长尾和极端样本影响较大
- `q01/q99` 更接近“有效工作区间”

所以从工程角度看，quantile 归一化更像是在问：

“模型真正经常看到的数据范围在哪里？”

而不是：

“假设数据像高斯分布那样围绕均值展开。”

---

## 5. 现有设计文档是怎么说的

`pi05_alig.md` 和 `pi05_alig_3.md` 都明确注意到了 OpenPI / LeRobot 的实现微差。

`pi05_alig.md` 的表述是：

```377:389:/home/Luogang/SRC/Robot/lerobot/bt/pi05/alig/pi05_alig.md
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
```

`pi05_alig_3.md` 的表述更简洁：

```807:820:/home/Luogang/SRC/Robot/lerobot/bt/pi05/alig/pi05_alig_3.md
#### P2-1: Quantile 归一化公式微差

| 项目 | OpenPI | LeRobot |
|------|--------|---------|
| 分母处理 | `q99 - q01 + 1e-6` (总是加) | `max(q99 - q01, 1e-8)` (仅零时替换) |
| 来源 | `transforms.py:144` | `normalize_processor.py:370-377` |

- 实际数据中 `q99 - q01 >> 1e-6`，差异可忽略

#### P2-2: Normalization Stats 来源

- OpenPI: `assets/r1_pro_data_convert_chassis/norm_stats.json`
- LeRobot: `dataset.meta.stats`
- 需要验证数值是否一致
```

这两份文档其实已经隐含了一个很重要的结论：

- **公式微差** 被认为可忽略
- **stats 来源差异** 并没有被认为可自动忽略，而是要求验证

---

## 6. 差异真的可以忽略吗：深入分析

这里要把两类问题分开。

## 6.1 第一类：公式实现差异

也就是这一对：

- OpenPI: `q99 - q01 + 1e-6`
- LeRobot: `denom == 0 ? 1e-8 : denom`

这类差异我认为 **通常可以忽略**，但要讲清楚为什么。

### 6.1.1 当 `q99 - q01` 很正常时

设：

```text
d = q99 - q01
```

如果 `d = 0.1`、`1.0`、`0.01` 这种正常量级：

- OpenPI 分母是 `d + 1e-6`
- LeRobot 分母是 `d`

相对误差大约是：

```text
1e-6 / d
```

例子：

| d | 相对误差量级 |
|---|-------------|
| `1.0` | `1e-6` |
| `0.1` | `1e-5` |
| `0.01` | `1e-4` |

这在训练中通常小到可以忽略。

### 6.1.2 当 `d` 很小但非零时

这才是真正值得注意的角落。

比如：

```text
d = 1e-5
```

则：

- OpenPI 分母 = `1.1e-5`
- LeRobot 分母 = `1e-5`

相对差异已经接近 `10%`

这时“可忽略”就不再是无条件成立的。

不过这种情况本身也说明一个问题：该维度几乎是常量维，已经缺少有效动态范围。对这种维度来说：

- 训练信息量本来就很低
- 是否用 `1e-6` 或 `1e-8` 往往不是主导问题
- 更大的问题是这维数据是否值得保留、是否应该特殊处理

所以这类情况我会判断为：

- **数值上不一定可忽略**
- **工程上通常也不是主导误差源**

### 6.1.3 当 `d = 0` 时

这是最极端情况。

若某维完全恒定：

- OpenPI 用分母 `1e-6`
- LeRobot 用分母 `1e-8`

如果 `x == q01`，两边都映射到 `-1`。

如果因为浮点误差或脏数据导致 `x != q01` 但 `q99 == q01`：

- LeRobot 的放大倍数会比 OpenPI 大 100 倍

但这仍然不是常规训练主路径，更像是：

- 统计量异常
- 某维退化成常量
- 数据清洗问题

结论是：

- **正常数据下**，公式差异可忽略
- **退化维度下**，差异不一定可忽略，但那已经是异常数据问题，而不是正常对齐问题

## 6.2 第二类：统计量来源差异

这类差异 **不能直接说可忽略**。

OpenPI 的 stats 来自：

- `assets/.../norm_stats.json`

LeRobot 的 stats 来自：

- `meta/stats.json`
- 运行时通过 `dataset.meta.stats` 读入

即便两边公式完全一样，只要 `q01/q99` 本身不一样，最终归一化结果就会明显不一样。

这是比 `1e-6` 和 `1e-8` 更值得担心的问题。

### 6.2.1 为什么 stats 差异更关键

归一化输出本质上取决于两个东西：

```text
output = f(x; q01, q99)
```

如果：

- OpenPI 用的 `q01 = -0.2, q99 = 0.3`
- LeRobot 用的 `q01 = -0.1, q99 = 0.4`

那即使公式完全相同，输出也会明显不同。

相反，如果两边 `q01/q99` 完全一致，只是一个用 `+1e-6`、一个用 `1e-8` fallback，那么大多数维度差异都极小。

### 6.2.2 LeRobot 和 OpenPI 的统计量计算方式并非完全同一实现

LeRobot 的 `compute_stats.py` 使用 `RunningQuantileStats`，也是基于 histogram 近似分位数：

```23:30:/home/Luogang/SRC/Robot/lerobot/src/lerobot/datasets/compute_stats.py
class RunningQuantileStats:
    """
    Maintains running statistics for batches of vectors, including mean,
    standard deviation, min, max, and approximate quantiles.

    Statistics are computed per feature dimension and updated incrementally
    as new batches are observed. Quantiles are estimated using histograms,
```

并默认计算：

```20:20:/home/Luogang/SRC/Robot/lerobot/src/lerobot/datasets/compute_stats.py
DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
```

OpenPI 的 `RunningStats` 也是 histogram-based quantile 估计。两边理念相近，但不是同一个实现、也未必有同样的 bin 设置、更新顺序或数据预处理路径。

所以“统计量应近似一致”是合理预期，但不是无需验证的数学定理。

## 6.3 第三类：模型选择差异

OpenPI 里：

- `PI0` 默认不用 quantile
- `PI05` 默认用 quantile

这件事本身说明：Quantile 归一化不是一个“放之四海而皆准”的中性小细节，而是跟模型族、输入建模方式、训练习惯绑定的设计选择。

所以如果要做 `PI05` 对齐，就应该认真核对这条链，而不是把它当成完全次要问题。

---

## 7. 我对“差异可忽略吗”的结论

这里必须分层回答，不能一句话带过。

### 7.1 公式层面

我的结论：**大多数正常数据下，可以忽略。**

条件是：

- `q99 - q01` 不是极小值
- 该维不是常量或近常量
- 没有明显脏数据导致统计量退化

在这个前提下：

- `+1e-6` 和 `denom==0 ? 1e-8 : denom` 的差异通常远小于训练中的其它误差源
- `pi05_alig.md` 和 `pi05_alig_3.md` 的这个判断基本站得住

### 7.2 工程整体对齐层面

我的结论：**不能直接说可忽略，必须验证 stats。**

更准确地说：

- “公式差异可忽略” 我基本同意
- “OpenPI 与 LeRobot 的 quantile 归一化整体差异可忽略” 我不同意直接下这个结论

因为整体对齐更依赖：

- `q01/q99` 是不是来自同一份数据
- 统计时是否经过相同 repack / 过滤 / padding / 维度选择
- state/action 维度定义是否完全一致

这些一旦不一致，误差远大于 `1e-6` vs `1e-8`。

### 7.3 风险分级

| 差异项 | 风险级别 | 我是否认为可忽略 | 原因 |
|---|---|---|---|
| `+1e-6` vs `zero->1e-8` | 低 | 通常可忽略 | 正常动态范围下误差极小 |
| `q99-q01` 极小的近常量维 | 中 | 不能一概忽略 | 相对误差会放大 |
| OpenPI `norm_stats.json` vs LeRobot `meta/stats.json` 数值不同 | 高 | 不能忽略 | 直接改变归一化结果 |
| 维度定义或特征顺序不一致 | 高 | 绝不能忽略 | 会把归一化应用到错误维度 |
| `VISUAL` 是否误用了 quantile | 高 | 绝不能忽略 | 会破坏图像输入语义 |

---

## 8. 结论摘要

### 8.1 什么是 Quantile 归一化

它是用 `q01/q99` 把 **state/action** 的主数据区间线性映射到 `[-1, 1]`，比 `mean/std` 更稳健，也更适合真实机器人数据的长尾和多峰分布。

### 8.2 它有什么作用

- 稳定数值范围
- 减少不同维度量纲差
- 提高离散化和策略学习的稳定性
- 降低异常值对缩放区间的破坏

### 8.3 OpenPI 和 LeRobot 的差异能不能忽略

最准确的结论是：

1. **公式微差通常可以忽略。**
2. **整体归一化对齐不能因为公式微差可忽略，就自动认为没问题。**
3. **真正需要重点验证的是 stats 内容是否一致，而不是 `1e-6` 和 `1e-8`。**

如果你要做 PI05/OpenPI 对齐，这件事最稳妥的判断标准不是“看公式差不多”，而是：

- 对同一批 state/action
- 分别喂给 OpenPI 和 LeRobot 的归一化实现
- 直接比较输出误差

只有这样，才能真正回答“这部分差异是否可忽略”。

