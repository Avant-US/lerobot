# StrGroot Policy 改良计划 — refactor1_c

> 日期: 2026-04-02
> 目标: 将 `str_groot` policy 从依赖本地 `./starVLA` 目录迁移到 pip 安装的 starVLA 包,
> 并修复现有设计问题、提升性能和健壮性。

---

## 一、背景

- **starVLA 代码来源**: `https://github.com/starVLA/starVLA` (唯一权威来源)
- **本地 `./starVLA` 目录**: 过时且错误, 将被删除
- **安装方式**: `pip install -e .` + `pip install -r requirements.txt` + `pip install flash-attn --no-build-isolation`
  (注: starVLA 的 `pyproject.toml` 中 `dependencies=[]` 为空, 实际依赖在 `requirements.txt` 中)
- **`deployment` 包**: 包含在 starVLA 仓库中, pip 安装后自动可用, 无需 mock
- **涉及文件范围**: 仅修改 `src/lerobot/policies/str_groot/` 和 `bt/str_groot_1/`

---

## 二、现有问题清单

### 问题 1: get_optim_params() 不支持差异化学习率 [严重]

**文件**: `modeling_str_groot.py:207-208`

```python
def get_optim_params(self):
    return self.parameters()  # 返回所有参数, 无分组
```

Config 中定义了 `optimizer_lr_vlm`、`optimizer_lr_action_head`、`tune_vlm`、`tune_action_head`,
但 `get_optim_params()` 完全忽略这些配置, 返回所有参数且不做分组。导致:
- VLM 和 Action Head 无法使用不同学习率
- 被冻结的参数虽然不会更新梯度, 但仍被传入优化器, 浪费内存

starVLA 原生支持:
```yaml
trainer:
  learning_rate:
    base: 2.5e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
```

### 问题 2: _batch_to_examples() 效率低下 [高]

**文件**: `modeling_str_groot.py:250-288`

每个 batch element 的每张图片都要经历 GPU Tensor -> clamp/uint8 -> CPU -> PIL Image 的转换。
当前逐元素 `.cpu()` 调用会触发多次 GPU-CPU 同步:

```
当前: 每张图片单独 batch[key][i].cpu() -> 多次 GPU-CPU 同步
改良: 先在 GPU 上批量处理, 一次性 batch[key].cpu() -> 仅一次同步
```

注: PIL 转换不可避免, 因为 starVLA 的 `QwenVL.build_qwenvl_inputs` -> `process_vision_info`
(来自 `qwen_vl_utils`) **必须接收 PIL Image**。

### 问题 3: sys.path hack 和 deployment mock 应移除 [高]

**文件**: `modeling_str_groot.py:26-62`

**Line 29-31** 的 sys.path hack:
```python
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
```
pip 安装后不需要, 且有害 (project root 加入 sys.path 可能导致模块名冲突)。

**Line 37-62** 的 deployment mock:
`deployment` 目录存在于 starVLA GitHub 仓库中, 且 `pyproject.toml` 没有 exclude 它。
pip 安装 starVLA 后 `deployment` 包自动可用, 整个 mock 代码块变为不必要的死代码。

### 问题 4: Config 静默覆盖 chunk_size/n_action_steps [中]

**文件**: `configuration_str_groot.py:82-83`

```python
self.chunk_size = self.future_action_window_size + 1
self.n_action_steps = self.chunk_size
```

用户设置的 `chunk_size=16` 会被静默覆盖为 `8`, 没有任何提示。

### 问题 5: state_dim 静默自动调整 [中]

**文件**: `configuration_str_groot.py:120-122`

```python
if actual_dim != self.state_dim:
    self.state_dim = actual_dim  # 静默覆盖
```

当数据集 state 维度与 config 不匹配且未指定 `state_indices` 时, 静默改变 `state_dim`。
如果已加载 checkpoint (期望特定 `state_dim`), 这会导致 shape 不匹配。

### 问题 6: 无梯度检查点支持 [中]

包裹了 4B 参数的 Qwen3-VL 模型, 但没有实现 gradient checkpointing。
PI0 policy 支持此功能 (`configuration_pi0.py:74`), 可大幅降低显存占用。
starVLA 原生支持: `trainer.enable_gradient_checkpointing: true`

### 问题 7: Scheduler 配置硬编码 [中]

**文件**: `configuration_str_groot.py:143-149`

```python
def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=int(10000 * self.warmup_ratio),  # 硬编码 10000
        num_decay_steps=10000,                              # 硬编码 10000
        ...
    )
```

warmup/decay steps 基于硬编码的 10000 而非实际训练步数。

### 问题 8: action_delta_indices 语义可能错误 [低]

**文件**: `configuration_str_groot.py:156-157`

```python
def action_delta_indices(self) -> list[int]:
    return list(range(self.chunk_size))
```

返回所有维度为 delta, 但例如 gripper 开合是绝对值而非增量。
StarVLA 使用的是绝对动作值, 不是增量。

### 问题 9: 无 PEFT/LoRA 支持 [低]

PI0 定义了 `_get_default_peft_targets()`, StrGroot 缺失此方法。

### 问题 10: 训练脚本创建无关目录 [低]

**文件**: `train_str_groot_libero.py:163-166`

```python
ckpt_path = Path(args.starvla_checkpoint)
if ckpt_path.is_absolute() and not ckpt_path.exists():
    ckpt_path.mkdir(parents=True, exist_ok=True)  # 错误: 这是 checkpoint 路径不是输出目录
```

### 问题 11: Checkpoint 加载安全风险 [低]

**文件**: `modeling_str_groot.py:192`

`torch.load(..., weights_only=False)` 允许反序列化任意 Python 对象。

---

## 三、改良方案

### Phase 0: starVLA 包迁移适配

#### 0.1 移除 sys.path hack 和 deployment mock

**修改文件**: `src/lerobot/policies/str_groot/modeling_str_groot.py`

**删除 line 26-31** (sys.path hack):
```python
# 删除以下代码:
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
```

**删除 line 37-62** (deployment mock 整个代码块)

**清理不再需要的 import**: `sys`, `types`, `Path`, `np` (numpy), `Image` (PIL.Image)

改动后 import 区域变为:
```python
import logging
import os
from collections import deque
from typing import TypeVar

import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.str_groot.configuration_str_groot import StrGrootConfig

logger = logging.getLogger(__name__)
T = TypeVar("T", bound="StrGrootPolicy")
```

#### 0.2 创建 bt/str_groot_1/setup_env.sh

初始化脚本, 负责安装 starVLA 及所有依赖。支持 `STARVLA_REPO_PATH` 变量指向已有代码库。

**冲突检测策略**: 安装每个依赖之前, 用 `pip install --dry-run` 做预检。
如果发现版本冲突 (已安装的包版本与 starVLA 要求不兼容), 脚本会:
1. 列出所有冲突的包名、当前版本、要求版本
2. 暂停安装, 让用户人工决定如何解决
3. 用户解决冲突后可重新运行脚本继续

```bash
#!/bin/bash
# str_groot 环境初始化脚本
# 在安装依赖时检测版本冲突, 冲突时停下来让用户解决
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 如果已有 starVLA 代码库, 设置此变量跳过克隆:
#   STARVLA_REPO_PATH=/path/to/starVLA bash bt/str_groot_1/setup_env.sh
STARVLA_REPO_PATH="${STARVLA_REPO_PATH:-}"

# ----------------------------------------------------------------
# 辅助函数: 冲突检测
# 使用 pip install --dry-run 预检, 捕获冲突信息
# ----------------------------------------------------------------
check_conflicts() {
  local req_file="$1"
  echo "  检测依赖冲突..."
  local conflict_output
  conflict_output=$(pip install --dry-run -r "$req_file" 2>&1) || true

  # 检查是否存在不兼容提示
  local conflicts
  conflicts=$(echo "$conflict_output" | grep -iE "(incompatible|conflict|ERROR)" || true)

  if [ -n "$conflicts" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ⚠  检测到依赖冲突, 请先手动解决再重新运行此脚本          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "冲突详情:"
    echo "$conflicts"
    echo ""
    echo "完整的 dry-run 输出:"
    echo "$conflict_output" | tail -30
    echo ""
    echo "建议:"
    echo "  1. 检查当前环境中冲突包的版本: pip show <package_name>"
    echo "  2. 决定保留哪个版本, 手动 pip install <package>==<version>"
    echo "  3. 或创建新的虚拟环境: python -m venv .venv && source .venv/bin/activate"
    echo "  4. 解决冲突后重新运行: bash $0"
    exit 1
  fi
  echo "  未检测到冲突。"
}

check_single_pkg_conflict() {
  local pkg="$1"
  local install_args="${2:-}"
  echo "  检测 $pkg 冲突..."
  local conflict_output
  conflict_output=$(pip install --dry-run $install_args "$pkg" 2>&1) || true

  local conflicts
  conflicts=$(echo "$conflict_output" | grep -iE "(incompatible|conflict|ERROR)" || true)

  if [ -n "$conflicts" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ⚠  安装 $pkg 与当前环境冲突                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "冲突详情:"
    echo "$conflicts"
    echo ""
    echo "建议: 手动解决冲突后重新运行此脚本。"
    exit 1
  fi
}

# ================================================================
# Step 1/4: 获取 starVLA 代码库
# ================================================================
echo "==> 1/4 获取 starVLA 代码库..."
if [ -z "$STARVLA_REPO_PATH" ]; then
  STARVLA_REPO_PATH="${REPO_ROOT}/.cache/starVLA_repo"
  if [ ! -d "$STARVLA_REPO_PATH" ]; then
    git clone --depth 1 https://github.com/starVLA/starVLA "$STARVLA_REPO_PATH"
  else
    echo "  已存在: $STARVLA_REPO_PATH, 跳过克隆。"
  fi
else
  echo "  使用已有代码库: $STARVLA_REPO_PATH"
fi

if [ ! -f "$STARVLA_REPO_PATH/requirements.txt" ]; then
  echo "ERROR: $STARVLA_REPO_PATH/requirements.txt 不存在, 请检查路径是否正确。"
  exit 1
fi

# ================================================================
# Step 2/4: 安装 starVLA 的依赖 (requirements.txt) — 先检测冲突
# ================================================================
echo "==> 2/4 安装 starVLA 的依赖 (requirements.txt)..."
check_conflicts "$STARVLA_REPO_PATH/requirements.txt"
pip install -r "$STARVLA_REPO_PATH/requirements.txt"

# ================================================================
# Step 3/4: 安装 flash-attn — 先检测冲突
# ================================================================
echo "==> 3/4 安装 flash-attn..."
check_single_pkg_conflict "flash-attn" "--no-build-isolation"
pip install flash-attn --no-build-isolation || {
  echo "WARNING: flash-attn 安装失败。"
  echo "  可能原因: 缺少 CUDA toolkit 或编译工具链。"
  echo "  可手动安装: pip install flash-attn --no-build-isolation"
  echo "  或跳过 (starVLA 可在无 flash-attn 下运行, 但性能较低)。"
}

# ================================================================
# Step 4/4: 安装 starVLA 包 — 先检测冲突
# ================================================================
echo "==> 4/4 安装 starVLA 包..."
check_single_pkg_conflict "$STARVLA_REPO_PATH"
pip install -e "$STARVLA_REPO_PATH"

# ================================================================
# 验证安装
# ================================================================
echo "==> 验证安装..."
python -c "from starVLA.model.framework.QwenGR00T import Qwen_GR00T; print('starVLA import OK')"
python -c "from deployment.model_server.tools.image_tools import to_pil_preserve; print('deployment import OK')"
echo ""
echo "==> 安装完成! starVLA 代码库路径: $STARVLA_REPO_PATH"
```

#### 0.3 创建 bt/str_groot_1/requirements.txt

```
# str_groot 额外依赖 (starVLA 及其依赖通过 setup_env.sh 安装)
qwen-vl-utils
omegaconf
```

---

### Phase 1: 核心正确性修复

#### 1.1 实现差异化学习率的参数分组

**修改文件**: `modeling_str_groot.py`

将 `get_optim_params()` 从返回 `self.parameters()` 改为返回参数分组列表:

```python
def get_optim_params(self) -> list[dict]:
    """Return parameter groups with per-component learning rates.

    Aligns with starVLA's native per-module LR support:
      qwen_vl_interface -> optimizer_lr_vlm
      action_model      -> optimizer_lr_action_head
    """
    groups = []
    if self.config.tune_vlm:
        vlm_params = [
            p for p in self._starvla_model.qwen_vl_interface.parameters()
            if p.requires_grad
        ]
        if vlm_params:
            groups.append({
                "params": vlm_params,
                "lr": self.config.optimizer_lr_vlm,
            })
    if self.config.tune_action_head:
        action_params = [
            p for p in self._starvla_model.action_model.parameters()
            if p.requires_grad
        ]
        if action_params:
            groups.append({
                "params": action_params,
                "lr": self.config.optimizer_lr_action_head,
            })
    if not groups:
        raise ValueError(
            "No trainable parameters: both tune_vlm and tune_action_head are False"
        )
    return groups
```

**原理**: lerobot 的 `optim/factory.py` 调用 `policy.get_optim_params()` 并传给 `AdamW`。
PyTorch 的 `AdamW` 原生支持参数组列表 (每组可指定独立 lr)。

#### 1.2 修复 Scheduler 硬编码

**修改文件**: `configuration_str_groot.py`

添加 `scheduler_total_steps` 字段, 用于计算 warmup/decay steps:

```python
scheduler_total_steps: int = 10000  # 默认值, 应由训练步数覆盖

def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=int(self.scheduler_total_steps * self.warmup_ratio),
        num_decay_steps=self.scheduler_total_steps,
        peak_lr=self.optimizer_lr,
        decay_lr=self.optimizer_lr * 0.1,
    )
```

#### 1.3 修复 action_delta_indices

**修改文件**: `configuration_str_groot.py`

将 `action_delta_indices` 改为可配置, 默认返回 `None` (表示不使用 delta):

```python
# 在 dataclass 字段中添加:
action_delta_dims: list[int] | None = None  # 默认 None = 所有动作都是绝对值

@property
def action_delta_indices(self) -> list[int] | None:
    return self.action_delta_dims
```

---

### Phase 2: 性能优化

#### 2.1 优化 _batch_to_examples 的图像转换

**修改文件**: `modeling_str_groot.py`

批量化 CPU 传输, 减少 GPU-CPU 同步次数:

```python
def _batch_to_examples(self, batch: dict[str, Tensor], inference: bool = False) -> list[dict]:
    image_keys = sorted(k for k in batch if k.startswith("observation.images."))
    if not image_keys:
        raise ValueError(f"No image keys found in batch. Available keys: {sorted(batch.keys())}")

    B = batch[image_keys[0]].shape[0]

    # --- 批量 GPU -> CPU 传输 (减少同步次数) ---
    images_cpu = {}
    for key in image_keys:
        imgs = batch[key]  # (B, C, H, W) on GPU
        if imgs.is_floating_point():
            imgs = (imgs.clamp(0, 1) * 255).to(torch.uint8)
        images_cpu[key] = imgs.cpu()  # 每个 key 仅一次 .cpu()

    actions_cpu = batch["action"].cpu().float().numpy() if not inference and "action" in batch else None
    state_cpu = batch["observation.state"].cpu().float().numpy() if "observation.state" in batch else None

    # --- 构建 examples ---
    examples: list[dict] = []
    for i in range(B):
        images = [to_pil_image(images_cpu[key][i]) for key in image_keys]

        example: dict = {
            "image": images,
            "lang": batch["task"][i] if "task" in batch else "",
        }

        if actions_cpu is not None:
            example["action"] = actions_cpu[i]

        if state_cpu is not None:
            state = state_cpu[i]
            if self.config.state_indices is not None:
                state = state[..., list(self.config.state_indices)]
            example["state"] = state.reshape(1, -1)

        examples.append(example)

    return examples
```

**优化效果**:
- 图像: 从 `B * num_image_keys` 次 `.cpu()` 减少到 `num_image_keys` 次
- action/state: 从 `B` 次减少到 `1` 次

#### 2.2 添加 Gradient Checkpointing 支持

**修改文件**: `configuration_str_groot.py` + `modeling_str_groot.py`

Config 添加:
```python
gradient_checkpointing: bool = False
```

Modeling `__init__` 中, 在 checkpoint 加载之后、冻结之前添加:
```python
if config.gradient_checkpointing:
    self._starvla_model.qwen_vl_interface.model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled for QwenVL backbone")
```

Qwen 系列 HuggingFace transformers 模型原生支持 `gradient_checkpointing_enable()`。

---

### Phase 3: 健壮性和可维护性

#### 3.1 Config 覆盖时添加警告

**修改文件**: `configuration_str_groot.py`

`__post_init__` 中:
```python
computed_chunk = self.future_action_window_size + 1
if self.chunk_size != computed_chunk:
    logger.warning(
        "chunk_size=%d overridden to %d (= future_action_window_size + 1)",
        self.chunk_size, computed_chunk,
    )
self.chunk_size = computed_chunk
self.n_action_steps = self.chunk_size
```

`validate_features()` 中 state_dim 自动调整时:
```python
if actual_dim != self.state_dim:
    logger.warning(
        "state_dim=%d auto-adjusted to %d to match dataset "
        "(set state_indices to select specific dims instead)",
        self.state_dim, actual_dim,
    )
    self.state_dim = actual_dim
```

#### 3.2 添加 PEFT 支持

**修改文件**: `modeling_str_groot.py`

```python
def _get_default_peft_targets(self) -> dict:
    return {
        "target_modules": [
            "q_proj", "v_proj",   # QwenVL attention
            "to_q", "to_v",      # DiT attention (action model)
        ],
        "modules_to_save": [],
    }
```

#### 3.3 添加 _build_starvla_config 中的映射注释

**修改文件**: `modeling_str_groot.py`

在 `_build_starvla_config()` 方法中添加注释, 说明各字段与 starVLA YAML config
(如 `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`) 的对应关系:

```python
def _build_starvla_config(self):
    """Build OmegaConf config for starVLA's Qwen_GR00T.

    Maps StrGrootConfig fields to the structure expected by
    starVLA.model.framework.QwenGR00T.Qwen_GR00T.__init__().

    Reference: starVLA YAML config (examples/LIBERO/train_files/starvla_cotrain_libero.yaml)
      framework.qwenvl.base_vlm           -> c.base_vlm
      framework.action_model.*             -> c.action_* / c.state_* / c.dit_*
      framework.action_model.diffusion_model_cfg.* -> DiT architecture params
      datasets.vla_data.image_size         -> c.image_size
    """
    ...
```

#### 3.4 修复 Checkpoint 加载安全

**修改文件**: `modeling_str_groot.py`

```python
# 改前:
state_dict = torch.load(local_path, map_location="cpu", weights_only=False)

# 改后:
state_dict = torch.load(local_path, map_location="cpu", weights_only=True)
```

#### 3.5 清理训练脚本

**修改文件**: `bt/str_groot_1/train_str_groot_libero.py`

1. **删除 line 163-166 的错误目录创建逻辑**:
```python
# 删除以下代码:
ckpt_path = Path(args.starvla_checkpoint)
if ckpt_path.is_absolute() and not ckpt_path.exists():
    ckpt_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("starvla_checkpoint 路径不存在，已创建目录: %s", ckpt_path)
```

2. **删除注释掉的参数** (line 68-74, 113, 116-117):
```python
# 删除以下注释代码:
# p.add_argument("--base-vlm", ...)
# p.add_argument("--action-dim", ...)
# p.add_argument("--state-dim", ...)
# base_vlm=args.base_vlm,
# action_dim=args.action_dim,
# state_dim=args.state_dim,
```

3. **修复 logging**: 移除 `logging.basicConfig(force=True)`, 改用 `logger = logging.getLogger(__name__)`

#### 3.6 清理评估脚本

**修改文件**: `bt/str_groot_1/eval_str_groot_libero.py`

1. **添加 `--stats-path` 参数**: 允许指定自定义 normalization stats 文件路径,
   不强制依赖 `--dataset-repo` 来获取 stats

2. **修复 logging**: 移除 `logging.basicConfig(force=True)`, 改用 `logger = logging.getLogger(__name__)`

---

## 四、架构设计

### 改良后的组件图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    StrGrootPolicy (PreTrainedPolicy)                │
│                                                                     │
│  ┌──────────────────────┐   ┌────────────────────────────────────┐  │
│  │  StrGrootConfig      │   │  Qwen_GR00T (_starvla_model)       │  │
│  │  (PreTrainedConfig)  │   │                                    │  │
│  │                      │   │  ┌──────────────┐ ┌─────────────┐  │  │
│  │  tune_vlm            │   │  │ QwenVL       │ │ FlowMatch   │  │  │
│  │  tune_action_head    │   │  │ Interface    │ │ ActionHead  │  │  │
│  │  optimizer_lr_vlm    │   │  │ (Qwen3-VL)  │ │ (DiT)       │  │  │
│  │  optimizer_lr_action │   │  │              │ │             │  │  │
│  │  gradient_ckpt       │   │  │  lr=lr_vlm  │ │ lr=lr_act   │  │  │
│  │  state_indices       │   │  └──────────────┘ └─────────────┘  │  │
│  └──────────────────────┘   └────────────────────────────────────┘  │
│                                                                     │
│  get_optim_params() -> [{params: vlm, lr: lr_vlm},                  │
│                         {params: act_head, lr: lr_action}]          │
│                                                                     │
│  forward(batch) -> loss, {"loss": loss.item()}                      │
│  select_action(batch) -> action (from action_queue)                 │
│  _get_default_peft_targets() -> {target_modules: [...]}             │
└─────────────────────────────────────────────────────────────────────┘
```

### 参数分组与冻结策略图

```
StrGrootPolicy
├── _starvla_model (Qwen_GR00T)
│   ├── qwen_vl_interface (QwenVL)          <- tune_vlm 控制
│   │   ├── model (Qwen3-VL-4B)            |  lr = optimizer_lr_vlm
│   │   │   ├── visual (vision tower)       |  (默认 optimizer_lr * 0.1)
│   │   │   ├── model (language model)      |
│   │   │   └── lm_head                     |
│   │   └── processor                       |  (不需要梯度)
│   │                                       |
│   └── action_model (FlowmatchingHead)     <- tune_action_head 控制
│       ├── dit_model (DiT-B)               |  lr = optimizer_lr_action_head
│       │   ├── transformer_blocks (x16)    |  (默认 = optimizer_lr)
│       │   └── norm/proj layers            |
│       ├── action_encoder                  |
│       ├── action_decoder                  |
│       ├── state_encoder (MLP)             |
│       └── future_tokens (learnable)       |
│
├── Gradient Checkpointing                  <- gradient_checkpointing 控制
│   └── 仅对 qwen_vl_interface.model 启用   |  节省显存, 增加计算
│
└── PEFT Targets                            <- LoRA fine-tuning
    ├── qwen_vl: q_proj, v_proj
    └── action_model: to_q, to_v
```

### 训练数据流序列图

```
User                 TrainPipeline        StrGrootPolicy        Qwen_GR00T
 │                       │                     │                     │
 │  train(cfg)           │                     │                     │
 │──────────────────────>│                     │                     │
 │                       │  make_policy(cfg)   │                     │
 │                       │────────────────────>│                     │
 │                       │                     │  __init__            │
 │                       │                     │  _build_starvla_cfg  │
 │                       │                     │────────────────────>│ Qwen_GR00T(cfg)
 │                       │                     │  _load_checkpoint   │
 │                       │                     │────────────────────>│ load_state_dict
 │                       │                     │  grad_ckpt (if)     │
 │                       │                     │  freeze_vlm (if)    │
 │                       │                     │<────────────────────│
 │                       │                     │                     │
 │                       │  get_optim_params() │                     │
 │                       │────────────────────>│                     │
 │                       │  [{vlm, lr_vlm},   │                     │
 │                       │   {act, lr_act}]    │                     │
 │                       │<────────────────────│                     │
 │                       │                     │                     │
 │                       │  for step in steps: │                     │
 │                       │  forward(batch)     │                     │
 │                       │────────────────────>│                     │
 │                       │                     │ _batch_to_examples  │
 │                       │                     │ (batch .cpu())      │
 │                       │                     │────────────────────>│ forward(examples)
 │                       │                     │                     │ VLM -> ActionHead
 │                       │                     │<────────────────────│ {action_loss}
 │                       │  loss, info         │                     │
 │                       │<────────────────────│                     │
 │                       │  loss.backward()    │                     │
 │                       │  optimizer.step()   │  vlm: lr_vlm        │
 │                       │                     │  act: lr_action_head │
```

### 推理数据流序列图

```
Environment          Preprocessor       StrGrootPolicy        Qwen_GR00T          Postprocessor
    │                     │                   │                    │                     │
    │  obs (raw)          │                   │                    │                     │
    │────────────────────>│                   │                    │                     │
    │                     │  normalize        │                    │                     │
    │                     │  device_move      │                    │                     │
    │                     │  batch (norm)     │                    │                     │
    │                     │──────────────────>│                    │                     │
    │                     │                   │ action_queue empty? │                    │
    │                     │                   │ YES:                │                    │
    │                     │                   │ _batch_to_examples  │                    │
    │                     │                   │───────────────────>│                     │
    │                     │                   │                    │ predict_action      │
    │                     │                   │                    │ VLM -> ActionHead   │
    │                     │                   │                    │ (N denoising steps) │
    │                     │                   │<───────────────────│ normalized_actions  │
    │                     │                   │ fill action_queue  │                     │
    │                     │                   │                    │                     │
    │                     │  action (norm)    │ queue.popleft()    │                     │
    │                     │<──────────────────│                    │                     │
    │                     │                   │                    │                     │
    │                     │──────────────────────────────────────────────────────────────>│
    │                     │                   │                    │    unnormalize       │
    │                     │                   │                    │    device_move(cpu)  │
    │  action (raw)       │                   │                    │                     │
    │<───────────────────────────────────────────────────────────────────────────────────│
    │  env.step(action)   │                   │                    │                     │
```

---

## 五、改良后的类图

```
                    ┌──────────────────────┐
                    │  PreTrainedConfig    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴──────────────────┐
              │         StrGrootConfig             │
              │ @register_subclass("str_groot")    │
              ├────────────────────────────────────┤
              │ + n_obs_steps: int = 1             │
              │ + chunk_size: int = 8              │
              │ + base_vlm: str                    │
              │ + action_model_type: str           │
              │ + action_dim / state_dim: int      │
              │ + state_indices: tuple | None      │
              │ + tune_vlm: bool                   │
              │ + tune_action_head: bool           │
              │ + optimizer_lr_vlm: float          │
              │ + optimizer_lr_action_head: float   │
              │ + gradient_checkpointing: bool     │  <- 新增
              │ + action_delta_dims: list | None   │  <- 新增
              │ + scheduler_total_steps: int       │  <- 新增
              ├────────────────────────────────────┤
              │ + validate_features()              │
              │ + get_optimizer_preset()            │
              │ + get_scheduler_preset()            │
              │ + observation_delta_indices         │
              │ + action_delta_indices              │  <- 改为可配置
              └────────────────┬───────────────────┘
                               │ uses
              ┌────────────────┴───────────────────┐
              │        StrGrootPolicy               │
              │   extends PreTrainedPolicy          │
              ├────────────────────────────────────-┤
              │ - _starvla_model: Qwen_GR00T       │
              │ - _action_queue: deque[Tensor]      │
              ├────────────────────────────────────┤
              │ + __init__(config)                  │
              │ + forward(batch) -> loss, info      │
              │ + select_action(batch) -> action    │
              │ + predict_action_chunk(batch)        │
              │ + get_optim_params() -> list[dict]  │  <- 改为参数分组
              │ + reset()                           │
              │ + _get_default_peft_targets()       │  <- 新增
              │ - _build_starvla_config()            │  <- 添加映射注释
              │ - _load_starvla_checkpoint()         │  <- 修复 weights_only
              │ - _batch_to_examples()               │  <- 优化批量传输
              └────────────────────────────────────┘
```

---

## 六、修改文件清单

| 文件 | 修改类型 | 改动内容 |
|------|---------|---------|
| `src/lerobot/policies/str_groot/modeling_str_groot.py` | 修改 | 删除 sys.path hack + deployment mock; 清理 import; 实现差异化 LR 参数分组; 优化 _batch_to_examples 批量传输; 添加 gradient checkpointing; 添加 PEFT targets; 修复 checkpoint weights_only; 添加 _build_starvla_config 映射注释 |
| `src/lerobot/policies/str_groot/configuration_str_groot.py` | 修改 | 添加 gradient_checkpointing, action_delta_dims, scheduler_total_steps 字段; 修复 scheduler 硬编码; 添加 chunk_size/state_dim 覆盖警告; 修复 action_delta_indices |
| `bt/str_groot_1/train_str_groot_libero.py` | 修改 | 删除错误目录创建; 清理注释代码; 修复 logging |
| `bt/str_groot_1/eval_str_groot_libero.py` | 修改 | 添加 --stats-path; 修复 logging |
| `bt/str_groot_1/setup_env.sh` | **新建** | 初始化脚本: 克隆 starVLA -> 安装依赖 -> flash-attn -> pip install -> 验证 |
| `bt/str_groot_1/requirements.txt` | **新建** | 记录 str_groot 额外依赖 |
| `bt/str_groot_1/readme.md` | 修改 | 补充环境初始化说明, 文件用途清单, 完整工作流 |

---

## 七、各文件完整修改 Diff

### 7.1 `src/lerobot/policies/str_groot/modeling_str_groot.py`

#### 7.1.1 Import 区域 (line 1-62 -> 1-16)

删除 `sys`, `types`, `Path`, `np`, `Image` import, 删除 sys.path hack 和 deployment mock:

```python
# === 改后 (完整 import 区域) ===
"""StrGroot Policy — adapts StarVLA's Qwen_GR00T as a LeRobot PreTrainedPolicy."""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import TypeVar

import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.str_groot.configuration_str_groot import StrGrootConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="StrGrootPolicy")
```

#### 7.1.2 __init__ 方法 (添加 gradient checkpointing)

```python
def __init__(self, config: StrGrootConfig, **kwargs):
    super().__init__(config)
    config.validate_features()
    self.config = config

    starvla_cfg = self._build_starvla_config()

    from starVLA.model.framework.QwenGR00T import Qwen_GR00T

    self._starvla_model = Qwen_GR00T(starvla_cfg)

    if config.starvla_checkpoint:
        self._load_starvla_checkpoint(config.starvla_checkpoint)

    if config.gradient_checkpointing:
        self._starvla_model.qwen_vl_interface.model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for QwenVL backbone")

    if config.freeze_vlm:
        for p in self._starvla_model.qwen_vl_interface.parameters():
            p.requires_grad = False

    self.reset()
```

#### 7.1.3 _build_starvla_config (添加映射注释)

```python
def _build_starvla_config(self):
    """Build OmegaConf config for starVLA's Qwen_GR00T.

    Maps StrGrootConfig fields to the structure expected by
    starVLA.model.framework.QwenGR00T.Qwen_GR00T.__init__().

    Reference YAML: starVLA/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
      framework.qwenvl.base_vlm                    <- c.base_vlm
      framework.action_model.action_dim             <- c.action_dim
      framework.action_model.state_dim              <- c.state_dim
      framework.action_model.action_hidden_dim      <- c.action_hidden_dim
      framework.action_model.future_action_window_size <- c.future_action_window_size
      framework.action_model.diffusion_model_cfg.*  <- c.dit_* / c.cross_attention_dim
      datasets.vla_data.image_size                  <- c.image_size
      trainer.repeated_diffusion_steps              <- c.repeated_diffusion_steps
    """
    from omegaconf import OmegaConf
    # ... (方法体不变)
```

#### 7.1.4 _load_starvla_checkpoint (修复 weights_only)

```python
# 改前:
state_dict = torch.load(local_path, map_location="cpu", weights_only=False)

# 改后:
state_dict = torch.load(local_path, map_location="cpu", weights_only=True)
```

#### 7.1.5 get_optim_params (差异化学习率)

```python
# === 改后 ===
def get_optim_params(self) -> list[dict]:
    """Return parameter groups with per-component learning rates.

    Aligns with starVLA's native per-module LR support:
      qwen_vl_interface -> optimizer_lr_vlm
      action_model      -> optimizer_lr_action_head
    """
    groups = []
    if self.config.tune_vlm:
        vlm_params = [
            p for p in self._starvla_model.qwen_vl_interface.parameters()
            if p.requires_grad
        ]
        if vlm_params:
            groups.append({
                "params": vlm_params,
                "lr": self.config.optimizer_lr_vlm,
            })
    if self.config.tune_action_head:
        action_params = [
            p for p in self._starvla_model.action_model.parameters()
            if p.requires_grad
        ]
        if action_params:
            groups.append({
                "params": action_params,
                "lr": self.config.optimizer_lr_action_head,
            })
    if not groups:
        raise ValueError(
            "No trainable parameters: both tune_vlm and tune_action_head are False"
        )
    return groups
```

#### 7.1.6 _batch_to_examples (优化批量传输)

```python
# === 改后 ===
def _batch_to_examples(
    self,
    batch: dict[str, Tensor],
    inference: bool = False,
) -> list[dict]:
    image_keys = sorted(k for k in batch if k.startswith("observation.images."))
    if not image_keys:
        raise ValueError(
            f"No image keys found in batch. Available keys: {sorted(batch.keys())}"
        )

    B = batch[image_keys[0]].shape[0]

    # Batch GPU -> CPU transfer: one .cpu() call per key instead of per element
    images_cpu = {}
    for key in image_keys:
        imgs = batch[key]  # (B, C, H, W)
        if imgs.is_floating_point():
            imgs = (imgs.clamp(0, 1) * 255).to(torch.uint8)
        images_cpu[key] = imgs.cpu()

    actions_np = batch["action"].cpu().float().numpy() if not inference and "action" in batch else None
    state_np = batch["observation.state"].cpu().float().numpy() if "observation.state" in batch else None

    examples: list[dict] = []
    for i in range(B):
        images = [to_pil_image(images_cpu[key][i]) for key in image_keys]

        example: dict = {
            "image": images,
            "lang": batch["task"][i] if "task" in batch else "",
        }

        if actions_np is not None:
            example["action"] = actions_np[i]

        if state_np is not None:
            state = state_np[i]
            if self.config.state_indices is not None:
                state = state[..., list(self.config.state_indices)]
            example["state"] = state.reshape(1, -1)

        examples.append(example)

    return examples
```

#### 7.1.7 添加 PEFT targets

```python
# 在 _batch_to_examples 之后, 类的末尾添加:
def _get_default_peft_targets(self) -> dict:
    return {
        "target_modules": [
            "q_proj", "v_proj",   # QwenVL attention
            "to_q", "to_v",      # DiT attention (action model)
        ],
        "modules_to_save": [],
    }
```

---

### 7.2 `src/lerobot/policies/str_groot/configuration_str_groot.py`

#### 7.2.1 新增字段

```python
# 在 starvla_checkpoint 字段之后 (约 line 69) 添加:
gradient_checkpointing: bool = False

# 在 state_indices 字段之后 (约 line 49) 添加:
action_delta_dims: list[int] | None = None

# 在 warmup_ratio 字段之后 (约 line 78) 添加:
scheduler_total_steps: int = 10000
```

#### 7.2.2 __post_init__ 添加覆盖警告

```python
# 在文件顶部添加 import:
import logging
logger = logging.getLogger(__name__)

# 修改 __post_init__:
def __post_init__(self):
    super().__post_init__()
    computed_chunk = self.future_action_window_size + 1
    if self.chunk_size != computed_chunk:
        logger.warning(
            "chunk_size=%d overridden to %d (= future_action_window_size + 1)",
            self.chunk_size, computed_chunk,
        )
    self.chunk_size = computed_chunk
    self.n_action_steps = self.chunk_size

    # ... 其余不变 ...
```

#### 7.2.3 validate_features 添加 state_dim 调整警告

```python
# 在 validate_features() 中, 修改 state_dim 自动调整部分:
else:
    if actual_dim != self.state_dim:
        logger.warning(
            "state_dim=%d auto-adjusted to %d to match dataset "
            "(set state_indices to select specific dims instead)",
            self.state_dim, actual_dim,
        )
        self.state_dim = actual_dim
```

#### 7.2.4 修复 get_scheduler_preset

```python
def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=int(self.scheduler_total_steps * self.warmup_ratio),
        num_decay_steps=self.scheduler_total_steps,
        peak_lr=self.optimizer_lr,
        decay_lr=self.optimizer_lr * 0.1,
    )
```

#### 7.2.5 修复 action_delta_indices

```python
@property
def action_delta_indices(self) -> list[int] | None:
    return self.action_delta_dims
```

---

### 7.3 `bt/str_groot_1/train_str_groot_libero.py`

#### 7.3.1 修复 logging (line 28-34)

```python
# 改前:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
LOGGER = logging.getLogger(__name__)

# 改后:
LOGGER = logging.getLogger(__name__)
```

#### 7.3.2 删除注释掉的参数 (line 68-74)

```python
# 删除以下注释代码:
# p.add_argument(
#     "--base-vlm",
#     default="Qwen/Qwen3-VL-4B-Instruct",
#     help="底层 VLM 模型名",
# )
# p.add_argument("--action-dim", type=int, default=7)
# p.add_argument("--state-dim", type=int, default=7)
```

#### 7.3.3 删除注释掉的 config 字段 (line 113, 116-117)

```python
# 在 build_train_config 中删除:
# base_vlm=args.base_vlm,
# action_dim=args.action_dim,
# state_dim=args.state_dim,
```

#### 7.3.4 删除错误目录创建 (line 163-166)

```python
# 删除以下代码:
ckpt_path = Path(args.starvla_checkpoint)
if ckpt_path.is_absolute() and not ckpt_path.exists():
    ckpt_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("starvla_checkpoint 路径不存在，已创建目录: %s", ckpt_path)
```

---

### 7.4 `bt/str_groot_1/eval_str_groot_libero.py`

#### 7.4.1 修复 logging (line 39-45)

```python
# 改前:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
LOGGER = logging.getLogger(__name__)

# 改后:
LOGGER = logging.getLogger(__name__)
```

#### 7.4.2 添加 --stats-path 参数

```python
# 在 parse_args 中添加:
p.add_argument(
    "--stats-path",
    default=None,
    help="Path to custom normalization stats JSON (overrides --dataset-repo stats)",
)
```

在 `main()` 中使用:
```python
# 改前:
LOGGER.info("Loading dataset stats from %s ...", args.dataset_repo)
ds_meta = LeRobotDatasetMetadata(repo_id=args.dataset_repo)
dataset_stats = ds_meta.stats

# 改后:
if args.stats_path:
    import json as _json
    LOGGER.info("Loading custom stats from %s ...", args.stats_path)
    with open(args.stats_path) as f:
        raw = _json.load(f)
    dataset_stats = {
        k: {sk: torch.tensor(sv) for sk, sv in v.items()}
        for k, v in raw.items()
    }
else:
    LOGGER.info("Loading dataset stats from %s ...", args.dataset_repo)
    ds_meta = LeRobotDatasetMetadata(repo_id=args.dataset_repo)
    dataset_stats = ds_meta.stats
```

---

## 八、验证计划

1. **环境初始化验证**: `bash bt/str_groot_1/setup_env.sh && python -c "from starVLA.model.framework.QwenGR00T import Qwen_GR00T; print('OK')"`

2. **参数分组验证**: 构建 policy 后打印 `policy.get_optim_params()`, 确认:
   - 返回 `list[dict]`
   - 每组有正确的 `lr` (vlm 组用 `optimizer_lr_vlm`, action 组用 `optimizer_lr_action_head`)
   - 仅包含 `requires_grad=True` 的参数

3. **前向传播验证**: `bash bt/str_groot_1/test_random37_lerobot_train.sh`

4. **完整训练验证** (冒烟测试):
   ```bash
   python bt/str_groot_1/train_str_groot_libero.py \
     --steps 2 --batch-size 1 --freeze-vlm --starvla-checkpoint "" --num-workers 0
   ```

5. **评估验证**:
   ```bash
   python bt/str_groot_1/eval_str_groot_libero.py --dry-run
   ```

6. **显存验证**: 对比 `gradient_checkpointing=True/False` 时的 `torch.cuda.max_memory_allocated()`

7. **回归测试**: `pytest tests/ -k "str_groot" -vv` (如有相关测试)

---

## 九、实施顺序

按 Phase 顺序执行, 每个 Phase 完成后验证:

1. **Phase 0** (迁移适配): setup_env.sh -> requirements.txt -> 删除 sys.path hack/mock -> 清理 import
2. **Phase 1** (正确性): get_optim_params -> scheduler -> action_delta_indices
3. **Phase 2** (性能): _batch_to_examples 优化 -> gradient checkpointing
4. **Phase 3** (健壮性): config 警告 -> PEFT -> config 注释 -> checkpoint 安全 -> 训练脚本清理 -> 评估脚本清理
5. **文档**: 更新 readme.md

---

## 十、TODO: RotationTransform 与 pipablepytorch3d

### 10.1 背景

starVLA 的数据加载管线 (`starVLA/dataloader/gr00t_lerobot/transform/state_action.py`)
使用 `pytorch3d.transforms` 实现 `RotationTransform`, 用于在不同旋转表示之间转换:

```
axis_angle (3D) ↔ matrix (3x3) ↔ rotation_6d (6D)
euler_angles (3D) ↔ matrix (3x3) ↔ quaternion (4D)
```

调用链: `lerobot_datasets.py` → `data_config.py` → `state_action.py` → `pytorch3d.transforms`

使用的 `pytorch3d.transforms` 函数:
- `axis_angle_to_matrix`, `matrix_to_axis_angle`
- `euler_angles_to_matrix`, `matrix_to_euler_angles`
- `quaternion_to_matrix`, `matrix_to_quaternion`
- `matrix_to_rotation_6d`, `rotation_6d_to_matrix`

### 10.2 pipablepytorch3d 现状

- `pipablepytorch3d` 是 pytorch3d 的 pip 可安装重打包版本
- **仅支持 Python < 3.12**, 当前 lerobot-venv 是 Python 3.12.13, 无法安装
- 官方 `pytorch3d` 也没有 3.12 的预编译 wheel, 只能从源码编译 (需 CUDA toolkit, 耗时 10-30 分钟)

### 10.3 与 str_groot 的关系

**当前不需要 pytorch3d.** 原因:

1. str_groot policy 使用 **lerobot 自己的 dataset pipeline** 加载数据,
   不经过 starVLA 的 `lerobot_datasets.py` → `data_config.py` 路径
2. `QwenGR00T` 模型的 `forward()` / `predict_action()` 不依赖 pytorch3d
3. `state_action.py` 的 import 在 `transform/__init__.py` 中已被注释掉:
   ```python
   # from .state_action import (
   #     StateActionDropout,
   #     ...
   # )
   ```
4. 只有使用 starVLA 自己的 dataloader (`starVLA.dataloader.lerobot_datasets`) 时才会触发 pytorch3d 导入

### 10.4 RotationTransform 对 VLA 效果的影响分析

**为什么旋转表示很重要:**

不同旋转表示在 SO(3) (三维旋转群) 上的拓扑性质不同:

| 表示 | 维度 | 连续性 | 问题 |
|------|------|--------|------|
| euler_angles | 3 | 不连续 | 万向锁 (gimbal lock), ±π 跳变 |
| axis_angle | 3 | 不连续 | 零旋转附近方向未定义, 2π 周期跳变 |
| quaternion | 4 | 双覆盖 | q 和 -q 表示同一旋转, 需额外约束 |
| rotation_6d | 6 | **连续** | 冗余但对网络友好 (Zhou et al. 2019) |

用不连续表示 (euler/axis_angle) 做回归时, 在不连续边界附近损失曲面剧烈变化,
网络难以学好。rotation_6d 是 SO(3) 上的连续映射, 损失曲面平滑, 学习更稳定。

**对当前 str_groot 的实际影响:**

- **短期影响小**: LIBERO 等桌面操作任务旋转幅度有限, 远离奇异点;
  且 str_groot 的 FlowMatching action head 是生成式模型, 比纯 MSE 回归对不连续性更鲁棒
- **长期泛化时影响大**: 如果要扩展到大旋转任务 (翻转物体、工具使用、灵巧手),
  不做旋转表示转换可能导致旋转预测精度下降, 成功率降低几到十几个百分点

### 10.5 后续方案 (待实施)

如果需要在 lerobot pipeline 中支持 RotationTransform, 有以下方案:

**方案 A: 纯 PyTorch 实现旋转转换函数 (推荐)**

`pytorch3d.transforms` 中用到的旋转转换函数本质上是纯数学运算, 不依赖 CUDA 扩展。
可用纯 PyTorch 在几十行内实现, 完全避免 pytorch3d 依赖:

```python
# 示例: rotation_6d_to_matrix (Zhou et al. 2019)
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].flatten(start_dim=-2)
```

类似地, `axis_angle_to_matrix`, `euler_angles_to_matrix`, `quaternion_to_matrix`
及其逆函数都可以用 Rodrigues 公式等纯 PyTorch 实现。

实现位置: `src/lerobot/policies/str_groot/rotation_utils.py` (新建)
或作为 lerobot 通用工具: `src/lerobot/utils/rotation.py`

**方案 B: 从 GitHub 源码编译 pytorch3d**

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

官方 pytorch3d 的 `setup.py` 没有 Python 版本限制, 可以在 3.12 上编译。
缺点: 需要 CUDA toolkit + C++ 编译器, 编译耗时 10-30 分钟, 增加环境复杂度。

**方案 C: 只拷贝 pytorch3d.transforms 子模块**

```bash
git clone --depth 1 https://github.com/facebookresearch/pytorch3d /tmp/pytorch3d
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/pytorch3d"
cp -r /tmp/pytorch3d/pytorch3d/transforms "$SITE/pytorch3d/"
touch "$SITE/pytorch3d/__init__.py"
```

`pytorch3d/transforms/` 是纯 Python 代码, 不依赖 C++/CUDA 扩展。
缺点: 非标准安装方式, 不便于版本管理和升级。

### 10.6 优先级与时间线

- **当前**: 不安装 pytorch3d, 不影响 str_groot 训练和评估
- **Phase 4 (后续)**: 如需支持大旋转任务, 实施方案 A (纯 PyTorch 实现),
  在 processor pipeline 中添加 RotationTransform 步骤
