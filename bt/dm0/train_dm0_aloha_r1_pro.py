#!/usr/bin/env python3
"""用 ALOHA 预训练 DM0 checkpoint 微调 R1-Pro 7-DoF 人形机器人.

ALOHA (src) 与 R1-Pro (tgt) 关节数/轴定义对比
-----------------------------------------------
  单臂  DoF   关节轴顺序
  ALOHA  6    Y, -Z, -Z, X, Y, X
  R1Pro  7    Y,  X,  Z, Y, Z, Y, X

单臂关节映射（含符号翻转）：
  tgt J0 (Y)  <- src J0 (Y)   sign=+1
  tgt J1 (X)  <- src J3 (X)   sign=+1
  tgt J2 (Z)  <- src J1 (-Z)  sign=-1
  tgt J3 (Y)  <- src J4 (Y)   sign=+1
  tgt J4 (Z)  <- src J2 (-Z)  sign=-1
  tgt J5 (Y)  <- None         新增，随机/零初始化
  tgt J6 (X)  <- src J5 (X)   sign=+1

DM0 的内部张量布局（重要）
--------------------------
DM0 在 ``modeling_dm0.py`` 里强制 ``pad_to_dim(state, max_state_dim=32)`` /
``pad_to_dim(action, max_action_dim=32)``。所以无论 ALOHA 还是 R1-Pro 的 ckpt，
内部所有 action 层都是 ``[32, hidden]`` 或 ``[hidden, 32]``：

  dm0_model.model.action_in_proj.weight  shape (hidden, 32)
  dm0_model.model.action_in_proj.bias    shape (hidden,)
  dm0_model.model.action_out_proj.weight shape (32, hidden)
  dm0_model.model.action_out_proj.bias   shape (32,)

「14 / 23 维」只是 32 个槽位中的「有效槽」，剩下是 padding 0。重映射做的是：
**在 32 个槽位里把前 14 个 ALOHA 槽换座到前 16 个 R1-Pro 槽（左右臂各 7 + 双夹爪）。**

槽位分配：
  ALOHA  (默认 14 个有效槽):
      [0:6]  left_arm     [6]  left_grip
      [7:13] right_arm    [13] right_grip
      [14:32] padding 0
  R1Pro  (默认 23 个有效槽):
      [0:7]   left_arm    [7:14]  right_arm
      [14]    left_grip   [15]    right_grip
      [16:23] other (waist / chassis 等，新维度)
      [23:32] padding 0
  (与 DEFAULT_NON_DELTA_MASK=[14,15] 一致)

DM0 架构本身**不消费 state**（``dexbotic/model/dm0/dm0_arch.py`` 的 ``forward``
里 ``states`` 仅用于取 batch_size/device/dtype），所以无 state_proj 需要重映射；
``_action_min/_action_max/_state_min/_state_max`` 都是 ``persistent=False`` 的
buffer，从 ``norm_stats_path`` 读，不在 ckpt 里。

用法
----
  # phase-1 freeze-ViT，从 ALOHA ckpt 起训
  accelerate launch bt/dm0/train_dm0_aloha_r1_pro.py \\
      --task=train \\
      --aloha-ckpt=./checkpoints/DM0-aloha \\
      --dataset-repo=your-org/r1pro_data \\
      --norm-stats-path=./norm_stats/r1_pro.json

  # phase-2 LoRA-on-ViT（接 phase-1 输出）
  accelerate launch bt/dm0/train_dm0_aloha_r1_pro.py \\
      --task=lora_train \\
      --phase1-ckpt=./outputs/bt/dm0/<run>/checkpoints/last/pretrained_model \\
      --dataset-repo=your-org/r1pro_data

  # full fine-tune，从 ALOHA ckpt 起训
  accelerate launch bt/dm0/train_dm0_aloha_r1_pro.py \\
      --task=full_train \\
      --aloha-ckpt=./checkpoints/DM0-aloha \\
      --dataset-repo=your-org/r1pro_data \\
      --norm-stats-path=./norm_stats/r1_pro.json \\
      --lr=1e-5

  # dry-run：只做 checkpoint 重映射，不启动训练（用于验证）
  python bt/dm0/train_dm0_aloha_r1_pro.py \\
      --task=train --dry-run \\
      --aloha-ckpt=./checkpoints/DM0-aloha \\
      --dataset-repo=dummy/dummy \\
      --norm-stats-path=./norm_stats/r1_pro.json
"""

from __future__ import annotations

import os
import torch

_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(_local_rank)

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    _cur_dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    _dev_cnt = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception as _e:  # noqa: BLE001 — sandbox / no-CUDA 环境下不要让脆弱的 print 阻塞 import
    _cur_dev, _dev_cnt = "n/a", 0
print(
    f"[rank={_local_rank}] "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
    f"device_count={_dev_cnt}, current={_cur_dev}"
)

from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.dm0.configuration_dm0 import DM0Config
from lerobot.scripts.lerobot_train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 训练超参默认值（与 train_dm0_r1_pro.py 一致）
# ---------------------------------------------------------------------------
DEFAULT_NUM_IMAGES = 3
DEFAULT_ORIGINAL_ACTION_DIM = 23       # R1-Pro 目标动作维度（"有效槽"数）
DEFAULT_MAX_ACTION_DIM = 32            # DM0 内部 action 层固定宽度（max_action_dim）
DEFAULT_NON_DELTA_MASK = [14, 15]      # gripper 索引（保持绝对值）
DEFAULT_CHUNK_SIZE = 50
DEFAULT_LR = 1e-4
DEFAULT_DECAY_LR = 1e-5
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_STEPS = 12500
DEFAULT_SAVE_FREQ = 2500
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_LORA_R = 16

# ---------------------------------------------------------------------------
# 关节映射常量
# ---------------------------------------------------------------------------

# 单臂关节映射：PER_ARM_JOINT_MAP[tgt_j] = (src_j | None, sign)
#   src_j=None 表示 R1-Pro 新增关节，无 ALOHA 对应轴
PER_ARM_JOINT_MAP: Dict[int, Tuple[Optional[int], int]] = {
    0: (0,    +1),   # tgt J0 Y  <- src J0 Y
    1: (3,    +1),   # tgt J1 X  <- src J3 X
    2: (1,    -1),   # tgt J2 Z  <- src J1 -Z，翻转符号
    3: (4,    +1),   # tgt J3 Y  <- src J4 Y
    4: (2,    -1),   # tgt J4 Z  <- src J2 -Z，翻转符号
    5: (None, +1),   # tgt J5 Y  新增，随机/零初始化
    6: (5,    +1),   # tgt J6 X  <- src J5 X
}

# ALOHA 单臂 DoF 数
ALOHA_ARM_DOF = 6
# R1-Pro 单臂 DoF 数
R1PRO_ARM_DOF = 7


# ---------------------------------------------------------------------------
# 全身动作/状态向量映射构建
# ---------------------------------------------------------------------------

@dataclass
class ActionLayoutConfig:
    """描述一个机器人完整动作/状态向量的布局。"""
    arm_dof: int
    # 双臂在向量里的起始索引：[left_start, right_start]
    left_arm_start: int = 0
    right_arm_start: int = -1          # -1 表示 left_arm_start + arm_dof + n_gripper_between
    # gripper 索引列表（按 [left_grip_idx, right_grip_idx] 顺序）
    gripper_indices: List[int] = field(default_factory=list)
    # 总维度（其余未映射的维度补零/随机初始化）
    total_dim: int = 0

    def __post_init__(self):
        if self.right_arm_start == -1:
            # ALOHA 默认：left_arm | left_grip | right_arm | right_grip
            self.right_arm_start = self.left_arm_start + self.arm_dof + 1


def _default_aloha_layout(arm_dof: int = ALOHA_ARM_DOF) -> ActionLayoutConfig:
    """ALOHA 默认布局：[left×6 | left_grip | right×6 | right_grip]"""
    return ActionLayoutConfig(
        arm_dof=arm_dof,
        left_arm_start=0,
        right_arm_start=arm_dof + 1,
        gripper_indices=[arm_dof, 2 * arm_dof + 1],
        total_dim=2 * arm_dof + 2,
    )


def _default_r1pro_layout(arm_dof: int = R1PRO_ARM_DOF, total_dim: int = 23) -> ActionLayoutConfig:
    """R1-Pro 默认布局：[left×7 | right×7 | left_grip | right_grip | other×7]"""
    return ActionLayoutConfig(
        arm_dof=arm_dof,
        left_arm_start=0,
        right_arm_start=arm_dof,
        gripper_indices=[2 * arm_dof, 2 * arm_dof + 1],
        total_dim=total_dim,
    )


def build_full_action_map(
    src_layout: ActionLayoutConfig,
    tgt_layout: ActionLayoutConfig,
    per_arm_map: Dict[int, Tuple[Optional[int], int]] = PER_ARM_JOINT_MAP,
) -> Tuple[Dict[int, Optional[int]], Dict[int, int]]:
    """构建完整动作向量的维度映射和符号字典.

    Returns:
        full_map:  {tgt_idx: src_idx | None}  (None = 新增维度)
        full_sign: {tgt_idx: +1 | -1}
    """
    full_map: Dict[int, Optional[int]] = {}
    full_sign: Dict[int, int] = {}

    # 左臂关节
    for tgt_j, (src_j, sign) in per_arm_map.items():
        tgt_idx = tgt_layout.left_arm_start + tgt_j
        src_idx = (src_layout.left_arm_start + src_j) if src_j is not None else None
        full_map[tgt_idx] = src_idx
        full_sign[tgt_idx] = sign

    # 右臂关节
    for tgt_j, (src_j, sign) in per_arm_map.items():
        tgt_idx = tgt_layout.right_arm_start + tgt_j
        src_idx = (src_layout.right_arm_start + src_j) if src_j is not None else None
        full_map[tgt_idx] = src_idx
        full_sign[tgt_idx] = sign

    # gripper（直接对应，无符号翻转）
    for tgt_grip_idx, src_grip_idx in zip(
        tgt_layout.gripper_indices, src_layout.gripper_indices
    ):
        full_map[tgt_grip_idx] = src_grip_idx
        full_sign[tgt_grip_idx] = +1

    # 其余 tgt 维度（other，如腰部/底盘）置 None
    for tgt_idx in range(tgt_layout.total_dim):
        if tgt_idx not in full_map:
            full_map[tgt_idx] = None
            full_sign[tgt_idx] = +1

    return full_map, full_sign


# ---------------------------------------------------------------------------
# 单张量重映射工具
# ---------------------------------------------------------------------------
#
# 注意：DM0 的 action_in_proj/action_out_proj 都是固定 ``max_action_dim=32`` 宽度，
# 重映射只在 32 个槽位里换座位，**不改变 tensor shape**。
#
# - action_out_proj.weight  shape (max_action_dim, hidden)  → 重排 dim=0 的前若干行
# - action_out_proj.bias    shape (max_action_dim,)         → 重排前若干元素
# - action_in_proj.weight   shape (hidden, max_action_dim)  → 重排 dim=1 的前若干列
# - action_in_proj.bias     shape (hidden,)                 → 不动（hidden 维）

def _remap_along_action_dim(
    W: torch.Tensor,
    axis: int,
    src_real_dim: int,
    tgt_real_dim: int,
    max_action_dim: int,
    full_map: Dict[int, Optional[int]],
    full_sign: Dict[int, int],
    new_weight_std: float = 0.02,
    new_slot_zero: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """沿 ``axis`` 把 W 的「action 槽」从 ALOHA 的前 src_real_dim 个，
    重排为 R1-Pro 的前 tgt_real_dim 个，保持 W.shape[axis] == max_action_dim。

    Args:
        W:                 原张量，``W.shape[axis] == max_action_dim``。
        axis:              0 或 1，标识 action 维所在轴（也支持 1-D bias 时 axis=0）。
        src_real_dim:      源「有效槽」数（ALOHA 14）。
        tgt_real_dim:      目标「有效槽」数（R1-Pro 23）。
        max_action_dim:    固定宽度（DM0 默认 32）。
        full_map:          {tgt_i: src_i | None} 映射。
        full_sign:         {tgt_i: ±1} 符号字典。
        new_weight_std:    新增槽（无对应 src 的 tgt 槽）的随机初始化标准差。
        new_slot_zero:     True 表示新增槽置零（用于 bias），False 用 normal_。
        generator:         可选 ``torch.Generator``，用于可复现的随机初始化。
    """
    if W.shape[axis] != max_action_dim:
        raise ValueError(
            f"axis={axis} 的维度 {W.shape[axis]} 与 max_action_dim={max_action_dim} 不一致。"
        )

    new_W = torch.zeros_like(W)

    def _src_slice(idx: int) -> Tuple:
        return (slice(None),) * axis + (idx,)

    for tgt_i in range(tgt_real_dim):
        src_i = full_map.get(tgt_i)
        sign = full_sign.get(tgt_i, 1)
        if src_i is not None and src_i < src_real_dim:
            new_W[_src_slice(tgt_i)] = sign * W[_src_slice(src_i)]
        else:
            if new_slot_zero or new_weight_std == 0.0:
                pass  # 已经是 0
            else:
                src_view = new_W[_src_slice(tgt_i)]
                if generator is not None:
                    src_view.normal_(mean=0.0, std=new_weight_std, generator=generator)
                else:
                    src_view.normal_(mean=0.0, std=new_weight_std)
    # 槽 [tgt_real_dim : max_action_dim]（padding 槽）保持 0。
    return new_W


# ---------------------------------------------------------------------------
# 模式匹配：自动识别需要重映射的层
# ---------------------------------------------------------------------------
#
# DM0 的 dexbotic 架构里只有 action_in_proj / action_out_proj 两个 action 维投影：
#   - action_in_proj : nn.Linear(action_dim=32, hidden) — 输入吃 (B, chunk, 32)
#   - action_out_proj: nn.Linear(hidden, action_dim=32) — 输出 (B, chunk, 32)
# action_time_mlp_in/out 是 hidden×hidden，不涉及 action 槽，不需要 remap。
# DM0 forward 里的 states 参数仅用于读 batch_size/device/dtype，无 state_proj 层。

ACTION_IN_PROJ_PATTERN = "action_in_proj"
ACTION_OUT_PROJ_PATTERN = "action_out_proj"


def _key_matches(key: str, patterns: List[str]) -> bool:
    key_lower = key.lower()
    return any(p.lower() in key_lower for p in patterns)


# ---------------------------------------------------------------------------
# 主 checkpoint 重映射函数
# ---------------------------------------------------------------------------

def _load_state_dict_from_dir(
    ckpt_dir: Path,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """从一个 checkpoint 目录加载完整 state_dict，支持四种格式：

    1. 单文件 ``model.safetensors``
    2. 分片 ``model.safetensors.index.json`` + ``model-*-of-*.safetensors``
    3. 单文件 ``pytorch_model.bin``
    4. 分片 ``pytorch_model.bin.index.json`` + ``pytorch_model-*-of-*.bin``

    Returns:
        (state_dict, fmt) 其中 fmt ∈ {"safetensors", "safetensors-sharded",
        "bin", "bin-sharded"}，用于后续按相同格式回写。
    """
    single_st = ckpt_dir / "model.safetensors"
    index_st = ckpt_dir / "model.safetensors.index.json"
    single_bin = ckpt_dir / "pytorch_model.bin"
    index_bin = ckpt_dir / "pytorch_model.bin.index.json"

    if single_st.is_file():
        from safetensors.torch import load_file
        return load_file(str(single_st)), "safetensors"

    if index_st.is_file():
        from safetensors.torch import load_file
        with index_st.open() as f:
            weight_map = json.load(f).get("weight_map") or {}
        shard_names = sorted(set(weight_map.values()))
        merged: Dict[str, torch.Tensor] = {}
        for sname in shard_names:
            merged.update(load_file(str(ckpt_dir / sname)))
        return merged, "safetensors-sharded"

    if single_bin.is_file():
        return torch.load(str(single_bin), map_location="cpu"), "bin"

    if index_bin.is_file():
        with index_bin.open() as f:
            weight_map = json.load(f).get("weight_map") or {}
        shard_names = sorted(set(weight_map.values()))
        merged = {}
        for sname in shard_names:
            merged.update(torch.load(str(ckpt_dir / sname), map_location="cpu"))
        return merged, "bin-sharded"

    raise FileNotFoundError(
        f"未在 {ckpt_dir} 找到任何已知格式的权重文件。"
    )


def _save_state_dict_to_dir(
    state_dict: Dict[str, torch.Tensor],
    ckpt_dir: Path,
    fmt: str,
) -> None:
    """按原格式回写 state_dict。分片格式会被合并成单文件（重映射后通常 <50GB）。"""
    if fmt.startswith("safetensors"):
        from safetensors.torch import save_file
        # 删除旧分片 / 单文件，统一写成单文件
        for f in ckpt_dir.glob("model*.safetensors"):
            f.unlink()
        idx = ckpt_dir / "model.safetensors.index.json"
        if idx.exists():
            idx.unlink()
        save_file(state_dict, str(ckpt_dir / "model.safetensors"))
    else:
        for f in ckpt_dir.glob("pytorch_model*.bin"):
            f.unlink()
        idx = ckpt_dir / "pytorch_model.bin.index.json"
        if idx.exists():
            idx.unlink()
        torch.save(state_dict, str(ckpt_dir / "pytorch_model.bin"))


def remap_aloha_checkpoint(
    src_ckpt_dir: str | Path,
    dst_dir: str | Path,
    src_layout: ActionLayoutConfig,
    tgt_layout: ActionLayoutConfig,
    per_arm_map: Dict[int, Tuple[Optional[int], int]] = PER_ARM_JOINT_MAP,
    max_action_dim: int = DEFAULT_MAX_ACTION_DIM,
    new_weight_std: float = 0.02,
    seed: int = 0,
) -> Path:
    """把 ALOHA DM0 ckpt 的 ``action_in_proj`` / ``action_out_proj`` 关节槽位重排为 R1-Pro 排布.

    DM0 的两个 action 投影层都是固定 ``max_action_dim=32`` 宽度。本函数只改这两个层
    的前 ``src_real_dim`` (=14) 个槽，重排为前 ``tgt_real_dim`` (=23) 个槽：

        action_out_proj.weight  (32, hidden) →  rows[0:23] 重排，rows[23:32]=0
        action_out_proj.bias    (32,)        →  elements[0:23] 重排，[23:32]=0
        action_in_proj.weight   (hidden, 32) →  cols[0:23] 重排，cols[23:32]=0
        action_in_proj.bias     (hidden,)    →  保持不动（hidden 维，与 action 槽无关）

    其他层（LLM / ViT / mm_projector / action_expert / action_time_mlp_*）原样保留。

    Args:
        src_ckpt_dir:    ALOHA pretrained checkpoint 目录。可以是 dexbotic-native 的
                         DM0-base dump（``model_type=dexbotic_dm0`` + 直接命名的
                         ``model.action_in_proj.weight`` 等），也可以是 LeRobot 保存的
                         policy ckpt（带 ``dm0_model.`` 前缀）。重映射模式自动识别。
        dst_dir:         输出目录（若不同于 src 则自动 copytree）。
        src_layout:      ALOHA 动作向量布局（``total_dim`` 为有效槽数，如 14）。
        tgt_layout:      R1-Pro 动作向量布局（``total_dim`` 为有效槽数，如 23）。
        per_arm_map:     单臂关节映射字典。
        max_action_dim:  DM0 内部 action 层固定宽度（默认 32）。
        new_weight_std:  R1-Pro 新增槽的随机初始化标准差。
        seed:            新增槽随机初始化的种子（保证可复现）。

    Returns:
        dst_dir (Path)。
    """
    src_ckpt_dir = Path(src_ckpt_dir).expanduser()
    dst_dir = Path(dst_dir).expanduser()

    src_real_dim = src_layout.total_dim
    tgt_real_dim = tgt_layout.total_dim
    if src_real_dim > max_action_dim or tgt_real_dim > max_action_dim:
        raise ValueError(
            f"src_real_dim={src_real_dim} 或 tgt_real_dim={tgt_real_dim} "
            f"超过 max_action_dim={max_action_dim}。"
        )

    full_map, full_sign = build_full_action_map(src_layout, tgt_layout, per_arm_map)

    LOGGER.info(
        "checkpoint 重映射：max_action_dim=%d | ALOHA real_dim=%d -> R1Pro real_dim=%d | src_dir=%s",
        max_action_dim, src_real_dim, tgt_real_dim, src_ckpt_dir,
    )
    LOGGER.info("槽位映射（tgt_slot: src_slot, sign）：")
    for tgt_i in sorted(full_map):
        src_i = full_map[tgt_i]
        LOGGER.info(
            "  tgt[%2d] <- src[%s]  sign=%+d  %s",
            tgt_i, str(src_i).rjust(4), full_sign[tgt_i],
            "(NEW)" if src_i is None else "",
        )

    # ---- 复制整个目录（保留 config.json / tokenizer / dm0_arch_config.json 等） ----
    # 排除 .git / __pycache__ 等无关大目录（dexbotic 发布的 ckpt 通常带 git LFS 历史，
    # 一并 copytree 会让目标目录翻倍）。
    def _ignore(_dir: str, names: List[str]) -> List[str]:
        skip = {".git", "__pycache__", ".DS_Store", ".gitattributes", ".gitignore"}
        return [n for n in names if n in skip]

    if dst_dir.resolve() != src_ckpt_dir.resolve():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_ckpt_dir, dst_dir, ignore=_ignore)
        LOGGER.info("已复制 checkpoint 目录: %s -> %s", src_ckpt_dir, dst_dir)

    # ---- 加载权重（可能是 dexbotic-native 或 LeRobot-policy 任一格式） ----
    state_dict, fmt = _load_state_dict_from_dir(dst_dir)
    LOGGER.info("加载权重：fmt=%s, n_keys=%d", fmt, len(state_dict))

    # 检测 dexbotic-native vs LeRobot-policy（决定查找前缀）
    has_lerobot_prefix = any(k.startswith("dm0_model.model.") for k in state_dict)
    has_dexbotic_prefix = any(
        k.startswith("model.action_in_proj") or k.startswith("model.action_out_proj")
        for k in state_dict
    )
    LOGGER.info(
        "key 命名风格：lerobot-policy(dm0_model.model.*)=%s, dexbotic-native(model.*)=%s",
        has_lerobot_prefix, has_dexbotic_prefix,
    )

    generator = torch.Generator().manual_seed(seed)
    new_state_dict: Dict[str, torch.Tensor] = {}
    remap_log: List[str] = []

    for key, tensor in state_dict.items():
        is_in_proj = ACTION_IN_PROJ_PATTERN in key
        is_out_proj = ACTION_OUT_PROJ_PATTERN in key
        # 排除 action_time_mlp_in / action_time_mlp_out 等非 action-dim 投影
        if "action_time_mlp" in key:
            new_state_dict[key] = tensor
            continue

        if not (is_in_proj or is_out_proj):
            new_state_dict[key] = tensor
            continue

        # 可能是 .weight / .bias / .lora_A.weight / .lora_B.weight 等
        # 仅对 weight/bias 处理（其它 PEFT 子张量不应该出现在原始 ALOHA ckpt 里）
        if not (key.endswith(".weight") or key.endswith(".bias")):
            new_state_dict[key] = tensor
            continue

        if is_out_proj:
            # action_out_proj.weight: (max_action_dim, hidden), 改 axis=0
            # action_out_proj.bias  : (max_action_dim,),         改 axis=0
            if tensor.dim() == 2 and tensor.shape[0] == max_action_dim:
                new_state_dict[key] = _remap_along_action_dim(
                    tensor, axis=0,
                    src_real_dim=src_real_dim, tgt_real_dim=tgt_real_dim,
                    max_action_dim=max_action_dim,
                    full_map=full_map, full_sign=full_sign,
                    new_weight_std=new_weight_std, new_slot_zero=False,
                    generator=generator,
                )
                remap_log.append(f"{key} weight rows[:{tgt_real_dim}] reordered")
            elif tensor.dim() == 1 and tensor.shape[0] == max_action_dim:
                new_state_dict[key] = _remap_along_action_dim(
                    tensor, axis=0,
                    src_real_dim=src_real_dim, tgt_real_dim=tgt_real_dim,
                    max_action_dim=max_action_dim,
                    full_map=full_map, full_sign=full_sign,
                    new_weight_std=0.0, new_slot_zero=True,
                )
                remap_log.append(f"{key} bias[:{tgt_real_dim}] reordered (new slots = 0)")
            else:
                LOGGER.warning(
                    "%s shape=%s 不符合 action_out_proj 预期 (%d, hidden) 或 (%d,)，原样保留。",
                    key, tuple(tensor.shape), max_action_dim, max_action_dim,
                )
                new_state_dict[key] = tensor
        else:  # is_in_proj
            # action_in_proj.weight: (hidden, max_action_dim), 改 axis=1
            # action_in_proj.bias  : (hidden,),                hidden 维，不动
            if tensor.dim() == 2 and tensor.shape[1] == max_action_dim:
                new_state_dict[key] = _remap_along_action_dim(
                    tensor, axis=1,
                    src_real_dim=src_real_dim, tgt_real_dim=tgt_real_dim,
                    max_action_dim=max_action_dim,
                    full_map=full_map, full_sign=full_sign,
                    new_weight_std=new_weight_std, new_slot_zero=False,
                    generator=generator,
                )
                remap_log.append(f"{key} weight cols[:{tgt_real_dim}] reordered")
            elif tensor.dim() == 1:
                # bias 是 hidden 维，不需要 remap
                new_state_dict[key] = tensor
                remap_log.append(f"{key} bias kept as-is (hidden-dim)")
            else:
                LOGGER.warning(
                    "%s shape=%s 不符合 action_in_proj 预期 (hidden, %d) 或 (hidden,)，原样保留。",
                    key, tuple(tensor.shape), max_action_dim,
                )
                new_state_dict[key] = tensor

    if not remap_log:
        LOGGER.error(
            "重映射结束但没有匹配到 action_in_proj/action_out_proj。"
            "确认 ckpt 是 DM0 ckpt（包含 dm0_model.model.action_*_proj 或 model.action_*_proj 这些 key）。"
        )
    else:
        LOGGER.info("已处理 %d 个 action 投影张量：", len(remap_log))
        for line in remap_log:
            LOGGER.info("  [remap] %s", line)

    # ---- 回写权重 ----
    _save_state_dict_to_dir(new_state_dict, dst_dir, fmt)
    LOGGER.info("已保存重映射权重至: %s (fmt=%s)", dst_dir, fmt)

    # ---- 同步更新 config.json 的相关字段（仅当字段存在且明显是「有效槽数」时） ----
    config_json = dst_dir / "config.json"
    if config_json.exists():
        with config_json.open() as f:
            cfg_dict = json.load(f)
        updated_fields: List[str] = []
        # 1. LeRobot policy config: original_action_dim 通常等于 src_real_dim
        if cfg_dict.get("original_action_dim") == src_real_dim:
            cfg_dict["original_action_dim"] = tgt_real_dim
            updated_fields.append(f"original_action_dim: {src_real_dim} -> {tgt_real_dim}")
        # 2. 同步 input_features / output_features 中的 STATE / ACTION shape（如果存在）
        for fkey in ("input_features", "output_features"):
            block = cfg_dict.get(fkey, {})
            if not isinstance(block, dict):
                continue
            for feat_name, feat in list(block.items()):
                if not isinstance(feat, dict):
                    continue
                shape = feat.get("shape")
                if shape and len(shape) == 1 and shape[0] == src_real_dim:
                    feat["shape"] = [tgt_real_dim]
                    updated_fields.append(f"{fkey}.{feat_name}.shape: [{src_real_dim}] -> [{tgt_real_dim}]")
        # 注意：dexbotic-native 的 config.json 里 `action_dim` = max_action_dim（通常 32），
        # 不应该改。max_state_dim/max_action_dim 也不改。

        if updated_fields:
            with config_json.open("w") as f:
                json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
            for line in updated_fields:
                LOGGER.info("config.json updated: %s", line)

    return dst_dir


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="用 ALOHA DM0 checkpoint 微调 R1-Pro 7-DoF 人形机器人.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # task / phase ------------------------------------------------------------
    p.add_argument(
        "--task",
        choices=["train", "lora_train", "full_train"],
        default="train",
        help=(
            "`train`: phase-1 freeze ViT.  "
            "`lora_train`: phase-2 LoRA on ViT（需 --phase1-ckpt）.  "
            "`full_train`: 全量微调，不冻结任何子模块。"
        ),
    )

    # dataset -----------------------------------------------------------------
    p.add_argument("--dataset-repo", required=True)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--episodes", type=int, nargs="*", default=None)

    # checkpoints -------------------------------------------------------------
    p.add_argument(
        "--aloha-ckpt",
        default=None,
        help=(
            "ALOHA 预训练 DM0 checkpoint 目录（含 model.safetensors 或 pytorch_model.bin）。"
            "phase-1 / full_train-from-aloha 使用；"
            "提供后自动做关节维度重映射并存入 <output-root>/remapped_ckpt/。"
        ),
    )
    p.add_argument(
        "--phase1-ckpt",
        default=None,
        help=(
            ".../checkpoints/last/pretrained_model 路径。"
            "phase-2 必填；full_train 可选（提供则续训，跳过 ALOHA 重映射）。"
        ),
    )
    p.add_argument(
        "--norm-stats-path",
        default=None,
        help="R1-Pro 的 norm_stats.json 路径（phase-1 / 从 aloha-ckpt 起训必填）。",
    )

    # ALOHA layout configuration ----------------------------------------------
    p.add_argument(
        "--aloha-arm-dof",
        type=int,
        default=ALOHA_ARM_DOF,
        help="ALOHA 单臂关节数（默认 6）。",
    )
    p.add_argument(
        "--aloha-action-dim",
        type=int,
        default=2 * ALOHA_ARM_DOF + 2,
        help="ALOHA 动作向量总维度（默认 14 = 6+1+6+1）。",
    )
    p.add_argument(
        "--aloha-left-arm-start",
        type=int,
        default=0,
        help="ALOHA 左臂关节在动作向量的起始索引（默认 0）。",
    )
    p.add_argument(
        "--aloha-right-arm-start",
        type=int,
        default=-1,
        help="ALOHA 右臂关节起始索引（-1 = left_arm_start + arm_dof + 1）。",
    )
    p.add_argument(
        "--aloha-left-grip",
        type=int,
        default=ALOHA_ARM_DOF,
        help="ALOHA 左手夹爪在动作向量的索引（默认 6）。",
    )
    p.add_argument(
        "--aloha-right-grip",
        type=int,
        default=2 * ALOHA_ARM_DOF + 1,
        help="ALOHA 右手夹爪在动作向量的索引（默认 13）。",
    )

    # R1-Pro layout configuration ---------------------------------------------
    p.add_argument(
        "--r1pro-arm-dof",
        type=int,
        default=R1PRO_ARM_DOF,
        help="R1-Pro 单臂关节数（默认 7）。",
    )
    p.add_argument(
        "--r1pro-action-dim",
        type=int,
        default=DEFAULT_ORIGINAL_ACTION_DIM,
        help="R1-Pro 动作向量总维度（默认 23）。",
    )
    p.add_argument(
        "--r1pro-left-arm-start",
        type=int,
        default=0,
        help="R1-Pro 左臂关节在动作向量的起始索引（默认 0）。",
    )
    p.add_argument(
        "--r1pro-right-arm-start",
        type=int,
        default=R1PRO_ARM_DOF,
        help="R1-Pro 右臂关节起始索引（默认 7）。",
    )
    p.add_argument(
        "--r1pro-left-grip",
        type=int,
        default=2 * R1PRO_ARM_DOF,
        help="R1-Pro 左手夹爪索引（默认 14）。",
    )
    p.add_argument(
        "--r1pro-right-grip",
        type=int,
        default=2 * R1PRO_ARM_DOF + 1,
        help="R1-Pro 右手夹爪索引（默认 15）。",
    )

    # checkpoint remap tuning -------------------------------------------------
    p.add_argument(
        "--max-action-dim",
        type=int,
        default=DEFAULT_MAX_ACTION_DIM,
        help="DM0 内部 action 投影层固定宽度（max_action_dim，默认 32）。",
    )
    p.add_argument(
        "--new-weight-std",
        type=float,
        default=0.02,
        help="R1-Pro 新增槽（J5 Y / 腰部 / 底盘等）的 action 投影权重随机初始化 std。",
    )
    p.add_argument(
        "--remap-seed",
        type=int,
        default=0,
        help="新增槽随机初始化的种子（保证可复现）。",
    )

    # policy hyper-params -----------------------------------------------------
    p.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    p.add_argument(
        "--non-delta-mask",
        type=int,
        nargs="*",
        default=DEFAULT_NON_DELTA_MASK,
        help="保持绝对值的 action 维度（如 gripper）。",
    )
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    # optim / schedule --------------------------------------------------------
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--decay-lr", type=float, default=DEFAULT_DECAY_LR)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)

    # training loop -----------------------------------------------------------
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    p.add_argument("--log-freq", type=int, default=1)
    p.add_argument("--seed", type=int, default=1000)

    # LoRA --------------------------------------------------------------------
    p.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)

    # image augmentation ------------------------------------------------------
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--aug-prob", type=float, default=0.5)
    p.add_argument("--aug-size", type=int, default=728)
    p.add_argument("--aug-head-cam-substr", default="head")
    p.add_argument("--aug-wrist-cam-substrs", nargs="*", default=["wrist", "hand"])

    # output / logging --------------------------------------------------------
    p.add_argument("--output-root", default="outputs/bt/dm0_aloha")
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--job-name", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="dm0_aloha_r1pro")
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="只执行 checkpoint 重映射，不启动训练。")
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--vit-lr-mult", type=float, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# 从参数构建布局配置
# ---------------------------------------------------------------------------

def _build_src_layout(args: argparse.Namespace) -> ActionLayoutConfig:
    right_start = args.aloha_right_arm_start
    if right_start == -1:
        right_start = args.aloha_left_arm_start + args.aloha_arm_dof + 1
    return ActionLayoutConfig(
        arm_dof=args.aloha_arm_dof,
        left_arm_start=args.aloha_left_arm_start,
        right_arm_start=right_start,
        gripper_indices=[args.aloha_left_grip, args.aloha_right_grip],
        total_dim=args.aloha_action_dim,
    )


def _build_tgt_layout(args: argparse.Namespace) -> ActionLayoutConfig:
    return ActionLayoutConfig(
        arm_dof=args.r1pro_arm_dof,
        left_arm_start=args.r1pro_left_arm_start,
        right_arm_start=args.r1pro_right_arm_start,
        gripper_indices=[args.r1pro_left_grip, args.r1pro_right_grip],
        total_dim=args.r1pro_action_dim,
    )


# ---------------------------------------------------------------------------
# Policy config builders（与 train_dm0_r1_pro.py 一致，入参改为 aloha-remapped path）
# ---------------------------------------------------------------------------

def _build_phase1_policy(args: argparse.Namespace, remapped_ckpt: Path) -> DM0Config:
    if not args.norm_stats_path:
        raise ValueError("--norm-stats-path 在 phase-1 是必填的。")
    return DM0Config(
        model_name_or_path=str(remapped_ckpt),
        norm_stats_path=args.norm_stats_path,
        freeze_vision_encoder=True,
        gradient_checkpointing=True,
        num_images=args.num_images,
        original_action_dim=args.r1pro_action_dim,
        non_delta_mask=list(args.non_delta_mask),
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        optimizer_lr=args.lr,
        scheduler_warmup_steps=args.warmup_steps,
        scheduler_decay_steps=args.steps,
        scheduler_decay_lr=args.decay_lr,
        push_to_hub=False,
        ema_decay=args.ema_decay,
        device=args.device,
        vit_lr_mult=args.vit_lr_mult,
    )


def _build_phase2_policy(args: argparse.Namespace) -> DM0Config:
    if not args.phase1_ckpt:
        raise ValueError("--phase1-ckpt 在 phase-2 是必填的。")
    cfg = DM0Config(
        gradient_checkpointing=True,
        num_images=args.num_images,
        original_action_dim=args.r1pro_action_dim,
        non_delta_mask=list(args.non_delta_mask),
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        optimizer_lr=args.lr,
        scheduler_warmup_steps=args.warmup_steps,
        scheduler_decay_steps=args.steps,
        scheduler_decay_lr=args.decay_lr,
        push_to_hub=False,
        ema_decay=args.ema_decay,
        device=args.device,
        vit_lr_mult=args.vit_lr_mult,
    )
    cfg.pretrained_path = str(Path(args.phase1_ckpt).expanduser())
    if args.norm_stats_path:
        cfg.norm_stats_path = args.norm_stats_path
    return cfg


def _build_full_policy(args: argparse.Namespace, remapped_ckpt: Optional[Path]) -> DM0Config:
    """全量微调，支持从 remapped_ckpt 或 phase1_ckpt 两种起点。"""
    if remapped_ckpt is None and not args.phase1_ckpt:
        raise ValueError(
            "full_train 需要 --aloha-ckpt（从 ALOHA 重映射起训）"
            "或 --phase1-ckpt（从 phase-1 续训）至少其一。"
        )
    cfg = DM0Config(
        model_name_or_path=str(remapped_ckpt) if remapped_ckpt else None,
        norm_stats_path=args.norm_stats_path,
        freeze_vision_encoder=False,
        train_expert_only=False,
        gradient_checkpointing=True,
        num_images=args.num_images,
        original_action_dim=args.r1pro_action_dim,
        non_delta_mask=list(args.non_delta_mask),
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        optimizer_lr=args.lr,
        scheduler_warmup_steps=args.warmup_steps,
        scheduler_decay_steps=args.steps,
        scheduler_decay_lr=args.decay_lr,
        push_to_hub=False,
        ema_decay=args.ema_decay,
        device=args.device,
        vit_lr_mult=args.vit_lr_mult,
    )
    if args.phase1_ckpt:
        cfg.pretrained_path = str(Path(args.phase1_ckpt).expanduser())
    return cfg


# ---------------------------------------------------------------------------
# 整体 TrainPipelineConfig 构建
# ---------------------------------------------------------------------------

def build_train_config(
    args: argparse.Namespace,
    remapped_ckpt: Optional[Path],
) -> TrainPipelineConfig:
    if args.task == "train":
        if remapped_ckpt is None:
            raise ValueError("phase-1 需要 --aloha-ckpt。")
        policy_cfg = _build_phase1_policy(args, remapped_ckpt)
    elif args.task == "lora_train":
        policy_cfg = _build_phase2_policy(args)
    elif args.task == "full_train":
        policy_cfg = _build_full_policy(args, remapped_ckpt)
    else:
        raise ValueError(f"未知 --task={args.task!r}")

    peft_cfg = PeftConfig(method_type="LORA", r=args.lora_r) if args.task == "lora_train" else None

    job_name = args.job_name or f"dm0_aloha_{args.task}"
    run_id = os.environ.get("DM0_RUN_ID") or args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"{args.task}-{run_id}"
    save_freq = max(1, min(args.save_freq, args.steps))

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo,
            root=args.dataset_root,
            episodes=args.episodes,
        ),
        policy=policy_cfg,
        peft=peft_cfg,
        output_dir=output_dir,
        job_name=job_name,
        resume=args.resume,
        seed=args.seed,
        batch_size=args.batch_size,
        steps=args.steps,
        num_workers=args.num_workers,
        eval_freq=0,
        log_freq=args.log_freq,
        save_checkpoint=True,
        save_freq=save_freq,
        wandb=WandBConfig(enable=args.wandb, project=f"{args.wandb_project}_{args.task}"),
    )


# ---------------------------------------------------------------------------
# 图像增强 hook（与 train_dm0_r1_pro.py 相同）
# ---------------------------------------------------------------------------

def _install_dm0_image_aug_hook(args: argparse.Namespace) -> None:
    import sys
    import lerobot.scripts.lerobot_train as _lt

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from dm0_augmentations import (  # type: ignore[import-not-found]
        DM0AugmentedDataset,
        build_dm0_per_camera_augs,
    )

    _orig_make_dataset = _lt.make_dataset

    def _make_dataset_with_aug(cfg):
        dataset = _orig_make_dataset(cfg)
        cam_keys = list(getattr(dataset, "meta").camera_keys)
        if not cam_keys:
            LOGGER.warning("数据集未识别到 camera_keys，跳过图像增强。")
            return dataset
        cam_to_aug = build_dm0_per_camera_augs(
            cam_keys=cam_keys,
            head_cam_substr=args.aug_head_cam_substr or None,
            wrist_cam_substrs=args.aug_wrist_cam_substrs or (),
            aug_prob=args.aug_prob,
            size=args.aug_size,
        )
        head_cams = [c for c, t in cam_to_aug.items() if "RandomResizedCrop" in repr(t._compose)]
        wrist_cams = [c for c in cam_to_aug if c not in head_cams]
        LOGGER.info(
            "图像增强已启用：head=%s -> policy_dm0, wrist=%s -> policy_color_dm0; p=%.2f, size=%d",
            head_cams or "(none)", wrist_cams or "(none)", args.aug_prob, args.aug_size,
        )
        return DM0AugmentedDataset(dataset, cam_to_aug=cam_to_aug)

    _lt.make_dataset = _make_dataset_with_aug


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    src_layout = _build_src_layout(args)
    tgt_layout = _build_tgt_layout(args)

    # ---- step 1: 如需 ALOHA 重映射，先做 ----
    # accelerate 多卡场景下，所有 rank 都会执行 main()。直接同时 copytree 5GB 到同一目录会触发
    # rmtree/unlink race（一个 rank 删掉了另一个 rank 正在写的文件）。
    # 解决：只让 rank 0 做 remap + 写 sentinel，其余 rank 阻塞等待 sentinel。
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main_process = rank == 0

    remapped_ckpt: Optional[Path] = None
    if args.aloha_ckpt and args.task != "lora_train":
        run_id = os.environ.get("DM0_RUN_ID") or args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dst = Path(args.output_root) / f"remapped_ckpt-{run_id}"
        sentinel = dst / ".remap_done"

        if is_main_process:
            # 若上次失败留下半成品目录，先清掉再 remap（仅 rank 0 持有写权限）。
            if dst.exists() and not sentinel.exists():
                LOGGER.info("发现残留目录 %s（无 sentinel），清理后重做。", dst)
                shutil.rmtree(dst)
            remapped_ckpt = remap_aloha_checkpoint(
                src_ckpt_dir=args.aloha_ckpt,
                dst_dir=dst,
                src_layout=src_layout,
                tgt_layout=tgt_layout,
                per_arm_map=PER_ARM_JOINT_MAP,
                max_action_dim=args.max_action_dim,
                new_weight_std=args.new_weight_std,
                seed=args.remap_seed,
            )
            sentinel.touch()
            LOGGER.info("[rank=0] 重映射完成，sentinel 已写: %s", sentinel)
        else:
            LOGGER.info("[rank=%d] 等待 rank-0 完成 ALOHA 重映射 -> %s", rank, dst)
            deadline = time.time() + 60 * 60  # 60 分钟超时（5GB ckpt copytree+save 约 1-3 分钟）
            while not sentinel.exists():
                if time.time() > deadline:
                    raise TimeoutError(
                        f"[rank={rank}] 等待 rank-0 重映射超时（>60min）：{sentinel} 一直未出现。"
                    )
                time.sleep(3)
            remapped_ckpt = dst
            LOGGER.info("[rank=%d] 检测到 sentinel，复用 rank-0 重映射结果: %s", rank, remapped_ckpt)

        # ---- N-way barrier：让所有 rank 同步到 train(cfg) 入口 ----
        # 否则 rank 0 比其他 rank 早 1-3 分钟（remap 耗时）进入 train()，会先 mkdir
        # output_dir，迟到的 rank 进 cfg.validate() 时撞上 FileExistsError。
        if world_size > 1:
            barrier_dir = Path(args.output_root) / f".barrier-{run_id}"
            barrier_dir.mkdir(parents=True, exist_ok=True)
            my_flag = barrier_dir / f"rank_{rank}.ready"
            my_flag.touch()
            LOGGER.info("[rank=%d] barrier flag 已写: %s", rank, my_flag)
            barrier_deadline = time.time() + 60 * 60
            while True:
                ready = sum(
                    1 for r in range(world_size) if (barrier_dir / f"rank_{r}.ready").exists()
                )
                if ready == world_size:
                    break
                if time.time() > barrier_deadline:
                    raise TimeoutError(
                        f"[rank={rank}] 等待 N-way barrier 超时（>60min）："
                        f"{ready}/{world_size} ranks ready in {barrier_dir}."
                    )
                time.sleep(1)
            LOGGER.info("[rank=%d] 全部 %d 个 rank 已对齐 barrier，进入 train()。", rank, world_size)
    elif args.task == "lora_train":
        LOGGER.info("phase-2 lora_train：跳过 ALOHA 重映射，使用 --phase1-ckpt。")

    # ---- step 2: 组装训练配置 ----
    cfg = build_train_config(args, remapped_ckpt)

    LOGGER.info("DM0-ALOHA->R1Pro | task=%s | output_dir=%s", args.task, cfg.output_dir)
    LOGGER.info(
        "policy: chunk_size=%d, num_images=%d, r1pro_action_dim=%d, non_delta_mask=%s",
        cfg.policy.chunk_size, cfg.policy.num_images,
        cfg.policy.original_action_dim, cfg.policy.non_delta_mask,
    )
    LOGGER.info(
        "optim: lr=%.2e -> %.2e (warmup=%d, decay=%d)",
        cfg.policy.optimizer_lr, cfg.policy.scheduler_decay_lr,
        cfg.policy.scheduler_warmup_steps, cfg.policy.scheduler_decay_steps,
    )
    LOGGER.info(
        "freeze: vision_encoder=%s, expert_only=%s | init_from=%s",
        cfg.policy.freeze_vision_encoder,
        cfg.policy.train_expert_only,
        cfg.policy.pretrained_path or cfg.policy.model_name_or_path,
    )
    if cfg.policy.vit_lr_mult is not None:
        LOGGER.info(
            "ViT lr_mult=%.4f -> vit_lr=%.2e (base_lr=%.2e)",
            cfg.policy.vit_lr_mult,
            cfg.policy.optimizer_lr * cfg.policy.vit_lr_mult,
            cfg.policy.optimizer_lr,
        )
    if cfg.peft is not None:
        LOGGER.info("PEFT: method=%s, r=%d", cfg.peft.method_type, cfg.peft.r)

    if args.no_aug:
        LOGGER.info("--no-aug：图像增强已关闭。")
    else:
        _install_dm0_image_aug_hook(args)

    if args.dry_run:
        LOGGER.info("dry-run：仅执行 checkpoint 重映射，不启动训练。remapped=%s", remapped_ckpt)
        return

    train(cfg)
    LOGGER.info("训练完成。输出目录: %s", cfg.output_dir)


if __name__ == "__main__":
    main()
