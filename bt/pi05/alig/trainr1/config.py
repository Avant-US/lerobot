#!/usr/bin/env python3
"""
OpenPI pi05_r1pro_chassis 对齐训练配置 — 集中式参数定义

本文件集中定义所有与 OpenPI JAX `pi05_r1pro_chassis` 对齐所需的参数。
训练脚本和验证脚本都从此文件导入，确保参数的一致性和可维护性。

参数来源:
  OpenPI config:      openpi/src/openpi/training/config.py:1024-1042
  OpenPI optimizer:    openpi/src/openpi/training/optimizer.py
  OpenPI train:        openpi/scripts/train.py
  LeRobot PI05Config:  lerobot/src/lerobot/policies/pi05/configuration_pi05.py
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────────────

# 项目根目录
LEROBOT_ROOT = Path("/home/Luogang/SRC/Robot/lerobot")
OPENPI_ROOT = Path("/mnt/r/share/lkx/pi/openpi")
TRAINR1_DIR = LEROBOT_ROOT / "bt" / "pi05" / "alig" / "trainr1"

# 虚拟环境
VENV_PATH = Path("/mnt/r/Venv/lerobot-venv")
VENV_PYTHON = VENV_PATH / "bin" / "python"

# 数据路径
RAW_DATA_DIR = Path("/mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis")
CONVERTED_DATA_DIR = TRAINR1_DIR / "data" / "r1_pro_chassis_v30"
OPENPI_NORM_STATS = (
    OPENPI_ROOT / "assets" / "pi05_r1pro_chassis"
    / "r1_pro_data_convert_chassis" / "norm_stats.json"
)

# 转换工具
CONVERT_SCRIPT = (
    LEROBOT_ROOT / "bt" / "pi05" / "alig" / "dataprocess"
    / "convert_r1pro_to_lerobot.py"
)

# 预训练权重 (LeRobot HuggingFace Hub)
PRETRAINED_MODEL_PATH = "lerobot/pi05_base"

# 输出目录
OUTPUT_DIR = TRAINR1_DIR / "outputs" / "r1pro_chassis_aligned"

# ──────────────────────────────────────────────────────────────────────
# 模型配置 (与 OpenPI Pi0Config(pi05=True) 对齐)
# ──────────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "dtype": "bfloat16",           # OpenPI 默认 bfloat16
    "chunk_size": 50,              # action_horizon
    "n_action_steps": 50,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "tokenizer_max_length": 200,   # pi0.5 专用
    "image_resolution": (224, 224),
}

# ──────────────────────────────────────────────────────────────────────
# 优化器配置 (与 OpenPI AdamW 默认对齐)
# ──────────────────────────────────────────────────────────────────────

OPTIMIZER_CONFIG = {
    "lr": 2.5e-5,                  # peak_lr (CosineDecaySchedule)
    "betas": (0.9, 0.95),          # Adam momentum coefficients
    "eps": 1e-8,                   # numerical stability
    "weight_decay": 1e-10,         # @#2 关键对齐项! OpenPI=1e-10, LeRobot默认=0.01
    "grad_clip_norm": 1.0,         # gradient clipping threshold
}

# ──────────────────────────────────────────────────────────────────────
# LR 调度配置 (与 OpenPI optax.warmup_cosine_decay_schedule 对齐)
# ──────────────────────────────────────────────────────────────────────

LR_SCHEDULE_CONFIG = {
    "warmup_steps": 1_000,         # 线性 warmup 阶段
    "decay_steps": 30_000,         # 总衰减步数 (含 warmup) — 注意: 不随 num_train_steps 变化!
                                   # 当 num_train_steps=100000 > decay_steps=30000 时,
                                   # cosine 在 step 30000 完成, LR 被钳位在 decay_lr 直到训练结束
    "peak_lr": 2.5e-5,             # 最大学习率
    "decay_lr": 2.5e-6,            # 最终学习率 (peak_lr 的 1/10)
    "phase_mode": "post_warmup",   # @#2 关键对齐项! 匹配 optax 的余弦相位语义
}

# ──────────────────────────────────────────────────────────────────────
# 训练配置 (与 OpenPI TrainConfig 对齐)
# ──────────────────────────────────────────────────────────────────────

TRAINING_CONFIG = {
    "batch_size": 256,             # @#2 CLI 覆盖: --batch_size 256
    "steps": 100_000,              # @#2 CLI 覆盖: --num_train_steps 100000
    "seed": 42,                    # @#2 OpenPI 默认 seed
    "log_freq": 100,               # 日志间隔 (OpenPI: log_interval=100)
    "save_freq": 500,              # @#2 CLI 覆盖: --save_interval 500
    "keep_period": 2_500,          # @#2 CLI 覆盖: --keep_period 2500 (保留 step%2500==0 的 checkpoint)
    "eval_freq": -1,               # 不做在线评估 (纯训练对齐)
    "num_workers": 2,              # DataLoader workers (OpenPI 默认 num_workers=2)
}

# ──────────────────────────────────────────────────────────────────────
# 对齐专属配置 (LeRobot PI05Config 中的 OpenPI 对齐开关)
# ──────────────────────────────────────────────────────────────────────

ALIGNMENT_CONFIG = {
    "ema_decay": 0.99,             # @#2 EMA 指数移动平均 (OpenPI 默认)
    "loss_include_padding": True,  # @#2 Loss 含 padding 维 (OpenPI 行为: 32 维 MSE)
    "augmentation_enabled": True,  # @#2 数据增强 (OpenPI JIT 内增强)
    "gradient_checkpointing": True,  # 内存优化 (bfloat16 + batch_size=64 需要)
}

# ──────────────────────────────────────────────────────────────────────
# 数据增强参数 (与 OpenPI model.py:168-187 对齐)
# ──────────────────────────────────────────────────────────────────────

AUGMENTATION_CONFIG = {
    "aug_crop_scale": 0.95,        # RandomCrop as fraction (非 wrist 相机)
    "aug_rotate_degrees": 5.0,     # ±5° rotation (非 wrist 相机)
    "aug_color_brightness": 0.3,   # ColorJitter brightness (所有相机)
    "aug_color_contrast": 0.4,     # ColorJitter contrast (所有相机)
    "aug_color_saturation": 0.5,   # ColorJitter saturation (所有相机)
    "aug_wrist_patterns": ("wrist",),  # wrist 相机匹配模式
}

# ──────────────────────────────────────────────────────────────────────
# 归一化配置 (@#2 Normalization Stats)
# ──────────────────────────────────────────────────────────────────────

NORMALIZATION_CONFIG = {
    "state_mode": "QUANTILES",     # Pi0.5 使用 quantile 归一化
    "action_mode": "QUANTILES",    # 动作同样使用 quantile
    "visual_mode": "IDENTITY",     # 图像不归一化
    # 归一化公式: (x - q01) / (q99 - q01) * 2 - 1  → [-1, 1]
    # 注入方式: 通过 convert_r1pro_to_lerobot.py --norm-stats-path
}

# ──────────────────────────────────────────────────────────────────────
# R1 Pro Chassis 机器人配置
# ──────────────────────────────────────────────────────────────────────

ROBOT_CONFIG = {
    "action_dim": 23,              # 实际动作维度
    "padded_action_dim": 32,       # 模型内部 padding 后维度
    "action_layout": {
        "left_arm": (0, 7),        # 7 DOF
        "right_arm": (7, 14),      # 7 DOF
        "left_gripper": (14, 15),  # 1 DOF
        "right_gripper": (15, 16), # 1 DOF
        "torso": (16, 20),         # 4 DOF
        "chassis": (20, 23),       # 3 DOF (linear_x, linear_y, angular_z)
    },
    "cameras": {
        "head_rgb": {"resolution": (360, 640), "is_wrist": False},
        "left_wrist_rgb": {"resolution": (480, 640), "is_wrist": True},
        "right_wrist_rgb": {"resolution": (480, 640), "is_wrist": True},
    },
    "fps": 14,
}

# ──────────────────────────────────────────────────────────────────────
# WandB 配置
# ──────────────────────────────────────────────────────────────────────

WANDB_CONFIG = {
    "enable": True,
    "project": "pi05_r1pro_chassis_alignment",
    # entity: 用户自行设置
}

# ──────────────────────────────────────────────────────────────────────
# OpenPI 参考值 (用于验证脚本)
# ──────────────────────────────────────────────────────────────────────

OPENPI_REFERENCE = {
    "lr_schedule": {
        # 关键步数处的 LR 理论值 (optax.warmup_cosine_decay_schedule)
        # Phase 1: 线性 warmup (step 0 → 1000)
        0: 2.497e-8,       # peak_lr / (warmup_steps + 1)
        500: 1.250e-5,     # warmup 中点
        1000: 2.500e-5,    # peak (warmup 结束)
        # Phase 2: cosine 衰减 (step 1000 → 30000)
        5000: 2.302e-5,    # 早期衰减
        10000: 1.819e-5,   # 中期衰减
        15000: 1.434e-5,   # 中点附近
        20000: 1.056e-5,   # 后期衰减
        25000: 5.31e-6,    # 接近尾声
        30000: 2.500e-6,   # cosine 终点 (decay_lr)
        # Phase 3: 钳位期 (step 30000 → 100000), LR 恒定 = decay_lr
        # decay_steps=30000 未被 CLI 覆盖, cosine 完成后 LR 被 min() 钳位
        35000: 2.500e-6,   # 钳位
        50000: 2.500e-6,   # 钳位
        75000: 2.500e-6,   # 钳位
        100000: 2.500e-6,  # 训练结束, 仍为 decay_lr
    },
    "dataset": {
        "total_episodes": 64,
        "total_frames": 61_923,
        "fps": 14,
    },
    "cli_command": (
        "uv run python scripts/train.py pi05_r1pro_chassis"
        " --exp_name $EXPNAME --batch_size 256"
        " --num_train_steps 100000 --save_interval 500 --keep_period 2500"
    ),
}


def build_train_cli_args() -> list[str]:
    """
    构建 lerobot_train.py 的 CLI 参数列表。

    返回完整的命令行参数，包含所有对齐项的覆盖。
    """
    args = [
        # 数据集
        f"--dataset.repo_id=local/{CONVERTED_DATA_DIR.name}",
        f"--dataset.root={CONVERTED_DATA_DIR}",

        # 预训练模型
        f"--policy.path={PRETRAINED_MODEL_PATH}",

        # 模型参数
        f"--policy.dtype={MODEL_CONFIG['dtype']}",

        # 优化器参数 (@#2 Weight Decay)
        f"--policy.optimizer_weight_decay={OPTIMIZER_CONFIG['weight_decay']}",

        # 对齐开关
        f"--policy.loss_include_padding={str(ALIGNMENT_CONFIG['loss_include_padding']).lower()}",
        f"--policy.ema_decay={ALIGNMENT_CONFIG['ema_decay']}",
        f"--policy.augmentation_enabled={str(ALIGNMENT_CONFIG['augmentation_enabled']).lower()}",
        f"--policy.gradient_checkpointing={str(ALIGNMENT_CONFIG['gradient_checkpointing']).lower()}",

        # 训练参数
        f"--batch_size={TRAINING_CONFIG['batch_size']}",
        f"--steps={TRAINING_CONFIG['steps']}",
        f"--seed={TRAINING_CONFIG['seed']}",
        f"--log_freq={TRAINING_CONFIG['log_freq']}",
        f"--save_freq={TRAINING_CONFIG['save_freq']}",
        f"--eval_freq={TRAINING_CONFIG['eval_freq']}",
        f"--num_workers={TRAINING_CONFIG['num_workers']}",

        # 输出
        f"--output_dir={OUTPUT_DIR}",

        # WandB
        f"--wandb.enable={str(WANDB_CONFIG['enable']).lower()}",
        f"--wandb.project={WANDB_CONFIG['project']}",
    ]
    return args


def print_alignment_summary():
    """打印对齐参数摘要。"""
    print("=" * 70)
    print("OpenPI pi05_r1pro_chassis 对齐训练 — 参数摘要")
    print("=" * 70)
    print()

    sections = [
        ("优化器 (P0 Critical)", [
            ("weight_decay", OPTIMIZER_CONFIG["weight_decay"], "0.01", "1e-10"),
            ("lr", OPTIMIZER_CONFIG["lr"], "2.5e-5", "2.5e-5"),
            ("betas", OPTIMIZER_CONFIG["betas"], "(0.9, 0.95)", "(0.9, 0.95)"),
            ("grad_clip", OPTIMIZER_CONFIG["grad_clip_norm"], "1.0", "1.0"),
        ]),
        ("Loss 计算 (P0 Critical)", [
            ("loss_include_padding", ALIGNMENT_CONFIG["loss_include_padding"], "False", "True(32维)"),
        ]),
        ("LR Schedule (P0 Critical)", [
            ("phase_mode", LR_SCHEDULE_CONFIG["phase_mode"], "absolute", "post_warmup"),
            ("warmup_steps", LR_SCHEDULE_CONFIG["warmup_steps"], "1000", "1000"),
            ("decay_steps", LR_SCHEDULE_CONFIG["decay_steps"], "30000", "30000"),
            ("peak_lr", LR_SCHEDULE_CONFIG["peak_lr"], "2.5e-5", "2.5e-5"),
            ("decay_lr", LR_SCHEDULE_CONFIG["decay_lr"], "2.5e-6", "2.5e-6"),
        ]),
        ("EMA (P1)", [
            ("ema_decay", ALIGNMENT_CONFIG["ema_decay"], "None", "0.99"),
        ]),
        ("数据增强 (P1)", [
            ("augmentation_enabled", ALIGNMENT_CONFIG["augmentation_enabled"], "False", "True"),
        ]),
        ("训练参数 (CLI 覆盖)", [
            ("batch_size", TRAINING_CONFIG["batch_size"], "8", "256"),
            ("steps", TRAINING_CONFIG["steps"], "100000", "100000"),
            ("save_freq", TRAINING_CONFIG["save_freq"], "20000", "500"),
            ("keep_period", TRAINING_CONFIG["keep_period"], "N/A", "2500"),
            ("seed", TRAINING_CONFIG["seed"], "1000", "42"),
            ("num_workers", TRAINING_CONFIG["num_workers"], "4", "2"),
        ]),
    ]

    for section_name, items in sections:
        print(f"  [{section_name}]")
        for name, value, lr_default, openpi_value in items:
            status = "OK" if str(value) == openpi_value else "ALIGNED"
            print(f"    {name:30s} = {str(value):15s}  (LeRobot默认={lr_default}, OpenPI={openpi_value}) [{status}]")
        print()

    print("=" * 70)


if __name__ == "__main__":
    print_alignment_summary()
    print("\nCLI 参数:")
    for arg in build_train_cli_args():
        print(f"    {arg}")
