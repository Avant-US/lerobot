#!/usr/bin/env python3
"""DM0 R1-Pro 两阶段训练入口（freeze_vit / lora_vit）.

LeRobot 版本的 ``dexbotic/playground/benchmarks/real/r1_pro_dm0_freeze_lora.py``：

- ``--task=train``      ：第 1 阶段，冻结 ViT，微调 LLM (+ projector / action head)。
- ``--task=lora_train`` ：第 2 阶段，冻结整模型，在 ViT 的 ``out_proj`` 上挂 LoRA。

LoRA 的 ``alpha=32 / dropout=0.05`` 写死在 ``DM0Policy._get_default_peft_targets``
里（``PeftConfig`` 没暴露这两个字段），与 dexbotic 配方对齐；这里只暴露 ``--lora-r``。

示例::

    # phase 1: freeze ViT
    accelerate launch bt/dm0/train_dm0_r1_pro.py \\
        --task=train \\
        --dataset-repo=your-org/r1_pro_chassis_v3_lerobot \\
        --norm-stats-path=./norm_stats/r1_pro_chassis_v3.json \\
        --dm0-base=./checkpoints/DM0-base

    # phase 2: LoRA on ViT (复用 phase-1 的 last 检查点)
    accelerate launch bt/dm0/train_dm0_r1_pro.py \\
        --task=lora_train \\
        --dataset-repo=your-org/r1_pro_chassis_v3_lerobot \\
        --phase1-ckpt=./outputs/bt/dm0/<phase1_run>/checkpoints/last/pretrained_model

``norm_stats.json`` 由 ``examples/dm0/compute_norm_stats.py`` 离线生成。
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

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


# r1_pro_dm0_freeze_lora.py 的常量（DM0DataConfig / DM0ActionConfig）
DEFAULT_NUM_IMAGES = 3
DEFAULT_ORIGINAL_ACTION_DIM = 23
DEFAULT_NON_DELTA_MASK = [14, 15]
DEFAULT_CHUNK_SIZE = 50
DEFAULT_LR = 1e-4
DEFAULT_DECAY_LR = 1e-5
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_STEPS = 30000
DEFAULT_SAVE_FREQ = 2500
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_LORA_R = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DM0 R1-Pro phased training (freeze ViT → LoRA on ViT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # task / phase ------------------------------------------------------------
    p.add_argument(
        "--task",
        choices=["train", "lora_train"],
        default="train",
        help="`train`: phase-1 freeze ViT.  `lora_train`: phase-2 LoRA on ViT.",
    )

    # dataset -----------------------------------------------------------------
    p.add_argument("--dataset-repo", required=True, help="LeRobotDataset repo id.")
    p.add_argument("--dataset-root", default=None, help="本地数据缓存路径（覆盖默认）.")
    p.add_argument("--episodes", type=int, nargs="*", default=None)

    # checkpoints -------------------------------------------------------------
    p.add_argument(
        "--dm0-base",
        default="./checkpoints/DM0-base",
        help="dexbotic 原生 DM0 权重目录（仅 phase-1 使用）.",
    )
    p.add_argument(
        "--phase1-ckpt",
        default=None,
        help="phase-2 必填：phase-1 输出的 .../checkpoints/last/pretrained_model .",
    )
    p.add_argument(
        "--norm-stats-path",
        default=None,
        help="dexbotic 风格 norm_stats.json；phase-1 必填，phase-2 一般继承自 phase-1 ckpt.",
    )

    # policy hyper-params (mirror r1_pro_dm0_freeze_lora.py) ------------------
    p.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    p.add_argument("--original-action-dim", type=int, default=DEFAULT_ORIGINAL_ACTION_DIM)
    p.add_argument(
        "--non-delta-mask",
        type=int,
        nargs="*",
        default=DEFAULT_NON_DELTA_MASK,
        help="保持绝对值（不做 delta）的 action 维度，例如 gripper.",
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
    p.add_argument(
        "--lora-r",
        type=int,
        default=DEFAULT_LORA_R,
        help="LoRA rank (alpha/dropout 写死在 DM0Policy._get_default_peft_targets 里).",
    )

    # image augmentation (mirrors dexbotic aug_policy=['dm0','color_dm0',...]) -
    p.add_argument(
        "--no-aug",
        action="store_true",
        help="关闭 dexbotic 风格的图像增强（默认开启：head=policy_dm0, wrist=policy_color_dm0）.",
    )
    p.add_argument(
        "--aug-prob",
        type=float,
        default=0.5,
        help="ColorJitter 的应用概率 p（与 dexbotic 默认一致）.",
    )
    p.add_argument(
        "--aug-size",
        type=int,
        default=728,
        help="增强后图像尺寸（与 dexbotic policy_dm0 / policy_color_dm0 一致，下游 ViT 会再缩放）.",
    )
    p.add_argument(
        "--aug-head-cam-substr",
        default="head",
        help="包含此子串（不区分大小写）的 camera key 使用 policy_dm0；空串表示不按子串识别 head.",
    )
    p.add_argument(
        "--aug-wrist-cam-substrs",
        nargs="*",
        default=["wrist", "hand"],
        help="包含其中任一子串的 camera key 使用 policy_color_dm0.",
    )

    # output / logging --------------------------------------------------------
    p.add_argument("--output-root", default="outputs/bt/dm0")
    p.add_argument("--job-name", default=None, help="不填则自动生成 dm0_<task>.")
    p.add_argument("--wandb", action="store_true", help="启用 W&B logging.")
    p.add_argument("--wandb-project", default="dm0_r1_pro_chassis_v3")
    p.add_argument("--device", default=None, help="例如 cuda / cpu / cuda:0.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只组装配置，不实际启动训练.",
    )

    p.add_argument("--ema-decay", type=float, default=None, help="EMA decay(0<decay<1). None = disabled.")
    if p.ema_decay is not None:
        if p.ema_decay <= 0 or p.ema_decay >= 1:
            raise ValueError("--ema-decay must be between 0 and 1.")

    return p.parse_args()


def _build_phase1_policy(args: argparse.Namespace) -> DM0Config:
    if not args.norm_stats_path:
        raise ValueError(
            "--norm-stats-path 在 phase-1 是必填的（用 examples/dm0/compute_norm_stats.py 生成）."
        )
    return DM0Config(
        model_name_or_path=args.dm0_base,
        norm_stats_path=args.norm_stats_path,
        freeze_vision_encoder=True,
        gradient_checkpointing=True,
        num_images=args.num_images,
        original_action_dim=args.original_action_dim,
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
    )


def _build_phase2_policy(args: argparse.Namespace) -> DM0Config:
    if not args.phase1_ckpt:
        raise ValueError("--phase1-ckpt 在 phase-2 是必填的.")
    cfg = DM0Config(
        gradient_checkpointing=True,
        num_images=args.num_images,
        original_action_dim=args.original_action_dim,
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
    )
    cfg.pretrained_path = str(Path(args.phase1_ckpt).expanduser())
    if args.norm_stats_path:
        cfg.norm_stats_path = args.norm_stats_path
    return cfg


def _build_peft(args: argparse.Namespace) -> PeftConfig | None:
    if args.task != "lora_train":
        return None
    return PeftConfig(method_type="LORA", r=args.lora_r)


def build_train_config(args: argparse.Namespace) -> TrainPipelineConfig:
    if args.task == "train":
        policy_cfg = _build_phase1_policy(args)
    else:
        policy_cfg = _build_phase2_policy(args)

    job_name = args.job_name or f"dm0_{args.task}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{args.task}-{run_id}"

    save_freq = max(1, min(args.save_freq, args.steps))

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo,
            root=args.dataset_root,
            episodes=args.episodes,
        ),
        policy=policy_cfg,
        peft=_build_peft(args),
        output_dir=output_dir,
        job_name=job_name,
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


def _install_dm0_image_aug_hook(args: argparse.Namespace) -> None:
    """Wrap LeRobot's ``make_dataset`` so the train DataLoader sees per-camera DM0 augmentation.

    Mirrors dexbotic's ``aug_policy=["dm0", "color_dm0", "color_dm0"]``: the head
    camera goes through geometric+color augmentation; wrist cameras get color-only.
    """
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
            "图像增强已启用（dexbotic 风格）：head=%s -> policy_dm0, wrist=%s -> policy_color_dm0; "
            "p=%.2f, size=%d",
            head_cams or "(none)",
            wrist_cams or "(none)",
            args.aug_prob,
            args.aug_size,
        )
        return DM0AugmentedDataset(dataset, cam_to_aug=cam_to_aug)

    _lt.make_dataset = _make_dataset_with_aug


def main() -> None:
    args = parse_args()
    cfg = build_train_config(args)

    LOGGER.info("DM0 task=%s | output_dir=%s", args.task, cfg.output_dir)
    LOGGER.info(
        "policy: chunk_size=%d, num_images=%d, action_dim=%d, non_delta_mask=%s",
        cfg.policy.chunk_size,
        cfg.policy.num_images,
        cfg.policy.original_action_dim,
        cfg.policy.non_delta_mask,
    )
    LOGGER.info(
        "optim: lr=%.2e → %.2e (warmup=%d, decay=%d)",
        cfg.policy.optimizer_lr,
        cfg.policy.scheduler_decay_lr,
        cfg.policy.scheduler_warmup_steps,
        cfg.policy.scheduler_decay_steps,
    )
    if cfg.peft is not None:
        LOGGER.info("PEFT: method=%s, r=%d (alpha/dropout from DM0Policy default targets)",
                    cfg.peft.method_type, cfg.peft.r)

    if args.no_aug:
        LOGGER.info("--no-aug：图像增强已关闭。")
    else:
        _install_dm0_image_aug_hook(args)

    if args.dry_run:
        LOGGER.info("dry-run：只组装配置，不启动训练。")
        return

    train(cfg)
    LOGGER.info("训练完成。输出目录: %s", cfg.output_dir)


if __name__ == "__main__":
    main()
