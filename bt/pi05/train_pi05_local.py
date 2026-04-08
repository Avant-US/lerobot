#!/usr/bin/env python3
"""
本地 Pi0.5 训练入口（绕过 lerobot-train 的全量策略导入链）。

功能：
1) 读取 selected_episodes.json
2) 加载 LeRobotDataset 子集
3) 构建 PI05Policy + 处理器
4) 运行训练循环并保存 checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import types
from pathlib import Path

import torch
from tqdm import tqdm

from bt.pi05.prepare_data import build_training_dataset, load_selected_episodes
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def bootstrap_lerobot_policies_package():
    """
    避免执行 lerobot/policies/__init__.py（该文件会提前导入 groot/groot2）。
    仅为当前脚本注入一个轻量 package stub，不修改源码。
    """
    if "lerobot.policies" in sys.modules:
        return

    import lerobot

    policies_dir = Path(lerobot.__file__).resolve().parent / "policies"
    pkg = types.ModuleType("lerobot.policies")
    pkg.__path__ = [str(policies_dir)]  # type: ignore[attr-defined]
    pkg.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = pkg


def parse_args():
    p = argparse.ArgumentParser(description="Train pi05 on selected LIBERO episodes")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--local-root", required=True)
    p.add_argument("--episode-file", required=True)
    p.add_argument("--pretrained-path", default="lerobot/pi05_base")
    p.add_argument("--tokenizer-name", default="google/paligemma-3b-pt-224")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=10)
    p.add_argument("--save-freq", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--normalization-mode", choices=["MEAN_STD", "QUANTILES"], default="QUANTILES")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--train-expert-only", action="store_true")
    p.add_argument("--freeze-vision-encoder", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def detect_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_normalization_mode(requested_mode: str, dataset) -> NormalizationMode:
    if requested_mode == "MEAN_STD":
        stats = getattr(dataset.meta, "stats", None)
        has_quantiles = bool(
            stats
            and any(
                isinstance(feature_stats, dict) and feature_stats.get("q01") is not None and feature_stats.get("q99") is not None
                for feature_stats in stats.values()
            )
        )
        if has_quantiles:
            logger.warning(
                "Normalization mode MEAN_STD was requested, but dataset quantile stats are available. "
                "Switching to QUANTILES."
            )
            return NormalizationMode.QUANTILES

    return NormalizationMode[requested_mode]

def make_dataset(args, episodes: list[int]):
    bootstrap_lerobot_policies_package()
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.local_root)
    tmp_cfg = PI05Config()
    delta_timestamps = resolve_delta_timestamps(tmp_cfg, ds_meta)
    dataset = build_training_dataset(
        repo_id=args.repo_id,
        local_root=args.local_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        ds_meta=ds_meta,
        log=logger,
    )

    logger.info("Dataset loaded: %s episodes, %s frames", dataset.num_episodes, dataset.num_frames)
    return dataset


def make_policy(args, dataset):
    bootstrap_lerobot_policies_package()
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = detect_device(args.device)
    norm_mode = resolve_normalization_mode(args.normalization_mode, dataset)

    features = dataset_to_policy_features(dataset.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    config = PI05Config(
        dtype=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
        device=device,
        input_features=input_features,
        output_features=output_features,
    )
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": norm_mode,
        "ACTION": norm_mode,
    }

    policy = PI05Policy.from_pretrained(args.pretrained_path, config=config)
    policy.to(device)
    n_total = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info("Policy loaded: total=%s trainable=%s", f"{n_total:,}", f"{n_trainable:,}")
    return policy, config, device


def make_optimizer(args, policy):
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    sched_cfg = CosineDecayWithWarmupSchedulerConfig(
        peak_lr=args.lr,
        decay_lr=2.5e-6,
        num_warmup_steps=args.warmup_steps,
        num_decay_steps=args.steps,
    )
    scheduler = sched_cfg.build(optimizer, args.steps)
    return optimizer, scheduler


def save_ckpt(output_dir: Path, step: int, policy, optimizer, scheduler, preprocessor, postprocessor):
    ckpt = output_dir / "checkpoints" / f"step_{step:06d}"
    model_dir = ckpt / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(model_dir)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        ckpt / "training_state.pt",
    )
    preprocessor.save_pretrained(model_dir)
    postprocessor.save_pretrained(model_dir)
    logger.info("Saved checkpoint: %s", ckpt)


def cycle(loader):
    while True:
        yield from loader


def train():
    args = parse_args()
    set_seed(args.seed)
    bootstrap_lerobot_policies_package()

    episodes = load_selected_episodes(args.episode_file)
    logger.info("Episodes from file: %s", episodes)

    dataset = make_dataset(args, episodes)
    policy, config, device = make_policy(args, dataset)
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
        PolicyAction,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TokenizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    # `lerobot/pi05_base` 只保存 policy 权重和 processor 配置，不包含 tokenizer 文件。
    # PI0.5 官方 processor 使用的是 `google/paligemma-3b-pt-224`，这里直接加载配置的 tokenizer。
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False, trust_remote_code=True)
        logger.info("Tokenizer loaded: %s", args.tokenizer_name)
    except Exception as exc:  # nosec B110
        raise RuntimeError(
            "Failed to load tokenizer "
            f"{args.tokenizer_name}. Pi0.5 expects a PaliGemma/Gemma-compatible tokenizer "
            "(for example `google/paligemma-3b-pt-224`)."
        ) from exc

    preprocessor = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset.meta.stats,
            ),
            Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
            TokenizerProcessorStep(
                tokenizer=tokenizer,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DeviceProcessorStep(device=config.device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=dataset.meta.stats,
            ),
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    optimizer, scheduler = make_optimizer(args, policy)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    data_iter = cycle(dataloader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    policy.train()
    bar = tqdm(range(1, args.steps + 1), desc="pi05-train", unit="step")
    loss_acc = 0.0
    time_acc = []

    for step in bar:
        t0 = time.perf_counter()
        batch = next(data_iter)
        batch = preprocessor(batch)

        autocast_dtype = torch.bfloat16 if (args.dtype == "bfloat16" and device == "cuda") else torch.float32
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            loss, _ = policy.forward(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        time_acc.append(elapsed)
        loss_value = float(loss.item())
        loss_acc += loss_value
        bar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if step % args.log_freq == 0:
            avg_loss = loss_acc / args.log_freq
            avg_t = sum(time_acc[-args.log_freq:]) / min(args.log_freq, len(time_acc))
            logger.info(
                "[%s/%s] loss=%.4f lr=%s time/step=%.2fs",
                step,
                args.steps,
                avg_loss,
                f"{optimizer.param_groups[0]['lr']:.2e}",
                avg_t,
            )
            loss_acc = 0.0

        if args.save_freq > 0 and (step % args.save_freq == 0 or step == args.steps):
            save_ckpt(output_dir, step, policy, optimizer, scheduler, preprocessor, postprocessor)


if __name__ == "__main__":
    train()
