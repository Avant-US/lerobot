"""
PI05 Fine-Tuning on physical-intelligence/libero (LeRobot v3.0 format).

Standalone training script that:
  1. Loads the libero dataset via LeRobotDataset (v3.0 parquet from HuggingFace Hub)
  2. Builds / loads a PI05Policy (optionally from pretrained weights)
  3. Constructs the PI05 pre/post-processor pipelines
  4. Runs a standard training loop with AdamW + cosine-decay LR

Usage:
    # Quick debug with 2 episodes, random weights
    python -m bt.sftpi05.train_pi05 --max-episodes 2 --steps 20 --no-pretrained

    # Fine-tune from pretrained base
    python -m bt.sftpi05.train_pi05 --pretrained-path lerobot/pi05_base --steps 3000
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="PI05 fine-tuning on libero")

    # Dataset
    p.add_argument("--repo-id", default="physical-intelligence/libero")
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="Specific episode indices to use")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Max number of episodes (from the start)")
    p.add_argument("--local-root", default=None,
                   help="Local root dir for dataset cache")

    # Policy
    p.add_argument("--pretrained-path", default="lerobot/pi05_base",
                   help="HF repo or local path; empty string = random init")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch (random weights)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])

    # Training
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    # Logging & saving
    p.add_argument("--log-freq", type=int, default=10)
    p.add_argument("--save-freq", type=int, default=50)
    p.add_argument("--no-save", action="store_true")

    # Model knobs
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--freeze-vision-encoder", action="store_true")
    p.add_argument("--train-expert-only", action="store_true")

    # Normalization
    p.add_argument("--normalization-mode", default="MEAN_STD",
                   choices=["MEAN_STD", "QUANTILES"])

    p.add_argument("--device", default=None)
    return p.parse_args()


def detect_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset (LeRobot v3.0)
# ---------------------------------------------------------------------------
def make_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.local_root)

    episodes = args.episodes
    if episodes is None and args.max_episodes is not None:
        episodes = list(range(min(args.max_episodes, ds_meta.total_episodes)))

    # PI05 action_delta_indices = [0, 1, ..., chunk_size-1]
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    tmp_cfg = PI05Config()
    delta_timestamps = resolve_delta_timestamps(tmp_cfg, ds_meta)
    logger.info(f"delta_timestamps: { {k: f'len={len(v)}' for k, v in delta_timestamps.items()} if delta_timestamps else None }")

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.local_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )

    logger.info(
        f"LeRobotDataset: {dataset.num_episodes} episodes, "
        f"{dataset.num_frames} frames, fps={dataset.fps}"
    )
    return dataset


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def make_policy(args, dataset):
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = detect_device(args.device)
    norm_mode = NormalizationMode[args.normalization_mode]

    features = dataset_to_policy_features(dataset.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    logger.info(f"Input features: {list(input_features.keys())}")
    logger.info(f"Output features: {list(output_features.keys())}")

    config = PI05Config(
        max_state_dim=32,
        max_action_dim=32,
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

    pretrained = args.pretrained_path if not args.no_pretrained else None
    if pretrained:
        logger.info(f"Loading pretrained: {pretrained}")
        policy = PI05Policy.from_pretrained(pretrained, config=config)
    else:
        logger.info("Random initialisation (no pretrained)")
        policy = PI05Policy(config)

    policy.to(device)
    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_total:,} total, {n_train:,} trainable")
    return policy, config


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
def make_processors(config, dataset_stats):
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    return make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------
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
    lr_scheduler = sched_cfg.build(optimizer, args.steps)
    return optimizer, lr_scheduler


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_ckpt(out_dir, step, policy, optimizer, scheduler, pre, post):
    ckpt = out_dir / "checkpoints" / f"step_{step:06d}"
    model_dir = ckpt / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(model_dir)
    state = {"step": step, "optimizer": optimizer.state_dict()}
    if scheduler:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, ckpt / "training_state.pt")
    pre.save_pretrained(model_dir)
    post.save_pretrained(model_dir)
    logger.info(f"Saved checkpoint -> {ckpt}")


# ---------------------------------------------------------------------------
# Infinite dataloader helper
# ---------------------------------------------------------------------------
def cycle(dl):
    while True:
        yield from dl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train(args):
    device = detect_device(args.device)
    logger.info(f"Device: {device}")

    from lerobot.utils.random_utils import set_seed
    set_seed(args.seed)

    # ---- dataset ----
    dataset = make_dataset(args)

    # ---- policy ----
    policy, config = make_policy(args, dataset)

    # ---- processors (stats come from dataset metadata) ----
    preprocessor, postprocessor = make_processors(config, dataset.meta.stats)

    # ---- optimizer ----
    optimizer, lr_scheduler = make_optimizer(args, policy)

    # ---- dataloader ----
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    dl_iter = cycle(dataloader)

    # ---- output dir ----
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        import datetime as dt
        out_dir = Path(f"outputs/sft_pi05/{dt.datetime.now():%Y-%m-%d_%H-%M-%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    # ---- training loop ----
    policy.train()
    logger.info(f"Training for {args.steps} steps, bs={args.batch_size}")

    loss_accum = 0.0
    t_accum = []
    bar = tqdm(range(1, args.steps + 1), desc="Train", unit="step")

    for step in bar:
        t0 = time.perf_counter()

        batch = next(dl_iter)
        batch = preprocessor(batch)

        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16 if (args.dtype == "bfloat16" and device == "cuda") else torch.float32,
        ):
            loss, loss_dict = policy.forward(batch)

        loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)

        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler:
            lr_scheduler.step()

        dt_s = time.perf_counter() - t0
        t_accum.append(dt_s)
        lv = loss.item()
        loss_accum += lv
        bar.set_postfix(loss=f"{lv:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if step % args.log_freq == 0:
            avg_loss = loss_accum / args.log_freq
            avg_t = sum(t_accum[-args.log_freq:]) / min(args.log_freq, len(t_accum))
            logger.info(
                f"[{step}/{args.steps}] loss={avg_loss:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  t={avg_t:.2f}s/step"
            )
            loss_accum = 0.0

        if not args.no_save and args.save_freq > 0 and (
            step % args.save_freq == 0 or step == args.steps
        ):
            save_ckpt(out_dir, step, policy, optimizer, lr_scheduler,
                      preprocessor, postprocessor)

    total_t = sum(t_accum)
    logger.info(f"Done. {total_t:.1f}s total ({total_t / args.steps:.2f}s/step)")
    return policy


if __name__ == "__main__":
    train(parse_args())
