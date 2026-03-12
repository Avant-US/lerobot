"""
SmolVLA Pre-Training on community data (LeRobot v3.0 format).

Standalone training script that:
  1. Loads a converted community dataset via LeRobotDataset (v3.0)
  2. Builds a SmolVLAPolicy (random init for pre-training, or from pretrained VLM)
  3. Constructs the SmolVLA pre/post-processor pipelines
  4. Runs a standard training loop with AdamW + cosine-decay LR

Usage:
    # Quick debug with random weights, 20 steps
    python -m bt.ptsmolvla.train_smolvla \
        --repo-id community_pt_smolvla --local-root data/community_pt_smolvla \
        --no-pretrained --steps 20 --batch-size 2

    # Pre-train from pretrained base
    python -m bt.ptsmolvla.train_smolvla \
        --repo-id community_pt_smolvla --local-root data/community_pt_smolvla \
        --pretrained-path lerobot/smolvla_base --steps 3000
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
    p = argparse.ArgumentParser(description="SmolVLA pre-training on community data")

    # Dataset
    p.add_argument("--repo-id", default="community_pt_smolvla")
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="Specific episode indices to use")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Max number of episodes (from the start)")
    p.add_argument("--local-root", default="data/community_pt_smolvla",
                   help="Local root dir for dataset")

    # Policy
    p.add_argument("--pretrained-path", default="lerobot/smolvla_base",
                   help="HF repo or local path for pretrained weights")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch (random weights)")
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])

    # Training
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-10)
    p.add_argument("--grad-clip-norm", type=float, default=10.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    # Logging & saving
    p.add_argument("--log-freq", type=int, default=5)
    p.add_argument("--save-freq", type=int, default=50)
    p.add_argument("--no-save", action="store_true")

    # Model knobs
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--freeze-vision-encoder", action="store_true", default=True)
    p.add_argument("--no-freeze-vision-encoder", dest="freeze_vision_encoder", action="store_false")
    p.add_argument("--train-expert-only", action="store_true", default=True)
    p.add_argument("--no-train-expert-only", dest="train_expert_only", action="store_false")

    # Normalization
    p.add_argument("--normalization-mode", default="MEAN_STD",
                   choices=["MEAN_STD", "QUANTILES"])

    p.add_argument("--device", default=None)
    return p.parse_args()


def detect_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.local_root)

    episodes = args.episodes
    if episodes is None and args.max_episodes is not None:
        episodes = list(range(min(args.max_episodes, ds_meta.total_episodes)))

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    tmp_cfg = SmolVLAConfig()
    delta_timestamps = resolve_delta_timestamps(tmp_cfg, ds_meta)
    logger.info(
        f"delta_timestamps: "
        f"{ {k: f'len={len(v)}' for k, v in delta_timestamps.items()} if delta_timestamps else None }"
    )

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


def make_policy(args, dataset):
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = detect_device(args.device)
    norm_mode = NormalizationMode[args.normalization_mode]

    features = dataset_to_policy_features(dataset.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    logger.info(f"Input features: {list(input_features.keys())}")
    logger.info(f"Output features: {list(output_features.keys())}")

    config = SmolVLAConfig(
        max_state_dim=32,
        max_action_dim=32,
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
        policy = SmolVLAPolicy.from_pretrained(pretrained, config=config)
    else:
        logger.info("Random initialisation (no pretrained weights)")
        policy = SmolVLAPolicy(config)

    policy.to(device)
    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_total:,} total, {n_train:,} trainable")
    return policy, config


def make_processors(config, dataset_stats):
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    return make_smolvla_pre_post_processors(config=config, dataset_stats=dataset_stats)


def make_optimizer(args, policy):
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
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


def cycle(dl):
    while True:
        yield from dl


def train(args):
    device = detect_device(args.device)
    logger.info(f"Device: {device}")

    from lerobot.utils.random_utils import set_seed
    set_seed(args.seed)

    dataset = make_dataset(args)
    policy, config = make_policy(args, dataset)
    preprocessor, postprocessor = make_processors(config, dataset.meta.stats)
    optimizer, lr_scheduler = make_optimizer(args, policy)

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

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        import datetime as dt
        out_dir = Path(f"outputs/pt_smolvla/{dt.datetime.now():%Y-%m-%d_%H-%M-%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    policy.train()
    logger.info(f"Pre-training for {args.steps} steps, bs={args.batch_size}")

    loss_accum = 0.0
    t_accum = []
    bar = tqdm(range(1, args.steps + 1), desc="PreTrain", unit="step")

    for step in bar:
        t0 = time.perf_counter()

        batch = next(dl_iter)
        batch = preprocessor(batch)

        with torch.autocast(
            device_type=device if device != "cpu" else "cpu",
            dtype=torch.bfloat16 if (args.dtype == "bfloat16" and device == "cuda") else torch.float32,
            enabled=(device == "cuda"),
        ):
            loss_out = policy.forward(batch)
            if isinstance(loss_out, tuple):
                loss, loss_dict = loss_out
            else:
                loss = loss_out
                loss_dict = {"loss": loss.item()}

        loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                args.grad_clip_norm,
            )

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
