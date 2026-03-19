#!/usr/bin/env python3
"""
使用 LeRobot 对 GR00T N1.5 进行小规模预训练：
1) 从 HuggingFaceVLA/libero 中随机抽取 37 个 episode
2) 使用抽样子集执行小规模预训练

示例：
python bt/ptgr00t15_3/train_groot_n15_libero_small.py --steps 1 --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.groot2.configuration_groot import GrootConfig
from lerobot.scripts.lerobot_train import train


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="随机抽样 37 个 libero episodes，并做 GR00T N1.5 小规模预训练"
    )
    parser.add_argument("--dataset-repo", default="HuggingFaceVLA/libero")
    parser.add_argument("--dataset-root", default="data/hf_libero_groot_n15_37")
    parser.add_argument("--sample-size", type=int, default=37)
    parser.add_argument("--sample-seed", type=int, default=20260314)
    parser.add_argument(
        "--episodes-file",
        default="bt/ptgr00t15_3/episodes_37.json",
        help="保存随机抽样结果的 JSON 文件",
    )

    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--output-root", default="outputs/bt/ptgr00t15_3")
    parser.add_argument("--job-name", default="groot_n15_libero37_small_pretrain")

    parser.add_argument("--base-model-path", default="nvidia/GR00T-N1.5-3B")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default=None, help="例如 cuda / cpu / cuda:0，不填则自动选择")

    parser.add_argument("--tune-llm", action="store_true", default=False)
    parser.add_argument("--tune-visual", action="store_true", default=False)
    parser.add_argument("--tune-projector", action="store_true", default=True)
    parser.add_argument("--no-tune-projector", dest="tune_projector", action="store_false")
    parser.add_argument("--tune-diffusion-model", action="store_true", default=False)

    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--no-use-bf16", dest="use_bf16", action="store_false")

    parser.add_argument("--save-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")

    parser.add_argument("--dry-run", action="store_true", help="只生成配置，不启动训练")
    return parser.parse_args()


def sample_episodes(repo_id: str, root: str | Path, sample_size: int, seed: int) -> tuple[list[int], int]:
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    total_episodes = int(meta.total_episodes)
    if sample_size <= 0:
        raise ValueError(f"sample_size 必须 > 0，当前为 {sample_size}")
    if sample_size > total_episodes:
        raise ValueError(
            f"sample_size({sample_size}) 超过数据集 episode 总数({total_episodes})，请减小 sample_size。"
        )

    rng = random.Random(seed)
    episodes = sorted(rng.sample(range(total_episodes), sample_size))
    return episodes, total_episodes


def save_sampled_episodes(
    output_path: Path,
    dataset_repo: str,
    total_episodes: int,
    sample_size: int,
    sample_seed: int,
    episodes: list[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_repo": dataset_repo,
        "total_episodes": total_episodes,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "episodes": episodes,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_train_config(args: argparse.Namespace, episodes: list[int], output_dir: Path) -> TrainPipelineConfig:
    policy_cfg = GrootConfig(
        push_to_hub=False,
        repo_id=None,
        base_model_path=args.base_model_path,
        embodiment_tag=args.embodiment_tag,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        tune_projector=args.tune_projector,
        tune_diffusion_model=args.tune_diffusion_model,
        use_bf16=args.use_bf16,
        device=args.device,
    )

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo,
        root=args.dataset_root,
        episodes=episodes,
        streaming=False,
    )

    save_freq = max(1, min(args.save_freq, args.steps))
    return TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=args.job_name,
        batch_size=args.batch_size,
        steps=args.steps,
        num_workers=args.num_workers,
        eval_freq=0,
        log_freq=args.log_freq,
        save_checkpoint=args.save_checkpoint,
        save_freq=save_freq,
        wandb=WandBConfig(enable=False),
    )


def main() -> None:
    args = parse_args()

    episodes, total_episodes = sample_episodes(
        repo_id=args.dataset_repo,
        root=args.dataset_root,
        sample_size=args.sample_size,
        seed=args.sample_seed,
    )
    episodes_path = Path(args.episodes_file)
    save_sampled_episodes(
        output_path=episodes_path,
        dataset_repo=args.dataset_repo,
        total_episodes=total_episodes,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        episodes=episodes,
    )

    LOGGER.info(
        "已随机抽样 %d/%d 个 episodes（seed=%d），保存到 %s",
        len(episodes),
        total_episodes,
        args.sample_seed,
        episodes_path,
    )
    LOGGER.info("episodes 前10个: %s", episodes[:10])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    cfg = build_train_config(args=args, episodes=episodes, output_dir=output_dir)

    LOGGER.info("训练输出目录: %s", output_dir)
    LOGGER.info(
        "开始小规模预训练: repo=%s, episodes=%d, steps=%d, batch_size=%d",
        args.dataset_repo,
        len(episodes),
        args.steps,
        args.batch_size,
    )

    if args.dry_run:
        LOGGER.info("dry-run 模式：仅完成抽样和配置构建，不执行训练。")
        return

    train(cfg)
    LOGGER.info("训练完成。输出目录: %s", output_dir)


if __name__ == "__main__":
    main()
