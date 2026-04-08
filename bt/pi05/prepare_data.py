#!/usr/bin/env python3
"""
从 HuggingFaceVLA/libero 数据集中随机抽取 37 个 episode，
下载到本地并验证 LeRobot 数据格式。

用法:
    python -m bt.pi05.prepare_data                # 默认: 37 episodes, seed=42
    python -m bt.pi05.prepare_data --num-episodes 5 --seed 123  # 自定义
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset as HFDataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices, hf_transform_to_torch

REPO_ID = "HuggingFaceVLA/libero"
TOTAL_EPISODES = 1693
DEFAULT_NUM_EPISODES = 37
DEFAULT_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent
HF_HOME = Path("~/hfhome").expanduser()
logger = logging.getLogger(__name__)


def sample_episodes(total: int, n: int, seed: int) -> list[int]:
    random.seed(seed)
    return sorted(random.sample(range(total), n))


def load_selected_episodes(episode_file: str | Path) -> list[int]:
    payload = json.loads(Path(episode_file).read_text())
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) == 0:
        raise ValueError(f"Invalid episodes in {episode_file}")
    return [int(ep) for ep in episodes]


class LocalParquetFallbackDataset(torch.utils.data.Dataset):
    """
    直接使用本地 parquet 缓存训练。

    某些数据集在 `LeRobotDataset(..., episodes=[...])` 路径下会错误地只加载极少数 episode。
    这里退回到“加载当前本地 data/*.parquet 中的全部样本”，并基于本地 `episode_index/index`
    重新构建 action chunk 所需的 episode 边界。
    """

    def __init__(self, ds_meta: LeRobotDatasetMetadata, hf_dataset, delta_timestamps):
        self.repo_id = ds_meta.repo_id
        self.root = ds_meta.root
        self.meta = ds_meta
        self.features = ds_meta.features
        self.hf_dataset = hf_dataset
        self.image_transforms = None
        self.delta_timestamps = delta_timestamps
        self._episode_bounds = self._build_episode_bounds()
        self.loaded_episode_ids = sorted(self._episode_bounds)
        absolute_indices = [int(idx) for idx in self.hf_dataset["index"]]
        self._absolute_to_relative_idx = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(absolute_indices)}
        self.delta_indices = None
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, 1e-4)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)
        self.hf_dataset.set_transform(hf_transform_to_torch)

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.loaded_episode_ids)

    def _build_episode_bounds(self) -> dict[int, tuple[int, int]]:
        bounds: dict[int, list[int]] = defaultdict(list)
        for ep_idx, abs_idx in zip(self.hf_dataset["episode_index"], self.hf_dataset["index"], strict=True):
            bounds[int(ep_idx)].append(int(abs_idx))
        return {ep_idx: (min(indices), max(indices) + 1) for ep_idx, indices in bounds.items()}

    def _get_query_indices(self, abs_idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start, ep_end = self._episode_bounds[ep_idx]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        result = {}
        for key, q_idx in query_indices.items():
            relative_indices = [self._absolute_to_relative_idx[idx] for idx in q_idx]
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        abs_idx = int(item["index"].item())

        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding, **query_result}

        task_idx = int(item["task_index"].item())
        item["task"] = self.meta.tasks.iloc[task_idx].name
        return item


def load_local_cached_dataset(local_root: str | Path, ds_meta: LeRobotDatasetMetadata, delta_timestamps, log=None):
    data_root = Path(local_root) / "data"
    paths = sorted(str(path) for path in data_root.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {data_root}")
    hf_dataset = HFDataset.from_parquet(paths)
    dataset = LocalParquetFallbackDataset(ds_meta, hf_dataset, delta_timestamps)
    active_logger = log or logger
    active_logger.warning(
        "Using local cached parquet fallback: %s cached episodes, %s frames from %s",
        dataset.num_episodes,
        dataset.num_frames,
        data_root,
    )
    return dataset


def build_training_dataset(
    repo_id: str,
    local_root: str | Path,
    episodes: list[int],
    delta_timestamps,
    ds_meta: LeRobotDatasetMetadata | None = None,
    log=None,
):
    active_logger = log or logger
    ds_meta = ds_meta or LeRobotDatasetMetadata(repo_id, root=local_root)

    try:
        dataset = LeRobotDataset(
            repo_id,
            root=local_root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            force_cache_sync=False,
            download_videos=False,
        )
        loaded_episodes = {
            int(ep.item()) if hasattr(ep, "item") else int(ep) for ep in dataset.hf_dataset.unique("episode_index")
        }
        if loaded_episodes != set(episodes):
            active_logger.warning(
                "Episode-filtered load mismatch: requested=%s loaded=%s. Falling back to local cached parquet data.",
                len(episodes),
                len(loaded_episodes),
            )
            dataset = load_local_cached_dataset(local_root, ds_meta, delta_timestamps, log=active_logger)
    except Exception as exc:
        active_logger.warning("Episode-filtered load failed (%s). Falling back to local cached parquet data.", exc)
        dataset = load_local_cached_dataset(local_root, ds_meta, delta_timestamps, log=active_logger)

    return dataset


def main():
    os.environ["HF_HOME"] = str(HF_HOME)
    HF_HOME.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="从 LIBERO 数据集随机抽取 episode")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-download", action="store_true", help="只生成 episode 列表，不下载数据")
    args = parser.parse_args()

    selected = sample_episodes(TOTAL_EPISODES, args.num_episodes, args.seed)
    print(f"随机抽取 {args.num_episodes} 个 episode (seed={args.seed}, 总计 {TOTAL_EPISODES} episodes)")
    print(f"  HF_HOME 缓存目录: {HF_HOME}")
    print(f"  选中的 episode 索引: {selected}")

    episode_file = OUTPUT_DIR / "selected_episodes.json"
    with open(episode_file, "w") as f:
        json.dump({"seed": args.seed, "num_episodes": args.num_episodes, "episodes": selected}, f, indent=2)
    print(f"  Episode 列表已保存到: {episode_file}")

    if args.skip_download:
        print("已跳过数据下载。")
        return

    print(f"\n正在从 {REPO_ID} 下载选中的 {args.num_episodes} 个 episode ...")
    dataset = LeRobotDataset(repo_id=REPO_ID, episodes=selected, force_cache_sync=True)

    print(f"\n===== 数据集摘要 =====")
    print(f"  来源:         {REPO_ID}")
    print(f"  Episode 数:   {args.num_episodes}")
    print(f"  总帧数:       {len(dataset)}")
    print(f"  FPS:          {dataset.fps}")
    print(f"  Robot type:   {dataset.meta.robot_type}")
    print(f"  Features:     {list(dataset.meta.features.keys())}")

    sample = dataset[0]
    print(f"\n===== 单帧样本 (index=0) =====")
    for key, val in sample.items():
        if hasattr(val, "shape"):
            print(f"  {key:40s} shape={val.shape}  dtype={val.dtype}")
        else:
            print(f"  {key:40s} value={val}")

    print(f"\n数据准备完成。可通过以下方式在训练中使用:")
    ep_str = str(selected)
    print(f'  lerobot-train --dataset.repo_id={REPO_ID} --dataset.episodes=\'{ep_str}\' ...')


if __name__ == "__main__":
    main()
