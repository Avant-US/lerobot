#!/usr/bin/env python3
"""
将 OpenPI 格式的 R1 Pro LeRobot v2.1 数据集转换为 LeRobot 标准 v3.0 格式。

转换步骤:
  Phase 0+1 (合并): 采样 episodes + 列名重命名（同时完成，避免双重 I/O）
  Phase 2: v2.1 → v3.0 格式升级（使用 LeRobot 官方脚本）
  Phase 2.5: Norm Stats — 精确计算 或 从已有文件导入
  Phase 3: 基本验证

Norm Stats 双模式:
  默认: 从转换后数据用 np.quantile() 精确计算分位数 (数学精确, float64)
  --norm-stats-path: 从已有 norm_stats.json 导入 q01/q99 (适用于与 OpenPI 精确对齐)

用法:
  # 模式 1: 精确计算 (默认)
  python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
      --input /mnt/r/share/lkx/pi/data/r1_pro_test_data \
      --output ./bt/pi05/alig/data/r1_pro_test_data_v30

  # 模式 2: 导入 OpenPI norm_stats.json
  python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
      --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
      --output ./bt/pi05/alig/data/r1_pro_chassis_v30 \
      --sample-episodes 10 \
      --norm-stats-path /mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Features, Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 列名映射: OpenPI 约定 → LeRobot 约定
# ──────────────────────────────────────────────────────────────────────
COLUMN_RENAME_MAP = {
    "head_rgb": "observation.images.head_rgb",
    "left_wrist_rgb": "observation.images.left_wrist_rgb",
    "right_wrist_rgb": "observation.images.right_wrist_rgb",
    "state": "observation.state",
    "actions": "action",
}

IMAGE_COLUMNS = [
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
]

# OpenPI norm_stats.json key → LeRobot stats.json key
NORM_STATS_KEY_MAP = {
    "state": "observation.state",
    "actions": "action",
}


def sample_episode_indices(total_episodes: int, n: int, seed: int) -> list[int]:
    """随机采样 n 个 episode 索引，返回排序后的列表。"""
    if n >= total_episodes:
        logger.info("采样数 %d >= 总 episodes %d, 使用全部", n, total_episodes)
        return list(range(total_episodes))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total_episodes), n))
    logger.info("采样 %d/%d episodes: %s", n, total_episodes, indices)
    return indices


def load_episodes_jsonl(path: Path) -> list[dict]:
    with jsonlines.open(path, "r") as reader:
        return sorted(list(reader), key=lambda x: x["episode_index"])


def load_episodes_stats_jsonl(path: Path) -> list[dict]:
    with jsonlines.open(path, "r") as reader:
        return sorted(list(reader), key=lambda x: x["episode_index"])


def load_tasks_jsonl(path: Path) -> list[dict]:
    with jsonlines.open(path, "r") as reader:
        return sorted(list(reader), key=lambda x: x["task_index"])


def write_parquet_with_image_features(df: pd.DataFrame, output_path: Path) -> None:
    """写入 parquet 文件，正确标记 Image 列。"""
    schema = pa.Schema.from_pandas(df)
    features = Features.from_arrow_schema(schema)
    for col in IMAGE_COLUMNS:
        if col in df.columns:
            features[col] = Image()
    schema = features.arrow_schema
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, schema=schema)


def phase01_sample_rename(
    input_dir: Path,
    output_dir: Path,
    sampled_indices: list[int] | None,
) -> None:
    """
    Phase 0+1 合并：采样 episodes 并同时重命名列。

    如果 sampled_indices 为 None，则处理全部 episodes（仅重命名列）。
    输出为有效的 v2.1 数据集（带 LeRobot 标准列名），可直接送入 Phase 2。
    """
    logger.info("Phase 0+1: 采样 + 列名重命名 → %s", output_dir)

    if output_dir.exists():
        logger.warning("输出目录已存在，清除: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # ── 加载源元数据 ─────────────────────────────────────────────
    with open(input_dir / "meta" / "info.json") as f:
        info = json.load(f)

    episodes_meta = load_episodes_jsonl(input_dir / "meta" / "episodes.jsonl")
    episodes_stats = load_episodes_stats_jsonl(input_dir / "meta" / "episodes_stats.jsonl")
    tasks = load_tasks_jsonl(input_dir / "meta" / "tasks.jsonl")

    total_episodes = info["total_episodes"]

    # 如果无采样，使用全部 episode 索引
    if sampled_indices is None:
        sampled_indices = list(range(total_episodes))

    # ── 构建 task 重映射 ────────────────────────────────────────
    # 收集采样 episodes 引用的所有 task
    sampled_ep_meta = [episodes_meta[i] for i in sampled_indices]
    referenced_tasks = set()
    for ep in sampled_ep_meta:
        for task_str in ep["tasks"]:
            for t in tasks:
                if t["task"] == task_str:
                    referenced_tasks.add(t["task_index"])

    # 创建旧 task_index → 新 task_index 的映射
    old_to_new_task = {}
    new_tasks = []
    for new_idx, old_idx in enumerate(sorted(referenced_tasks)):
        old_to_new_task[old_idx] = new_idx
        orig_task = next(t for t in tasks if t["task_index"] == old_idx)
        new_tasks.append({"task_index": new_idx, "task": orig_task["task"]})

    # ── 处理 parquet 数据文件 ──────────────────────────────────
    global_offset = 0
    new_episodes_meta = []
    new_episodes_stats = []

    for new_idx, orig_idx in enumerate(sampled_indices):
        src_path = input_dir / "data" / "chunk-000" / f"episode_{orig_idx:06d}.parquet"
        dst_path = output_dir / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"

        logger.info(
            "  [%d/%d] episode %d → %d (%s)",
            new_idx + 1,
            len(sampled_indices),
            orig_idx,
            new_idx,
            src_path.name,
        )

        df = pd.read_parquet(src_path)
        num_frames = len(df)

        # 重编号 episode 和全局索引
        df["episode_index"] = new_idx
        df["index"] = range(global_offset, global_offset + num_frames)
        # frame_index 是 episode 内的索引，保持不变

        # 重映射 task_index
        df["task_index"] = df["task_index"].map(old_to_new_task)

        # 列重命名 (Phase 1)
        df = df.rename(columns=COLUMN_RENAME_MAP)

        # 写入带 Image 标记的 parquet
        write_parquet_with_image_features(df, dst_path)

        # ── 更新 episode 元数据 ──────────────────────────────
        orig_ep = episodes_meta[orig_idx]
        new_episodes_meta.append({
            "episode_index": new_idx,
            "tasks": orig_ep["tasks"],
            "length": num_frames,
        })

        # ── 更新 episode 统计 ────────────────────────────────
        orig_stats = episodes_stats[orig_idx]
        new_stats_dict = {}
        for key, value in orig_stats["stats"].items():
            new_key = COLUMN_RENAME_MAP.get(key, key)
            new_stats_dict[new_key] = value

        # 更新 episode_index 统计
        new_stats_dict["episode_index"] = {
            "min": [new_idx],
            "max": [new_idx],
            "mean": [float(new_idx)],
            "std": [0.0],
            "count": [num_frames],
        }
        # 更新 index 统计
        new_stats_dict["index"] = {
            "min": [global_offset],
            "max": [global_offset + num_frames - 1],
            "mean": [float(global_offset + (num_frames - 1) / 2)],
            "std": [float(((num_frames**2 - 1) / 12) ** 0.5)] if num_frames > 1 else [0.0],
            "count": [num_frames],
        }

        new_episodes_stats.append({
            "episode_index": new_idx,
            "stats": new_stats_dict,
        })

        global_offset += num_frames

    # ── 写入元数据文件 ──────────────────────────────────────────
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json - 重命名 features + 更新计数
    new_features = {}
    for key, value in info["features"].items():
        new_key = COLUMN_RENAME_MAP.get(key, key)
        new_features[new_key] = value
    info["features"] = new_features
    info["total_episodes"] = len(sampled_indices)
    info["total_frames"] = global_offset
    info["total_tasks"] = len(new_tasks)
    info["splits"] = {"train": f"0:{len(sampled_indices)}"}
    # 保留 total_chunks 和 total_videos（Phase 2 需要删除它们）

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    logger.info("  写入 meta/info.json")

    # tasks.jsonl
    with jsonlines.open(meta_dir / "tasks.jsonl", "w") as writer:
        writer.write_all(new_tasks)
    logger.info("  写入 meta/tasks.jsonl (%d tasks)", len(new_tasks))

    # episodes.jsonl
    with jsonlines.open(meta_dir / "episodes.jsonl", "w") as writer:
        writer.write_all(new_episodes_meta)
    logger.info("  写入 meta/episodes.jsonl (%d episodes)", len(new_episodes_meta))

    # episodes_stats.jsonl
    with jsonlines.open(meta_dir / "episodes_stats.jsonl", "w") as writer:
        writer.write_all(new_episodes_stats)
    logger.info("  写入 meta/episodes_stats.jsonl")

    logger.info("Phase 0+1 完成: %d episodes, %d frames", len(sampled_indices), global_offset)


def phase2_convert_v21_to_v30(dataset_dir: Path) -> None:
    """Phase 2: v2.1 → v3.0 格式升级（调用 LeRobot 官方转换器）。"""
    logger.info("Phase 2: v2.1 → v3.0 格式升级: %s", dataset_dir)

    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    repo_id = f"local/{dataset_dir.name}"
    convert_dataset(
        repo_id=repo_id,
        root=str(dataset_dir),
        push_to_hub=False,
    )

    # 清理 _old 目录
    old_dir = dataset_dir.parent / f"{dataset_dir.name}_old"
    if old_dir.exists():
        logger.info("清理中间目录: %s", old_dir)
        shutil.rmtree(old_dir)

    logger.info("Phase 2 完成")


def phase2_5_compute_quantiles(dataset_dir: Path) -> None:
    """
    Phase 2.5: 计算分位数统计并更新 stats.json 和 episode metadata。

    官方 v2.1→v3.0 转换器的 aggregate_stats 只聚合源 episodes_stats.jsonl
    中已有的键 (min/max/mean/std/count)。v2.1 源数据没有分位数，
    所以转换后的 stats.json 也没有。

    此步骤从 v3.0 数据 parquet 直接计算分位数，更新：
    1. meta/stats.json — 添加全局 q01/q10/q50/q90/q99
    2. meta/episodes/ parquet — 添加 per-episode 分位数列
    """
    logger.info("Phase 2.5: 计算分位数统计: %s", dataset_dir)

    from lerobot.datasets.utils import load_info

    info = load_info(dataset_dir)
    features = info["features"]
    quantiles = [0.01, 0.10, 0.50, 0.90, 0.99]
    q_keys = [f"q{int(q * 100):02d}" for q in quantiles]

    # 找出需要计算分位数的数值特征
    numeric_features = []
    image_features = []
    for key, ft in features.items():
        if ft["dtype"] in ("float32", "float64", "int64"):
            if key not in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                numeric_features.append(key)
        elif ft["dtype"] == "image":
            image_features.append(key)

    logger.info("  数值特征: %s", numeric_features)
    logger.info("  图像特征: %s (将采样计算)", image_features)

    # 读取所有数据文件
    data_dir = dataset_dir / "data"
    data_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    logger.info("  数据文件数: %d", len(data_files))

    # 收集所有数值数据 (per-feature)
    all_data = {key: [] for key in numeric_features}
    for df_path in data_files:
        df = pd.read_parquet(df_path, columns=numeric_features)
        for key in numeric_features:
            col = df[key]
            # 将 Series of arrays 转为 2D numpy array
            if hasattr(col.iloc[0], '__len__'):
                arr = np.stack(col.values)
            else:
                arr = col.values.reshape(-1, 1)
            all_data[key].append(arr)

    # 计算全局分位数
    quantile_results = {}
    for key in numeric_features:
        concatenated = np.concatenate(all_data[key], axis=0).astype(np.float64)
        for q, q_key in zip(quantiles, q_keys):
            q_val = np.quantile(concatenated, q, axis=0)
            quantile_results[f"{key}/{q_key}"] = q_val

    # 更新 stats.json
    stats_path = dataset_dir / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    for key in numeric_features:
        if key not in stats:
            continue
        for q, q_key in zip(quantiles, q_keys):
            q_val = quantile_results[f"{key}/{q_key}"]
            stats[key][q_key] = q_val.tolist()

    # 为图像特征添加默认分位数 (基于 min/max 近似)
    for key in image_features:
        if key in stats:
            for q, q_key in zip(quantiles, q_keys):
                # 图像分位数：用 min/max 的线性插值近似
                s_min = np.array(stats[key]["min"])
                s_max = np.array(stats[key]["max"])
                s_mean = np.array(stats[key]["mean"])
                if q <= 0.01:
                    stats[key][q_key] = s_min.tolist()
                elif q >= 0.99:
                    stats[key][q_key] = s_max.tolist()
                else:
                    stats[key][q_key] = (s_min + q * (s_max - s_min)).tolist()

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info("  stats.json 更新完成 (添加分位数)")

    # 更新 episode metadata parquet — 添加 per-episode 分位数列
    episodes_dir = dataset_dir / "meta" / "episodes"
    ep_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    for ep_file in ep_files:
        ep_df = pd.read_parquet(ep_file)
        updated = False

        # 读取对应的数据，按 episode 计算分位数
        for key in numeric_features:
            for q, q_key in zip(quantiles, q_keys):
                col_name = f"stats/{key}/{q_key}"
                if col_name not in ep_df.columns:
                    # 使用全局分位数作为 per-episode 近似（精确计算需要按 episode 分组读取数据）
                    q_val = quantile_results[f"{key}/{q_key}"]
                    ep_df[col_name] = [q_val.tolist()] * len(ep_df)
                    updated = True

        for key in image_features:
            for q, q_key in zip(quantiles, q_keys):
                col_name = f"stats/{key}/{q_key}"
                if col_name not in ep_df.columns:
                    ep_df[col_name] = [stats[key][q_key]] * len(ep_df)
                    updated = True

        # 为 metadata 特征也添加分位数
        for meta_key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            for q, q_key in zip(quantiles, q_keys):
                col_name = f"stats/{meta_key}/{q_key}"
                if col_name not in ep_df.columns:
                    # 从 min/max 近似
                    min_col = f"stats/{meta_key}/min"
                    max_col = f"stats/{meta_key}/max"
                    if min_col in ep_df.columns and max_col in ep_df.columns:
                        approx = [
                            (np.array(mn) + q * (np.array(mx) - np.array(mn))).tolist()
                            for mn, mx in zip(ep_df[min_col], ep_df[max_col])
                        ]
                        ep_df[col_name] = approx
                        updated = True

        if updated:
            ep_df.to_parquet(ep_file, index=False)

    logger.info("Phase 2.5 完成: 分位数统计已添加 (精确计算)")


def phase2_5_import_norm_stats(dataset_dir: Path, norm_stats_path: Path) -> None:
    """
    Phase 2.5 (导入模式): 从已有的 norm_stats.json 导入 q01/q99 到 stats.json。

    适用于与 OpenPI 精确对齐的场景。OpenPI norm_stats.json 格式:
      {"norm_stats": {"state": {"mean":[], "std":[], "q01":[], "q99":[]},
                       "actions": {...}}}

    Key 映射: "state" → "observation.state", "actions" → "action"
    """
    logger.info("Phase 2.5: 导入已有 norm stats: %s", norm_stats_path)

    with open(norm_stats_path) as f:
        src = json.load(f)

    # 支持 OpenPI 格式 (嵌套在 "norm_stats" 下) 和直接格式
    if "norm_stats" in src:
        src = src["norm_stats"]

    stats_path = dataset_dir / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    imported_count = 0
    for src_key, dst_key in NORM_STATS_KEY_MAP.items():
        if src_key not in src:
            logger.warning("  norm stats 文件缺少 key: %s, 跳过", src_key)
            continue
        if dst_key not in stats:
            logger.warning("  stats.json 缺少 key: %s, 跳过", dst_key)
            continue

        src_stats = src[src_key]
        for q_key in ["q01", "q99"]:
            if q_key in src_stats:
                val = src_stats[q_key]
                # 确保是 list (可能来自 numpy 序列化)
                if not isinstance(val, list):
                    val = list(val)
                stats[dst_key][q_key] = val
                logger.info("  %s.%s → %s.%s (%d dims)",
                            src_key, q_key, dst_key, q_key, len(val))
                imported_count += 1

        # 导入其它可用的分位数 (q10/q50/q90)
        for q_key in ["q10", "q50", "q90"]:
            if q_key in src_stats:
                val = src_stats[q_key]
                if not isinstance(val, list):
                    val = list(val)
                stats[dst_key][q_key] = val

    if imported_count == 0:
        raise ValueError(f"未能从 {norm_stats_path} 导入任何分位数。"
                         f"文件 keys: {list(src.keys())}")

    # 补充 q10/q50/q90 如果源文件没有 (OpenPI 只有 q01/q99)
    quantiles_needed = [0.01, 0.10, 0.50, 0.90, 0.99]
    q_keys_needed = [f"q{int(q * 100):02d}" for q in quantiles_needed]
    for dst_key in NORM_STATS_KEY_MAP.values():
        if dst_key not in stats:
            continue
        feat_stats = stats[dst_key]
        q01 = np.array(feat_stats.get("q01", feat_stats.get("min", [0])))
        q99 = np.array(feat_stats.get("q99", feat_stats.get("max", [1])))
        for q, q_key in zip(quantiles_needed, q_keys_needed):
            if q_key not in feat_stats:
                # 线性插值近似
                feat_stats[q_key] = (q01 + q * (q99 - q01)).tolist()
                logger.info("  %s.%s 通过线性插值补充", dst_key, q_key)

    # 为图像和 metadata 特征添加默认分位数 (与 compute 模式相同)
    from lerobot.datasets.utils import load_info
    info = load_info(dataset_dir)
    features = info["features"]
    image_features = [k for k, ft in features.items() if ft["dtype"] == "image"]
    meta_keys = ("timestamp", "frame_index", "episode_index", "index", "task_index")

    for key in image_features:
        if key in stats:
            for q, q_key in zip(quantiles_needed, q_keys_needed):
                if q_key not in stats[key]:
                    s_min = np.array(stats[key]["min"])
                    s_max = np.array(stats[key]["max"])
                    if q <= 0.01:
                        stats[key][q_key] = s_min.tolist()
                    elif q >= 0.99:
                        stats[key][q_key] = s_max.tolist()
                    else:
                        stats[key][q_key] = (s_min + q * (s_max - s_min)).tolist()

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info("  stats.json 更新完成 (导入分位数)")

    # 更新 episode metadata parquet — 添加 per-episode 分位数列
    episodes_dir = dataset_dir / "meta" / "episodes"
    ep_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    for ep_file in ep_files:
        ep_df = pd.read_parquet(ep_file)
        updated = False

        # 数值特征 (observation.state, action)
        for dst_key in NORM_STATS_KEY_MAP.values():
            if dst_key not in stats:
                continue
            for q_key in q_keys_needed:
                col_name = f"stats/{dst_key}/{q_key}"
                if col_name not in ep_df.columns:
                    ep_df[col_name] = [stats[dst_key][q_key]] * len(ep_df)
                    updated = True

        # 图像特征
        for key in image_features:
            if key in stats:
                for q_key in q_keys_needed:
                    col_name = f"stats/{key}/{q_key}"
                    if col_name not in ep_df.columns:
                        ep_df[col_name] = [stats[key][q_key]] * len(ep_df)
                        updated = True

        # Metadata 特征
        for meta_key in meta_keys:
            for q, q_key in zip(quantiles_needed, q_keys_needed):
                col_name = f"stats/{meta_key}/{q_key}"
                if col_name not in ep_df.columns:
                    min_col = f"stats/{meta_key}/min"
                    max_col = f"stats/{meta_key}/max"
                    if min_col in ep_df.columns and max_col in ep_df.columns:
                        approx = [
                            (np.array(mn) + q * (np.array(mx) - np.array(mn))).tolist()
                            for mn, mx in zip(ep_df[min_col], ep_df[max_col])
                        ]
                        ep_df[col_name] = approx
                        updated = True

        if updated:
            ep_df.to_parquet(ep_file, index=False)

    logger.info("Phase 2.5 完成: 分位数统计已添加 (从 %s 导入)", norm_stats_path.name)


def phase3_verify(dataset_dir: Path) -> None:
    """Phase 3: 基本验证。"""
    logger.info("Phase 3: 验证数据集: %s", dataset_dir)

    # 1. 检查 info.json
    with open(dataset_dir / "meta" / "info.json") as f:
        info = json.load(f)

    assert info["codebase_version"] == "v3.0", f"版本不正确: {info['codebase_version']}"

    expected_keys = [
        "observation.images.head_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
        "observation.state",
        "action",
    ]
    for key in expected_keys:
        assert key in info["features"], f"缺少特征: {key}"
    logger.info("  info.json 验证通过 (v3.0, features 正确)")

    # 2. 检查 stats.json 存在且包含分位数
    stats_path = dataset_dir / "meta" / "stats.json"
    assert stats_path.exists(), "缺少 meta/stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    for feat_key in ["observation.state", "action"]:
        assert feat_key in stats, f"stats.json 缺少 {feat_key}"
        assert "q01" in stats[feat_key], f"stats.json {feat_key} 缺少 q01 分位数"
        assert "q99" in stats[feat_key], f"stats.json {feat_key} 缺少 q99 分位数"
    logger.info("  stats.json 验证通过 (含分位数)")

    # 3. 用 LeRobotDataset 加载
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = f"local/{dataset_dir.name}"
    ds = LeRobotDataset(repo_id, root=str(dataset_dir))
    sample = ds[0]
    logger.info("  LeRobotDataset 加载成功, 样本 keys: %s", list(sample.keys()))

    for key in expected_keys:
        assert key in sample, f"样本中缺少: {key}"

    # 检查 shapes
    state = sample["observation.state"]
    action = sample["action"]
    logger.info("  state shape: %s, action shape: %s", state.shape, action.shape)
    assert state.shape[-1] == 23, f"state 维度错误: {state.shape}"
    assert action.shape[-1] == 23, f"action 维度错误: {action.shape}"

    logger.info("Phase 3 验证通过")


def main():
    parser = argparse.ArgumentParser(
        description="将 OpenPI R1 Pro 数据集转换为 LeRobot v3.0 标准格式"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入数据集目录 (OpenPI v2.1 格式)")
    parser.add_argument("--output", type=Path, required=True, help="输出数据集目录 (LeRobot v3.0 格式)")
    parser.add_argument("--sample-episodes", type=int, default=None, help="随机采样的 episode 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip-v30", action="store_true", help="跳过 Phase 2 (v2.1→v3.0)")
    parser.add_argument("--skip-verify", action="store_true", help="跳过 Phase 3 验证")
    parser.add_argument(
        "--norm-stats-path", type=Path, default=None,
        help="导入已有的 norm stats 文件 (OpenPI norm_stats.json 格式)。"
             "若不指定，则从转换后数据用 np.quantile() 精确计算分位数。"
    )
    args = parser.parse_args()

    # 确定采样索引
    sampled_indices = None
    if args.sample_episodes is not None:
        with open(args.input / "meta" / "info.json") as f:
            total = json.load(f)["total_episodes"]
        sampled_indices = sample_episode_indices(total, args.sample_episodes, args.seed)

    # Phase 0+1: 采样 + 列名重命名
    phase01_sample_rename(args.input, args.output, sampled_indices)

    # Phase 2: v2.1 → v3.0
    if not args.skip_v30:
        phase2_convert_v21_to_v30(args.output)

        # Phase 2.5: Norm Stats — 导入 或 精确计算
        if args.norm_stats_path:
            phase2_5_import_norm_stats(args.output, args.norm_stats_path)
        else:
            phase2_5_compute_quantiles(args.output)

    # Phase 3: 验证
    if not args.skip_verify and not args.skip_v30:
        phase3_verify(args.output)

    logger.info("全部转换完成!")


if __name__ == "__main__":
    main()
