"""
Convert physical-intelligence/libero from v2.0 to v3.0 format locally.

Downloads only the requested episodes from the v2.0 Hub dataset,
restructures files into the LeRobot v3.0 layout, and writes them locally
so that LeRobotDataset can load them.

Usage:
    python -m bt.sftpi05.convert_libero --max-episodes 2 --output-dir data/libero_v30
    python -m bt.sftpi05.convert_libero --episodes 0 1 5 --output-dir data/libero_v30
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import datasets
import jsonlines
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "physical-intelligence/libero"
V20_REVISION = "v2.0"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100

# v2.0 column name → v3.0 canonical name
COLUMN_RENAME = {
    "image": "observation.images.top",
    "wrist_image": "observation.images.wrist",
    "state": "observation.state",
    "actions": "action",
}

# v2.0 feature name → v3.0 feature definition
FEATURE_RENAME = {
    "image": ("observation.images.top", {
        "dtype": "image",
        "shape": [3, 256, 256],
        "names": ["channels", "height", "width"],
    }),
    "wrist_image": ("observation.images.wrist", {
        "dtype": "image",
        "shape": [3, 256, 256],
        "names": ["channels", "height", "width"],
    }),
    "state": ("observation.state", None),
    "actions": ("action", None),
}


def load_jsonlines(fpath: Path) -> list[dict]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def download_v20_meta(repo_id: str, revision: str) -> dict:
    """Download and parse v2.0 meta files from the Hub."""
    info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset", revision=revision)
    with open(info_path) as f:
        info = json.load(f)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])

    tasks_path = hf_hub_download(repo_id, "meta/tasks.jsonl", repo_type="dataset", revision=revision)
    tasks_list = load_jsonlines(Path(tasks_path))
    tasks = {t["task_index"]: t["task"] for t in sorted(tasks_list, key=lambda x: x["task_index"])}

    episodes_path = hf_hub_download(repo_id, "meta/episodes.jsonl", repo_type="dataset", revision=revision)
    episodes_list = load_jsonlines(Path(episodes_path))
    episodes = {e["episode_index"]: e for e in sorted(episodes_list, key=lambda x: x["episode_index"])}

    stats_path = hf_hub_download(repo_id, "meta/stats.json", repo_type="dataset", revision=revision)
    with open(stats_path) as f:
        stats = json.load(f)

    return {"info": info, "tasks": tasks, "episodes": episodes, "stats": stats}


def download_episode_parquet(repo_id: str, ep_idx: int, revision: str) -> Path:
    """Download a single episode parquet from v2.0 layout: data/chunk-XXX/episode_XXXXXX.parquet"""
    chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE
    filename = f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision))


def convert_libero(
    repo_id: str = REPO_ID,
    output_dir: str = "data/libero_v30",
    episodes: list[int] | None = None,
    max_episodes: int | None = None,
):
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download v2.0 meta
    logger.info("Downloading v2.0 metadata...")
    v20 = download_v20_meta(repo_id, V20_REVISION)
    v20_info = v20["info"]
    v20_tasks = v20["tasks"]
    v20_episodes = v20["episodes"]
    v20_stats = v20["stats"]

    total_available = v20_info["total_episodes"]

    # Resolve which episodes to convert
    if episodes is not None:
        ep_indices = sorted(episodes)
    elif max_episodes is not None:
        ep_indices = list(range(min(max_episodes, total_available)))
    else:
        ep_indices = list(range(total_available))

    logger.info(f"Converting {len(ep_indices)} episodes out of {total_available}")

    # Detect image keys from info features
    image_keys = [k for k, ft in v20_info["features"].items() if ft["dtype"] == "image"]
    logger.info(f"Image keys: {image_keys}")

    # 2. Download and concatenate episode parquets into v3.0 file-based layout
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    global_frame_idx = 0
    all_dfs = []

    for i, ep_idx in enumerate(ep_indices):
        logger.info(f"  Downloading episode {ep_idx} ({i+1}/{len(ep_indices)})...")
        ep_path = download_episode_parquet(repo_id, ep_idx, V20_REVISION)
        df = pd.read_parquet(ep_path)

        ep_len = len(df)
        ep_meta = v20_episodes[ep_idx]
        task_idx = ep_meta.get("task_index", 0)
        task_text = v20_tasks.get(task_idx, "manipulation task")

        # Reindex to global frame indices
        df["index"] = range(global_frame_idx, global_frame_idx + ep_len)
        df["episode_index"] = i  # re-map to sequential 0..N-1
        df["frame_index"] = range(ep_len)

        episodes_meta.append({
            "episode_index": i,
            "tasks": json.dumps([task_text]),
            "length": ep_len,
            "dataset_from_index": global_frame_idx,
            "dataset_to_index": global_frame_idx + ep_len,
            "data/chunk_index": 0,
            "data/file_index": 0,
        })

        global_frame_idx += ep_len
        all_dfs.append(df)

    total_frames = global_frame_idx

    # Concatenate and rename columns
    logger.info("Concatenating episode data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    rename_map = {old: new for old, new in COLUMN_RENAME.items() if old in combined_df.columns}
    combined_df = combined_df.rename(columns=rename_map)

    new_image_keys = [COLUMN_RENAME.get(k, k) for k in image_keys]

    # Write with proper HF image encoding
    if new_image_keys:
        schema_from_pandas = Features.from_arrow_schema(
            pd.io.parquet.get_engine("pyarrow").api.Schema.from_pandas(combined_df)
        )
        for key in new_image_keys:
            schema_from_pandas[key] = Image()
        hf_ds = Dataset.from_pandas(combined_df, features=schema_from_pandas)
        hf_ds.to_parquet(data_dir / "file-000.parquet")
    else:
        combined_df.to_parquet(data_dir / "file-000.parquet", index=False)

    # 3. Write v3.0 info.json (with renamed features)
    new_features = {}
    for old_key, ft in v20_info["features"].items():
        ft_copy = dict(ft)
        ft_copy["shape"] = list(ft_copy["shape"])
        if old_key in FEATURE_RENAME:
            new_key, override = FEATURE_RENAME[old_key]
            if override is not None:
                ft_copy = override
            new_features[new_key] = ft_copy
        else:
            new_features[old_key] = ft_copy

    v30_info = {
        "codebase_version": "v3.0",
        "robot_type": v20_info.get("robot_type", "unknown"),
        "total_episodes": len(ep_indices),
        "total_frames": total_frames,
        "total_tasks": len(set(v20_tasks.values())),
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": 200,
        "fps": int(v20_info["fps"]),
        "splits": {"train": f"0:{total_frames}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": new_features,
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(v30_info, f, indent=4)

    # 4. Write v3.0 tasks.parquet
    unique_tasks = sorted(set(v20_tasks.values()))
    tasks_df = pd.DataFrame(
        {"task_index": range(len(unique_tasks))},
        index=pd.Index(unique_tasks, name="task"),
    )
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    # 5. Write v3.0 episodes parquet
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    ep_ds = Dataset.from_dict({
        k: [r[k] for r in episodes_meta] for k in episodes_meta[0]
    })
    ep_ds.to_parquet(episodes_dir / "file-000.parquet")

    # 6. Write stats.json (rename keys to match v3.0 feature names)
    v30_stats = {}
    for old_key, val in v20_stats.items():
        if old_key in COLUMN_RENAME:
            v30_stats[COLUMN_RENAME[old_key]] = val
        else:
            v30_stats[old_key] = val
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(v30_stats, f, indent=4)

    logger.info(f"Conversion complete: {root}")
    logger.info(f"  Episodes: {len(ep_indices)}, Frames: {total_frames}")
    return str(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert physical-intelligence/libero v2.0 → v3.0")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--output-dir", default="data/libero_v30")
    p.add_argument("--episodes", type=int, nargs="+", default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()
    convert_libero(args.repo_id, args.output_dir, args.episodes, args.max_episodes)
